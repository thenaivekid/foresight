"""
input_ingester.py — the INPUT INGESTER.

It does NOT think, plan, or control. It only:
  1. seeds the shared KV cache with the system prompt (the pinned eviction sink);
  2. ingests the streaming visual tokens (from the encoder's vis_q) into that
     cache -- it is the SOLE writer of the primary cache (single-writer invariant
     => no write-write conflict), i.e. it builds the prefill/KV the controller reads;
  3. bounds memory (StreamingLLM eviction past the KV budget);
  4. publishes each frame's video time on the shared clock so the controller can
     self-pace ("check again in next_check_s of video time").

All proactivity (input gate = fps steering, output gate = emitting an answer) is
owned by the controller thread, which reads this cache via an MVCC snapshot.

DETERMINISTIC LOCKSTEP (cfg.deterministic): the clock is published only AFTER a
frame is fully in the cache, and the ingester then WAITS until the controller has
completed every tick due at <= vt (clock.get_next_check() > vt) before feeding
the next frame. Combined with the encoder's blocking put + fixed fps + greedy
decode, the whole walk is frame-indexed: every tick sees exactly frames [0..vt],
identically on every run — the async snapshot race is gone.
"""
import queue
import time

from util import log


def input_ingester_thread(cfg, mgr, vis_q, ctrl, stop, prof=None, clock=None,
                          feed_done=None, writer_q=None, evaluator=None):
    system_prompt = cfg.system_prompt.replace("{instruction}", cfg.instruction)
    # ICL IN THE SINK (cfg.icl_in_sink) — PRIVILEGE THE PRESENT.
    # The task ICL is CONSTANT, but it was being spliced fresh after the video
    # tokens on every tick, which put ~1400 tokens of instruction prose BETWEEN
    # the newest frame and the point of generation:
    #     [~100k vision tokens] .. [newest frame] -> [1400 tok of ICL] -> {"seen":"
    # so the last thing the model saw before answering was the manual, not the
    # video. Seeding it into the pinned eviction sink instead means the newest
    # frame is adjacent to the decision, and the ICL is prefilled ONCE per run
    # rather than re-prefilled every tick.
    if cfg.icl_in_sink and cfg.controller_prompt:
        system_prompt = system_prompt.rstrip() + "\n" + cfg.controller_prompt.rstrip() + "\n"
    sink = mgr.seed(system_prompt)
    log("ingester", 0.0, f"seeded cache, sink={sink} tokens, budget={cfg.kv_budget} "
                         f"(deterministic={cfg.deterministic}, gate_mode={cfg.gate_mode})")

    # PROBE-GATE (gate_mode="probe"): yes/no logit probe + hysteresis, fires writer_q.
    probe_mode = cfg.gate_mode == "probe" and writer_q is not None
    gate_q = cfg.goal_question.replace("{event}", cfg.event or cfg.instruction)
    n_frames = 0
    armed = True
    last_trigger_vt = -1e9

    lockstep_ok = True        # latched False by a handshake timeout
    try:
        while not stop.is_set() or not vis_q.empty():
            try:
                vt, embeds = vis_q.get(timeout=0.5)
            except queue.Empty:
                continue
            n_frames += 1
            if prof is not None:
                prof.observe("visq_depth", vis_q.qsize())
                prof.incr("frames_ingested")     # frames the ingester actually wrote to cache

            compact_request = mgr.pop_compaction_request()
            if compact_request is not None:
                target_len = max(mgr.sink + 1, int(cfg.kv_budget * cfg.compact_target_occupancy))
                keep_recent = max(1, int((target_len - mgr.sink) * cfg.compact_keep_recent_fraction))
                compact_started = time.perf_counter()
                removed = mgr.compact(target_len=target_len, keep_recent=keep_recent)
                compact_duration_ms = 1000.0 * (time.perf_counter() - compact_started)
                log("ingester.compact", vt,
                    f"requested_at={compact_request['vt']:.1f}s removed={removed} "
                    f"target={target_len} occ={compact_request['occupancy']:.3f} "
                    f"p={compact_request['confidence']:.3f} "
                    f"duration_ms={compact_duration_ms:.1f}")
                if prof is not None:
                    prof.incr("ingester_compact_calls")
                    prof.incr("ingester_compact_tokens_removed", removed)
                    prof.observe("ingester_compact_ms", compact_duration_ms)

            # ingest streaming visual tokens (optionally prefixed by a text timestamp
            # so the model has a real-time signal, not just token order)
            if cfg.timestamp_tokens:
                mgr.ingest(mgr.b.embed_text(cfg.timestamp_fmt.format(t=vt)))
            mgr.ingest(embeds)
            if prof is not None:
                prof.mark("frame_ingested", vt)

            dropped = mgr.evict()                # bounded memory
            if dropped:
                log("ingester.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

            # ---- PROBE-GATE: one forward pass, Schmitt/hysteresis edge, fire writer
            if probe_mode and n_frames % cfg.goal_gate_every == 0:
                share = mgr.probe(gate_q, "probe.goal")
                if cfg.gate_hysteresis:
                    if not armed and (share < cfg.gate_low_thr or
                                      (cfg.gate_rearm_s > 0
                                       and (vt - last_trigger_vt) >= cfg.gate_rearm_s)):
                        armed = True
                    fire = (armed and share >= cfg.gate_high_thr
                            and (vt - last_trigger_vt) > cfg.debounce_s)
                    if fire:
                        armed = False
                else:
                    fire = (share >= cfg.goal_threshold
                            and (vt - last_trigger_vt) > cfg.debounce_s)
                log("ingester.gate", vt, f"[{cfg.video_id or '?'}] share={share:.2f} "
                                         f"armed={armed} fire={fire}")
                if fire:
                    last_trigger_vt = vt
                    if probe_mode and evaluator is not None:
                        evaluator.record_trigger(vt, share)
                    if cfg.deterministic:
                        writer_q.put(vt)         # block, then WAIT for the write to
                        writer_q.join()          # finish -> snapshot is frame-indexed
                    else:
                        try:
                            writer_q.put_nowait(vt)
                        except queue.Full:
                            pass

            # publish vt only now: a controller tick at vt is guaranteed to see
            # every frame up to and including vt in the cache.
            if clock is not None:
                clock.set(vt)

            # LOCKSTEP: hold the next frame until every tick due at <= vt has
            # COMPLETED, so which frames each tick sees is frame-indexed, not a
            # thread race. (Guard against a dead controller with a long timeout.)
            if cfg.deterministic and clock is not None and lockstep_ok:
                t_wait = time.time()
                while clock.get_next_check() <= vt:
                    time.sleep(0.002)
                    if time.time() - t_wait > 600:
                        lockstep_ok = False
                        log("ingester", vt,
                            "LOCKSTEP TIMEOUT waiting on controller — abandoning "
                            "the handshake for the REST of this run. A dead "
                            "controller must cost ONE timeout, not one per frame.")
                        break
    finally:
        if feed_done is not None:
            feed_done.set()                      # tell the controller the feed is drained
    stop.set()
    log("ingester", 0.0, "input ingester stopped")
