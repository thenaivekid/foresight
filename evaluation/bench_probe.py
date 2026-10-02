"""bench_probe.py -- single-pass, multi-probe driver for StreamingBench and OVO-Bench.

WHY THIS IS NOT run_sample(). OmniPro is PROACTIVE: the controller decides when
to emit, so the pipeline must run in video time and the gate steers the
trajectory. StreamingBench (17 of 18 tasks) and OVO-Bench (BT + RVP + FAR) are
REACTIVE: the benchmark names the timestamp and asks a question there. There is
no emission decision, so:

  * the three-thread encoder/ingester/controller pipeline is unnecessary --
    nothing needs to self-pace,
  * realtime pacing is pure cost (it floors wall time at 1x video duration),
  * and one streaming pass can answer EVERY question in a video, because
    mgr.probe() / borrow_begin() splice onto the primary cache, read, truncate,
    and restore the logical clock -- leaving no trace.

So this module mirrors input_ingester_thread's per-frame work exactly (seed ->
[timestamp text] -> visual tokens -> audio -> evict -> compaction) in ONE thread,
and splices a probe whenever video time reaches the next probe timestamp.

FIDELITY: the ingest sequence below is a line-for-line mirror of
input_ingester.py. If that file changes, this must change with it. Everything
behavioural still comes from foresight/config.py.

Probe kinds:
  mcq    -- splice question + options, read argmax over the A/B/C/D token ids
  yesno  -- splice the dataset's own yes/no template, read yes_share AND p_hit
  count  -- splice the dataset's REC template, generate a short integer
"""
from __future__ import annotations

import dataclasses
import os
import sys

from utils import FORESIGHT_DIR, log

if FORESIGHT_DIR not in sys.path:
    sys.path.insert(0, FORESIGHT_DIR)

LETTERS = ["A", "B", "C", "D"]


def _word_ids(tok, words):
    """Mirror of backend._word_ids: first token id of each surface form."""
    out = []
    for w in words:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("", "0", "false", "False")


def build_cfg(base_cfg, sample, *, max_seconds, instruction):
    """Per-sample cfg for a REACTIVE probe run.

    deterministic=True / realtime=False on purpose: with probe timestamps given
    by the benchmark there is nothing to self-pace, and realtime would floor wall
    time at the video duration for no benefit. Every other knob still comes from
    config.py or the same OMNIPRO_* env overrides foresight_adapter.py honours, so
    an arm here is configured the same way an OmniPro arm is.
    """
    # The VL checkout's AsyncOmniConfig is vision-only and has NO `use_audio` /
    # `audio_window_s` field, while the omni checkout's does. dataclasses.replace
    # forwards every kwarg to __init__, so passing an absent field raises
    # TypeError: unexpected keyword argument -- which is exactly how all 47
    # ovo_far35 videos failed in 0.0 s each. Filter the overrides to the fields
    # this config actually declares, so one bench_probe.py serves both checkouts.
    _fields = {f.name for f in dataclasses.fields(base_cfg)}
    _over = dict(
        instruction=instruction,
        event=instruction,
        video_path=sample.video_path,
        video_id=sample.video_id,
        max_seconds=max_seconds,
        deterministic=True,
        realtime=False,
        decode_mode=os.environ.get("OMNIPRO_DECODE_MODE", base_cfg.decode_mode),
        model_id=os.environ.get("OMNIPRO_MODEL_ID", base_cfg.model_id),
        use_audio=_env_bool("OMNIPRO_USE_AUDIO", getattr(base_cfg, "use_audio", False)),
        audio_window_s=float(os.environ.get(
            "OMNIPRO_AUDIO_WINDOW_S", getattr(base_cfg, "audio_window_s", 4.0))),
        kv_budget=int(os.environ.get("OMNIPRO_KV_BUDGET", base_cfg.kv_budget)),
        prune_mode=os.environ.get("OMNIPRO_PRUNE_MODE", base_cfg.prune_mode),
        plan_compact=_env_bool("OMNIPRO_PLAN_COMPACT", base_cfg.plan_compact),
        # the controller ICL is proactive-emission scaffolding; a reactive probe
        # run must not carry it, or the model is primed to volunteer events.
        controller_prompt="",
        icl_in_sink=False,
    )
    dropped = sorted(set(_over) - _fields)
    if dropped:
        log(f"build_cfg: config {type(base_cfg).__name__} has no {dropped} "
            f"-- not overriding (this checkout does not declare them)", tag="bench")
    return dataclasses.replace(base_cfg,
                               **{k: v for k, v in _over.items() if k in _fields})


class ProbeRunner:
    """One streaming pass per video; probes spliced at benchmark timestamps."""

    def __init__(self, adapter):
        self.a = adapter
        self.torch = adapter.torch

    # ---- the probe readers --------------------------------------------------
    # All three use the borrow protocol (manager.borrow_begin/borrow_end), the
    # same splice-read-erase path the controller uses: append to the PRIMARY at
    # the pre-borrow (pos, phys), then erase every appended token and restore the
    # logical clock. borrow_begin also refuses a second concurrent reader, which
    # is the guard that turns a silent cache corruption into a named exception.
    def _step(self, mgr, embeds, state):
        """One forward at the borrow cursor. Mirrors controller.py's step()."""
        logits, cache = mgr.b.forward(embeds, mgr.cache,
                                      pos_start=state["pos"], phys_start=state["phys"],
                                      want_logits=True)
        # forward() may hand back a DIFFERENT cache object; the manager must keep
        # pointing at the one that actually grew or borrow_end() truncates a stale
        # object and leaves our tokens in the primary.
        mgr.cache = cache
        state["pos"] += int(embeds.shape[1])
        state["phys"] += int(embeds.shape[1])
        return logits

    def _letter_id_map(self, mgr):
        ids = getattr(self, "_letter_ids", None)
        if ids is None:
            ids = {L: _word_ids(mgr.b.tok, [L, f" {L}", f"{L}.", f" {L}."])
                   for L in LETTERS}
            missing = [L for L, v in ids.items() if not v]
            if missing:
                raise RuntimeError(f"no token id for option letters {missing}")
            self._letter_ids = ids
        return ids

    def _read_mcq(self, mgr, prompt, n_options):
        """Read the argmax over the option-letter token ids only.

        Constrained to the letters rather than free generation: the dataset
        scorer is `int(gt in response)` on a single letter, so an unconstrained
        decode that emitted prose would score wrong for a formatting reason
        rather than a perception one.
        """
        ids = self._letter_id_map(mgr)
        pos, phys = mgr.borrow_begin("bench.mcq")
        try:
            logits = self._step(mgr, mgr.b.embed_text(prompt),
                                {"pos": pos, "phys": phys})
            probs = self.torch.softmax(logits.float(), dim=-1)
            scores = {L: max(probs[i].item() for i in ids[L])
                      for L in LETTERS[:n_options]}
        finally:
            mgr.borrow_end()
        best = max(scores, key=scores.get)
        return best, scores

    def _read_yesno(self, mgr, prompt):
        """yes_share on the dataset's own yes/no template. mgr.probe() is exactly
        this splice-read-erase for a single forward, so we call it rather than
        re-implement the clock restore."""
        return mgr.probe(prompt, "bench.yesno")

    def _read_count(self, mgr, prompt, max_tokens=6):
        """Short greedy generation for REC's integer answer."""
        pos, phys = mgr.borrow_begin("bench.count")
        try:
            st = {"pos": pos, "phys": phys}
            embeds = mgr.b.embed_text(prompt)
            out_ids = []
            for _ in range(max_tokens):
                logits = self._step(mgr, embeds, st)
                nxt = int(self.torch.argmax(logits).item())
                if nxt == mgr.b.eos_id or nxt in mgr.b.newline_ids:
                    break
                out_ids.append(nxt)
                embeds = mgr.b.embed_token(nxt)
            return mgr.b.decode(out_ids)
        finally:
            mgr.borrow_end()

    # ---- the single streaming pass ----------------------------------------
    def run(self, sample, probes, *, instruction=None):
        """probes: list of dicts, each {t, kind, prompt, n_options?, meta}.
        Returns the same list with `response` / `scores` / `p_hit` filled in.

        Probes are sorted by t; the pass streams to the LAST probe time only.
        """
        from util import Profiler, seed_everything

        probes = sorted(probes, key=lambda p: float(p["t"]))
        if not probes:
            return []
        last_t = float(probes[-1]["t"])

        cfg = build_cfg(self.a.base_cfg, sample,
                        max_seconds=last_t + 1.0,
                        instruction=instruction or sample.question)
        seed_everything(cfg.seed, cfg.deterministic)
        prof = Profiler(enabled=True)
        mgr = self.a._KVCacheManager(self.a.backend, kv_budget=cfg.kv_budget, prof=prof,
                                     on_evict="rebase")

        audio = None
        if getattr(cfg, "use_audio", False) and cfg.video_path:
            from audio_stream import AudioIngestor
            audio = AudioIngestor(cfg, mgr.b, cfg.video_path)
            if not audio.ok:
                audio = None          # no track: behaves exactly vision-only
        # A required-audio task with a dead track is a broken run, not a
        # fallback -- the score would look like a perception failure.
        if any(p.get("audio_required") for p in probes) and audio is None:
            raise RuntimeError(
                f"audio_dependency=required but no audio track decoded for "
                f"{sample.video_path}")

        sink = mgr.seed(cfg.system_prompt.replace("{instruction}", cfg.instruction))
        log(f"[{sample.video_id}] seeded sink={sink} probes={len(probes)} "
                          f"to_t={last_t:.1f}s audio={'on' if audio else 'off'}", tag="bench")

        from dataset import iter_frames
        nxt = 0
        n_frames = 0
        last_vt = 0.0
        n_window = 0
        for vt, img in iter_frames(cfg.video_path, fps=cfg.fps,
                                   max_seconds=cfg.max_seconds):
            n_frames += 1
            last_vt = vt
            # --- mirror of input_ingester_thread's per-frame work -------------
            # SLIDING WINDOW (StreamingLLM), replacing the occupancy-triggered
            # compaction this file used to run.
            #
            # mgr.evict() keeps the pinned sink plus the most recent
            # (kv_budget - sink) tokens and drops the middle. With
            # on_evict="rebase" the survivors are re-rotated back onto a
            # contiguous window and next_pos is reset to the physical length, so
            # RoPE positions stay inside [0, kv_budget) for a video of ANY
            # length instead of climbing past the trained range.
            #
            # kv_budget is therefore a WINDOW, sized in config/env as N seconds
            # of video x tokens-per-second. Eviction is now routine and cheap
            # (one cat per layer) rather than an exceptional event, so there is
            # no occupancy threshold, no target occupancy and no thrash: the
            # cache sits pinned at the budget once the window fills.
            n_evicted = mgr.evict()
            if n_evicted:
                n_window += 1
                if n_window == 1 or n_window % 60 == 0:
                    log(f"[{sample.video_id}] window evict #{n_window} at "
                        f"vt={vt:.0f}s dropped={n_evicted} -> len={mgr._len()} "
                        f"next_pos={mgr.next_pos}", tag="bench")
            if cfg.timestamp_tokens:
                mgr.ingest(mgr.b.embed_text(cfg.timestamp_fmt.format(t=vt)))
            mgr.ingest(self.a.encoder_backend.embed_frame(img))
            if audio is not None:
                a = audio.maybe_embed(vt)
                if a is not None:
                    mgr.ingest(a)
            mgr.evict()
            # --- fire every probe whose timestamp this frame has reached ------
            while nxt < len(probes) and float(probes[nxt]["t"]) <= vt:
                p = probes[nxt]
                try:
                    if p["kind"] == "mcq":
                        resp, scores = self._read_mcq(mgr, p["prompt"],
                                                      p.get("n_options", 4))
                        p["response"], p["scores"] = resp, scores
                    elif p["kind"] == "yesno":
                        share = self._read_yesno(mgr, p["prompt"])
                        p["p_hit"] = share
                        p["response"] = "Yes" if share >= 0.5 else "No"
                    elif p["kind"] == "count":
                        p["response"] = self._read_count(mgr, p["prompt"])
                    else:
                        raise ValueError(f"unknown probe kind {p['kind']!r}")
                    p["probe_vt"] = vt
                except Exception as exc:        # one bad probe must not kill the video
                    p["response"] = None
                    p["error"] = f"{type(exc).__name__}: {exc}"
                    log(f"[{sample.video_id}] probe {nxt} FAILED: {exc}", tag="bench")
                nxt += 1

        # Probes past the last decoded frame still get answered, against the FULL
        # cache. Both benchmarks mean "the video up to time t", so a t at or past
        # the end means the model should see the whole video -- and t is often
        # only a fraction of a frame interval past the last pts (a probe at
        # 20.0 s on a clip whose last frame lands at 19.6 s). Marking these
        # unanswered would silently zero real probes and quietly shrink the
        # numerator while the denominator stayed put.
        if nxt < len(probes) and n_frames > 0:
            for p in probes[nxt:]:
                try:
                    if p["kind"] == "mcq":
                        resp, scores = self._read_mcq(mgr, p["prompt"],
                                                      p.get("n_options", 4))
                        p["response"], p["scores"] = resp, scores
                    elif p["kind"] == "yesno":
                        share = self._read_yesno(mgr, p["prompt"])
                        p["p_hit"] = share
                        p["response"] = "Yes" if share >= 0.5 else "No"
                    elif p["kind"] == "count":
                        p["response"] = self._read_count(mgr, p["prompt"])
                    p["probe_vt"] = last_vt
                    p["past_end"] = True        # flagged, not dropped
                except Exception as exc:
                    p["response"] = None
                    p["error"] = f"{type(exc).__name__}: {exc}"
            nxt = len(probes)
        while nxt < len(probes):                # no frames decoded at all
            probes[nxt]["response"] = None
            probes[nxt]["error"] = "no frames decoded from video"
            nxt += 1

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        log(f"[{sample.video_id}] done frames={n_frames} window_evicts={n_window} "
                             f"answered={sum(1 for p in probes if p.get('response') is not None)}"
                             f"/{len(probes)}", tag="bench")
        return probes
