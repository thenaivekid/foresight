"""
vision_stream.py — the streaming VISION ENCODER (its own async component).

Unlike v2 (where this thread was CPU-only and the orchestrator did the encode),
here the encoder OWNS the vision GPU work and is a first-class concurrent
component:

  * It decodes the video and paces itself to REAL wall-clock time (so a 120 s
    clip takes ~120 s at speed 1.0) — true streaming, not batch.
  * It runs the ViT + multimodal projector (`backend.embed_frame`) and pushes
    PROJECTED visual tokens `[1, N, H]` (already in LLM space) onto `vis_q`.
    `vis_q` is the buffer "in between" the encoder and the orchestrator.
  * Its frame rate is controlled live by the orchestrator through
    `EncoderControl` — the proactive INPUT gate. Focus => encode more frames
    right now; boring => fewer. The orchestrator never has to wait for the
    encoder and vice-versa.
    This is the CONSUMER of the plan's `fps` field (see controller),
    gated on `cfg.plan_fps`; with it off the encoder walks at `cfg.fps` exactly
    as before. Honoring a controller-written value under lockstep requires the
    encoder to be synchronised with the controller too, not just the ingester —
    see `_await_controller` below for why, and for the wiring it needs.

The encode touches NO KV cache, so it runs concurrently with the orchestrator
and writer with zero locking.
"""
import queue
import time

import av
import torch

from util import log

try:
    from clip_pruner import (_get_clip_state as _clip_state_fn,
                             prune_frame_clip as _clip_fn)
except Exception:
    _clip_state_fn = None
    _clip_fn = None
try:
    from dsh_pruner import DSHState as _DSHState, prune_frame_dsh as _dsh_fn
except Exception:
    _DSHState = None
    _dsh_fn = None
try:
    from class_pruner import prune_frame as _prune_frame_fn
except ImportError:
    _prune_frame_fn = None


def _await_controller(cfg, clock, upto_vt, stop):
    """LOCKSTEP HANDSHAKE, encoder side. Block until the ingester has published
    `upto_vt` AND every controller tick due at <= that vt has COMPLETED.

    Why this exists: the ingester's own lockstep wait (input_ingester.py, the
    `while clock.get_next_check() <= vt` loop) bounds the INGESTER, not the
    ENCODER. The encoder is decoupled from it by `vis_q` (depth 8 in run.py,
    **256** in evaluation/foresight_adapter.py — the eval path) and paces itself
    against the WALL clock, so it routinely runs many frames ahead of the
    ingester. That does not matter while fps is a constant; the moment the
    encoder reads a controller-written value (`ctrl.get_fps()`), *which* fps a
    frame is sampled under becomes a function of how far the encoder happened to
    run ahead — i.e. a thread race, exactly the one `cfg.deterministic` exists to
    kill.

    WHY THE GUARANTEE DOES NOT DEPEND ON frame_q_size: the wait is placed before
    the fps read for the frame AFTER `last_emit`, and `clock` only reaches
    `last_emit` once the ingester has actually ingested that frame. So the
    encoder cannot emit frame k+1 until frame k has left the queue and been
    written to the cache — in-flight depth is <= 1 whatever `frame_q_size` says,
    and the 8-vs-256 difference between run.py and foresight_adapter.py stops
    mattering. Without this wait the depth-256 eval path is the WORST case, not
    an equivalent one.

    IT ALSO DEPENDS ON A CROSS-FILE ORDERING: within a tick the controller calls
    `ctrl.set_fps()` (controller.py, in the apply block) strictly before
    `clock.set_next_check()` (the last statement of the tick, deliberately after
    `borrow_end` — see its comment there). Publishing the release AFTER the write
    is what makes "the fps we read is the fps this tick decided" true rather than
    likely. If that ordering is ever reversed, this handshake degrades back into
    a race and the fps trace stops being reproducible.

    INVARIANT 1 (never observe the future) holds: we only ever wait on ticks that
    are due at a video time we have already passed. Nothing here can be answered
    only by a clip that exists in full.

    Returns True if the handshake completed, False if it gave up (stop, or the
    dead-peer timeout). The caller latches a False and stops waiting: a dead
    controller must cost ONE timeout, not one per frame.
    """
    timeout_s = getattr(cfg, "lockstep_timeout_s", 600.0)
    t_wait = time.time()
    while not stop.is_set():
        if clock.get() >= upto_vt and clock.get_next_check() > clock.get():
            return True
        time.sleep(0.002)
        if time.time() - t_wait > timeout_s:     # mirrors the ingester's dead-peer guard
            log("encoder", upto_vt,
                f"LOCKSTEP TIMEOUT after {timeout_s:g}s waiting on the controller — "
                f"it is not ticking. Abandoning the fps handshake for the REST of this "
                f"run and pinning fps to cfg.fps ({cfg.fps}); this run's fps trace is "
                f"not comparable.")
            return False
    return False


def _fps_source(cfg, clock):
    """Decide, ONCE per run, where the encoder's fps comes from. Returns
    ("plan"|"pinned", note) and logs the reason. Kept separate from the hot loop
    so the decision is stated in the log of every run instead of inferred."""
    if not getattr(cfg, "plan_fps", False):
        # DEFAULT / today's behaviour, bit-for-bit: the fixed base fps under
        # lockstep, the live steer only in the async demo.
        return ("pinned" if cfg.deterministic else "plan"), "plan_fps=False"
    if not cfg.deterministic:
        return "plan", "plan_fps=True, async"
    if clock is not None:
        return "plan", "plan_fps=True, lockstep handshake via clock"
    # plan_fps asked for, but the encoder has no way to observe controller
    # progress -> honoring it would be a live race. Refuse LOUDLY rather than
    # publish numbers that do not reproduce.
    #
    # The gate_mode="probe" arm reaches here BY DESIGN and the refusal is the
    # right answer there for a second, stronger reason: probe mode runs no
    # controller thread at all, so nothing ever calls ctrl.set_fps() and the
    # steer is inert by construction. Pinning is not merely equivalent, it is
    # SAFER — EncoderControl clamps its initial value into
    # [encoder_idle_fps, encoder_focus_fps], so on a config with
    # cfg.fps < encoder_idle_fps, honoring ctrl.get_fps() there would silently
    # change the probe arm's frame rate versus every previous probe run.
    if getattr(cfg, "plan_fps_require_lockstep", True):
        log("encoder", 0.0,
            "WARNING: PLAN_FPS INERT — plan_fps=True but encoder_thread got "
            "clock=None under cfg.deterministic, so the fps steer would be a "
            "wall-clock race (vis_q, depth 8/256, decouples the encoder from the "
            "ingester's lockstep). PINNING fps to cfg.fps: the plan's `fps` field "
            "has NO consumer in this run. Fix: pass `clock` into encoder_thread "
            "(run.py + evaluation/foresight_adapter.py), or set "
            "cfg.plan_fps_require_lockstep=False to accept non-reproducible runs.")
        return "pinned", "plan_fps=True but no clock (refused)"
    log("encoder", 0.0,
        "WARNING: plan_fps honored WITHOUT the lockstep handshake "
        "(plan_fps_require_lockstep=False) — this run is NOT bit-reproducible.")
    return "plan", "plan_fps=True, unsynchronised (explicitly allowed)"


def encoder_thread(cfg, backend, vis_q, ctrl, stop, prof=None, clock=None, pruner_state=None):
    container = av.open(cfg.video_path)
    vstream = container.streams.video[0]
    last_emit = -1e9
    wall_start = time.time()
    if prof is not None:
        prof.mark("source_start", 0.0)
    printed_tok = False                      # print real tokens/frame once
    # ---- THE INPUT GATE, made real ------------------------------------------
    # Was: `interval = 1.0 / (cfg.fps if cfg.deterministic else ctrl.get_fps())`.
    # cfg.deterministic is True for EVERY eval, so ctrl.get_fps() was never read:
    # set_fps() was a write with no reader and the steered encoder was, on
    # the benchmark, the constant 1.0 fps. This is that dead path's consumer.
    fps_source, fps_note = _fps_source(cfg, clock)
    log("encoder", 0.0, f"fps source={fps_source} ({fps_note}) "
                        f"base={cfg.fps} bounds=[{cfg.encoder_idle_fps},{cfg.encoder_focus_fps}]")
    eff_fps = cfg.fps if fps_source == "pinned" else ctrl.get_fps()
    lockstep_ok = True                       # latched False by a handshake timeout
    # ---- CLASS PRUNER (OFF by default) --------------
    _prune_enabled = (pruner_state is not None
                      and getattr(cfg, "prune_classes", False)
                      and _prune_frame_fn is not None)
    _embed_cache = {}
    _printed_prune = False
    _prune_retention = getattr(cfg, "prune_retention_floor", 0.30)
    _prune_lambda = getattr(cfg, "prune_ignore_lambda", 0.5)
    if _prune_enabled:
        log("encoder", 0.0, f"class pruner ARMED (retention_floor={_prune_retention}, "
                            f"ignore_lambda={_prune_lambda})")
    # ---- DSH TEMPORAL PRUNER (route 2: vision-vision, no text space) --------
    # Per-position EMA history + cosine novelty gate. Unlike the class pruner
    # this never touches the text embedding table, so probe_class_score's
    # negative result does not apply to it.
    _prune_mode = getattr(cfg, "prune_mode", "off")
    _dsh_state = None
    _printed_dsh = False
    if _prune_mode == "dsh" and _dsh_fn is not None:
        _dsh_state = _DSHState()
        log("encoder", 0.0,
            "DSH pruner ARMED (rule=%s keep_frac=%.2f alpha=%.2f tau=%.2f)" % (
                getattr(cfg, "dsh_rule", "rank"),
                getattr(cfg, "dsh_keep_frac", 0.30),
                getattr(cfg, "dsh_alpha", 0.10),
                getattr(cfg, "dsh_tau_temp", 0.90)))
    elif _prune_mode == "dsh":
        log("encoder", 0.0, "DSH pruner REQUESTED but dsh_pruner import failed -> OFF")
    # ---- CLIP-QDP PRUNER (route 1: query-conditioned, OpenCLIP-aligned) ----
    # probe_clip_qdp_steps.json: per-patch CLIP scores ARE query-sensitive on
    # real frames (spearman vs an unrelated query = -0.23), so unlike the Qwen
    # text-embedding route this space carries real image-text alignment.
    _clip_state = None
    _printed_clip = False
    _clip_query = (getattr(cfg, "instruction", "") or "")
    if _prune_mode == "clip" and _clip_fn is not None:
        _clip_state = _clip_state_fn(
            getattr(cfg, "clip_model_id", "openai/clip-vit-large-patch14"),
            str(getattr(backend, "device", "cuda")))
        log("encoder", 0.0,
            "CLIP-QDP pruner ARMED (floor=%.2f query=%r)" % (
                getattr(cfg, "clip_retention_floor", 0.30), _clip_query[:60]))
    elif _prune_mode == "clip":
        log("encoder", 0.0, "CLIP pruner REQUESTED but clip_pruner import failed -> OFF")

    # ---- RANDOM CONTROL + PERMUTATION DIAGNOSTIC ---------------------------
    _rand_keep = float(getattr(cfg, "random_keep_frac", 0.50))
    _permute = bool(getattr(cfg, "permute_frame_tokens", False))
    _printed_rand = False
    # CPU generator seeded from cfg.seed: the drop pattern must be identical
    # across arms, otherwise this control adds its own variance.
    _rng = torch.Generator(device="cpu")
    _rng.manual_seed(int(getattr(cfg, "seed", 0)))
    if _prune_mode == "random":
        log("encoder", 0.0,
            "RANDOM prune CONTROL ARMED (keep_frac=%.2f)" % _rand_keep)
    if _permute:
        log("encoder", 0.0,
            "PERMUTE DIAGNOSTIC ARMED: per-frame token order shuffled")

    for frame in container.decode(video=0):
        if stop.is_set():
            break
        vt = float(frame.pts * vstream.time_base)
        if vt > cfg.max_seconds:
            break
        if fps_source == "pinned":
            new_fps = cfg.fps
        else:
            if cfg.deterministic and clock is not None and lockstep_ok and last_emit > -1e8:
                # A dead controller must cost ONE timeout, not one per frame: once
                # the handshake gives up we stop waiting and stop steering, so the
                # rest of the run is the pinned walk rather than a half-synchronised
                # one that is neither reproducible nor fast.
                lockstep_ok = _await_controller(cfg, clock, last_emit, stop)
                if not lockstep_ok:
                    fps_source = "pinned"
            new_fps = cfg.fps if fps_source == "pinned" else ctrl.get_fps()
        interval = 1.0 / new_fps
        if vt - last_emit < interval:
            continue
        if new_fps != eff_fps:
            # ACCEPTANCE-TEST SIGNAL ("a run with fps honored and a run with
            # it pinned must come out different"). Logging only the CHANGES keeps
            # the log small while making the fps trajectory a diffable artifact:
            # if two arms produce the same trace, the steer did not take.
            log("encoder", vt, f"fps steer {eff_fps:.2f} -> {new_fps:.2f} "
                               f"(interval {interval:.3f}s)")
            eff_fps = new_fps
        last_emit = vt
        if prof is not None:
            prof.observe("encoder_fps", new_fps)
        if cfg.realtime:                         # pace to wall clock
            delay = (wall_start + vt / cfg.speed) - time.time()
            if delay > 0:
                time.sleep(delay)

        img = frame.to_image()
        t = time.time()
        embeds = backend.embed_frame(img)        # ViT + projector (GPU)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not printed_tok:                      # one-time: real tokens/frame
            ip = getattr(backend.processor, "image_processor", None)
            px_per_tok = (getattr(ip, "patch_size", 16) * getattr(ip, "merge_size", 2)) ** 2
            n_tok = embeds.shape[1]
            log("encoder", vt,
                f"FIRST FRAME: {n_tok} vision tokens/frame "
                f"(orig img {img.size[0]}x{img.size[1]}px, "
                f"~{px_per_tok}px/token, max_pixels={cfg.max_pixels})")
            printed_tok = True
        if prof is not None:
            prof.observe("vis_encode_ms", 1000 * (time.time() - t))
            prof.observe("vis_tokens_per_frame", embeds.shape[1])
            prof.incr("frames_emitted")
            prof.mark("frame_encoded", vt)

        # Same budget as the CLIP arm, chosen at random with order preserved.
        if _prune_mode == "random":
            _n_in = embeds.shape[1]
            _keep_n = max(1, int(round(_n_in * _rand_keep)))
            _idx = torch.randperm(_n_in, generator=_rng)[:_keep_n].sort().values
            embeds = embeds[:, _idx.to(embeds.device), :]
            if not _printed_rand:
                log("encoder", vt,
                    "FIRST RANDOM PRUNE: %d -> %d tokens (keep_rate=%.3f)" % (
                        _n_in, _keep_n, _keep_n / _n_in))
                _printed_rand = True
            if prof is not None:
                prof.observe("random_keep_rate", _keep_n / _n_in)

        if _permute:
            _pidx = torch.randperm(embeds.shape[1], generator=_rng)
            embeds = embeds[:, _pidx.to(embeds.device), :]

        # ---- CLASS-BASED TOKEN PRUNING --------
        # Between embed_frame and vis_q: upstream of the KV cache, causal,
        # per-frame. Invariant 2: one matmul, does not block perception.
        if _prune_enabled:
            _keep, _ignore, _ = pruner_state.get_classes()
            if _keep:
                embeds, _pstats = _prune_frame_fn(
                    embeds, _keep, _ignore, backend.embed_text,
                    _embed_cache, retention_floor=_prune_retention,
                    ignore_lambda=_prune_lambda)
                if _pstats is not None:
                    if not _printed_prune:
                        log("encoder", vt,
                            "FIRST PRUNE: %d -> %d tokens "
                            "(retention=%.2f, keep=%s, ignore=%s)" %
                            (_pstats["n_original"], _pstats["n_kept"],
                             _pstats["retention"], _pstats["keep_classes"],
                             _pstats["ignore_classes"]))
                        _printed_prune = True
                    if prof is not None:
                        prof.observe("prune_retention", _pstats["retention"])

        if _clip_state is not None:
            embeds, _cstats = _clip_fn(
                embeds, img, _clip_query, _clip_state, cfg,
                grid=getattr(backend, "last_grid_hw", None))
            if _cstats is not None:
                if not _printed_clip:
                    log("encoder", vt,
                        "FIRST CLIP PRUNE: %d -> %d tokens (keep_rate=%.3f, "
                        "grid=%s, inferred=%s)" % (
                            _cstats["n_in"], _cstats["n_out"],
                            _cstats["keep_rate"], _cstats["grid"],
                            _cstats.get("grid_inferred")))
                    _printed_clip = True
                if prof is not None:
                    prof.observe("clip_keep_rate", _cstats["keep_rate"])

        if _dsh_state is not None:
            embeds, _dstats = _dsh_fn(embeds, _dsh_state, cfg)
            if _dstats is not None:
                if not _printed_dsh:
                    log("encoder", vt,
                        "FIRST DSH PRUNE: %d -> %d tokens "
                        "(keep_rate=%.3f, rule=%s, mean_cos=%.4f)" % (
                            _dstats["n_in"], _dstats["n_out"],
                            _dstats["keep_rate"], _dstats.get("rule"),
                            _dstats.get("mean_cos", float("nan"))))
                    _printed_dsh = True
                if prof is not None:
                    prof.observe("dsh_keep_rate", _dstats["keep_rate"])

        if cfg.deterministic:
            vis_q.put((vt, embeds))              # block: process EVERY frame -> reproducible
        else:
            try:
                vis_q.put_nowait((vt, embeds))
            except queue.Full:
                try:
                    vis_q.get_nowait()           # drop oldest (bounded latency)
                    if prof is not None:
                        prof.incr("frames_dropped")
                except queue.Empty:
                    pass
                vis_q.put_nowait((vt, embeds))
    container.close()
    if prof is not None:
        prof.mark("source_end", max(0.0, last_emit))
    stop.set()
    log("encoder", cfg.max_seconds, "video stream ended")
