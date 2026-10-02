"""
foresight_adapter.py — run the real foresight
pipeline per sample.

FIDELITY CONTRACT
-----------------
This driver does NOT reimplement the system. It spins up the exact three-thread
pipeline from run.py — encoder_thread -> input_ingester_thread -> controller_thread,
sharing one linear KV cache through KVCacheManager — and lets it run unmodified.
The controller self-paces in VIDEO time, so it MUST run realtime (batch would
fast-forward the whole clip before the controller could walk it).

The ONLY things set per-sample are: system_prompt (templated from the sample),
video_path, and max_seconds. Everything else keeps foresight's config defaults.

Emissions are captured via the evaluator hook the controller already invokes
(record_trigger / record_write) — no code in foresight is touched.
"""
from __future__ import annotations

import dataclasses
import queue
import sys
import threading

from utils import FORESIGHT_DIR, log

if FORESIGHT_DIR not in sys.path:
    sys.path.insert(0, FORESIGHT_DIR)


class CaptureEvaluator:
    """Collects (vt, share) triggers and (vt, text, latency) writes for any
    video. Thread-safe (the controller calls it from its own thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.triggers: list[tuple[float, float]] = []
        self.writes: list[tuple[float, str, float]] = []
        self.gates: list[tuple[float, float, float | None]] = []

    def record_gate(self, vt, share, thr=None):
        with self._lock:
            self.gates.append((float(vt), float(share), thr))

    def record_trigger(self, vt, share):
        with self._lock:
            self.triggers.append((float(vt), float(share)))

    def record_write(self, vt, text, wall_latency):
        with self._lock:
            self.writes.append((float(vt), str(text), float(wall_latency)))

    def emissions(self) -> list[dict]:
        """Merge triggers with their writer text (matched by trigger vt)."""
        with self._lock:
            writes_by_vt = {}
            for vt, text, lat in self.writes:
                writes_by_vt.setdefault(vt, (text, lat))
            out = []
            for vt, share in sorted(self.triggers):
                text, lat = writes_by_vt.get(vt, ("", None))
                out.append({"t_sec": vt, "share": share, "raw": text,
                            "writer_latency_s": lat})
            return out


class ForesightRunner:
    def __init__(self):
        from config import AsyncOmniConfig
        from backend import Qwen3VLBackend
        from manager import KVCacheManager
        from vision_stream import encoder_thread
        from input_ingester import input_ingester_thread
        from controller import controller_thread
        from util import EncoderControl, VideoClock

        self._KVCacheManager = KVCacheManager
        self._encoder_thread = encoder_thread
        self._ingester_thread = input_ingester_thread
        self._controller_thread = controller_thread
        self._EncoderControl = EncoderControl
        self._VideoClock = VideoClock

        # Everything (model_id, dtype, device(s), kv_budget, fps, prompts, sampling,
        # ...) comes from config.py — the single source of truth. The eval injects
        # only per-video data (instruction, video_path) in run_sample.
        self.base_cfg = AsyncOmniConfig()
        cfg = self.base_cfg

        import dataclasses as _dc
        import torch
        self.torch = torch

        # Optional multi-GPU placement (config.writer_device / encoder_device). When
        # set, the controller's generation and/or the encoder's vision run on their
        # OWN GPU so decode doesn't time-share with frame encode/ingest.
        has_enc = bool(cfg.encoder_device and cfg.encoder_device != cfg.device)
        has_ctrl = bool(cfg.writer_device and cfg.writer_device != cfg.device)
        primary_role = "language" if has_enc else "full"   # primary needs vision only if it also encodes
        log(f"loading backends: primary role={primary_role} on {cfg.device}"
            f"{' | controller on '+cfg.writer_device if has_ctrl else ''}"
            f"{' | encoder on '+cfg.encoder_device if has_enc else ''}", tag="runner")

        self.backend = Qwen3VLBackend(cfg, role=primary_role)          # ingester + shared cache
        self.encoder_backend = self.backend
        if has_enc:
            self.encoder_backend = Qwen3VLBackend(_dc.replace(cfg, device=cfg.encoder_device), role="vision")
        self.controller_backend = self.backend
        if has_ctrl:
            self.controller_backend = Qwen3VLBackend(_dc.replace(cfg, device=cfg.writer_device), role="language")
        log("backends loaded; async pipeline (3 threads, shared cache).",
            tag="runner")

    def run_sample(self, sample, *, max_seconds: float | None = None) -> dict:
        """Run the real async pipeline on one video; return captured emissions.

        The eval injects ONLY per-video DATA — the task `instruction` and the
        `video_path` (plus how much of the clip to run). EVERYTHING behavioural
        (prompts, fps, realtime, sampling, kv_budget, timestamps) comes from
        foresight/config.py, which is the single source of truth."""
        # pick the task-specific controller ICL prompt if we have one for this task,
        # else fall back to the generic controller_prompt (both live in config.py)
        controller_prompt = self.base_cfg.task_controller_prompts.get(
            sample.task, self.base_cfg.controller_prompt)
        # A/B switch for the head-to-head): OMNIPRO_GATE_MODE
        # overrides config.gate_mode so both systems run from one checkout.
        import os
        def env_bool(name, default):
            value = os.environ.get(name)
            if value is None:
                return default
            return value not in ("", "0", "false", "False")

        gate_mode = os.environ.get("OMNIPRO_GATE_MODE", self.base_cfg.gate_mode)
        # A/B switch for the schema-walk decoder: OMNIPRO_DECODE_MODE=schema|free
        # lets both decoders run from one checkout (same commit, same weights), so
        # the comparison is matched.
        decode_mode = os.environ.get("OMNIPRO_DECODE_MODE", self.base_cfg.decode_mode)
        # PER-TASK firing threshold. A single global 0.5 made two tasks score
        # time_f1 0.000 -- not a perception failure, the gate simply could never
        # fire. Fitted offline by auc.py; OMNIPRO_HIT_THRESHOLD forces one value
        # across all tasks (used to reproduce the old global-0.5 behaviour).
        hit_threshold = self.base_cfg.task_hit_thresholds.get(
            sample.task, self.base_cfg.hit_threshold)
        if os.environ.get("OMNIPRO_HIT_THRESHOLD"):
            hit_threshold = float(os.environ["OMNIPRO_HIT_THRESHOLD"])
        gate_strategy = self.base_cfg.task_gate_modes.get(
            sample.task, self.base_cfg.gate_strategy)
        # All three fitted knobs are ONE config; overriding only the threshold
        # silently runs a mode/refractory the fit never chose.
        if os.environ.get("OMNIPRO_GATE_STRATEGY"):
            gate_strategy = os.environ["OMNIPRO_GATE_STRATEGY"]
        refractory_s = self.base_cfg.task_refractory_s.get(
            sample.task, self.base_cfg.debounce_s)
        if os.environ.get("OMNIPRO_REFRACTORY_S"):
            refractory_s = float(os.environ["OMNIPRO_REFRACTORY_S"])
        # A/B switch for the KV-cache read path: OMNIPRO_CACHE_MODE=snapshot|inplace.
        # "inplace" drops the per-tick deep copy of the cache. It must produce
        # BYTE-IDENTICAL output to "snapshot" -- see , which asserts
        # exactly that -- so this is a memory/latency switch, never an accuracy one.
        cache_mode = os.environ.get("OMNIPRO_CACHE_MODE",
                                    self.base_cfg.controller_cache_mode)
        # A/B switch for the sampler: OMNIPRO_WRITER_GREEDY=1 forces argmax.
        # Seeded multinomial sampling is the leading suspect for the run-to-run
        # divergence observed between identical-config repeat runs: a
        # near-tie resolved differently by bf16 kernel jitter changes one token,
        # and the stream diverges from there. This knob is what tests that.
        _g = os.environ.get("OMNIPRO_WRITER_GREEDY")
        writer_greedy = (_g not in (None, "", "0", "false", "False")
                         if _g is not None else self.base_cfg.writer_greedy)
        _async = os.environ.get("OMNIPRO_ASYNC")
        deterministic = (not env_bool("OMNIPRO_ASYNC", True) if _async is not None
                 else self.base_cfg.deterministic)
        cfg = dataclasses.replace(
            self.base_cfg,
            decode_mode=decode_mode,
            hit_threshold=hit_threshold,
            gate_strategy=gate_strategy,
            debounce_s=refractory_s,
            controller_cache_mode=cache_mode,
            writer_greedy=writer_greedy,
            deterministic=deterministic,
            # signal-quality sweep knobs (all change the forward pass, so none can
            # be screened offline the way gate strategies can)
            seen_mode=os.environ.get("OMNIPRO_SEEN_MODE", self.base_cfg.seen_mode),
            schema_max_seen_tokens=int(os.environ.get(
                "OMNIPRO_SEEN_TOKENS", self.base_cfg.schema_max_seen_tokens)),
            # ---- token pruning arm (ablation): off | dsh | clip -----------
            prune_mode=os.environ.get("OMNIPRO_PRUNE_MODE",
                                      self.base_cfg.prune_mode),
            dsh_tau_temp=float(os.environ.get(
                "OMNIPRO_DSH_TAU", self.base_cfg.dsh_tau_temp)),
            dsh_alpha=float(os.environ.get(
                "OMNIPRO_DSH_ALPHA", self.base_cfg.dsh_alpha)),
            dsh_retention_floor=float(os.environ.get(
                "OMNIPRO_DSH_FLOOR", self.base_cfg.dsh_retention_floor)),
            dsh_rule=os.environ.get("OMNIPRO_DSH_RULE", self.base_cfg.dsh_rule),
            dsh_keep_frac=float(os.environ.get(
                "OMNIPRO_DSH_KEEP_FRAC", self.base_cfg.dsh_keep_frac)),
            clip_retention_floor=float(os.environ.get(
                "OMNIPRO_CLIP_FLOOR", self.base_cfg.clip_retention_floor)),
            random_keep_frac=float(os.environ.get(
                "OMNIPRO_RANDOM_KEEP", self.base_cfg.random_keep_frac)),
            permute_frame_tokens=env_bool("OMNIPRO_PERMUTE_TOKENS",
                                          self.base_cfg.permute_frame_tokens),
            plan_compact=env_bool("OMNIPRO_PLAN_COMPACT", self.base_cfg.plan_compact),
            compact_occupancy_thr=float(os.environ.get(
                "OMNIPRO_COMPACT_OCCUPANCY_THR", self.base_cfg.compact_occupancy_thr)),
            compact_p_thr=float(os.environ.get(
                "OMNIPRO_COMPACT_P_THR", self.base_cfg.compact_p_thr)),
            compact_novelty_max=float(os.environ.get(
                "OMNIPRO_COMPACT_NOVELTY_MAX", self.base_cfg.compact_novelty_max)),
            plan_classes=env_bool("OMNIPRO_PLAN_CLASSES", self.base_cfg.plan_classes),
            prune_classes=env_bool("OMNIPRO_PRUNE_CLASSES", self.base_cfg.prune_classes),
            ev0_dedup=env_bool("OMNIPRO_EV0_DEDUP", self.base_cfg.ev0_dedup),
            event_dedup_window_s=float(os.environ.get(
                "OMNIPRO_EVENT_DEDUP_WINDOW_S", self.base_cfg.event_dedup_window_s)),
            event_identity_in_prompt=env_bool(
                "OMNIPRO_EVENT_IDENTITY_PROMPT", self.base_cfg.event_identity_in_prompt),
            kv_budget=int(os.environ.get("OMNIPRO_KV_BUDGET", self.base_cfg.kv_budget)),
            instruction=sample.question,
            task=sample.task,
            event=(sample.event or sample.question),
            video_path=sample.video_path,
            video_id=sample.video_id,
            max_seconds=(max_seconds if max_seconds else 10 ** 9),
            controller_prompt=controller_prompt,
            writer_prompt=self.base_cfg.task_writer_prompts.get(
                sample.task, self.base_cfg.writer_prompt),
            gate_mode=gate_mode,
            # deterministic => frame-indexed lockstep walk: no wall-clock pacing
            # (batch), blocking queues, ingester waits on each due tick. This is
            # what makes runs bit-reproducible (the async snapshot race is gone).
            realtime=(self.base_cfg.realtime and not deterministic),
        )

        # re-seed per sample so every video starts from the same RNG state and
        # (deterministic=True) CUDA kernels are deterministic
        from util import seed_everything
        seed_everything(cfg.seed, cfg.deterministic)

        # TELEMETRY: a Profiler collects per-run counts + timings (encoder frames
        # emitted/dropped, ingester frames ingested, per-op GPU times, controller
        # gen time + tokens). Passed to the manager + all three threads; summary
        # printed below into the run log.
        from util import Profiler
        prof = Profiler(enabled=True)
        prof.capture_timeline = env_bool("OMNIPRO_CAPTURE_DIAGNOSTICS", False)
        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=prof)
        in_q = queue.Queue(maxsize=max(256, cfg.frame_q_size))
        stop = threading.Event()
        feed_done = threading.Event()   # ingester -> controller: stream fully drained
        ctrl = self._EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
        clock = self._VideoClock()
        ev = CaptureEvaluator()
        pruner_state = None
        if cfg.prune_classes:
            from class_pruner import ClassPrunerState
            pruner_state = ClassPrunerState()
        ctrl._pruner_state = pruner_state

        if cfg.gate_mode == "probe":
            # PROBE-GATE head-to-head arm: yes/no logit gate inside the ingester
            # fires a separate writer. No controller/clock; in deterministic mode
            # the ingester blocks on writer_q.join() -> frame-indexed, reproducible.
            from writer import writer_thread
            writer_q = queue.Queue(maxsize=4)
            threads = [
                # No `clock` here deliberately: probe mode has no controller, so nothing
                # ever calls ctrl.set_fps() and plan-fps steering is inert by construction.
                # vision_stream logs PLAN_FPS INERT and pins fps, which is correct here.
                threading.Thread(target=self._encoder_thread,
                                 args=(cfg, self.encoder_backend, in_q, ctrl, stop, prof),
                                 kwargs={"pruner_state": pruner_state},
                                 name="encoder", daemon=True),
                threading.Thread(target=self._ingester_thread,
                                 args=(cfg, mgr, in_q, ctrl, stop, prof, None, feed_done,
                                       writer_q, ev),
                                 name="input_ingester", daemon=True),
                threading.Thread(target=writer_thread,
                                 args=(cfg, mgr, writer_q, stop, prof, ev,
                                       self.controller_backend, feed_done),
                                 name="writer", daemon=True),
            ]
        else:
            threads = [
                threading.Thread(target=self._encoder_thread,
                                 args=(cfg, self.encoder_backend, in_q, ctrl, stop, prof),
                                 kwargs={"clock": clock, "pruner_state": pruner_state},
                                 name="encoder", daemon=True),
                threading.Thread(target=self._ingester_thread,
                                 args=(cfg, mgr, in_q, ctrl, stop, prof, clock, feed_done),
                                 name="input_ingester", daemon=True),
                threading.Thread(target=self._controller_thread,
                                 args=(cfg, mgr, ctrl, clock, stop, prof, ev,
                                       self.controller_backend, feed_done),
                                 name="controller", daemon=True),
            ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        # dump the telemetry for THIS video into the run log
        log(f"TELEMETRY {sample.video_id}\n{prof.summary()}", tag="prof")

        emits = ev.emissions()
        result = {
            "id": sample.id, "task": sample.task, "video_id": sample.video_id,
            "question": sample.question, "event": sample.event,
            "audio_dependency": sample.audio_dependency,
            "ground_truth": sample.ground_truth,
            "predictions": emits,
            "n_triggers": len(ev.triggers), "n_writes": len(ev.writes),
            "n_gates": len(ev.gates),
            "eval_mode": "online", "realtime": cfg.realtime,
        }
        if env_bool("OMNIPRO_CAPTURE_DIAGNOSTICS", False):
            result["gate_trace"] = [
                {"t_sec": vt, "p_hit": p, "threshold": thr}
                for vt, p, thr in ev.gates
            ]
            result["effective_config"] = dataclasses.asdict(cfg)
            result["telemetry"] = prof.summary()
            result["wall_timeline"] = prof.timeline
        return result
