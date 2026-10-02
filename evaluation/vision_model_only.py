#!/usr/bin/env python3
"""Unpruned, untuned vision diagnostic; uses the real three-thread pipeline.

prepare selects 18 distinct videos without consulting GT or predictions.
run executes one fixed shard; plot reads saved observations, never video.
No API judge, threshold fit, replay, or truncated clips are involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ALLOWED_AUDIO = {"none": {"none"}, "helpful": {"helpful"},
                 "none_helpful": {"none", "helpful"}}


def validate_input_manifest(root, shard):
    """Fail before model loading if the supplied shard violates the experiment."""
    manifest = json.loads((root / "manifest.json").read_text())
    audio = manifest.get("audio_filter", "none")
    if audio not in ALLOWED_AUDIO:
        raise ValueError("Vision study accepts only none/helpful data, never audio=required")
    records = json.loads((root / f"shard{shard}.json").read_text())
    ids = [r["id"] for r in records]
    expected = manifest["ids"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Invalid experiment sample IDs")
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected):
        raise ValueError("Shard IDs are duplicated or outside the manifest")
    if any(r.get("audio_dependency") not in ALLOWED_AUDIO[audio] for r in records):
        raise ValueError("Shard includes forbidden audio-dependency samples")
    return manifest, records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def model_only_config(cfg):
    """Last-word overrides: no fitted output filters or input/cache pruning."""
    return dataclasses.replace(
        cfg, model_id="Qwen/Qwen3-VL-8B-Instruct", task_hit_thresholds={},
        task_gate_modes={}, task_refractory_s={}, hit_threshold=0.5,
        gate_strategy="level", debounce_s=0.0, ev0_dedup=False,
        prune_mode="off", prune_classes=False, plan_classes=False,
        permute_frame_tokens=False, plan_compact=False, kv_budget=262144,
        controller_cache_mode="snapshot", gate_mode="controller",
        decode_mode="schema", deterministic=False, realtime=True, speed=1.0,
    )


def prepare(args):
    from dataset import ALL_TASKS
    source = Path(args.benchmark)
    raw = json.loads(source.read_text())
    chosen, used = [], set()
    for task in sorted(ALL_TASKS):
        eligible = sorted(
            (r for r in raw if r["task"] == task
             and r.get("audio_dependency") in ALLOWED_AUDIO[args.audio]
             and (Path(args.dataset) / r["video_path"]).is_file()),
            key=lambda r: (float(r["duration"]), r["id"]),
        )
        picked = []
        for record in eligible:
            if record["video_id"] in used:
                continue
            picked.append(record)
            used.add(record["video_id"])
            if len(picked) == 2:
                break
        if len(picked) != 2:
            raise ValueError(f"Not enough distinct audio={args.audio} videos for {task}")
        chosen.extend(picked)
    assert len(chosen) == len(used) == 18
    if (args.out / "manifest.json").exists():
        raise FileExistsError("Manifest already exists; reuse it or choose a new output directory")
    shards, loads = [[] for _ in range(4)], [0.0] * 4
    for record in sorted(chosen, key=lambda r: -float(r["duration"])):
        gpu = min(range(4), key=lambda i: loads[i])
        shards[gpu].append(record)
        loads[gpu] += float(record["duration"])
    write_json(args.out / "benchmark.json", chosen)
    for gpu, records in enumerate(shards):
        write_json(args.out / f"shard{gpu}.json", records)
    write_json(args.out / "manifest.json", {
        "name": "vision_model_only_18", "selection":
        f"Task alphabetical order; two shortest existing audio={args.audio} videos, "
        "excluding video IDs already selected for another task; no GT/score selection.",
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset": args.dataset, "ids": [r["id"] for r in chosen],
        "audio_filter": args.audio,
        "shard_duration_s": loads, "n_samples": 18, "n_unique_videos": 18,
        "interpretation": "Qualitative short-clip diagnostic, not representative benchmark performance.",
    })
    for r in chosen:
        print(r["task"], r["video_id"], r["duration"])
    print("shard duration totals", loads)


def run(args):
    run_started = time.monotonic()
    if args.wall_budget_s < 0:
        raise ValueError("Wall budget must be nonnegative")
    # Do not allow inherited experiment switches to silently change this profile.
    for name in list(os.environ):
        if name.startswith("OMNIPRO_"):
            del os.environ[name]
    os.environ["FORESIGHT_DIR"] = str(REPO / "foresight")
    os.environ["OMNIPRO_CAPTURE_DIAGNOSTICS"] = "1"
    sys.path.insert(0, str(REPO / "foresight"))
    from dataset import load_samples
    from foresight_adapter import ForesightRunner
    from metrics import match_emits_to_gt
    manifest, shard_records = validate_input_manifest(args.out, args.shard)
    samples = load_samples(audio=manifest.get("audio_filter", "none"),
                           benchmark_json=str(args.out / f"shard{args.shard}.json"),
                           dataset_dir=manifest["dataset"])
    if {s.id for s in samples} != {r["id"] for r in shard_records}:
        raise ValueError("Dataset loader dropped declared samples; check video files and filters")
    if not samples:
        print("Declared empty shard; no model loaded", flush=True)
        return
    dest = args.out / f"g{args.shard}"
    dest.mkdir(exist_ok=True)
    pred_path = dest / "online_pred.jsonl"
    previous = [json.loads(s) for s in pred_path.read_text().splitlines()] if pred_path.exists() else []
    if any(bool(r.get("realtime")) != args.realtime or
           bool(r["effective_config"]["deterministic"]) == args.realtime for r in previous):
        raise ValueError("Existing predictions use a different execution mode; choose a fresh output directory")
    if any(r.get("experiment_arm", "raw") != args.arm for r in previous):
        raise ValueError("Existing predictions use a different ablation arm")
    from config import AsyncOmniConfig
    from trigger_ablation_profiles import configure_arm
    profile = configure_arm(model_only_config(AsyncOmniConfig()), args.arm)
    profile = dataclasses.replace(profile, deterministic=not args.realtime, realtime=args.realtime)
    profile_hash = hashlib.sha256(json.dumps(dataclasses.asdict(profile), sort_keys=True).encode()).hexdigest()
    if any(r.get("profile_sha256") != profile_hash for r in previous):
        raise ValueError("Saved profile is different or unversioned; do not mix old runs with a changed configuration")
    if (len(previous) != len({r['id'] for r in previous}) or
            not {r['id'] for r in previous} <= {s.id for s in samples} or
            any(not r.get('gate_trace') for r in previous)):
        raise ValueError("Existing shard predictions are incomplete, duplicated, or belong to a different shard")
    done = {r["id"] for r in previous}
    if all(s.id in done for s in samples):
        print("Shard already complete", flush=True)
        return
    runner = ForesightRunner()
    runner.base_cfg = profile
    write_json(dest / "profile.json", dataclasses.asdict(runner.base_cfg))
    errors = []
    old_hook = threading.excepthook

    def capture_error(event):
        errors.append(f"{event.thread.name}: {event.exc_type.__name__}: {event.exc_value}")
        old_hook(event)

    threading.excepthook = capture_error
    try:
        for i, sample in enumerate(samples):
            if sample.id in done:
                continue
            if args.wall_budget_s:
                remaining = args.wall_budget_s - (time.monotonic() - run_started)
                reserve = sample.duration + max(120.0, 0.15 * sample.duration)
                if remaining < reserve:
                    print(f"BUDGET STOP before {sample.id}; complete samples preserved, "
                          f"remaining={remaining:.1f}s reserve={reserve:.1f}s", flush=True)
                    return
            errors.clear()
            print(f"===== [{i+1}/{len(samples)}] {sample.id}", flush=True)
            started = time.time()
            result = runner.run_sample(sample)
            if errors:
                raise RuntimeError("Pipeline thread failure: " + "; ".join(errors))
            if not result["gate_trace"]:
                raise RuntimeError("No controller observations; refuse to save as completed")
            result["wall_s"] = time.time() - started
            result["execution_host"] = os.uname().nodename
            result["experiment_arm"] = args.arm
            result["profile_sha256"] = profile_hash
            result["audio_filter"] = manifest.get("audio_filter", "none")
            result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
            result["protocol"] = ("unpruned_model_bool_level_no_ev0_no_refractory"
                                  if args.arm == "raw" else f"unpruned_trigger_ablation_{args.arm}")
            if args.realtime:
                result["protocol"] += "_wallclock_async"
            matches, fp, fn = match_emits_to_gt(
                [p["t_sec"] for p in result["predictions"]], sample.gt_times, 3.0)
            result["timing"] = {"tp": len(matches), "fp": len(fp), "fn": len(fn)}
            result["content_scoring"] = "not performed; checkmarks indicate timing only"
            with pred_path.open("a") as stream:
                stream.write(json.dumps(result, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(f"COMPLETED {sample.id} ticks={len(result['gate_trace'])} "
                  f"timing={result['timing']} wall_s={result['wall_s']:.1f}", flush=True)
    finally:
        threading.excepthook = old_hook


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from metrics import match_emits_to_gt

    records = [json.loads(line) for path in sorted(args.out.glob("g*/online_pred.jsonl"))
               for line in path.read_text().splitlines()]
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert len(records) == 18 and len({r["id"] for r in records}) == 18, "Need all 18 completed samples"
    assert {r["id"] for r in records} == set(manifest["ids"])
    assert all(r["audio_dependency"] in ALLOWED_AUDIO[manifest.get("audio_filter", "none")] for r in records)
    assert set(Counter(r["task"] for r in records).values()) == {2}
    records.sort(key=lambda r: (r["task"], r["video_id"]))
    fig, axes = plt.subplots(9, 2, figsize=(20, 27))
    handles = [
        Line2D([], [], color="#1f77b4", marker=".", ms=4, label="p_hit (dots = actual observations)"),
        Line2D([], [], color="red", ls="--", label="Ground-truth event"),
        Line2D([], [], color="#2ca02c", marker="v", ls="none", label="Actual system emission"),
        Line2D([], [], color="#166534", marker="$✓$", ms=11, ls="none", label="Correct timing (±3 s, one-to-one)"),
        Line2D([], [], color="grey", ls=":", label="True/false decision boundary (0.5; not fitted)"),
    ]
    timings = []
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)

    def panel(ax, record):
        ticks = record["gate_trace"]
        ts = [t["t_sec"] for t in ticks]
        ps = [t["p_hit"] for t in ticks]
        gts = [float(g["t_sec"]) for g in record["ground_truth"]]
        emits = [float(p["t_sec"]) for p in record["predictions"]]
        matches, _, _ = match_emits_to_gt(emits, gts, 3.0)
        # Scatter avoids suggesting that the model was queried between ticks.
        ax.plot(ts, ps, color="#1f77b4", marker=".", ms=2.5, lw=0.65)
        for gt in gts:
            ax.axvline(gt, color="red", ls="--", lw=1, alpha=0.8)
        ax.plot(emits, [1.04] * len(emits), "v", color="#2ca02c", ms=5)
        ax.plot([emits[i] for i, _, _ in matches], [1.15] * len(matches),
                marker="$✓$", ls="none", ms=10, color="#166534")
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set(xlim=(0, max(ts + gts + emits) + 1), ylim=(-0.02, 1.24),
               xlabel="Video time (s)", ylabel="p_hit", yticks=[0, 0.5, 1])
        ax.set_title(f"{record['task']} | {record['video_id']}\n"
                     f"{len(gts)} GT · {len(emits)} emits · {len(matches)} timing matches", loc="left", fontsize=9)
        ax.tick_params(labelsize=8)
        return {"id": record["id"], "n_gt": len(gts), "n_emits": len(emits),
                "n_matches": len(matches), "matched_pairs": matches}

    for ax, record in zip(axes.flat, records):
        timings.append(panel(ax, record))
        small, a = plt.subplots(figsize=(12, 4))
        panel(a, record)
        small.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
        small.tight_layout(rect=(0, 0.20, 1, 1))
        small.savefig(figures / f"{record['task']}__{record['video_id']}.png", dpi=150)
        plt.close(small)
    fig.suptitle(f"Vision-only diagnostic · 18 audio={manifest.get('audio_filter', 'none')} videos · no pruning\n"
                 f"Model true/false choice; arm={records[0].get('experiment_arm', 'raw')}",
                 fontsize=15, y=0.995)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.018))
    fig.text(0.5, 0.008, "Two shortest distinct videos per task; qualitative selection, not a benchmark estimate. "
             "✓ indicates timing only. Blue lines connect observations, not continuous inference.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.975))
    for ext in ("png", "pdf"):
        path = figures / f"ALL_vision_model_only_18panels.{ext}"
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)
    write_json(args.out / "timing_matches.json", timings)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "run", "plot"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, choices=range(4), default=0)
    from trigger_ablation_profiles import ARMS
    ap.add_argument("--arm", choices=tuple(ARMS), default="raw")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--realtime", dest="realtime", action="store_true", default=True,
                      help="Default: async controller, input paced at 1x wall time")
    mode.add_argument("--lockstep", dest="realtime", action="store_false",
                      help="Explicit legacy evaluation mode; not a live-stream test")
    ap.add_argument("--benchmark", default="/path/to/work/omnipro_data/benchmark.json")
    ap.add_argument("--dataset", default="/path/to/work/omnipro_data")
    ap.add_argument("--audio", choices=tuple(ALLOWED_AUDIO), default="none_helpful")
    ap.add_argument("--wall-budget-s", type=float, default=0,
                    help="Stop between full clips before this worker budget; 0=unlimited")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "plot": plot}[args.action](args)


if __name__ == "__main__":
    main()