#!/usr/bin/env python3
"""Repeat a completed model-only run with unchanged seeds, then overlay traces.

The orchestrator waits for the first launcher to exit, not for partial output
counts. Source hashes and manifests are checked before starting fresh processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def source_hashes():
    paths = list((REPO / "foresight").glob("*.py"))
    paths += [HERE / name for name in ("foresight_adapter.py", "dataset.py", "utils.py",
                                       "vision_model_only.py", "metrics.py",
                                       "trigger_ablation_profiles.py", "trigger_live_matrix.py",
                                       "none_helpful_study.py")]
    paths.append(REPO / "run_vision_model_only.sbatch")
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def load_run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest["ids"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError(f"{root}: manifest must contain distinct nonempty sample IDs")
    rows = [json.loads(line) for p in sorted(root.glob("g*/online_pred.jsonl"))
            for line in p.read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(expected) or len(by_id) != len(expected) or set(by_id) != set(expected):
        raise ValueError(f"{root}: requires exactly the {len(expected)} completed manifest samples")
    allowed = {"none": {"none"}, "helpful": {"helpful"},
               "none_helpful": {"none", "helpful"}}[manifest.get("audio_filter", "none")]
    for r in rows:
        if r["audio_dependency"] not in allowed or not r.get("gate_trace"):
            raise ValueError(f"Invalid diagnostic sample {r['id']}")
        if not r.get("effective_config", {}).get("deterministic", True):
            delivery_times(r)  # Missing live delivery timestamps must never fall back to tick time.
    return by_id


def compare_sample(a, b):
    if a["effective_config"] != b["effective_config"]:
        raise ValueError(f"Config mismatch for {a['id']}")
    if a["ground_truth"] != b["ground_truth"] or a["protocol"] != b["protocol"]:
        raise ValueError(f"Protocol/GT mismatch for {a['id']}")
    ta, tb = a["gate_trace"], b["gate_trace"]

    def indexed(trace):
        times = [p["t_sec"] for p in trace]
        if any(y <= x for x, y in zip(times, times[1:])):
            raise ValueError("Tick timestamps must be strictly increasing")
        return {p["t_sec"]: p["p_hit"] for p in trace}

    ia, ib = indexed(ta), indexed(tb)
    common = sorted(ia.keys() & ib.keys())
    delta = [abs(ia[t] - ib[t]) for t in common]
    first_diff = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first_diff is None and len(ta) != len(tb):
        first_diff = min(len(ta), len(tb))
    ea = [p["t_sec"] for p in a["predictions"]]
    eb = [p["t_sec"] for p in b["predictions"]]
    ca = [(p["t_sec"], p["raw"]) for p in a["predictions"]]
    cb = [(p["t_sec"], p["raw"]) for p in b["predictions"]]
    return {
        "id": a["id"], "task": a["task"], "video_id": a["video_id"],
        "ticks_run1": len(ta), "ticks_run2": len(tb),
        "identical_time_grid": list(ia) == list(ib), "identical_curve": ta == tb,
        "first_different_tick_index": first_diff,
        "first_difference_times_s": None if first_diff is None else [
            trace[first_diff]["t_sec"] if first_diff < len(trace) else None for trace in (ta, tb)],
        "exact_common_times": len(common),
        "only_run1_times": len(ia.keys() - ib.keys()),
        "only_run2_times": len(ib.keys() - ia.keys()),
        "mean_abs_phit_difference_common_times": sum(delta) / len(delta) if delta else None,
        "max_abs_phit_difference_common_times": max(delta) if delta else None,
        "bool_disagreements_common_times": sum((ia[t] >= 0.5) != (ib[t] >= 0.5) for t in common),
        "identical_emission_times": ea == eb, "identical_emitted_text_and_times": ca == cb,
        "identical_text_sequence": [p["raw"] for p in a["predictions"]] ==
                       [p["raw"] for p in b["predictions"]],
        "emits_run1": len(ea), "emits_run2": len(eb),
        "timing_run1": a["timing"], "timing_run2": b["timing"],
    }


def delivery_times(record):
    """Actual delivered time in live runs; legacy tick time only in lockstep."""
    if record["effective_config"]["deterministic"]:
        return [p["t_sec"] for p in record["predictions"]]
    events = record["wall_timeline"]
    origin = next(e["monotonic_s"] for e in events if e["event"] == "source_start")
    times = [e["monotonic_s"] - origin for e in events if e["event"] == "response_delivered"]
    if len(times) != len(record["predictions"]):
        raise ValueError("Delivery events and predictions differ in length")
    return times


def compare(first, second):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from metrics import match_emits_to_gt

    aa, bb = load_run(first), load_run(second)
    if aa.keys() != bb.keys():
        raise ValueError("Sample IDs differ")
    ids = sorted(aa, key=lambda k: (aa[k]["task"], aa[k]["video_id"]))
    stats = [compare_sample(aa[k], bb[k]) for k in ids]
    out = second / "comparison"
    out.mkdir(exist_ok=True)
    colors = ("#1f77b4", "#e67e22")
    handles = [
        Line2D([], [], color=colors[0], marker=".", lw=1, label="Run 1 p_hit / ▼ emissions"),
        Line2D([], [], color=colors[1], marker=".", ls="--", lw=1, label="Run 2 p_hit / ▼ emissions"),
        Line2D([], [], color="red", ls="--", label="Ground-truth event"),
        Line2D([], [], color="#166534", marker="$✓$", ls="none", ms=10,
               label="Timing match (±3 s, one-to-one; each run separately)"),
        Line2D([], [], color="grey", ls=":", label="0.5 true/false boundary (not fitted)"),
    ]

    def panel(ax, k, stat):
        a, b = aa[k], bb[k]
        gts = [float(g["t_sec"]) for g in a["ground_truth"]]
        all_times = gts[:]
        for j, r in enumerate((a, b)):
            ticks = r["gate_trace"]
            ts = [p["t_sec"] for p in ticks]
            ps = [p["p_hit"] for p in ticks]
            # Show each run on its own observed timestamps, never resample.
            ax.plot(ts, ps, color=colors[j], ls="-" if j == 0 else "--",
                    lw=0.8, marker=".", ms=2, alpha=0.85)
            emits = delivery_times(r)
            matches, _, _ = match_emits_to_gt(emits, gts, tolerance=3.0)
            y = 1.05 + j * 0.22
            ax.plot(emits, [y] * len(emits), "v", color=colors[j], ms=4)
            ax.plot([emits[i] for i, _, _ in matches], [y + 0.09] * len(matches),
                    marker="$✓$", ls="none", color="#166534", ms=8)
            all_times += ts + emits
        for gt in gts:
            ax.axvline(gt, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set(xlim=(0, max(all_times) + 1), ylim=(-0.02, 1.48), yticks=[0, 0.5, 1],
               xlabel="Video time (s)", ylabel="p_hit")
        d = stat["mean_abs_phit_difference_common_times"]
        label = "n/a" if d is None else f"{d:.4f}"
        ax.set_title(f"{a['task']} | {a['video_id']}\n"
                     f"Identical curve: {stat['identical_curve']} · mean |Δp|={label} "
                     f"({stat['exact_common_times']} shared times)\n"
                     f"R1/R2 ticks {stat['ticks_run1']}/{stat['ticks_run2']} · "
                     f"emits {stat['emits_run1']}/{stat['emits_run2']} · "
                     f"check-time matches {stat['timing_run1']['tp']}/{stat['timing_run2']['tp']}",
                     loc="left", fontsize=8)
        ax.tick_params(labelsize=7)

    fig, axes = plt.subplots(9, 2, figsize=(20, 29))
    for ax, k, stat in zip(axes.flat, ids, stats):
        panel(ax, k, stat)
        small, a = plt.subplots(figsize=(12, 4.5))
        panel(a, k, stat)
        small.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
        small.tight_layout(rect=(0, 0.20, 1, 1))
        small.savefig(out / f"{stat['task']}__{stat['video_id']}__overlay.png", dpi=150)
        plt.close(small)
    fig.suptitle("Same-seed repeatability · 18 vision-only audio=none videos\n"
                 "Unchanged model/config/seeds/shards · no pruning or fitted gates · fresh processes",
                 fontsize=15, y=0.996)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.017))
    fig.text(0.5, 0.008, "Dots = observed checks; connectors are visual guides. Differences use exact shared "
             "timestamps only. Live-run triangles/✓ use actual wall delivery time. Login-node diagnostic.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.977))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ALL_vision_model_only_repeat_18panels.{ext}", dpi=150)
    plt.close(fig)
    result = {"run1": str(first), "run2": str(second), "n_samples": len(stats),
              "identical_curves": sum(s["identical_curve"] for s in stats),
              "identical_time_grids": sum(s["identical_time_grid"] for s in stats),
              "identical_emission_times": sum(s["identical_emission_times"] for s in stats),
              "identical_emitted_text_and_times": sum(s["identical_emitted_text_and_times"] for s in stats),
              "per_sample": stats}
    (out / "repeatability.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Same-seed repeatability diagnostic", "",
             "Two sequential fresh-process runs, identical selected videos, seeds and settings. "
             "This is a repeatability check, not a multi-seed variance estimate or a latency benchmark.", "",
             f"- Exactly identical probability curves: {result['identical_curves']}/18",
             f"- Identical check-time grids: {result['identical_time_grids']}/18",
             f"- Identical emission times: {result['identical_emission_times']}/18",
             f"- Identical emission text and times: {result['identical_emitted_text_and_times']}/18", "",
             "| Task / video | Ticks R1/R2 | Shared times | Mean absolute Δp | Timing TP R1/R2 |",
             "|---|---:|---:|---:|---:|"]
    for s in stats:
        lines.append(f"| {s['task']} / {s['video_id']} | {s['ticks_run1']}/{s['ticks_run2']} | "
                     f"{s['exact_common_times']} | {s['mean_abs_phit_difference_common_times']} | "
                     f"{s['timing_run1']['tp']}/{s['timing_run2']['tp']} |")
    lines += ["", "Probability deltas use only identical video timestamps, without interpolation. "
              "Even at a shared timestamp, earlier decisions may have changed input FPS, memory, "
              "or prompts. These are whole-system trajectory differences, not isolated kernel errors."]
    (out / "REPEATABILITY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, indent=2), flush=True)
    print(out / "ALL_vision_model_only_repeat_18panels.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--wait-pid", type=int)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    if not args.compare_only:
        import psutil
        fingerprint = source_hashes()
        if args.second.exists():
            raise FileExistsError("Repeat output must be fresh; do not resume into an old repeat")
        args.second.mkdir(parents=True)
        (args.second / "source_hashes.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
        if args.wait_pid:
            try:
                proc = psutil.Process(args.wait_pid)
            except psutil.NoSuchProcess:
                proc = None
            if proc is not None:
                if "run_vision_model_only.sbatch" not in " ".join(proc.cmdline()):
                    raise ValueError("Wait PID is not the diagnostic launcher")
                print(f"Waiting for first launcher PID {args.wait_pid} to exit", flush=True)
                proc.wait(timeout=6000)
        first_records = load_run(args.first)  # fail rather than repeat an incomplete run
        if source_hashes() != fingerprint:
            raise RuntimeError("Inference source changed while waiting; repeat would not be controlled")
        for name in ["manifest.json", "benchmark.json"] + [f"shard{i}.json" for i in range(4)]:
            shutil.copy2(args.first / name, args.second / name)
        live = not next(iter(first_records.values()))["effective_config"]["deterministic"]
        env = dict(os.environ, OUT=str(args.second), ALLOW_LOGIN_DIAGNOSTIC="1",
               REALTIME_DIAGNOSTIC="1" if live else "0")
        print("Starting fresh same-seed repeat on the same four GPUs", flush=True)
        with (args.second / "launch.log").open("w") as log:
            subprocess.run(["timeout", "5400", "bash", str(REPO / "run_vision_model_only.sbatch")],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if source_hashes() != fingerprint:
            raise RuntimeError("Inference source changed during repeat")
    compare(args.first, args.second)


if __name__ == "__main__":
    main()