#!/usr/bin/env python3
"""Report the pruning sweep with replicates, so deltas are judged against noise.

Run 1 of each arm lives in the original ablation dirs; replicates land in
abl_rep/. Arms with >1 run get a spread, which is the only thing that makes a
delta interpretable at n=18.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = {
    "baseline":   ["abl_base", "abl_rep/base_r2", "abl_rep/base_r3"],
    "clip_50":    ["abl_clip", "abl_rep/clip50_r2", "abl_rep/clip50_r3"],
    "clip_70":    ["abl_clip_sweep/f0.70", "abl_rep/clip70_r2"],
    "clip_80":    ["abl_clip_sweep/f0.80"],
    "clip_90":    ["abl_clip_sweep/f0.90", "abl_rep/clip90_r2"],
    "random_50":  ["abl_rep/random50"],
    "permute":    ["abl_rep/permute"],
}
METRICS = ("time_f1", "content_acc", "joint_f1", "macro_joint_f1")


def load(rel):
    path = os.path.join(HERE, rel, "online_metrics.json")
    if not os.path.exists(path):
        return None
    blob = json.load(open(path))
    overall = blob["overall"]
    overall["_n"] = blob.get("n", overall.get("n_samples"))
    return overall


def main():
    runs = {arm: [(d, load(d)) for d in dirs] for arm, dirs in ARMS.items()}

    print(f"{'arm':<11}{'run':<22}{'n':>4}{'time_f1':>9}{'content':>9}"
          f"{'joint_f1':>10}{'macro_j':>9}")
    print("-" * 74)
    for arm, entries in runs.items():
        for rel, m in entries:
            if m is None:
                print(f"{arm:<11}{rel:<22}{'--':>4}{'pending':>9}")
                continue
            print(f"{arm:<11}{rel:<22}{m['_n']:>4}{m['time_f1']:>9.4f}"
                  f"{m['content_acc']:>9.4f}{m['joint_f1']:>10.4f}"
                  f"{m['macro_joint_f1']:>9.4f}")

    print("\n=== NOISE FLOOR (spread across identical configs) ===")
    floor = {}
    for arm, entries in runs.items():
        vals = [m for _, m in entries if m is not None]
        if len(vals) < 2:
            continue
        print(f"\n{arm}  ({len(vals)} runs)")
        for k in METRICS:
            xs = [v[k] for v in vals]
            spread = max(xs) - min(xs)
            floor.setdefault(k, []).append(spread)
            print(f"  {k:<16} min={min(xs):.4f} max={max(xs):.4f} "
                  f"spread={spread:.4f}")

    if floor:
        print("\n=== VERDICT ===")
        for k in METRICS:
            if k not in floor:
                continue
            worst = max(floor[k])
            base = [m for _, m in runs["baseline"] if m is not None]
            arm50 = [m for _, m in runs["clip_50"] if m is not None]
            if base and arm50:
                delta = (statistics.mean(v[k] for v in arm50)
                         - statistics.mean(v[k] for v in base))
                verdict = "NOISE" if abs(delta) <= worst else "possibly real"
                print(f"  {k:<16} clip50-baseline={delta:+.4f}  "
                      f"noise={worst:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
