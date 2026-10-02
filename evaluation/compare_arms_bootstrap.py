#!/usr/bin/env python3
"""Paired bootstrap over videos for the matched-code pruning arms.

Determinism is established (identical code -> identical emissions), so the only
remaining uncertainty is which 18 videos we happened to pick. Resample videos
with replacement, recompute pooled metrics, report the CI of each delta.
"""
import argparse
import json
import os
import random

from metrics import ContentJudge, aggregate, score_sample
from utils import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "baseline": "abl_rep/base_r2",
    "clip_50": "abl_rep/clip50_r2",
    "clip_70": "abl_rep/clip70_r2",
    "clip_80": "abl_clip_sweep/f0.80",
    "clip_90": "abl_rep/clip90_r2",
    "random_50": "abl_rep/random50",
    "permute": "abl_rep/permute",
}
METRICS = ("time_f1", "content_acc", "joint_f1", "macro_joint_f1")


def scored(rel, judge, tol):
    path = os.path.join(HERE, rel, "online_pred.jsonl")
    if not os.path.exists(path):
        return None
    return {r["id"]: score_sample(r, tolerance=tol, judge=judge)
            for r in read_jsonl(path)}


def pooled(rows):
    return aggregate(rows)["overall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--min_n", type=int, default=18,
                    help="drop arms with fewer completed videos")
    args = ap.parse_args()

    judge = ContentJudge()
    arms = {n: scored(d, judge, args.tolerance) for n, d in ARMS.items()}
    arms = {n: v for n, v in arms.items() if v and len(v) >= args.min_n}
    base = arms.get("baseline")
    if not base:
        print("baseline missing")
        return
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"arms: {', '.join(arms)}\npaired videos: {len(ids)}\n")

    print(f"{'arm':<11}" + "".join(f"{m:>16}" for m in METRICS))
    for name, sc in arms.items():
        o = pooled([sc[i] for i in ids])
        print(f"{name:<11}" + "".join(f"{o[m]:>16.4f}" for m in METRICS))

    rng = random.Random(1234)
    draws = [[rng.choice(ids) for _ in ids] for _ in range(args.iters)]
    print(f"\n=== paired bootstrap vs baseline ({args.iters} resamples) ===")
    for name, sc in arms.items():
        if name == "baseline":
            continue
        print(f"\n{name}")
        for m in METRICS:
            deltas = []
            for d in draws:
                a = pooled([sc[i] for i in d])[m]
                b = pooled([base[i] for i in d])[m]
                deltas.append(a - b)
            deltas.sort()
            lo = deltas[int(0.025 * len(deltas))]
            hi = deltas[int(0.975 * len(deltas)) - 1]
            point = pooled([sc[i] for i in ids])[m] - pooled([base[i] for i in ids])[m]
            sig = "" if lo <= 0 <= hi else "  SIGNIFICANT"
            print(f"  {m:<16}{point:+.4f}  CI[{lo:+.4f},{hi:+.4f}]{sig}")


if __name__ == "__main__":
    main()
