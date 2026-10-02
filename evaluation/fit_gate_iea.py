#!/usr/bin/env python3
"""Fit (hit_threshold, gate_mode, refractory_s) by replaying logged p_hit.

The trace runs force OMNIPRO_HIT_THRESHOLD=1.01 so the gate never fires, which
keeps `reported` empty and stops the prompt feedback at controller.py:849 from
baking one threshold's decisions into the trace. Every candidate config is then
replayed against the same clean p_hit sequence.

This is a SCREEN, not the final number: firing would perturb later ticks, and the
replay cannot model that. Its job is to pick a config to run for real.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TICK = re.compile(
    r"^\[\s*[\d.]+s \| vid\s+([\d.]+)s\] ctrl\.gate\s+\[([^\]]+)\].*?p_hit=([\d.]+)")


def load_traces(roots):
    """{video_id: [(vt, p_hit)]}, de-duplicated and time-ordered.

    Resume + rechain means a video can appear in several logs; keep one reading
    per (video, vt) so a resumed video is not scored twice.
    """
    by_vid = defaultdict(dict)
    for root in roots:
        for path in glob.glob(os.path.join(root, "s*", "run_*.log")):
            with open(path, errors="replace") as fh:
                for line in fh:
                    m = TICK.match(line)
                    if m:
                        vt, vid, p = m.group(1), m.group(2), m.group(3)
                        by_vid[vid][round(float(vt), 3)] = float(p)
    return {v: sorted(d.items()) for v, d in by_vid.items()}


def load_gt(split_glob):
    gt, dur = {}, {}
    for path in glob.glob(split_glob):
        for e in json.load(open(path)):
            times = [float(g["trigger_time_sec"]) for g in e.get("ground_truth", [])]
            gt[e["video_id"]] = sorted(times)
            dur[e["video_id"]] = float(e.get("duration", 0.0))
    return gt, dur


def simulate(trace, thr, mode, refractory):
    """Emission times under one config. Mirrors the controller's gate."""
    fires, last, prev = [], -1e9, False
    for vt, p in trace:
        level = p >= thr
        want = (level and not prev) if mode == "edge" else level
        if want and (vt - last) >= refractory:
            fires.append(vt)
            last = vt
        prev = level
    return fires


def score(fires, truth, tol):
    """Greedy one-to-one temporal match, same rule as metrics.score_sample."""
    used, tp = set(), 0
    for f in fires:
        best, bd = None, tol
        for i, g in enumerate(truth):
            if i in used:
                continue
            if abs(f - g) <= bd:
                best, bd = i, abs(f - g)
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(fires) - tp, len(truth) - tp


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def evaluate(cfg, vids, traces, gt, tol):
    thr, mode, refr = cfg
    tp = fp = fn = 0
    for v in vids:
        a, b, c = score(simulate(traces[v], thr, mode, refr), gt[v], tol)
        tp, fp, fn = tp + a, fp + b, fn + c
    return f1(tp, fp, fn) + (tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=[os.path.join(HERE, "iea_all"),
                             os.path.join(HERE, "iea_partition-b")])
    ap.add_argument("--splits", default=os.path.join(HERE, "splits_iea", "all_s*.json"))
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    traces = load_traces(args.roots)
    gt, dur = load_gt(args.splits)
    vids = sorted(v for v in traces if v in gt and len(traces[v]) > 1)
    print(f"videos with traces: {len(vids)}   ticks: {sum(len(traces[v]) for v in vids):,}")
    print(f"gt events: {sum(len(gt[v]) for v in vids)}   tolerance: ±{args.tolerance}s\n")

    grid = [(t, m, r)
            for t in [round(0.05 * i, 2) for i in range(1, 20)] + [0.95, 0.98, 0.99]
            for m in ("edge", "level")
            for r in (0.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)]

    scored = sorted(((evaluate(c, vids, traces, gt, args.tolerance), c) for c in grid),
                    key=lambda x: -x[0][0])
    print(f"grid: {len(grid)} configs\n")
    print("=== top 10 fit-on-all (optimistic, selection not held out) ===")
    print(f"{'thr':>6}{'mode':>7}{'refr':>7}{'F1':>8}{'prec':>7}{'rec':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    for (fs, p, r, tp, fp, fn), (t, m, rf) in scored[:10]:
        print(f"{t:>6.2f}{m:>7}{rf:>7.0f}{fs:>8.4f}{p:>7.3f}{r:>7.3f}{tp:>5}{fp:>5}{fn:>5}")

    # Grouped K-fold: the config is chosen without ever seeing the fold it is
    # scored on, so this is the number that survives contact with new videos.
    k = args.folds
    folds = [vids[i::k] for i in range(k)]
    tp = fp = fn = 0
    picks = []
    for i in range(k):
        test = folds[i]
        train = [v for j, f in enumerate(folds) if j != i for v in f]
        best = max(grid, key=lambda c: evaluate(c, train, traces, gt, args.tolerance)[0])
        picks.append(best)
        a, b, c = 0, 0, 0
        for v in test:
            x, y, z = score(simulate(traces[v], *best), gt[v], args.tolerance)
            a, b, c = a + x, b + y, c + z
        tp, fp, fn = tp + a, fp + b, fn + c
        print(f"  fold {i}: picked thr={best[0]:.2f} {best[1]} refr={best[2]:.0f}"
              f"  -> test TP={a} FP={b} FN={c}")
    cv, cp, cr = f1(tp, fp, fn)
    print(f"\n=== {k}-fold CV (honest) ===")
    print(f"F1={cv:.4f}  precision={cp:.4f}  recall={cr:.4f}  TP={tp} FP={fp} FN={fn}")
    if len(set(picks)) > 1:
        print(f"NOTE: folds disagreed on the config {sorted(set(picks))} — the surface is flat, "
              f"do not over-trust the exact constants.")

    print("\n=== null controls ===")
    for name, cfg in (("never fire", (1.01, "edge", 0.0)),
                      ("always fire (thr=0)", (0.0, "level", 0.0)),
                      ("current config", (0.45, "edge", 600.0))):
        fs, p, r, a, b, c = evaluate(cfg, vids, traces, gt, args.tolerance)
        print(f"  {name:<22}F1={fs:.4f} prec={p:.3f} rec={r:.3f} TP={a} FP={b} FN={c}")


if __name__ == "__main__":
    main()
