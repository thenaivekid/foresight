#!/usr/bin/env python3
"""Post-hoc gate analysis per (task x audio_dependency).

The protocol this implements:
  1. Inference runs with the thresholds already in config.py. Those emissions are
     the SHIPPED number -- chosen without seeing the test set.
  2. Afterwards, replay the logged p_hit per (task, audio_dependency) group and
     find the best (threshold, mode, refractory). That is the ORACLE ceiling.
  3. Report both. The gap is the cost of not knowing the threshold in advance.

AUC-ROC decides how to read the gap. High AUC means the information is present and
only the operating point was unknown, so the oracle is a fair ceiling. AUC near 0.5
means no threshold exists and the oracle is fitting noise -- the two cases look
identical if you only report F1, which is why AUC is printed alongside.

Caveat, stated because it is easy to forget: inference FIRES, and firing appends to
`reported` which controller.py:849 splices into later prompts. So the replayed p_hit
is not exactly what a different threshold would have produced. The oracle is a
screen, not a promise.

    python analyze_gate_by_group.py --roots full8 iea_all iea_partition-b
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit_gate_iea import evaluate, load_traces, score, simulate  # noqa: E402

GRID = [(t, m, r)
        for t in [round(0.05 * i, 2) for i in range(1, 20)] + [0.95, 0.98, 0.99]
        for m in ("edge", "level")
        for r in (0.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)]


def meta(split_glob):
    gt, task, audio = {}, {}, {}
    for path in glob.glob(split_glob):
        for e in json.load(open(path)):
            v = e["video_id"]
            gt[v] = sorted(float(g["trigger_time_sec"]) for g in e.get("ground_truth", []))
            task[v] = e["task"]
            audio[v] = e.get("audio_dependency")
    return gt, task, audio


def auc_roc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sr, rank, i = s[order], np.arange(1, len(s) + 1, dtype=float), 0
    while i < len(s):                                    # average ranks over ties
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        rank[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = rank
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def labels(vids, traces, gt, tol):
    y, s = [], []
    for v in vids:
        for vt, p in traces[v]:
            y.append(1 if any(abs(vt - x) <= tol for x in gt[v]) else 0)
            s.append(p)
    return y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["full8", "iea_all", "iea_partition-b"])
    ap.add_argument("--splits", default=os.path.join(HERE, "splits_*", "*.json"))
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--min_videos", type=int, default=5,
                    help="skip groups too small for a fit to mean anything")
    ap.add_argument("--out", default=os.path.join(HERE, "gate_by_group.json"))
    args = ap.parse_args()

    traces = load_traces([os.path.join(HERE, r) if not os.path.isabs(r) else r
                          for r in args.roots])
    gt, task, audio = meta(args.splits)
    vids = [v for v in traces if v in gt and len(traces[v]) > 1]

    groups = defaultdict(list)
    for v in vids:
        groups[(task[v], audio[v])].append(v)

    print(f"videos {len(vids)}   ticks {sum(len(traces[v]) for v in vids):,}   "
          f"groups {len(groups)}   tol +-{args.tolerance}s\n")
    hdr = (f"{'task':<28}{'audio':<9}{'vid':>4}{'ev':>5}{'AUC':>7}"
           f"{'shipped':>9}{'oracle':>8}{'gap':>7}  best cfg")
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for (t, a), vs in sorted(groups.items()):
        if len(vs) < args.min_videos:
            continue
        n_ev = sum(len(gt[v]) for v in vs)
        if n_ev == 0:
            continue
        y, s = labels(vs, traces, gt, args.tolerance)
        au = auc_roc(y, s)
        best = max(GRID, key=lambda c: evaluate(c, vs, traces, gt, args.tolerance)[0])
        orc = evaluate(best, vs, traces, gt, args.tolerance)[0]
        # "shipped" is recomputed by replay under the config that actually ran, so
        # it sits on the same footing as the oracle -- a like-for-like gap.
        shipped = evaluate((0.45, "edge", 600.0), vs, traces, gt, args.tolerance)[0]
        print(f"{t:<28}{str(a):<9}{len(vs):>4}{n_ev:>5}{au:>7.3f}"
              f"{shipped:>9.4f}{orc:>8.4f}{orc - shipped:>7.4f}  "
              f"thr={best[0]:.2f} {best[1]} refr={best[2]:.0f}")
        out[f"{t}|{a}"] = {"videos": len(vs), "events": n_ev, "auc_roc": au,
                           "shipped_f1": shipped, "oracle_f1": orc,
                           "best": {"threshold": best[0], "mode": best[1],
                                    "refractory_s": best[2]}}

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("AUC ~0.5 => no threshold works and the oracle is noise; "
          "AUC >=0.7 => the operating point was the only unknown.")


if __name__ == "__main__":
    main()
