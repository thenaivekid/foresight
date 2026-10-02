"""
gates.py — screen firing strategies OFFLINE over saved p_hit traces.

Every gate is just a function over the p_hit sequence the controller already
logged, so a dozen ideas can be compared in seconds instead of a dozen GPU runs.

THE POINT: a single global threshold cannot work. Fitted per-task optima ranged
0.12 (instant_event_alert) to 0.85 (semantic_condition_alert) -- the model's
confidence SCALE shifts with how the task was phrased, so an absolute cut-off is
measuring the wrong thing. Strategies here ask instead: "is p_hit unusually high
FOR THIS STREAM, so far?" That uses only the model's own outputs -- no ground
truth -- so nothing is fitted to the benchmark and there is nothing to leak.

All adaptive gates are CAUSAL: statistics come only from ticks already seen, so
they are causal (never observe the future).

⚠️ IMPORTANT LIMITATION — this is a SCREEN, not a verdict.
Offline replay assumes p_hit does not depend on the gate. It does, weakly: firing
appends to `reported`, which is fed back into the prompt, which changes later
p_hit. So rankings here are approximate. Take the top candidates to the GPU and
confirm. Cheap screen -> expensive confirmation.

Usage:
    python gates.py output_smoke/dedup_counting [more dirs ...]
    python gates.py --all
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auc import greedy_match, load_gt, parse_scores   # noqa: E402


# --------------------------------------------------------------------------
# Each gate takes the ordered [(vt, p_hit)] for ONE video and returns emit times.
# Every one of them fires on a RISING EDGE (condition false -> true), mirroring
# controller.py: a condition that merely STAYS true must not re-fire.
# --------------------------------------------------------------------------
def _edges(ticks, decide, warmup=0):
    """Walk the stream once, firing on rising edges of `decide(i, vt, p, hist)`."""
    emits, prev, hist = [], False, []
    for i, (vt, p) in enumerate(sorted(ticks)):
        cur = bool(decide(i, vt, p, hist)) if i >= warmup else False
        if cur and not prev:
            emits.append(vt)
        prev = cur
        hist.append(p)                      # append AFTER deciding -> causal
    return emits


def fixed(thr):
    return lambda ticks: _edges(ticks, lambda i, vt, p, h: p >= thr)


def percentile(q, floor=0.05, warmup=5):
    """Fire when p_hit is in the top (100-q)% of THIS stream so far, and above a
    small absolute floor so a 0.01 -> 0.02 jump (2x, but meaningless) cannot fire."""
    def gate(ticks):
        def decide(i, vt, p, h):
            if p < floor or len(h) < warmup:
                return False
            k = max(0, min(len(h) - 1, int(q / 100.0 * (len(h) - 1))))
            return p > sorted(h)[k]
        return _edges(ticks, decide, warmup=warmup)
    return gate


def zscore(k, floor=0.05, warmup=8):
    """Fire when p_hit is k standard deviations above this stream's running mean."""
    def gate(ticks):
        def decide(i, vt, p, h):
            if p < floor or len(h) < warmup:
                return False
            sd = st.pstdev(h)
            return sd > 0 and (p - st.mean(h)) / sd > k
        return _edges(ticks, decide, warmup=warmup)
    return gate


def rel_median(mult, floor=0.05, warmup=5):
    """Fire when p_hit is `mult` times this stream's running median."""
    def gate(ticks):
        def decide(i, vt, p, h):
            if p < floor or len(h) < warmup:
                return False
            m = st.median(h)
            return p > max(m * mult, floor)
        return _edges(ticks, decide, warmup=warmup)
    return gate


def running_max(margin=1.0, floor=0.05, warmup=5):
    """Fire only on a NEW running maximum (times `margin`). Very conservative."""
    def gate(ticks):
        def decide(i, vt, p, h):
            return p >= floor and len(h) >= warmup and p > max(h) * margin
        return _edges(ticks, decide, warmup=warmup)
    return gate


def pct_hysteresis(q_hi, q_lo, floor=0.05, warmup=5):
    """Schmitt gate with FLOATING levels: arm above the q_hi percentile, re-arm
    below q_lo. Same discipline as the tuned hyst2b, but the levels track the
    stream instead of being nailed to 0.5/0.40."""
    def gate(ticks):
        emits, armed, hist = [], True, []
        for vt, p in sorted(ticks):
            if len(hist) >= warmup and p >= floor:
                s = sorted(hist)
                hi = s[max(0, min(len(s) - 1, int(q_hi / 100.0 * (len(s) - 1))))]
                lo = s[max(0, min(len(s) - 1, int(q_lo / 100.0 * (len(s) - 1))))]
                if armed and p > hi:
                    emits.append(vt)
                    armed = False
                elif not armed and p < lo:
                    armed = True
            hist.append(p)
        return emits
    return gate


STRATEGIES = {
    "fixed@0.5 (current)":      fixed(0.50),
    "fixed@0.2":                fixed(0.20),
    "fixed@0.1":                fixed(0.10),
    "pct>90":                   percentile(90),
    "pct>95":                   percentile(95),
    "pct>99":                   percentile(99),
    "pct>95 floor0.02":         percentile(95, floor=0.02),
    "pct>95 floor0.15":         percentile(95, floor=0.15),
    "zscore>2":                 zscore(2.0),
    "zscore>3":                 zscore(3.0),
    "median x3":                rel_median(3.0),
    "median x10":               rel_median(10.0),
    "running max":              running_max(),
    "pct hyst 95/70":           pct_hysteresis(95, 70),
    "pct hyst 90/50":           pct_hysteresis(90, 50),
}


def score(scores, gt, gate, tol):
    TP = FP = FN = NE = 0
    for vid, ticks in scores.items():
        g = gt.get(vid)
        if not g:
            continue
        em = gate(ticks)
        a, b, c = greedy_match(em, g, tol)
        TP += a; FP += b; FN += c; NE += len(em)
    pr = TP / (TP + FP) if TP + FP else 0.0
    rc = TP / (TP + FN) if TP + FN else 0.0
    return NE, TP, FP, FN, pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--benchmark", default="/path/to/work/"
                                           "omnipro_data/benchmark.json")
    ap.add_argument("--tolerance", type=float, default=3.0)
    args = ap.parse_args()

    dirs = args.dirs or (sorted(d for d in glob.glob(os.path.join(here, "output_*"))
                                if os.path.isdir(d)) if args.all else [])
    if not dirs:
        ap.error("give run dirs, or --all")

    overall = {k: [0, 0, 0, 0] for k in STRATEGIES}      # NE,TP,FP,FN pooled
    for d in dirs:
        paths = sorted(glob.glob(os.path.join(d, "**", "run_*.log"), recursive=True))
        scores = parse_scores(paths) if paths else {}
        if not scores:
            continue
        task = os.path.basename(d.rstrip("/"))
        gt = load_gt(args.benchmark, task=task)
        if not gt:
            print(f"\n=== {task} === (no ground truth for this task; skipped)")
            continue
        ngt = sum(len(gt[v]) for v in scores if v in gt)
        print(f"\n=== {task} ===  videos={len(scores)}  gt_events={ngt}")
        print(f"  {'strategy':22s} {'emits':>6s} {'tp':>4s} {'fp':>4s} {'fn':>4s} "
              f"{'prec':>6s} {'rec':>6s} {'F1':>7s}")
        rows = []
        for name, gate in STRATEGIES.items():
            NE, TP, FP, FN, pr, rc, f1 = score(scores, gt, gate, args.tolerance)
            rows.append((name, f1, NE, TP, FP, FN, pr, rc))
            o = overall[name]
            o[0] += NE; o[1] += TP; o[2] += FP; o[3] += FN
            print(f"  {name:22s} {NE:6d} {TP:4d} {FP:4d} {FN:4d} "
                  f"{pr:6.3f} {rc:6.3f} {f1:7.3f}")
        best = max(rows, key=lambda r: r[1])
        print(f"  -> best: {best[0]}  F1={best[1]:.3f}")

    print("\n\n===== POOLED ACROSS ALL TASKS =====")
    print(f"  {'strategy':22s} {'emits':>6s} {'tp':>4s} {'fp':>4s} {'fn':>4s} "
          f"{'prec':>6s} {'rec':>6s} {'F1':>7s}")
    pooled = []
    for name, (NE, TP, FP, FN) in overall.items():
        pr = TP / (TP + FP) if TP + FP else 0.0
        rc = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        pooled.append((name, f1, NE, TP, FP, FN, pr, rc))
    for name, f1, NE, TP, FP, FN, pr, rc in sorted(pooled, key=lambda r: -r[1]):
        print(f"  {name:22s} {NE:6d} {TP:4d} {FP:4d} {FN:4d} "
              f"{pr:6.3f} {rc:6.3f} {f1:7.3f}")
    print("\n  NOTE: a SCREEN, not a verdict — offline replay assumes p_hit is "
          "independent of the gate, but firing feeds `reported` back into the "
          "prompt. Confirm the top candidates on GPU.")


if __name__ == "__main__":
    main()
