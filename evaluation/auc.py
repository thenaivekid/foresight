"""
auc.py — the DENSE metric. No GPU, no judge, no re-running anything.

THE PROBLEM IT SOLVES
---------------------
Today the only signal on a change is time_f1 over ~33 scoring events across 11
videos. Run-to-run variance on one config has been measured at F1 0.255 vs 0.051,
so almost every "improvement" we have ever recorded was inside the noise. You
cannot prompt-engineer against a metric whose error bars exceed the effect.

The schema decoder gives us something better for free: `p_hit`, the CONTINUOUS
confidence read off the logits at every tick. Every tick also has a ground-truth
label (is a target event within +-tolerance of this video-second?). That is
~3,000 labelled decisions per run instead of ~33 events.

WHAT IT REPORTS
---------------
  AUC   -- pick a random tick where an event IS due and one where it is NOT;
           how often does the model score the first higher? 0.5 = coin flip.
  AP    -- average precision (better behaved when positives are rare, which they
           very much are here).
  sweep -- what precision/recall you would get at each firing threshold, computed
           offline over saved numbers. Tuning `hit_threshold` this way costs
           milliseconds instead of a full re-run per value.

WHY BOTH THIS AND F1: AUC measures whether PERCEPTION is right. F1 measures
perception AND whether the gate is tuned. Today a mistuned gate and a bad prompt
look identical. This separates them. **AUC is for iteration; every REPORTED number
stays OmniPro F1.**

Usage:
    python auc.py output_smoke/dedup_counting [more_dirs ...]
    python auc.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# [ 40.5s | vid  1.0s] ctrl.gate  [VIDEOID] fps=1.0 level=False ... p_hit=0.021 ...
_GATE = re.compile(
    r"\|\s*vid\s*([\d.]+)s\]\s+ctrl\.gate\s+\[([^\]]*)\].*?p_hit=([\d.]+)")


def parse_scores(paths):
    """-> {video_id: [(vt, p_hit), ...]} for every tick that logged a logit read."""
    out = defaultdict(list)
    for p in paths:
        with open(p, errors="replace") as fh:
            for line in fh:
                m = _GATE.search(line)
                if m:
                    out[m.group(2)].append((float(m.group(1)), float(m.group(3))))
    return out


def simulate_gate(ticks, thr):
    """Replay controller.py's firing rule offline: RISING EDGE of (p_hit >= thr).

    Sweeping per-tick labels is misleading — on dense-event tasks most ticks sit
    inside some +-tol window, so "always fire" scores well. What we actually care
    about is the emit TIMES the real gate would have produced."""
    emits, prev = [], False
    for vt, p in sorted(ticks):
        cur = p >= thr
        if cur and not prev:
            emits.append(vt)
        prev = cur
    return emits


def greedy_match(emits, gts, tol):
    """OmniPro's scoring: each ground-truth event can be claimed by at most one
    emit, nearest first, within +-tol."""
    used, tp = set(), 0
    for e in sorted(emits):
        best, bd = None, None
        for i, g in enumerate(gts):
            if i in used:
                continue
            d = abs(e - g)
            if d <= tol and (bd is None or d < bd):
                bd, best = d, i
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(emits) - tp, len(gts) - tp


def load_gt(benchmark_json, task=None, dataset_dir=None):
    """-> {video_id: [trigger_time_sec, ...]}.

    Read straight off the benchmark entries rather than via utils.parse_ground_truth,
    which raises on these records ("dictionary update sequence element #0 has length
    1; 2 is required" — it expects a different shape). The field we need is plain:
    ground_truth is a list of dicts each carrying trigger_time_sec, with
    trigger_time ("MM:SS") as the fallback."""
    gt = defaultdict(list)
    for e in json.load(open(benchmark_json)):
        # CRITICAL: one video appears under SEVERAL tasks with DIFFERENT ground
        # truth (e.g. 6rho9EbHtyc has 1 event as snapshot_counting, 5 as
        # realtime_state_monitor, 4 as event_narration). Merging them inflates the
        # positive rate and makes every number meaningless.
        if task and e.get("task") != task:
            continue
        for g in e.get("ground_truth") or []:
            if not isinstance(g, dict):
                continue
            t = g.get("trigger_time_sec")
            if t is None:
                raw = str(g.get("trigger_time") or "")
                if ":" in raw:
                    try:
                        m, s = raw.split(":")[-2:]
                        t = int(m) * 60 + float(s)
                    except (TypeError, ValueError):
                        t = None
            if t is not None:
                try:
                    gt[e["video_id"]].append(float(t))
                except (TypeError, ValueError):
                    pass
    return gt


def label_ticks(scores, gt, tol):
    """A tick is POSITIVE if a ground-truth event falls within +-tol of it — i.e.
    exactly the window in which firing would have been credited by OmniPro."""
    ys, ps, per_video = [], [], {}
    for vid, ticks in scores.items():
        times = gt.get(vid)
        if not times:
            continue                                  # no GT for this clip; skip
        y = [1 if any(abs(vt - t) <= tol for t in times) else 0 for vt, _ in ticks]
        p = [s for _, s in ticks]
        per_video[vid] = (y, p)
        ys += y
        ps += p
    return ys, ps, per_video


def auc_score(y, p):
    """Mann-Whitney AUC with tie handling. Pure python — no sklearn in this env."""
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):                             # average ranks within ties
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    s = sum(r for r, yy in zip(ranks, y) if yy)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def average_precision(y, p):
    """Area under precision-recall. More informative than AUC when positives are
    rare, which they are: only a few seconds of a video are near an event."""
    n1 = sum(y)
    if n1 == 0:
        return None
    order = sorted(range(len(p)), key=lambda i: -p[i])
    tp = 0
    ap = 0.0
    for rank, i in enumerate(order, start=1):
        if y[i]:
            tp += 1
            ap += tp / rank
    return ap / n1


def sweep(y, p, thresholds):
    rows = []
    for t in thresholds:
        tp = sum(1 for yy, pp in zip(y, p) if pp >= t and yy)
        fp = sum(1 for yy, pp in zip(y, p) if pp >= t and not yy)
        fn = sum(1 for yy, pp in zip(y, p) if pp < t and yy)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((t, tp, fp, fn, prec, rec, f1))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--benchmark", default=os.environ.get(
        "OMNIPRO_BENCHMARK_JSON",
        "/path/to/work/omnipro_data/benchmark.json"))
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--task", default=None, help="override task inferred from dir name")
    args = ap.parse_args()

    dirs = args.dirs or (sorted(d for d in glob.glob(os.path.join(here, "output_*"))
                                if os.path.isdir(d)) if args.all else [])
    if not dirs:
        ap.error("give one or more run dirs, or --all")

    for d in dirs:
        paths = sorted(glob.glob(os.path.join(d, "**", "run_*.log"), recursive=True))
        if not paths:
            continue
        # the run dir is named after the task; use it to select the right GT
        task = args.task or os.path.basename(d.rstrip("/"))
        gt = load_gt(args.benchmark, task=task)
        if not gt:
            gt = load_gt(args.benchmark, task=None)
            print(f"  (no GT for task {task!r}; falling back to ALL tasks — "
                  f"numbers will be inflated)")
        scores = parse_scores(paths)
        if not scores:
            print(f"\n=== {os.path.basename(d.rstrip('/'))} ===")
            print("  no p_hit in logs (free-decode run, or verify_logit_read off)")
            continue
        y, p, per_video = label_ticks(scores, gt, args.tolerance)
        print(f"\n=== {os.path.basename(d.rstrip('/'))} ===")
        if not y:
            print("  ticks found but no matching ground truth")
            continue
        a, apr = auc_score(y, p), average_precision(y, p)
        pos = sum(y)
        print(f"  videos={len(per_video)}  ticks={len(y)}  "
              f"positive={pos} ({100.0*pos/len(y):.1f}%)  "
              f"base_rate_AP={pos/len(y):.3f}")
        print(f"  AUC = {a:.4f}" if a is not None else "  AUC = n/a")
        print(f"  AP  = {apr:.4f}" if apr is not None else "  AP  = n/a")
        lo, hi = min(p), max(p)
        print(f"  p_hit range observed: {lo:.4f} .. {hi:.4f}")
        if hi < 0.5:
            print("  *** every p_hit is BELOW the default hit_threshold=0.5 -> the "
                  "gate can never fire. Threshold is mis-calibrated, not the model. ***")
        # ---- threshold sweep on the REAL gate (rising edge + greedy match) ----
        # This predicts time_f1 directly, and costs milliseconds instead of one
        # full GPU re-run per threshold value.
        cand = sorted({round(x, 4) for x in
                       [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
                       + [lo + (hi - lo) * k / 8 for k in range(1, 8)]})
        print("   thr    emits     tp     fp     fn   prec    rec  time_f1")
        rows = []
        for t in cand:
            TP = FP = FN = NE = 0
            for vid, ticks in scores.items():
                g = gt.get(vid)
                if not g:
                    continue
                em = simulate_gate(ticks, t)
                a, b, c = greedy_match(em, g, args.tolerance)
                TP += a; FP += b; FN += c; NE += len(em)
            pr = TP / (TP + FP) if TP + FP else 0.0
            rc = TP / (TP + FN) if TP + FN else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            rows.append((t, NE, TP, FP, FN, pr, rc, f1))
            print(f"  {t:6.4f} {NE:6d} {TP:6d} {FP:6d} {FN:6d}  {pr:5.3f}  {rc:5.3f}   {f1:5.3f}")
        best = max(rows, key=lambda r: r[-1])
        cur = [r for r in rows if abs(r[0] - 0.5) < 1e-9]
        print(f"  BEST threshold {best[0]:.4f} -> time_f1 {best[-1]:.3f}"
              + (f"   (current 0.5 gives {cur[0][-1]:.3f})" if cur else ""))


if __name__ == "__main__":
    main()
