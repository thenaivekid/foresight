"""
resweep.py — re-derive the firing threshold OFFLINE from a finished run.

WHY THIS EXISTS SEPARATELY FROM dump_scores.py
----------------------------------------------
dump_scores.py keys every tick on video_id alone and stamps the task as the
SHARD NAME ("g00"). That is fine for a single-task run and wrong for this one:
in a full development run each shard holds all 9 tasks, and 13 of g00's 44 videos appear
under MORE THAN ONE task with different ground truth. Keyed on video_id, their
ticks merge and the labels become meaningless.

The run log already disambiguates them. Every sample opens with

    [online] ===== [40/63] cumulative_counting::3MI7OZ-BlgE::40 gt_times=[23.0, ...]

so ticks are attributed by POSITION IN THE LOG — the header that precedes them —
not by video_id. GT comes off the same header, so no split file is needed and the
task/video pairing cannot drift.

Streaming discipline: this reads only what the controller already emitted, in the
order it emitted it. Nothing here looks ahead; the sweep is a post-hoc analysis of
a causal run, not a re-simulation with future knowledge.

Usage:
    python resweep.py RUN_DIR            # sweep, per task + overall
    python resweep.py RUN_DIR --dev 0.5  # fit on a dev half, report on held-out
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

TOL = 3.0  # OmniPro's +-3s temporal tolerance

_HEADER = re.compile(
    r"\[online\]\s+=====\s+\[\d+/\d+\]\s+(\S+?)::(\S+?)::(\S+?)\s+gt_times=\[([^\]]*)\]")
_GATE = re.compile(
    r"\|\s*vid\s*([\d.]+)s\]\s+ctrl\.gate\s+\[([^\]]*)\].*?p_hit=([\d.]+)")


def parse_run(run_dir):
    """-> {sample_id: {"task":..., "video":..., "gt":[...], "ticks":[(vt,p_hit)]}}

    Later generations re-run resumed samples, so a sample can appear in several
    log files. Keep the LAST occurrence: that is the one whose emits the final
    online_metrics.json was scored on.
    """
    samples = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "run_*.log"),
                                 recursive=True)):
        cur = None
        with open(path, errors="replace") as fh:
            for line in fh:
                h = _HEADER.search(line)
                if h:
                    task, vid, sid, gts = h.group(1), h.group(2), h.group(3), h.group(4)
                    key = f"{task}::{vid}::{sid}"
                    cur = {"task": task, "video": vid, "ticks": [],
                           "gt": [float(x) for x in gts.split(",") if x.strip()]}
                    samples[key] = cur           # overwrite = keep the latest run
                    continue
                if cur is None:
                    continue
                g = _GATE.search(line)
                if g:
                    cur["ticks"].append((float(g.group(1)), float(g.group(3))))
    return samples


def simulate(ticks, thr, mode="edge", refractory=0.0):
    """Replay a firing rule over the tick stream, strictly left to right.

    edge      -- fire on the RISING edge of p_hit >= thr (what controller.py does)
    level     -- fire on EVERY tick above thr
    Both honour an optional refractory period: suppress a fire within `refractory`
    seconds of the previous one. That is a legal streaming rule (it looks only at
    the past) and is the cheapest way to attack a 6.4x over-fire rate.
    """
    emits, prev, last = [], False, None
    for vt, p in sorted(ticks):
        cur = p >= thr
        want = (cur and not prev) if mode == "edge" else cur
        if want and (last is None or vt - last >= refractory):
            emits.append(vt)
            last = vt
        prev = cur
    return emits


def greedy_match(emits, gts, tol=TOL):
    """OmniPro scoring: each GT event claimed by at most one emit, nearest first."""
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


def score(samples, thr, mode, refractory):
    """Pool TP/FP/FN over samples -> (precision, recall, f1, n_emits)."""
    tp = fp = fn = 0
    for s in samples:
        e = simulate(s["ticks"], thr, mode, refractory)
        a, b, c = greedy_match(e, s["gt"])
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp + fp


GRID = [(m, t, ref)
        for m in ("edge", "level")
        for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
        for ref in (0.0, 3.0, 5.0, 10.0, 20.0, 30.0)]


def best_config(samples):
    best = None
    for mode, thr, ref in GRID:
        p, r, f, n = score(samples, thr, mode, ref)
        if best is None or f > best[0]:
            best = (f, mode, thr, ref, p, r, n)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dev", type=float, default=0.0,
                    help="fraction of samples to FIT on; the rest is held out. "
                         "0 = fit and report on everything (diagnostic only).")
    ap.add_argument("--out", default="thresholds.json")
    args = ap.parse_args()

    samples = parse_run(args.run_dir)
    by_task = defaultdict(list)
    for s in samples.values():
        if s["ticks"] and s["gt"]:
            by_task[s["task"]].append(s)

    print(f"{args.run_dir}: {len(samples)} samples, "
          f"{sum(len(v) for v in by_task.values())} usable, "
          f"{sum(len(s['ticks']) for v in by_task.values() for s in v)} ticks\n")

    # Baseline = what the run actually shipped (whatever the live gate produced).
    hdr = (f"{'task':<28}{'n':>4}  {'mode':<6}{'thr':>6}{'refr':>6}"
           f"{'P':>7}{'R':>7}{'F1':>7}   {'held-out F1':>11}")
    print(hdr); print("-" * len(hdr))

    chosen = {}
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda s: s["video"])   # deterministic
        if args.dev > 0:
            k = max(1, int(len(rows) * args.dev))
            fit, test = rows[:k], rows[k:]
        else:
            fit, test = rows, []
        f, mode, thr, ref, p, r, n = best_config(fit)
        ho = f"{score(test, thr, mode, ref)[2]:.3f}" if test else "-"
        print(f"{task:<28}{len(rows):>4}  {mode:<6}{thr:>6.2f}{ref:>6.0f}"
              f"{p:>7.3f}{r:>7.3f}{f:>7.3f}   {ho:>11}")
        chosen[task] = {"mode": mode, "hit_threshold": thr, "refractory_s": ref,
                        "fit_precision": round(p, 4), "fit_recall": round(r, 4),
                        "fit_f1": round(f, 4), "n_fit": len(fit),
                        "heldout_f1": None if not test else round(
                            score(test, thr, mode, ref)[2], 4)}

    allr = [s for v in by_task.values() for s in v]
    f, mode, thr, ref, p, r, n = best_config(allr)
    print("-" * len(hdr))
    print(f"{'ALL (single global config)':<28}{len(allr):>4}  {mode:<6}{thr:>6.2f}"
          f"{ref:>6.0f}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
    chosen["__global__"] = {"mode": mode, "hit_threshold": thr,
                            "refractory_s": ref, "fit_f1": round(f, 4)}

    out = os.path.join(args.run_dir, args.out)
    with open(out, "w") as fh:
        json.dump({"tol_s": TOL, "run_dir": args.run_dir,
                   "dev_fraction": args.dev, "per_task": chosen}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
