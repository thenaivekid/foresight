"""
refit.py — re-threshold a finished run and RESCORE IT FULLY (time + content).

Difference from resweep.py: resweep only replays emit TIMES, so it can report
time-F1 but not content_acc or joint-F1. This one filters the run's ACTUAL
predictions -- which carry the writer's text -- and pushes them back through
metrics.score_sample, the same function evaluate.py uses. Every column is
therefore directly comparable to the shipped online_metrics.json.

WHY IT CAN ONLY SUBTRACT EMITS
------------------------------
The writer only ran on ticks the live gate fired, so text exists only for those.
A swept config that fires somewhere the live gate did not has no text to score
and would be scored blind. So the candidate set is exactly the live emits, and
the sweep chooses which to KEEP. That is not a limitation here: this run emits
6.4x more than there are ground-truth events, so the whole problem is removing
emits, not adding them.

The rule applied to each live emit:
    keep if p_hit(at its tick) >= thr  AND  >= refractory seconds since the last
    kept emit.
Both terms look only at the past, so the filter is a legal streaming gate --
it could be run online unchanged.

NOTE ON `--dev`: default 0 fits and reports on the SAME samples. That is a
CEILING, not a reportable number: it tells you what the signal supports if the
threshold were chosen perfectly. Use --dev 0.5 for an honest held-out figure.

Usage:
    python refit.py RUN_DIR                 # ceiling (fit on all)
    python refit.py RUN_DIR --dev 0.5       # honest held-out
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics
from resweep import parse_run

TOL = 3.0
THRS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
REFRS = [0.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]


def load_preds(run_dir):
    """-> {sample_id: pred_record} from every completed shard."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "**", "online_pred.jsonl"),
                              recursive=True)):
        with open(p, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["id"]] = r          # later shard/gen wins, same as scoring
    return out


def attach_phit(preds, ticks_by_sample):
    """Give every prediction the p_hit of the tick it was emitted on.

    Emit times are tick times, but float formatting can differ by a hair, so
    match to the nearest tick within half a tick.
    """
    missing = 0
    for sid, rec in preds.items():
        ticks = sorted(ticks_by_sample.get(sid, {}).get("ticks", []))
        for e in rec.get("predictions", []):
            t = float(e["t_sec"])
            best, bd = None, None
            for vt, p in ticks:
                d = abs(vt - t)
                if bd is None or d < bd:
                    bd, best = d, p
            if best is not None and bd <= 0.51:
                e["_p"] = best
            else:
                e["_p"] = 1.0            # no tick found: never filtered out
                missing += 1
    return missing


def filter_preds(rec, thr, refr):
    """Apply the streaming keep-rule; return a copy with predictions filtered.

    De-duplicates identical timestamps first: the run emits several predictions
    at the same t_sec (writer shares), and the live scorer counts each as its own
    emit. Keeping that behaviour makes the thr=0/refr=0 row reproduce the shipped
    numbers exactly, which is how we verify this file is faithful.
    """
    kept, last = [], None
    for e in sorted(rec.get("predictions", []), key=lambda x: float(x["t_sec"])):
        t = float(e["t_sec"])
        if e.get("_p", 1.0) < thr:
            continue
        if last is not None and t - last < refr:
            continue
        kept.append(e)
        last = t
    out = dict(rec)
    out["predictions"] = kept
    return out


def score(records, thr, refr, judge):
    per = [metrics.score_sample(filter_preds(r, thr, refr), TOL, judge)
           for r in records]
    return metrics.aggregate(per)


def best_for(records, judge, key="time_f1"):
    best = None
    for thr in THRS:
        for refr in REFRS:
            agg = score(records, thr, refr, judge)["overall"]
            if best is None or agg[key] > best[0][key]:
                best = (agg, thr, refr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dev", type=float, default=0.0,
                    help="fraction fitted on; rest held out. 0 = fit on all (ceiling)")
    ap.add_argument("--metric", default="time_f1", choices=["time_f1", "joint_f1"])
    ap.add_argument("--out", default="thresholds_refit.json")
    args = ap.parse_args()

    preds = load_preds(args.run_dir)
    ticks = parse_run(args.run_dir)
    miss = attach_phit(preds, ticks)
    judge = metrics.ContentJudge()

    by_task = defaultdict(list)
    for sid, r in preds.items():
        by_task[r["task"]].append(r)
    allrec = [r for v in by_task.values() for r in v]

    print(f"{args.run_dir}: {len(allrec)} samples, "
          f"{sum(len(r.get('predictions', [])) for r in allrec)} live emits, "
          f"{miss} emits without a matching tick | judge={judge.mode}")

    base = score(allrec, 0.0, 0.0, judge)["overall"]
    print(f"baseline (keep every emit) = tF1 {base['time_f1']:.3f} "
          f"jF1 {base['joint_f1']:.3f} cAcc {base['content_acc']:.3f}"
          "   <- must match the shipped online_metrics.json\n")

    hdr = (f"{'task':<28}{'n':>4}{'thr':>7}{'refr':>6}{'emits':>7}"
           f"{'tP':>7}{'tR':>7}{'tF1':>7}{'jF1':>7}{'cAcc':>7}")
    print(hdr); print("-" * len(hdr))

    chosen, held = {}, []
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda r: r["id"])
        if args.dev > 0:
            k = max(1, int(len(rows) * args.dev))
            fit, test = rows[:k], rows[k:]
        else:
            fit, test = rows, rows
        agg, thr, refr = best_for(fit, judge, args.metric)
        rep = score(test, thr, refr, judge)["overall"] if test else agg
        held += [filter_preds(r, thr, refr) for r in test]
        print(f"{task:<28}{len(rows):>4}{thr:>7.3f}{refr:>6.0f}{rep['n_emits']:>7}"
              f"{rep['time_precision']:>7.3f}{rep['time_recall']:>7.3f}"
              f"{rep['time_f1']:>7.3f}{rep['joint_f1']:>7.3f}"
              f"{rep['content_acc']:>7.3f}")
        chosen[task] = {"hit_threshold": thr, "refractory_s": refr,
                        "n_samples": len(rows),
                        **{k2: rep[k2] for k2 in
                           ("time_precision", "time_recall", "time_f1",
                            "joint_f1", "content_acc", "n_emits")}}

    pooled = metrics.aggregate(
        [metrics.score_sample(r, TOL, judge) for r in held])["overall"]
    print("-" * len(hdr))
    print(f"{'POOLED (per-task thresholds)':<28}{len(held):>4}{'':>7}{'':>6}"
          f"{pooled['n_emits']:>7}{pooled['time_precision']:>7.3f}"
          f"{pooled['time_recall']:>7.3f}{pooled['time_f1']:>7.3f}"
          f"{pooled['joint_f1']:>7.3f}{pooled['content_acc']:>7.3f}")

    gagg, gthr, grefr = best_for(allrec, judge, args.metric)
    print(f"{'GLOBAL (one threshold)':<28}{len(allrec):>4}{gthr:>7.3f}{grefr:>6.0f}"
          f"{gagg['n_emits']:>7}{gagg['time_precision']:>7.3f}"
          f"{gagg['time_recall']:>7.3f}{gagg['time_f1']:>7.3f}"
          f"{gagg['joint_f1']:>7.3f}{gagg['content_acc']:>7.3f}")

    out = os.path.join(args.run_dir, args.out)
    with open(out, "w") as fh:
        json.dump({"tol_s": TOL, "run_dir": args.run_dir, "dev_fraction": args.dev,
                   "fit_metric": args.metric, "judge_mode": judge.mode,
                   "baseline_no_filter": base, "pooled": pooled,
                   "global": {"hit_threshold": gthr, "refractory_s": grefr, **gagg},
                   "per_task": chosen}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
