#!/usr/bin/env python3
"""fit_bench_gates.py -- fit the (few) thresholds these benchmarks actually admit.

THREE thresholds total across both benchmarks; everything else runs gate-off:

  StreamingBench PO  -- 1 threshold. The harness pins the other two knobs: it
     breaks on the first fire (=> edge mode, refractory irrelevant) and bounds
     polling at ground_truth+4. Only 3,284 ticks exist across the whole task,
     ~50x less than OmniPro gave PER TASK, so the grid here is deliberately
     COARSE -- a fine grid overfits exactly the way the 60-videos x 352-config
     fit did on OmniPro.

  OVO-Bench SSR, CRR -- 1 threshold each, fitted against the HUMAN `type` label
     rather than a temporal-proximity surrogate. These can be fitted offline
     from a single probe pass because the gate does not steer the trajectory:
     we probe at benchmark-specified timestamps rather than choosing when to
     emit. PO cannot be fitted that way, which is why it needs disjoint splits.

    python fit_bench_gates.py --bench streamingbench --arm po --coarse \
        --pred $OUT/sb_po_calib --out fitted_gates_sb_po.json
    python fit_bench_gates.py --bench ovobench --arm far --tasks SSR,CRR \
        --holdout 0.3 --seeds 1234 7 99 2024 31337 \
        --pred $OUT/ovo_far_probe --out fitted_gates_ovo_far.json --auc-report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import math
import random
import statistics as st

COARSE = [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05 .. 0.95
FINE = [round(0.01 * i, 2) for i in range(1, 100)]


def _rows(pred_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "g*", "pred.jsonl"))) or \
             sorted(glob.glob(os.path.join(pred_dir, "pred.jsonl"))):
        with open(f) as fh:
            for ln in fh:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    return rows


# ---- PO: pick the threshold maximising accuracy at the reported tolerance ---
def fit_po(rows, grid, tol):
    items = []
    for r in rows:
        for p in r.get("probes", []):
            if p.get("kind") == "po" and p.get("ticks"):
                items.append(p)
    if not items:
        raise SystemExit("[fit] no PO probes with tick traces in --pred")

    def acc(thr, subset):
        hits = []
        for p in subset:
            fired = None
            for t in p["ticks"]:                # first crossing wins, as the
                v = t.get("p_hit")              # benchmark's loop does
                if v is not None and v >= thr:
                    fired = t["t"]
                    break
            hits.append(0 if fired is None
                        else int(abs(float(fired) - float(p["gt_t"])) <= tol))
        return sum(hits) / len(hits) if hits else 0.0

    scored = [(acc(t, items), t) for t in grid]
    best_acc, best_thr = max(scored)
    return {"threshold": best_thr, "mode": "edge", "refractory_s": 0.0,
            "fit_acc": best_acc, "tolerance_s": tol, "n_fit": len(items),
            "grid": "coarse" if grid is COARSE else "fine",
            "curve": {str(t): a for a, t in scored},
            "note": "mode/refractory are PINNED by the harness (it breaks on "
                    "first fire), not fitted"}


# ---- SSR / CRR: labelled binary classification -----------------------------
# Hard floor on EITHER side of the fit/report split, as a fraction of the videos
# available to that task. Enforced regardless of --holdout.
MIN_SPLIT_FRAC = 0.05


def fit_far(rows, tasks, grid, holdout, seeds):
    by_task = {}
    for r in rows:
        for p in r.get("probes", []):
            if p.get("task") in tasks and p.get("p_hit") is not None:
                by_task.setdefault(p["task"], []).append(
                    (float(p["p_hit"]), int(p.get("type", 0)), r["video_id"]))
    out = {}
    for t, pts in sorted(by_task.items()):
        def acc(thr, sub):
            if not sub:
                return 0.0
            return sum(int((v >= thr) == bool(y)) for v, y, _ in sub) / len(sub)

        # in-sample (all probes)
        scored = [(acc(th, pts), th) for th in grid]
        insample_acc, insample_thr = max(scored)
        # held-out by VIDEO, so probes from one video never straddle the split
        vids = sorted({v for _, _, v in pts})
        runs = []
        for sd in seeds:
            rnd = random.Random(sd)
            sh = list(vids)
            rnd.shuffle(sh)
            # CALIBRATION FLOOR (both sides). The fit set is what calibrates the
            # threshold and the report set is what validates it; either one falling
            # below MIN_SPLIT_FRAC of the videos makes the number meaningless.
            k = max(1, int(round(holdout * len(sh))))
            k_min = int(math.ceil(MIN_SPLIT_FRAC * len(sh)))
            k = max(k, k_min)                       # report side >= floor
            k = min(k, len(sh) - k_min)             # fit side    >= floor
            k = max(1, min(k, len(sh) - 1))
            fit_v, rep_v = set(sh[k:]), set(sh[:k])
            fit_pts = [x for x in pts if x[2] in fit_v]
            rep_pts = [x for x in pts if x[2] in rep_v]
            _, th = max((acc(th, fit_pts), th) for th in grid)
            runs.append({"seed": sd, "threshold": th, "report_acc": acc(th, rep_pts),
                         "n_fit": len(fit_pts), "n_report": len(rep_pts)})
        ha = [r["report_acc"] for r in runs]
        out[t] = {
            "threshold": insample_thr, "mode": "level", "refractory_s": 0.0,
            "n_probes": len(pts),
            "n_pos": sum(1 for _, y, _ in pts if y == 1),
            "n_neg": sum(1 for _, y, _ in pts if y == 0),
            "majority_baseline": max(
                sum(1 for _, y, _ in pts if y == 1),
                sum(1 for _, y, _ in pts if y == 0)) / len(pts),
            "insample_acc": insample_acc,
            "heldout_acc_mean": st.mean(ha) if ha else None,
            "heldout_acc_sd": (st.stdev(ha) if len(ha) > 1 else 0.0),
            "heldout_runs": runs,
            "note": "held-out split is BY VIDEO; report the held-out mean, not "
                    "insample_acc. Compare against majority_baseline -- a "
                    "threshold that cannot beat it has no signal.",
        }
    if not out:
        raise SystemExit("[fit] no SSR/CRR probes with p_hit in --pred")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["streamingbench", "ovobench"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="SSR,CRR")
    ap.add_argument("--coarse", action="store_true",
                    help="19-point 0.05 grid (default for PO: only 3,284 ticks exist)")
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="PO: the tolerance the fit optimises (headline is <=3s)")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 7, 99, 2024, 31337])
    ap.add_argument("--auc-report", action="store_true")
    args = ap.parse_args()

    grid = FINE if args.fine else COARSE
    rows = _rows(args.pred)
    if not rows:
        raise SystemExit(f"[fit] no predictions under {args.pred}")

    if args.bench == "streamingbench":
        gates = {"Proactive Output": fit_po(rows, grid, args.tolerance)}
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        gates = fit_far(rows, tasks, grid, args.holdout, args.seeds)
        if args.auc_report:
            import score_bench
            gates["_auc"] = score_bench.auc_far(rows)

    with open(args.out, "w") as fh:
        json.dump(gates, fh, indent=2)
    for k, v in gates.items():
        if k.startswith("_"):
            continue
        extra = (f" heldout={v['heldout_acc_mean']:.4f}+-{v['heldout_acc_sd']:.4f}"
                 f" (majority {v['majority_baseline']:.4f})"
                 if "heldout_acc_mean" in v and v["heldout_acc_mean"] is not None
                 else f" fit_acc={v.get('fit_acc')}")
        print(f"[fit] {k}: thr={v['threshold']} mode={v['mode']}{extra}")
    print(f"[fit] wrote {args.out}")


if __name__ == "__main__":
    main()
