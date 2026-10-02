#!/usr/bin/env python3
"""score_bench.py -- score one arm into JSON, and roll all arms into RESULTS.json.

Aggregation follows each benchmark's OWN scorer, not a convenient average:

OVO-Bench (utils/OVOBenchScore.py):
  * BT / RVP : int(ground_truth in response) per question -> accuracy per task
               -> MACRO over the tasks in the mode
  * FAR      : scored PER PROBE, pooled across probes (629 SSR / 240 CRR /
               698 REC -- NOT 42 / 48 / 82 questions) -> macro over the 3 tasks
  * Total    : (backward + realtime + forward) / 3
StreamingBench: option-letter accuracy per task; PO is |answered - GT| <= x for
  x in 1..4 (above 4 is meaningless: the harness stops polling at GT+4).

No LLM judge is involved anywhere -- every metric here is deterministic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

OVO_BACKWARD = ["EPM", "ASI", "HLD"]
OVO_REALTIME = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
OVO_FORWARD = ["REC", "SSR", "CRR"]


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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _macro(per_task):
    vals = [v["acc"] for v in per_task.values() if v["acc"] is not None]
    return sum(vals) / len(vals) if vals else None


# ---- OVO -------------------------------------------------------------------
def score_ovo(rows, gates=None):
    """gates: {task: {"threshold": t}} from fit_bench_gates.py.

    WHY THIS MATTERS. bench_probe.py records `response` using a FIXED 0.5
    threshold, because it has to write something at probe time. Scoring the
    fitted gate therefore CANNOT reuse that field -- it must re-derive Yes/No
    from the stored p_hit at the fitted threshold. Without this the fit would
    have no effect on the reported number, which is the quiet way a threshold
    experiment reports the pre-fit result.
    """
    per_task = {}
    for r in rows:
        for p in r.get("probes", []):
            t = p.get("task")
            if t is None:
                continue
            b = per_task.setdefault(t, {"hits": [], "n": 0})
            b["n"] += 1
            resp = p.get("response")
            if t == "REC":
                # OVOBenchScore: digits joined must equal str(count) exactly
                if resp is None:
                    b["hits"].append(0)
                else:
                    digits = "".join(re.findall(r"\d+", str(resp)))
                    b["hits"].append(int(digits == str(p.get("count"))))
            elif t in ("SSR", "CRR"):
                gt = "No" if int(p.get("type", 0)) == 0 else "Yes"
                thr = (gates or {}).get(t, {}).get("threshold")
                if thr is not None and p.get("p_hit") is not None:
                    resp = "Yes" if float(p["p_hit"]) >= float(thr) else "No"
                b["hits"].append(0 if resp is None else int(gt in str(resp)))
            else:
                gt = p.get("answer")
                b["hits"].append(0 if (resp is None or gt is None)
                                 else int(str(gt) in str(resp)))
    for t, b in per_task.items():
        b["acc"] = (sum(b["hits"]) / len(b["hits"])) if b["hits"] else None
        b["n_scored"] = len(b["hits"])
        del b["hits"]

    modes = {}
    for name, tasks in (("backward", OVO_BACKWARD), ("realtime", OVO_REALTIME),
                        ("forward", OVO_FORWARD)):
        sub = {t: per_task[t] for t in tasks if t in per_task}
        modes[name] = {"per_task": sub, "macro": _macro(sub)}
    present = [m["macro"] for m in modes.values() if m["macro"] is not None]
    return {"per_task": per_task, "modes": modes,
            "total_avg": (sum(present) / len(present)) if present else None,
            "note": "per-probe denominators for FAR; macro-over-task per mode; "
                    "total_avg = mean of the modes present"}


# ---- StreamingBench --------------------------------------------------------
def score_sb_mcq(rows):
    per_task, per_cat = {}, {}
    for r in rows:
        for p in r.get("probes", []):
            t, c = p.get("task"), p.get("category", "unknown")
            if t is None or p.get("kind") != "mcq":
                continue
            b = per_task.setdefault(t, {"hits": [], "category": c})
            gt, resp = p.get("answer"), p.get("response")
            b["hits"].append(0 if (resp is None or gt is None)
                             else int(str(gt) in str(resp)))
    for t, b in per_task.items():
        b["acc"] = (sum(b["hits"]) / len(b["hits"])) if b["hits"] else None
        b["n_scored"] = len(b["hits"])
        per_cat.setdefault(b["category"], {})[t] = b
        del b["hits"]
    cats = {c: {"per_task": d, "macro": _macro(d)} for c, d in per_cat.items()}
    return {"per_task": per_task, "categories": cats}


def score_sb_po(rows):
    tols = [1, 2, 3, 4]
    recs = []
    for r in rows:
        for p in r.get("probes", []):
            if p.get("kind") != "po":
                continue
            recs.append(p)
    out = {"n": len(recs), "protocol": recs[0].get("protocol") if recs else None,
           "threshold": recs[0].get("threshold") if recs else None,
           "n_fired": sum(1 for p in recs if p.get("answered") is not None),
           "acc_by_tolerance": {}}
    for x in tols:
        hits = []
        for p in recs:
            a, gt = p.get("answered"), p.get("gt_t")
            hits.append(0 if a is None else int(abs(float(a) - float(gt)) <= x))
        out["acc_by_tolerance"][f"<={x}s"] = (sum(hits) / len(hits)) if hits else None
    # |answered - GT| distribution, for the timing-error histogram
    errs = [abs(float(p["answered"]) - float(p["gt_t"]))
            for p in recs if p.get("answered") is not None]
    out["abs_error_s"] = {"n": len(errs), "mean": _mean(errs),
                          "median": (sorted(errs)[len(errs) // 2] if errs else None)}
    out["note"] = ("tolerances above 4 s are meaningless in protocol b1: the "
                   "benchmark stops polling at ground_truth+4")
    return out


# ---- the labelled have_enough_info AUC (the headline diagnostic) -----------
def auc_far(rows):
    """AUC of p_hit against the human `type` label on SSR and CRR.

    This is the measurement OVO-Bench exists here for: 869 labelled binary
    points, versus the +-3 s temporal-proximity surrogate the OmniPro work had
    to fit against (which gave AUC 0.541 vs 0.5 chance).
    """
    def _auc(pos, neg):
        if not pos or not neg:
            return None
        # rank-based (Mann-Whitney U), ties get average rank
        xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
        ranks, i = {}, 0
        vals = [v for v, _ in xs]
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[j + 1] == vals[i]:
                j += 1
            r = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[k] = r
            i = j + 1
        s = sum(ranks[k] for k, (_, y) in enumerate(xs) if y == 1)
        n1, n0 = len(pos), len(neg)
        return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    buckets = {}
    for r in rows:
        for p in r.get("probes", []):
            if p.get("task") not in ("SSR", "CRR") or p.get("p_hit") is None:
                continue
            b = buckets.setdefault(p["task"], {"pos": [], "neg": []})
            (b["pos"] if int(p.get("type", 0)) == 1 else b["neg"]).append(float(p["p_hit"]))
    out = {}
    allp, alln = [], []
    for t, b in buckets.items():
        out[t] = {"n_pos": len(b["pos"]), "n_neg": len(b["neg"]),
                  "auc_p_hit": _auc(b["pos"], b["neg"])}
        allp += b["pos"]
        alln += b["neg"]
    out["pooled"] = {"n_pos": len(allp), "n_neg": len(alln),
                     "auc_p_hit": _auc(allp, alln)}
    out["reference"] = {"omnipro_surrogate_auc": 0.541, "chance": 0.5,
                        "note": "OmniPro fitted against a +-3s proximity "
                                "surrogate; this is a human label"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["streamingbench", "ovobench"])
    ap.add_argument("--arm")
    ap.add_argument("--pred")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", help="roll every results_*.json in this dir up")
    ap.add_argument("--gates", help="fitted gates json; SSR/CRR are then scored "
                                    "at the fitted threshold, re-derived from p_hit")
    ap.add_argument("--expected", type=int, default=0,
                    help="videos this arm will ultimately cover; records progress "
                         "and marks the result PARTIAL until it is reached")
    args = ap.parse_args()

    if args.combine:
        combined = {}
        for f in sorted(glob.glob(os.path.join(args.combine, "results_*.json"))):
            arm = os.path.basename(f)[len("results_"):-len(".json")]
            try:
                combined[arm] = json.load(open(f))
            except Exception as exc:
                combined[arm] = {"error": str(exc)}
        with open(args.out, "w") as fh:
            json.dump(combined, fh, indent=2)
        print(f"[score] combined {len(combined)} arms -> {args.out}")
        return

    if not (args.bench and args.arm and args.pred):
        sys.exit("need --bench --arm --pred (or --combine)")
    rows = _rows(args.pred)
    import datetime
    res = {"arm": args.arm, "bench": args.bench, "n_videos": len(rows),
           "n_errors": sum(1 for r in rows if "error" in r),
           "n_probes": sum(len(r.get("probes", [])) for r in rows),
           "scored_at": datetime.datetime.now().isoformat(timespec="seconds")}
    # A mid-run arm is scored on the videos finished SO FAR. Shards are balanced
    # and consumed in manifest order, so a partial number is a real sample rather
    # than a biased prefix -- but it is still a sample, and must never be read as
    # the final figure. Hence the explicit flag rather than a silent number.
    if args.expected:
        res["expected_videos"] = args.expected
        res["progress"] = round(len(rows) / args.expected, 4)
        res["partial"] = len(rows) < args.expected
    else:
        res["partial"] = None
    if args.bench == "ovobench":
        gates = None
        if args.gates and os.path.exists(args.gates):
            gates = {k: v for k, v in json.load(open(args.gates)).items()
                     if not k.startswith("_")}
            res["scored_at_fitted_gates"] = {
                k: v.get("threshold") for k, v in gates.items()}
        res.update(score_ovo(rows, gates))
        if any(p.get("task") in ("SSR", "CRR")
               for r in rows for p in r.get("probes", [])):
            res["have_enough_info_auc"] = auc_far(rows)
    else:
        if "po" in args.arm:
            res.update(score_sb_po(rows))
        else:
            res.update(score_sb_mcq(rows))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[score] {args.arm}: {res['n_videos']} videos, {res['n_probes']} probes "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
