"""
report.py — ONE table with everything: time-F1, content-F1, emits (with times),
ground truth (with times), and AUC-ROC of the per-tick confidence.

Scoring is delegated to metrics.py (the real OmniPro scorer: greedy ±3s temporal
match, then task-appropriate content check — exact-match parse or Gemini judge
with the persisted cache), so these numbers are the same ones the eval reports.
AUC comes from scores.jsonl, joined to the real task via online_pred.jsonl
(scores.jsonl's own `task` field holds the SHARD DIR name, not the task).

    python report.py                      # every shard under output_all
    python report.py output_all output_x  # explicit dirs
    python report.py --md REPORT.md       # also write markdown
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys


def fmt(v, nd=3, dash="UNJUDGED"):
    """Render a metric that may be withheld. metrics.aggregate returns None for
    content_acc/joint_* when any matched emit went unjudged, so a plain :.3f
    would crash — and silently substituting 0.0 would be exactly the kind of
    fake number this change exists to eliminate."""
    return dash if v is None else f"{v:.{nd}f}"



def _load(dirs):
    preds, scores = [], []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "online_pred.jsonl"), recursive=True):
            for line in open(f):
                line = line.strip()
                if line:
                    preds.append(json.loads(line))
        for f in glob.glob(os.path.join(d, "**", "scores.jsonl"), recursive=True):
            for line in open(f):
                line = line.strip()
                if line:
                    scores.append(json.loads(line))
    return preds, scores


def auc_roc(pairs):
    """Rank-based AUC with tie correction. pairs = [(score, label)]."""
    pos = sum(1 for _, y in pairs if y)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    srt = sorted(pairs, key=lambda p: p[0])
    ranks, i = {}, 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1][0] == srt[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s = sum(ranks[k] for k in range(len(srt)) if srt[k][1] == 1)
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def _fmt_times(ts, cap=6):
    if not ts:
        return "—"
    ts = sorted(float(t) for t in ts)
    shown = ", ".join(f"{t:.0f}" for t in ts[:cap])
    return shown + (f" …+{len(ts) - cap}" if len(ts) > cap else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=None)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--md", default=None, help="also write a markdown report here")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dirs = args.dirs or [os.path.join(here, "output_all")]
    preds, scores = _load(dirs)
    if not preds:
        print("no online_pred.jsonl found under: " + ", ".join(dirs))
        return 1

    from metrics import ContentJudge, aggregate, score_sample
    judge = ContentJudge()
    per_sample = [score_sample(p, tolerance=args.tolerance, judge=judge) for p in preds]
    agg = aggregate(per_sample)

    # ---- AUC per task: join scores.jsonl -> real task via video_id ------------
    vid2task, vid2gt = {}, {}
    for p in preds:
        vid2task[p["video_id"]] = p["task"]
        vid2gt[p["video_id"]] = [float(g["t_sec"]) for g in p.get("ground_truth", [])
                                 if isinstance(g, dict) and g.get("t_sec") is not None]
    by_task_pairs = collections.defaultdict(list)
    ticks_by_task = collections.Counter()
    for s in scores:
        v = s.get("video_id")
        if v not in vid2task:            # video still in flight
            continue
        t = vid2task[v]
        y = 1 if any(abs(s["vt"] - g) <= args.tolerance for g in vid2gt[v]) else 0
        by_task_pairs[t].append((s["p_hit"], y))
        ticks_by_task[t] += 1

    # ---- emit / GT times per task -------------------------------------------
    emits_by_task, gt_by_task, vids_by_task = (collections.defaultdict(list),
                                               collections.defaultdict(list),
                                               collections.defaultdict(set))
    for p in preds:
        t = p["task"]
        vids_by_task[t].add(p["video_id"])
        emits_by_task[t] += [float(e["t_sec"]) for e in p.get("predictions", [])]
        gt_by_task[t] += [float(g["t_sec"]) for g in p.get("ground_truth", [])
                          if isinstance(g, dict) and g.get("t_sec") is not None]

    tasks = sorted(agg["per_task"], key=lambda t: -agg["per_task"][t]["n_samples"])

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out(f"# Eval report — {len(preds)} samples, {len(scores)} ticks "
        f"(tolerance ±{args.tolerance:g}s)")
    out()
    out("| task | vids | GT | emits | time-F1 | content-F1 | content-acc | "
        "prec | rec | AUC | ticks | pos% |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in tasks:
        b = agg["per_task"][t]
        pr = by_task_pairs.get(t, [])
        a = auc_roc(pr)
        pos = (100.0 * sum(y for _, y in pr) / len(pr)) if pr else 0.0
        out(f"| {t} | {len(vids_by_task[t])} | {b['n_gt']} | {b['n_emits']} | "
            f"{b['time_f1']:.3f} | {fmt(b['joint_f1'])} | {fmt(b['content_acc'])} | "
            f"{b['time_precision']:.3f} | {b['time_recall']:.3f} | "
            f"{'—' if a is None else f'{a:.3f}'} | {ticks_by_task[t]} | {pos:.1f}% |")
    o = agg["overall"]
    allp = [p for v in by_task_pairs.values() for p in v]
    a = auc_roc(allp)
    out(f"| **OVERALL** | {len(set().union(*vids_by_task.values()))} | {o['n_gt']} | "
        f"{o['n_emits']} | **{o['time_f1']:.3f}** | **{fmt(o['joint_f1'])}** | "
        f"{fmt(o['content_acc'])} | {o['time_precision']:.3f} | {o['time_recall']:.3f} | "
        f"{'—' if a is None else f'{a:.3f}'} | {len(allp)} | "
        f"{100.0*sum(y for _,y in allp)/len(allp) if allp else 0:.1f}% |")
    out()
    out("`time-F1` = ±tol temporal match only. `content-F1` = match must ALSO be "
        "content-correct (OmniPro joint). `content-acc` = of temporally matched "
        "emits, the fraction also content-correct. `AUC` = ROC of per-tick `p_hit` "
        "against a ±tol positive label; **0.5 = chance**.")
    out()
    out("## Emit times vs ground-truth times")
    out()
    out("| task | GT times (s) | emit times (s) |")
    out("|---|---|---|")
    for t in tasks:
        out(f"| {t} | {_fmt_times(gt_by_task[t])} | {_fmt_times(emits_by_task[t])} |")
    out()
    out("## Per sample")
    out()
    out("| task | video | GT | emits | tp_time | tp_content | fp | fn |")
    out("|---|---|---:|---:|---:|---:|---:|---:|")
    idx = {p["id"]: p for p in preds}
    for s in sorted(per_sample, key=lambda x: (x["task"], x["id"])):
        p = idx[s["id"]]
        out(f"| {s['task']} | {p['video_id']} | {s['n_gt']} | {s['n_emits']} | "
            f"{s['tp_time']} | {s['tp_content']} | {s['fp']} | {s['fn']} |")

    if args.md:
        with open(args.md, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n[report] wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
