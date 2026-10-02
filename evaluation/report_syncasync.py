#!/usr/bin/env python3
"""report_syncasync.py -- per-task metrics and wall-clock/real-time for the
sync vs async A/B, plus the RT-factor distribution.

Two sources, deliberately:

  * METRICS come from each shard's `online_metrics.json`, which the harness
    writes with the judge active. Re-scoring the JSONL here with judge=None
    would silently report every content verdict as UNJUDGED, so the counts are
    aggregated across shards instead and the rates recomputed from them.
  * TIMING comes from `online_pred.jsonl` (`wall_s`, written by evaluate.py:91)
    joined to the video duration from the split file, because the prediction
    record does not carry duration.

The report is a pure function of those files, so it can be run at any point in
the chain and rebuilt after every resume.

RT = wall_s / video duration; a streaming system must consume one second of
footage in at most one second of compute, so RT < 1 keeps pace. The two arms
are NOT comparable on RT in the same sense:

  sync   no wall-clock pacing (adapter:235 realtime=False), so RT measures pure
         compute -- "how fast could it go".
  async  the encoder paces to the stream, so RT is bounded below by ~1.0 on any
         video it keeps up with, and RT > 1 means it fell behind -- "did it
         keep up".

    python report_syncasync.py --out output_syncasync
    python report_syncasync.py --out output_syncasync --csv rt.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["cumulative_counting", "dedup_counting", "event_narration",
         "explicit_target_grounding", "instant_event_alert",
         "realtime_state_monitor", "semantic_condition_alert",
         "sequential_step_instruction", "snapshot_counting"]
SHORT = {t: t.replace("_counting", "_cnt").replace("_instruction", "_inst")[:21]
         for t in TASKS}
N_EXPECTED = 135


def durations(split_dir):
    """sample id -> video duration, from the split that defined the run."""
    p = os.path.join(split_dir, "pct5_all.json")
    if not os.path.isfile(p):
        return {}
    return {e["id"]: float(e["duration"]) for e in json.load(open(p))
            if e.get("duration")}


def load_metrics(arm_dir):
    """Aggregate per-task counts across shards -> task -> dict of totals."""
    agg = defaultdict(lambda: defaultdict(float))
    for p in sorted(glob.glob(os.path.join(arm_dir, "g*", "online_metrics*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for t, v in (d.get("per_task") or {}).items():
            a = agg[t]
            a["n"] += v.get("n_samples", 0)
            a["gt"] += v.get("n_gt", 0)
            a["emits"] += v.get("n_emits", 0)
            a["matched"] += v.get("n_matched", 0)
            nj = v.get("n_judged", 0)
            a["judged"] += nj
            ca = v.get("content_acc")
            if ca is not None and nj:
                a["correct"] += ca * nj          # back out the count
    return agg


def load_timing(arm_dir, dur):
    """task -> list of (wall_s, duration_s, rt), deduped on sample id."""
    seen = {}
    for p in sorted(glob.glob(os.path.join(arm_dir, "g*", "online_pred.jsonl"))):
        for line in open(p, errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue                      # a link killed mid-write
            if r.get("id"):
                seen[r["id"]] = r             # last write wins, as resume does
    per = defaultdict(list)
    for sid, r in seen.items():
        w, d = r.get("wall_s"), dur.get(sid)
        if w and d:
            per[r["task"]].append((float(w), float(d), float(w) / float(d)))
    return per, len(seen)


def f1(tp, n_emits, n_gt):
    """Both F1s collapse to 2*tp/(emits+gt): fp = emits-tp, fn = gt-tp."""
    den = n_emits + n_gt
    return (2.0 * tp / den) if den else float("nan")


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def fm(x, n=3, w=6):
    return f"{'--':>{w}}" if x != x else f"{x:.{n}f}".rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "output_syncasync"))
    ap.add_argument("--splits", default=os.path.join(HERE, "splits_pct5"))
    ap.add_argument("--csv", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    dur = durations(a.splits)
    M, T, N = {}, {}, {}
    for m in ("sync", "async"):
        d = os.path.join(a.out, m)
        M[m] = load_metrics(d) if os.path.isdir(d) else {}
        T[m], N[m] = load_timing(d, dur) if os.path.isdir(d) else ({}, 0)

    print(f"\n{'='*84}\nsync vs async  --  5% OmniPro split, 135 samples (15/task), "
          f"Qwen3-VL-8B + shipped gates\n{'='*84}")
    for m in ("sync", "async"):
        pct = 100 * N[m] / N_EXPECTED
        bar = "#" * int(pct / 4) + "." * (25 - int(pct / 4))
        print(f"  {m:<6} [{bar}] {N[m]:>3}/{N_EXPECTED}  ({pct:>3.0f}%)")
    if not any(N.values()):
        print(f"\n  nothing yet -- no online_pred.jsonl under {a.out}\n")
        return

    # ---- per-task metrics --------------------------------------------------
    print(f"\nPER-TASK METRICS            sync: time / cAcc / joint"
          f"        async: time / cAcc / joint")
    print("-" * 84)
    mac = defaultdict(list)
    for t in TASKS:
        cells = []
        for m in ("sync", "async"):
            v = M[m].get(t)
            if not v or not v["gt"]:
                cells.append(" " * 28)
                continue
            tf = f1(v["matched"], v["emits"], v["gt"])
            ca = v["correct"] / v["judged"] if v["judged"] else float("nan")
            jf = (f1(v["correct"], v["emits"], v["gt"])
                  if v["judged"] else float("nan"))
            mac[(m, "t")].append(tf)
            if ca == ca:
                mac[(m, "c")].append(ca); mac[(m, "j")].append(jf)
            cells.append(f"  {fm(tf)} / {fm(ca)} / {fm(jf)}")
        print(f"{SHORT[t]:<22}{cells[0]}   {cells[1]}")
    print("-" * 84)
    cells = []
    for m in ("sync", "async"):
        g = lambda k: st.mean(mac[(m, k)]) if mac[(m, k)] else float("nan")
        cells.append(f"  {fm(g('t'))} / {fm(g('c'))} / {fm(g('j'))}")
    print(f"{'MACRO':<22}{cells[0]}   {cells[1]}")

    # ---- wall clock vs real time ------------------------------------------
    print(f"\nWALL CLOCK vs REAL TIME     (RT = wall_s / video_s;  <1 keeps pace)")
    print(f"{'task':<22}{'footage':>9}{'sync wall':>11}{'sync RT':>9}"
          f"{'async wall':>12}{'async RT':>10}")
    print("-" * 73)
    allrt = {"sync": [], "async": []}
    for t in TASKS:
        foot = None
        cells = {}
        for m in ("sync", "async"):
            v = T[m].get(t, [])
            if v:
                cells[m] = (st.mean(x[0] for x in v), st.mean(x[2] for x in v))
                allrt[m] += [x[2] for x in v]
                foot = st.mean(x[1] for x in v)
        line = f"{SHORT[t]:<22}{(f'{foot:.0f}s' if foot else '--'):>9}"
        line += (f"{cells['sync'][0]:>11.0f}{cells['sync'][1]:>9.2f}"
                 if "sync" in cells else f"{'--':>11}{'--':>9}")
        line += (f"{cells['async'][0]:>12.0f}{cells['async'][1]:>10.2f}"
                 if "async" in cells else f"{'--':>12}{'--':>10}")
        print(line)
    print("-" * 73)
    for m in ("sync", "async"):
        if allrt[m]:
            print(f"{m+' OVERALL':<22}{'':>9}"
                  f"{'':>11}{'' if m=='async' else f'{st.mean(allrt[m]):>9.2f}'}"
                  f"{'':>12}{f'{st.mean(allrt[m]):>10.2f}' if m=='async' else ''}")

    # ---- RT distribution ---------------------------------------------------
    print(f"\nRT-FACTOR DISTRIBUTION")
    print(f"{'arm':<8}{'n':>5}{'min':>8}{'p25':>8}{'median':>8}{'p75':>8}"
          f"{'p90':>8}{'max':>8}{'RT>1':>7}")
    print("-" * 68)
    for m in ("sync", "async"):
        xs = allrt[m]
        if not xs:
            print(f"{m:<8}{'--':>5}"); continue
        over = 100 * sum(1 for x in xs if x > 1.0) / len(xs)
        print(f"{m:<8}{len(xs):>5}{min(xs):>8.3f}{q(xs,.25):>8.3f}"
              f"{q(xs,.5):>8.3f}{q(xs,.75):>8.3f}{q(xs,.9):>8.3f}"
              f"{max(xs):>8.3f}{over:>6.0f}%")
    # coarse histogram, so the tail is visible without a plot
    for m in ("sync", "async"):
        xs = allrt[m]
        if not xs:
            continue
        edges = [0, .25, .5, .75, 1.0, 1.5, 2.0, 3.0, 1e9]
        lab = ["<.25", ".25-.5", ".5-.75", ".75-1", "1-1.5", "1.5-2", "2-3", ">3"]
        print(f"\n  {m} histogram")
        for lo, hi, L in zip(edges, edges[1:], lab):
            k = sum(1 for x in xs if lo <= x < hi)
            print(f"    {L:>7} |{'#' * int(40 * k / len(xs)):<40} {k:>3}")
    print()

    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("arm,task,wall_s,duration_s,rt\n")
            for m in ("sync", "async"):
                for t, v in T[m].items():
                    for w, d, rt in v:
                        fh.write(f"{m},{t},{w:.2f},{d:.2f},{rt:.4f}\n")
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
