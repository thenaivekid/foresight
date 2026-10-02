"""
retime.py — what does the emission TIMESTAMP choice cost us?

`controller.py:596` does not stamp an emission with the time it was emitted. It
stamps it with the model's own claim about when the event happened, floored 10 s
in the past:

    t_rec = min(vt, max(vt - 10.0, float(ev)))        # ev = model's event_time_s

That constant was introduced without a derivation or an experiment.
OmniPro defines the response timestamp as the video-second of EMISSION, so this is
ours, not the benchmark's -- and at 10 s it is more than 3x the +-3 s matching
tolerance, which means it can flip a miss into a match on its own.

This re-scores a completed run under several timestamp policies, offline, from the
saved logs. No GPU, no video, no re-inference: the logs already contain both `vt`
and the model's `event_time_s` for every fired tick, so this is analysis of saved
numbers -- consistent with the causality invariant.

Policies:
    vt         stamp when we actually spoke.  PROTOCOL-FAITHFUL.
    clamp10    today's behaviour, for reference.
    clamp3     the same idea sized to the tolerance instead of 3x it.
    ev         the model's raw claim, unclamped. NOT proactivity -- this measures
               retrospective localisation, and is here to show the ceiling.

TIME metrics are exact and need no judge. CONTENT is scored only where the rules
are mechanical (count / position / state / time_only); the two free-text tasks
need an LLM judge and are reported as UNJUDGED rather than guessed (there is
deliberately no fallback judge; see metrics.py).

Usage:
    python retime.py --ticks ticks.jsonl --run RUN_DIR
    python retime.py --ticks ticks.jsonl --run RUN_DIR --task dedup_counting
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M  # noqa: E402


class NoJudge:
    """A judge that judges nothing.

    The two free-text tasks (event_narration, sequential_step_instruction) need an
    LLM verdict. Calling a real judge here would mean one call per matched emit per
    POLICY -- and the matches differ per policy, so the cache cannot absorb it.
    Returning None makes metrics.py mark them UNJUDGED and withhold joint-F1 for
    those tasks, which is the documented behaviour and is honest. Time metrics are
    unaffected: they never involve the judge."""

    def score(self, *a, **k):
        return None


POLICIES = {
    "vt":      lambda vt, ev: vt,
    "clamp10": lambda vt, ev: vt if ev is None else min(vt, max(vt - 10.0, ev)),
    "clamp3":  lambda vt, ev: vt if ev is None else min(vt, max(vt - 3.0, ev)),
    "ev":      lambda vt, ev: vt if ev is None else min(vt, ev),
}


def load_gt(run_dir, split_glob):
    """Sample records (with ground_truth) keyed by id.

    Prefer the run's own online_pred.jsonl -- it is what was actually scored -- and
    fall back to the split files it was generated from."""
    gt = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "online_pred.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            gt[r["id"]] = r
    if not gt:
        for p in sorted(glob.glob(split_glob)):
            for s in json.load(open(p)):
                s.setdefault("ground_truth", [])
                gt[s["id"]] = s
    # normalise gt times to t_sec, the field score_sample reads
    for r in gt.values():
        for g in r.get("ground_truth", []):
            if "t_sec" not in g and "trigger_time_sec" in g:
                g["t_sec"] = float(g["trigger_time_sec"])
    return gt


def load_fires(ticks_path, keep_tasks=None):
    """-> {sample_id: [(vt, ev, answer_text), ...]} for every FIRED tick."""
    out = defaultdict(list)
    n_fire = n_trunc = 0
    for line in open(ticks_path):
        r = json.loads(line)
        if not r.get("fire"):
            continue
        if keep_tasks and r.get("task") not in keep_tasks:
            continue
        n_fire += 1
        if r.get("raw_state") == "truncated":
            n_trunc += 1
        d = r.get("diff") or {}
        ev = d.get("event_time_s")
        try:
            ev = float(ev)
        except (TypeError, ValueError):
            ev = None
        sid = f'{r["task"]}::{r["video"]}::{r["sample"]}'
        out[sid].append((float(r["vt"]), ev, str(d.get("answer") or "")))
    return out, n_fire, n_trunc


def score_all(gt, fires, policy, tolerance):
    fn = POLICIES[policy]
    judge = NoJudge()
    rows = []
    for sid, ticks in fires.items():
        rec = gt.get(sid)
        if rec is None:
            continue
        pred = dict(rec)
        # sort on vt only: `ev` is None on ticks where the model emitted no
        # event_time_s, and tuple ordering would compare None to a float.
        pred["predictions"] = [{"t_sec": fn(vt, ev), "raw": ans}
                               for vt, ev, ans in sorted(ticks, key=lambda x: x[0])]
        rows.append(M.score_sample(pred, tolerance=tolerance, judge=judge))
    return M.aggregate(rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", required=True, help="jsonl from fields.py --jsonl")
    ap.add_argument("--run", default="output", help="run dir with online_pred")
    ap.add_argument("--splits", default="splits_all/*.json")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--task", action="append")
    ap.add_argument("--dedup", action="store_true",
                    help="replay emission-suppression rules")
    a = ap.parse_args()

    gt = load_gt(a.run, a.splits)
    fires, n_fire, n_trunc = load_fires(a.ticks, set(a.task) if a.task else None)
    matched = sum(1 for s in fires if s in gt)
    print(f"[retime] {len(gt)} samples with ground truth")
    print(f"[retime] {n_fire} fired ticks over {len(fires)} samples "
          f"({matched} joined to GT; {n_trunc} from truncated log lines)")
    print(f"[retime] tolerance +-{a.tolerance}s\n")

    results = {}
    for pol in ("vt", "clamp3", "clamp10", "ev"):
        agg, _ = score_all(gt, fires, pol, a.tolerance)
        results[pol] = agg

    o = lambda p: results[p]["overall"]
    print("OVERALL — time metrics are exact (no judge involved)\n")
    print(f"{'policy':<10}{'what it means':<34}"
          f"{'time_P':>8}{'time_R':>8}{'time_F1':>9}{'tp':>7}{'fp':>7}{'fn':>7}")
    print("-" * 90)
    label = {"vt": "when we actually spoke",
             "clamp3": "back-date, capped at tolerance",
             "clamp10": "back-date, capped at 10s  <-- TODAY",
             "ev": "model's raw claim (localisation)"}
    for p in ("vt", "clamp3", "clamp10", "ev"):
        r = o(p)
        print(f"{p:<10}{label[p]:<34}"
              f"{r['time_precision']:>8.3f}{r['time_recall']:>8.3f}"
              f"{r['time_f1']:>9.3f}"
              f"{r['n_matched']:>7}{r['n_emits'] - r['n_matched']:>7}"
              f"{r['n_gt'] - r['n_matched']:>7}")

    # THE HEADLINE IS MACRO. metrics.py:aggregate documents why: the micro pool
    # weights a task by how many emits it happens to have, and dedup_counting alone
    # is a third of all emits, so a micro "overall" is largely dedup wearing that
    # name. macro_time_f1 is the number comparable to the paper's table.
    print(f"\n{'policy':<10}{'MACRO time_F1 (paper-comparable)':<36}{'delta vs vt':>13}")
    print("-" * 60)
    mbase = o("vt")["macro_time_f1"]
    for p in ("vt", "clamp3", "clamp10", "ev"):
        m = o(p)["macro_time_f1"]
        print(f"{p:<10}{m:<36.4f}{m - mbase:>+13.4f}")

    base, today = o("vt")["time_f1"], o("clamp10")["time_f1"]
    print(f"\n  micro time_F1: today's back-dating moves it "
          f"{today - base:+.3f} ({100 * (today / base - 1) if base else float('nan'):+.1f}%) "
          f"vs the protocol-faithful number")
    print(f"  macro time_F1: today's back-dating moves it "
          f"{o('clamp10')['macro_time_f1'] - mbase:+.4f}")

    print("\n\nPER TASK — time_F1 under each policy\n")
    tasks = sorted(results["vt"]["per_task"],
                   key=lambda t: -results["vt"]["per_task"][t]["n_gt"])
    print(f"{'task':<30}{'n_gt':>6}{'vt':>9}{'clamp3':>9}{'clamp10':>9}{'ev':>9}"
          f"{'  clamp10 - vt':>15}")
    print("-" * 90)
    for t in tasks:
        row = [results[p]["per_task"][t]["time_f1"] for p in
               ("vt", "clamp3", "clamp10", "ev")]
        n_gt = int(results["vt"]["per_task"][t]["n_gt"])
        print(f"{t:<30}{n_gt:>6}" + "".join(f"{v:>9.3f}" for v in row)
              + f"{row[2] - row[0]:>+15.3f}")

    print("\n\nCONTENT-GATED (joint) — rule-scored tasks only; the two free-text")
    print("tasks are UNJUDGED here by design and withheld, not zeroed.\n")
    print(f"{'task':<30}{'vt':>9}{'clamp3':>9}{'clamp10':>9}{'ev':>9}")
    print("-" * 70)
    for t in tasks:
        if t in M.JUDGE_TASKS:
            continue
        row = [results[p]["per_task"][t].get("joint_f1")
               or results[p]["per_task"][t].get("joint_f1_lb")
               for p in ("vt", "clamp3", "clamp10", "ev")]
        print(f"{t:<30}" + "".join(
            f"{(f'{v:.3f}' if isinstance(v, (int, float)) else '—'):>9}"
            for v in row))

    if a.dedup:
        dedup_report(gt, fires, a.tolerance)




# ---------------------------------------------------------------------------
# DEDUP REPLAY — the emission time is always `vt`; the question is which
# emissions we SUPPRESS.
# ---------------------------------------------------------------------------
# The idea under test: the model's `event_time_s` is a good IDENTIFIER for an
# event even though it is a bad TIMESTAMP for the response. If a tick reports an
# event time we have already spoken about, it is the same occurrence -- so stay
# quiet. And because the schema walk emits `event_time_s` BEFORE `answer`, the
# decision can be made before the expensive decode: a suppressed tick costs zero
# answer tokens.
#
# CONTROL (do not skip): `gap<k>` is a plain refractory timer -- suppress anything
# within k seconds of the last thing we said. If ev-dedup does not beat the timer,
# then `event_time_s` earned nothing and a two-line timer is the real fix.
#
# HONEST LIMIT. This is a post-hoc filter over the emissions a real run produced.
# Suppressing an emission would have changed `reported`, hence the next prompt,
# hence later ticks -- so this is a SCREEN, not a prediction of what a live run
# scores. It is the same class of offline sweep as the other replay tools (replay
# saved per-tick numbers), and any winner must be confirmed by a real run.

def _dedup_stream(ticks, mode, param):
    """ticks: [(vt, ev, answer)] sorted by vt -> [(vt, answer)] kept."""
    kept, said_ev, last_vt = [], [], None
    for vt, ev, ans in ticks:
        if mode == "none":
            pass
        elif mode == "gap":
            if last_vt is not None and (vt - last_vt) < param:
                continue
        elif mode == "ev":
            if ev is None:
                pass                       # no identifier -> cannot dedup, keep
            elif any(abs(ev - p) <= param for p in said_ev):
                continue
        elif mode == "ev+gap":
            if last_vt is not None and (vt - last_vt) < 1.0:
                continue
            if ev is not None and any(abs(ev - p) <= param for p in said_ev):
                continue
        kept.append((vt, ans))
        last_vt = vt
        if ev is not None:
            said_ev.append(ev)
    return kept


def dedup_report(gt, fires, tolerance):
    judge = NoJudge()
    combos = [("none", 0), ("gap", 2), ("gap", 5), ("gap", 10), ("gap", 20),
              ("ev", 0.0), ("ev", 1.0), ("ev", 3.0), ("ev", 5.0),
              ("ev+gap", 0.0), ("ev+gap", 3.0)]
    print("\n\nDEDUP REPLAY — emission time is always `vt` (protocol-faithful);")
    print("only WHICH emissions survive changes.\n")
    print(f"{'rule':<14}{'emits':>8}{'tp':>7}{'time_P':>9}{'time_R':>9}"
          f"{'micro_F1':>10}{'MACRO_F1':>10}{'answer tok saved':>18}")
    print("-" * 90)
    base_emits = None
    for mode, param in combos:
        rows = []
        n_emit = 0
        for sid, ticks in fires.items():
            rec = gt.get(sid)
            if rec is None:
                continue
            keep = _dedup_stream(sorted(ticks, key=lambda x: x[0]), mode, param)
            n_emit += len(keep)
            pred = dict(rec)
            pred["predictions"] = [{"t_sec": vt, "raw": ans} for vt, ans in keep]
            rows.append(M.score_sample(pred, tolerance=tolerance, judge=judge))
        agg = M.aggregate(rows)
        o = agg["overall"]
        if base_emits is None:
            base_emits = n_emit
        name = mode if mode == "none" else f"{mode}{param:g}"
        # a suppressed tick skips the `answer` slot: 32 tokens at ~100 ms each
        saved = (base_emits - n_emit) * 32
        print(f"{name:<14}{n_emit:>8}{o['n_matched']:>7}"
              f"{o['time_precision']:>9.3f}{o['time_recall']:>9.3f}"
              f"{o['time_f1']:>10.3f}{o['macro_time_f1']:>10.4f}"
              f"{saved:>15,} tok")


if __name__ == "__main__":
    main()
