"""
grid.py — dense per-task gate search, with time AND content metrics, fitted on a
dev half and reported on a held-out half.

WHAT IT SWEEPS (all four replayed from saved logs; none needs a GPU)

    mode        edge   fire on the RISING edge of p_hit >= thr  (controller.py)
                level  fire on EVERY tick above thr
    thr         the p_hit threshold
    refractory  suppress a fire within N seconds of the previous one
    dedup_ev    suppress a fire whose `event_time_s` is within D of one we have
                ALREADY spoken about -- the identity-key rule. `off`
                disables it; 0 means exact repeats only.

This extends resweep.py, which sweeps (mode, thr, refractory) on time-F1 alone.
Two things are new: the dedup_ev axis, and content/joint columns.

FITTING USES EVERY SAMPLE. NO HELD-OUT SPLIT. EVER.
---------------------------------------------------
This is a deliberate design decision: fit the thresholds on all
available samples. Do not add a dev/test split back into this tool.

The rationale is that the vision-only subset is small (four flagship
tasks hold 358 samples; explicit_target_grounding holds 6), and halving it makes
the per-task numbers noise -- a held-out half of explicit_target_grounding is
3 samples. The precedent is already in config.py: the shipped per-task thresholds
were fitted the same way, over a full development run.

Label these numbers for what they are: FIT-ON-ALL, i.e. in-sample. They measure
how much headroom the gate has, not what a fresh run would score.

CONTENT AND JOINT
-----------------
Seven tasks are rule-scored (time_only x2, count x3, position, state) -- exact and
mechanical. The two free-text tasks (event_narration, sequential_step_instruction)
are judged by the REAL LLM judge (`ContentJudge(backend="openai")`, gpt-5-mini with
Structured Outputs). No lexical fallback ever a failed call
returns UNJUDGED and the number is withheld, not estimated.

TWO PHASES, because the judge is an API and the grid is ~1,300 configs per task:

  SEARCH   runs with the judge DISABLED. Rule-scored tasks still fit on joint-F1
           (free, exact); the two free-text tasks fit on time-F1, since their
           joint is unavailable without spending ~40k API calls on configurations
           that will be thrown away. Each row prints which objective it used.
  REPORT   re-scores the CHOSEN configs with the real judge, so every content and
           joint cell in the tables below is a genuine gpt-5-mini verdict.

Verdicts are cached in judge_cache.json keyed by (question, gt, pred), so repeats
across the three tables cost nothing.

HONEST LIMITS
-------------
1. This is a REPLAY of one run's tick stream. Suppressing an emission would have
   changed `reported`, hence the next prompt, hence later ticks. A winner here is
   a candidate, not a result, and must be confirmed by a real run.
2. Only thresholds where an `answer` was actually decoded can be scored for
   content. Below that the schema never produced text, so those ticks are counted
   as time-only matches with unknown content -- reported as `n_noans`.
3. The search is over a causal rule applied left-to-right. Nothing looks ahead.

Usage:
    python grid.py --ticks ticks.jsonl --run RUN_DIR
    python grid.py --ticks ticks.jsonl --run RUN_DIR --objective time_f1
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

TOL = 3.0

# WHAT WE ACTUALLY SHIP, from foresight/config.py. The naive edge/0.5 default
# is NOT the production gate -- these per-task values were themselves fitted by
# resweep.py on this same corpus. Comparing a fresh search against edge/0.5 would
# overstate the gain by crediting it with work already done. Both baselines are
# reported: `prod_test` (edge/0.5, the pre-fitting default) and `cfg_test` (the
# shipped, already-fitted config). `cfg_test` is the honest one to beat.
PROD_THR = {"cumulative_counting": 0.925, "dedup_counting": 0.992,
            "event_narration": 0.10, "explicit_target_grounding": 0.50,
            "instant_event_alert": 0.45, "realtime_state_monitor": 0.80,
            "semantic_condition_alert": 0.98,
            "sequential_step_instruction": 0.01, "snapshot_counting": 0.985}
PROD_MODE = {"cumulative_counting": "edge", "dedup_counting": "edge",
             "event_narration": "level", "explicit_target_grounding": "edge",
             "instant_event_alert": "edge", "realtime_state_monitor": "level",
             "semantic_condition_alert": "level",
             "sequential_step_instruction": "level", "snapshot_counting": "edge"}
PROD_REF = {"cumulative_counting": 7.0, "dedup_counting": 5.0,
            "event_narration": 7.0, "explicit_target_grounding": 300.0,
            "instant_event_alert": 600.0, "realtime_state_monitor": 7.0,
            "semantic_condition_alert": 10.0,
            "sequential_step_instruction": 7.0, "snapshot_counting": 600.0}

MODES = ("edge", "level")
THRS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
        0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
REFS = (0.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0)
DEDUPS = (None, 0.0, 3.0, 5.0, 10.0, 20.0)


def load(ticks_path, run_dir, splits_glob):
    """-> {task: [sample, ...]}, each sample carrying gt + its full tick stream."""
    gt = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "online_pred.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            gt[r["id"]] = r
    if not gt:
        for p in sorted(glob.glob(splits_glob)):
            for s in json.load(open(p)):
                gt[s["id"]] = s
    for r in gt.values():
        for g in r.get("ground_truth", []):
            if "t_sec" not in g and "trigger_time_sec" in g:
                g["t_sec"] = float(g["trigger_time_sec"])

    ticks = defaultdict(list)
    for line in open(ticks_path):
        r = json.loads(line)
        if r.get("p_hit") is None:
            continue
        d = r.get("diff") or {}
        ev = d.get("event_time_s")
        try:
            ev = float(ev)
        except (TypeError, ValueError):
            ev = None
        sid = f'{r["task"]}::{r["video"]}::{r["sample"]}'
        ticks[sid].append((float(r["vt"]), float(r["p_hit"]), ev,
                           str(d.get("answer") or "")))

    by_task = defaultdict(list)
    for sid, tk in ticks.items():
        rec = gt.get(sid)
        if rec is None:
            continue
        by_task[rec["task"]].append({
            "id": sid, "task": rec["task"], "question": rec.get("question", ""),
            "gts": rec.get("ground_truth", []),
            "gtt": [float(g["t_sec"]) for g in rec.get("ground_truth", [])],
            "ticks": sorted(tk, key=lambda x: x[0]),
        })
    return by_task


def simulate(ticks, mode, thr, refractory, dedup):
    """Replay the firing rule left to right. -> [(vt, answer_text)]

    Mirrors resweep.py:simulate and adds the identity-key suppression. Strictly
    causal: every decision uses only ticks already seen."""
    emits, prev, last, said = [], False, None, []
    for vt, p, ev, ans in ticks:
        cur = p >= thr
        want = (cur and not prev) if mode == "edge" else cur
        prev = cur
        if not want:
            continue
        if last is not None and vt - last < refractory:
            continue
        if dedup is not None and ev is not None:
            if any(abs(ev - q) <= dedup for q in said):
                continue
        emits.append((vt, ans))
        last = vt
        if ev is not None:
            said.append(ev)
    return emits


class _NoJudge:
    """Judge used during the SEARCH phase only. Returns UNJUDGED for everything,
    which makes the free-text tasks fall back to a time-F1 objective instead of
    firing ~40k API calls at configurations that will be discarded."""

    def score(self, *a, **k):
        return None


class Scorer:
    """Time + content scoring with a memo on the expensive content verdict.

    Content correctness depends only on (task, answer text, the GT item matched),
    never on the gate config -- so the same (text, gt) pair recurs across all ~1300
    grid points and is judged once. Swapping the judge clears the memo, because a
    verdict of None (search phase) must not be reused as a real verdict."""

    def __init__(self, task, judge=None):
        self.task = task
        self.rule_scored = task not in M.JUDGE_TASKS
        self.set_judge(judge or _NoJudge())

    def set_judge(self, judge):
        self.judge = judge
        self._memo = {}
        # can this task produce a content number at all, with THIS judge?
        self.judged = self.rule_scored or not isinstance(judge, _NoJudge)

    def content_ok(self, ans, gt_item, question):
        key = (ans, id(gt_item))
        if key not in self._memo:
            self._memo[key] = M._content_correct(self.task, ans, gt_item,
                                                 question, self.judge)
        return self._memo[key]

    def score(self, samples, mode, thr, ref, dedup):
        tp = fp = fn = tpc = n_noans = n_unj = 0
        for s in samples:
            emits = simulate(s["ticks"], mode, thr, ref, dedup)
            et = [e[0] for e in emits]
            matches, un_e, un_g = M.match_emits_to_gt(et, s["gtt"], TOL)
            tp += len(matches); fp += len(un_e); fn += len(un_g)
            for ei, gj, _ in matches:
                ans = emits[ei][1]
                if not ans:
                    n_noans += 1          # matched in time, no text was decoded
                    continue
                ok = self.content_ok(ans, s["gts"][gj], s["question"])
                if ok is None:
                    n_unj += 1            # judge could not rule -- WITHHELD
                elif ok:
                    tpc += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        # joint denominators follow metrics.py: valid / all responses, valid / all GT
        jp = tpc / (tp + fp) if tp + fp else 0.0
        jr = tpc / (tp + fn) if tp + fn else 0.0
        jf = 2 * jp * jr / (jp + jr) if jp + jr else 0.0
        # WITHHOLD, never estimate: if any matched emit went unjudged, tpc is only
        # a lower bound and the content/joint numbers are not what they claim.
        ok = self.judged and n_unj == 0
        return {"time_p": p, "time_r": r, "time_f1": f,
                "content_acc": (tpc / tp if tp else 0.0) if ok else None,
                "joint_p": jp if ok else None,
                "joint_r": jr if ok else None,
                "joint_f1": jf if ok else None,
                "tp": tp, "fp": fp, "fn": fn, "n_emits": tp + fp,
                "n_noans": n_noans, "n_unjudged": n_unj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", required=True)
    ap.add_argument("--run", default="output")
    ap.add_argument("--splits", default="splits_all/*.json")
    ap.add_argument("--objective", default="auto",
                    choices=["auto", "time_f1", "joint_f1"],
                    help="auto = joint_f1 where content is rule-scored, else time_f1")
    ap.add_argument("--out", default="grid_best.json")
    ap.add_argument("--task", action="append", help="restrict to task(s)")
    ap.add_argument("--judge-backend", default="openai",
                    help='judge for the REPORT phase; "openai" = gpt-5-mini')
    a = ap.parse_args()

    by_task = load(a.ticks, a.run, a.splits)
    n_grid = len(MODES) * len(THRS) * len(REFS) * len(DEDUPS)
    print(f"[grid] {sum(len(v) for v in by_task.values())} samples, "
          f"{len(by_task)} tasks, {n_grid} configs/task\n")

    chosen = {}
    if a.task:
        by_task = {k: v for k, v in by_task.items() if k in set(a.task)}
    for task in sorted(by_task, key=lambda t: -len(by_task[t])):
        # FIT ON ALL SAMPLES -- see the module docstring. No split.
        fit = by_task[task]
        sc = Scorer(task)
        obj = a.objective
        if obj == "auto":
            obj = "joint_f1" if sc.judged else "time_f1"

        # sweep once, keep every result; choosing comes after, so a degenerate
        # objective can be detected instead of silently returning grid point #1.
        results = []
        for mode in MODES:
            for thr in THRS:
                for ref in REFS:
                    for ded in DEDUPS:
                        results.append(((mode, thr, ref, ded),
                                        sc.score(fit, mode, thr, ref, ded)))

        # DEGENERATE-OBJECTIVE GUARD. realtime_state_monitor scores content_acc
        # EXACTLY 0.000 on this corpus -- a known format mismatch fixed
        # after that run was produced. Its joint_f1
        # is therefore 0.0 for every configuration, `v > best_v` never fires after
        # the first, and the search silently returns the FIRST grid point while
        # looking like it optimised something. That is how RSM ended up worse than
        # the production gate. If the objective never varies, fall back to time_f1
        # and say so.
        vals = {round(m[obj], 9) for _, m in results if m[obj] is not None}
        degenerate = len(vals) <= 1
        if degenerate:
            obj = "time_f1"
        best = max(results, key=lambda kv: kv[1][obj])[0]
        mode, thr, ref, ded = best
        chosen[task] = {
            "mode": mode, "thr": thr, "refractory_s": ref, "dedup_ev_s": ded,
            "objective": obj, "objective_degenerate": degenerate,
            "n_samples": len(fit),
            "_scorer": sc, "_fit": fit, "_best": best,
        }
        print(f"  {task:<30} fit on {obj:<9} -> mode={mode:<5} thr={thr:<6} "
              f"ref={ref:<5} dedup={'off' if ded is None else ded}"
              + ("   [objective was CONSTANT -> fell back to time_f1]"
                 if degenerate else ""))

    # ---- REPORT PHASE: re-score the chosen configs with the REAL judge --------
    # Every content/joint cell printed below is a genuine gpt-5-mini verdict.
    # Rule-scored tasks never touch the API; only the two free-text tasks do, and
    # only for the configs we actually kept.
    judge = M.ContentJudge(backend=a.judge_backend)
    print(f"\n[grid] report-phase judge: backend={a.judge_backend} "
          f"mode={getattr(judge, 'mode', '?')}")
    if getattr(judge, "mode", "unavailable") == "unavailable":
        print("[grid] WARNING: judge unavailable -> content/joint for the two "
              "free-text tasks will be WITHHELD (--), never estimated.")
    for task, c in chosen.items():
        sc, best, fit = c.pop("_scorer"), c.pop("_best"), c.pop("_fit")
        sc.set_judge(judge)
        c["fitted"] = sc.score(fit, *best)
        c["shipped"] = sc.score(fit, PROD_MODE[task], PROD_THR[task],
                                PROD_REF[task], None)
        c["naive"] = sc.score(fit, "edge", 0.5, 0.0, None)
    print(f"[grid] judge verdicts: {getattr(judge, 'n_judged', 0)} judged, "
          f"{getattr(judge, 'n_unjudged', 0)} unjudged")

    json.dump(chosen, open(a.out, "w"), indent=2)

    def table(which, title):
        print(f"\n\n{title}\n")
        print(f"{'task':<29}{'time_P':>8}{'time_R':>8}{'time_F1':>9}"
              f"{'cAcc':>8}{'joint_P':>9}{'joint_R':>9}{'joint_F1':>10}"
              f"{'emits':>8}{'tp':>6}")
        print("-" * 104)
        f = lambda v: "--" if v is None else f"{v:.3f}"
        accum = defaultdict(list)
        for t in sorted(chosen):
            m = chosen[t][which]
            print(f"{t:<29}{m['time_p']:>8.3f}{m['time_r']:>8.3f}"
                  f"{m['time_f1']:>9.3f}{f(m['content_acc']):>8}"
                  f"{f(m['joint_p']):>9}{f(m['joint_r']):>9}{f(m['joint_f1']):>10}"
                  f"{m['n_emits']:>8}{m['tp']:>6}")
            accum["time_f1"].append(m["time_f1"])
            if m["joint_f1"] is not None:
                accum["joint_f1"].append(m["joint_f1"])
                accum["content_acc"].append(m["content_acc"])
        print("-" * 104)
        mac = lambda k: sum(accum[k]) / len(accum[k]) if accum[k] else float("nan")
        print(f"{'MACRO (9 tasks)':<29}{'':>8}{'':>8}{mac('time_f1'):>9.4f}"
              f"{mac('content_acc'):>8.3f}{'':>9}{'':>9}"
              f"{mac('joint_f1'):>10.4f}   <- content/joint over the 7 rule-scored")

    table("fitted", "BEST PER-TASK CONFIG (fitted on all samples)")
    table("shipped", "SHIPPED CONFIG — config.py per-task thr/mode/refractory, no dedup"
                     "  <- THE BASELINE TO BEAT")
    table("naive", "NAIVE DEFAULT — edge / 0.5 / no refractory / no dedup")

    print("\n\nGAIN OVER THE SHIPPED CONFIG (same objective, same samples)\n")
    print(f"{'task':<30}{'objective':<10}{'fitted':>8}{'shipped':>9}{'gain':>8}"
          f"{'naive':>8}{'n_gt':>7}{'n':>5}")
    print("-" * 86)
    for t in sorted(chosen):
        c = chosen[t]
        k = "joint_f1" if c["objective"] == "joint_f1" else "time_f1"
        f_, sh, nv = c["fitted"][k], c["shipped"][k], c["naive"][k]
        n_gt = c["fitted"]["tp"] + c["fitted"]["fn"]
        print(f"{t:<30}{c['objective']:<10}{f_:>8.3f}{sh:>9.3f}{f_ - sh:>+8.3f}"
              f"{nv:>8.3f}{n_gt:>7}{c['n_samples']:>5}")
    print(f"\nbest configs -> {a.out}")


if __name__ == "__main__":
    main()
