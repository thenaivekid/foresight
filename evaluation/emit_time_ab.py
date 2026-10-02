"""
emit_time_ab.py — does `event_time_s` work better as the EMISSION TIMESTAMP than
as an identity key alone?

THIS IS A PROTOCOL-VIOLATING ARM AND IS LABELLED AS ONE. OmniPro defines the
response timestamp as the video-second of EMISSION. Scoring an
emission at the model's own `event_time_s` reports a time we did not speak at, so
the `emit=ev` rows below are NOT benchmark-legal numbers. They answer a different
question -- "how well does the model localise the event it is talking about?" --
and are here to size that, not to be quoted as a score.

Four arms, a clean 2x2 over (emission timestamp) x (ev identity-key dedup):

    A  emit=vt, dedup=off   the protocol-faithful baseline
    B  emit=vt, dedup=ev    the proposal -- ev as IDENTIFIER only
    C  emit=ev, dedup=off   ev as TIMESTAMP only
    D  emit=ev, dedup=ev    ev as both

Every arm is REFIT PER TASK over the full gate grid (mode x thr x refractory,
plus the dedup window where the arm has one), fit on ALL samples, objective
time_f1 -- the same protocol as grid.py, so the arms are compared at each one's
own best gate rather than at a gate tuned for a different arm. Per grid.py's
header this is FIT-ON-ALL / in-sample by design; these numbers
measure headroom, not what a fresh run scores.

Time metrics only. No judge, no API calls, no GPU: content is unaffected by the
timestamp (metrics.py can only match on t_sec), and joint-F1 would need a judge
per arm per config. Offline replay of saved logs -- legal under INVARIANT 1.

`emit=ev` uses min(vt, ev): the model may claim the past, never the future, so
the arm stays causal. Ticks with no `event_time_s` fall back to vt in every arm
and are never dedup-suppressed (no identifier -> no claim of duplication).

Usage:
    python emit_time_ab.py --ticks ticks.jsonl --run RUN_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M  # noqa: E402
from grid import (MODES, THRS, REFS, DEDUPS, TOL, PROD_THR, PROD_MODE,  # noqa: E402
                  PROD_REF, load)

# grid.py sweeps DEDUPS = (None, 0, 3, 5, 10, 20). Split it: the off-arms take
# None only, the dedup-arms take the real windows.
DEDUP_WINDOWS = tuple(d for d in DEDUPS if d is not None)

# REFRACTORY GRID — WIDENED, and this is a correction to grid.py, not a taste.
# grid.py's REFS top out at 30 s, but the SHIPPED gate uses 300 s on
# explicit_target_grounding and 600 s on instant_event_alert / snapshot_counting
# -- i.e. "speak at most once per video". Those configs are OUTSIDE grid.py's own
# search space, so a "refit" there could not reach the gate already in production
# and scored BELOW it (snapshot_counting 0.206 refit vs 0.360 shipped on the first
# run of this script). A search that cannot express the incumbent understates every
# arm that depends on a long refractory -- which is exactly the dedup=off arms,
# since suppression is the only other way to get there. Extended so each arm is
# fitted over a space that contains the shipped policy.
REFS_EXT = tuple(sorted(set(REFS) | set(PROD_REF.values()) | {60.0, 120.0}))
# Same argument for the threshold axis: grid.py's THRS start at 0.05, but the
# shipped gate runs sequential_step_instruction at 0.01. Union in every shipped
# value so the search space provably CONTAINS the incumbent on every task, which
# is what makes "refit >= shipped" a check we can actually fail.
THRS_EXT = tuple(sorted(set(THRS) | set(PROD_THR.values())))


def candidates(ticks, mode, thr):
    """Ticks that pass the (mode, thr) firing rule -> [(vt, ev)].

    Hoisted out of the (refractory, dedup) loops: those two filter this list and
    never re-read the full tick stream, which is what makes a 4-arm sweep cheap.
    Strictly causal -- `edge` needs only the previous tick's state."""
    out, prev = [], False
    for vt, p, ev, _ans in ticks:
        cur = p >= thr
        want = (cur and not prev) if mode == "edge" else cur
        prev = cur
        if want:
            out.append((vt, ev))
    return out


def emits(cands, refractory, dedup, emit_ev):
    """Apply refractory + identity-key dedup, then stamp. -> [t_sec]

    Order matters and mirrors grid.py:simulate -- refractory first, then dedup,
    and `said` only records identifiers we actually spoke. The STAMP is applied
    last: which ticks survive is decided identically in every arm, so the arms
    differ in exactly one thing."""
    out, last, said = [], None, []
    for vt, ev in cands:
        if last is not None and vt - last < refractory:
            continue
        if dedup is not None and ev is not None:
            if any(abs(ev - q) <= dedup for q in said):
                continue
        last = vt
        if ev is not None:
            said.append(ev)
        out.append(min(vt, ev) if (emit_ev and ev is not None) else vt)
    return out


def score(samples, cands_by_sample, refractory, dedup, emit_ev):
    """Pooled tp/fp/fn over a task -> (p, r, f1, n_emits). Time only."""
    tp = fp = fn = n = 0
    for s, cands in zip(samples, cands_by_sample):
        et = emits(cands, refractory, dedup, emit_ev)
        n += len(et)
        m, ue, ug = M.match_emits_to_gt(et, s["gtt"], TOL)
        tp += len(m); fp += len(ue); fn += len(ug)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, n, tp, fp, fn


ARMS = [("A", "emit=vt  dedup=off", False, False),
        ("B", "emit=vt  dedup=ev ", False, True),
        ("C", "emit=ev  dedup=off", True, False),
        ("D", "emit=ev  dedup=ev ", True, True)]


def fit_task(samples):
    """Refit the gate per arm. -> {arm: (best_cfg, metrics)}"""
    best = {a[0]: (None, (0.0, 0.0, -1.0, 0, 0, 0, 0)) for a in ARMS}
    for mode in MODES:
        for thr in THRS_EXT:
            cands = [candidates(s["ticks"], mode, thr) for s in samples]
            for ref in REFS_EXT:
                for arm, _lbl, emit_ev, use_dedup in ARMS:
                    windows = DEDUP_WINDOWS if use_dedup else (None,)
                    for d in windows:
                        got = score(samples, cands, ref, d, emit_ev)
                        if got[2] > best[arm][1][2]:
                            best[arm] = ((mode, thr, ref, d), got)
    return best


def at_shipped(samples, task):
    """The SHIPPED gate (config.py, itself fitted on this corpus) under each arm,
    with no refit. This is the row that connects to the dedup-only analysis,
    which never refit the threshold."""
    mode, thr, ref = PROD_MODE[task], PROD_THR[task], PROD_REF[task]
    cands = [candidates(s["ticks"], mode, thr) for s in samples]
    out = {}
    for arm, _lbl, emit_ev, use_dedup in ARMS:
        if not use_dedup:
            out[arm] = ((mode, thr, ref, None),
                        score(samples, cands, ref, None, emit_ev))
        else:                      # ev0 = the untuned identity key
            out[arm] = ((mode, thr, ref, 0.0),
                        score(samples, cands, ref, 0.0, emit_ev))
    return out


def table(title, per_task, note):
    print(f"\n\n{title}\n{note}\n")
    hdr = f"{'task':<30}{'n_gt':>6}  "
    for arm, lbl, _e, _d in ARMS:
        hdr += f"{arm + ' ' + lbl.strip():>22}"
    print(hdr)
    print("-" * len(hdr))
    macro = defaultdict(list)
    for task in sorted(per_task, key=lambda t: -per_task[t]["n_gt"]):
        row = f"{task:<30}{per_task[task]['n_gt']:>6}  "
        for arm, _lbl, _e, _d in ARMS:
            p, r, f = per_task[task][arm][1][:3]
            macro[arm].append(f)
            row += f"{f'{p:.3f}/{r:.3f}/{f:.3f}':>22}"
        print(row)
    print("-" * len(hdr))
    row = f"{'MACRO time_F1':<30}{'':>6}  "
    for arm, _lbl, _e, _d in ARMS:
        row += f"{sum(macro[arm]) / len(macro[arm]):>22.4f}"
    print(row + "\n(cells are time_P / time_R / time_F1)")
    return {a: sum(macro[a]) / len(macro[a]) for a in macro}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", default="ticks.jsonl")
    ap.add_argument("--run", default="output")
    ap.add_argument("--splits", default="splits_all/*.json")
    ap.add_argument("--out", default="emit_time_ab.json")
    a = ap.parse_args()

    by_task = load(a.ticks, a.run, a.splits)
    print(f"[ab] {sum(len(v) for v in by_task.values())} samples, "
          f"{len(by_task)} tasks, tolerance +-{TOL}s")
    print("[ab] time metrics only -- no judge, no GPU, offline replay\n")

    fitted, shipped = {}, {}
    for task in sorted(by_task):
        samples = by_task[task]
        n_gt = sum(len(s["gtt"]) for s in samples)
        f = fit_task(samples)
        s = at_shipped(samples, task)
        f["n_gt"] = s["n_gt"] = n_gt
        fitted[task], shipped[task] = f, s
        print(f"  fitted {task:<30} n={len(samples):>4} gt={n_gt:>5}  "
              + "  ".join(f"{arm}={f[arm][1][2]:.3f}" for arm, *_ in ARMS))

    m_ship = table(
        "AT THE SHIPPED GATE (config.py) — no refit, dedup window = ev0 (untuned)",
        shipped,
        "Connects to the dedup-only analysis, which also did not refit.")
    m_fit = table(
        "REFIT PER ARM (mode x thr x refractory x dedup window), FIT-ON-ALL",
        fitted,
        "Each arm at its OWN best gate. In-sample: headroom, not a fresh-run score.")

    print("\n\nSUMMARY — macro time_F1\n")
    print(f"{'arm':<22}{'shipped gate':>14}{'refit':>10}{'refit delta vs A':>20}")
    print("-" * 66)
    for arm, lbl, _e, _d in ARMS:
        print(f"{arm + '  ' + lbl:<22}{m_ship[arm]:>14.4f}{m_fit[arm]:>10.4f}"
              f"{m_fit[arm] - m_fit['A']:>+20.4f}")
    print("\nREMINDER: arms C and D stamp emissions at the model's own claimed "
          "event time.\nThat is not the OmniPro response timestamp — those two rows "
          "are diagnostic,\nnot reportable scores.")

    with open(a.out, "w") as fh:
        json.dump({"fitted": {t: {k: (v if k == "n_gt" else
                                      {"cfg": v[0], "p": v[1][0], "r": v[1][1],
                                       "f1": v[1][2], "n_emits": v[1][3],
                                       "tp": v[1][4], "fp": v[1][5], "fn": v[1][6]})
                                  for k, v in d.items()} for t, d in fitted.items()},
                   "shipped": {t: {k: (v if k == "n_gt" else
                                       {"cfg": v[0], "p": v[1][0], "r": v[1][1],
                                        "f1": v[1][2], "n_emits": v[1][3],
                                        "tp": v[1][4], "fp": v[1][5], "fn": v[1][6]})
                                   for k, v in d.items()} for t, d in shipped.items()},
                   "macro_shipped": m_ship, "macro_fitted": m_fit}, fh, indent=1)
    print(f"\n[ab] wrote {a.out}")


if __name__ == "__main__":
    main()
