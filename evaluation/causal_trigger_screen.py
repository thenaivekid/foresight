#!/usr/bin/env python3
"""Fixed, causal suppression-only screen on saved async model-only emissions.

No inference, fitting, lookahead, interpolation, new answers, or time backdating.
Every policy receives the same recorded observations and can only retain an
already decoded/delivered response. This is NOT a counterfactual live rollout:
suppression would change reported history, future frames/checks, and answers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from metrics import match_emits_to_gt, TIME_ONLY, _extract_count, _extract_state
from repeat_vision_model_only import delivery_times, load_run


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    window_s: float = 0.0
    tau_s: float = 0.0
    high: float = 0.5
    low: float = 0.5
    cooldown_s: float = 0.0
    identity_window_s: float = 0.0


# Illustrative fixed global constants; no task/video tuning or optimizer.
# Both runs contain the same 18 videos, so run2 is NOT a held-out dataset.
POLICIES = (
    Policy("raw_level_0.5", "level"),
    Policy("cooldown_5s", "level", cooldown_s=5.0),
    Policy("cooldown_10s", "level", cooldown_s=10.0),
    Policy("rising_edge_0.5", "edge"),
    Policy("event_identity_ev0", "identity"),
    Policy("event_identity_ev1", "identity", identity_window_s=1.0),
    Policy("event_identity_ev3", "identity", identity_window_s=3.0),
    Policy("ev0_cooldown_5s", "identity", cooldown_s=5.0),
    Policy("exact_answer_dedup", "text_identity"),
    Policy("count_phase_change", "structured_change"),
    Policy("hysteresis_0.5_0.3", "edge", low=0.3),
    Policy("sample_mean_2s_edge", "mean_points", window_s=2.0),
    Policy("sample_mean_4s_edge", "mean_points", window_s=4.0),
    Policy("ema_point_0.5s_edge", "ema_point", tau_s=0.5),
    Policy("ema_point_1s_edge", "ema_point", tau_s=1.0),
    Policy("trailing_mean_2s_edge", "mean", window_s=2.0),
    Policy("trailing_mean_4s_edge", "mean", window_s=4.0),
    Policy("ema_0.5s_edge", "ema", tau_s=0.5),
    Policy("ema_1s_hysteresis", "ema", tau_s=1.0, low=0.3),
    Policy("two_positive_checks_edge", "confirm"),
    Policy("leaky_evidence_2s", "integral", tau_s=2.0, high=0.5, low=0.3),
)


def replay(ticks, policy):
    """Consume observations in arrival order; output retained emission indexes.

    tick = {arrival_s, p_hit, emission_index}. arrival_s is the measured end of
    the controller check, NOT the earlier video/snapshot timestamp. For continuous
    time filters, the PREVIOUS observation is held between ticks; the current
    observation must not be projected backwards over time when it was unknown.
    """
    window = deque()
    last_t, last_p, smooth = None, None, None
    armed, positive, evidence, last_fire = True, 0, 0.0, -math.inf
    retained, seen_events, seen_text = [], set(), set()
    last_structured = None
    for tick in ticks:
        t, p = tick["arrival_s"], tick["p_hit"]
        if not math.isfinite(t) or not 0 <= p <= 1:
            raise ValueError("Invalid observation")
        if last_t is not None and t <= last_t:
            raise ValueError("Observations must have strictly increasing arrival times")
        dt = 0.0 if last_t is None else t - last_t
        if policy.kind in ("ema", "integral"):
            decay = math.exp(-dt / policy.tau_s)
            if policy.kind == "ema":
                # No invented prehistory: initialise to the first measured value.
                smooth = p if smooth is None else decay * smooth + (1 - decay) * last_p
            else:
                # Signed evidence above/below neutral; exact leaky integration
                # under previous-value hold, with a floor at zero.
                if last_p is not None:
                    evidence = max(0.0, decay * evidence +
                                   policy.tau_s * (1 - decay) * (last_p - 0.5))
        if policy.kind in ("mean", "mean_points"):
            window.append((t, p))
        if policy.kind == "mean_points":
            while window[0][0] < t - policy.window_s:
                window.popleft()
            # Standard observation-weighted trailing average, including p NOW.
            score = sum(val for _, val in window) / len(window)
        elif policy.kind == "ema_point":
            # Discrete time-aware EWMA at the current observation. Unlike the
            # hold/integral policies below, this does not model the area between
            # measurements, and can respond immediately after a long quiet gap.
            alpha = -math.expm1(-dt / policy.tau_s)
            smooth = p if smooth is None else smooth + alpha * (p - smooth)
            score = smooth
        elif policy.kind == "mean":
            cutoff = max(window[0][0], t - policy.window_s)
            # Keep the observation immediately preceding the left boundary.
            while len(window) >= 2 and window[1][0] <= cutoff:
                window.popleft()
            points = list(window)
            area = sum(max(0.0, end - max(start, cutoff)) * val
                       for (start, val), (end, _) in zip(points, points[1:]))
            score = area / (t - cutoff) if t > cutoff else p
        elif policy.kind == "ema":
            score = smooth
        elif policy.kind == "integral":
            score = evidence
        elif policy.kind == "confirm":
            positive = positive + 1 if p >= 0.5 else 0
            score = float(positive >= 2)
        else:
            score = p

        if policy.kind == "integral":
            if p < policy.low:
                armed = True
                evidence = 0.0
                score = 0.0
        elif score < policy.low:
            armed = True
        event = tick.get("event_time_s")
        event = round(event, 3) if event is not None and math.isfinite(event) else None
        if policy.kind == "identity":
            duplicate = event is not None and any(
                abs(event - old) <= policy.identity_window_s + 1e-9 for old in seen_events)
            candidate = p >= policy.high and not duplicate
        elif policy.kind == "text_identity":
            candidate = p >= policy.high and (tick.get("task") in TIME_ONLY or
                tick.get("answer_text") not in seen_text)
        elif policy.kind == "structured_change":
            key = tick.get("structured_key")
            candidate = p >= policy.high and (key is None or key != last_structured)
        else:
            candidate = score >= policy.high and (policy.kind == "level" or armed)
        ei = tick["emission_index"]
        if candidate and ei is not None and t - last_fire >= policy.cooldown_s:
            retained.append(ei)
            armed = False
            last_fire = t
            if event is not None:
                seen_events.add(event)
            seen_text.add(tick.get("answer_text"))
            if tick.get("structured_key") is not None:
                last_structured = tick["structured_key"]
        last_t, last_p = t, p
    return retained


def observations(record):
    cfg = record["effective_config"]
    assert record["realtime"] and not cfg["deterministic"] and cfg["speed"] == 1.0
    assert cfg["hit_threshold"] == 0.5 and cfg["gate_strategy"] == "level"
    assert not cfg["ev0_dedup"] and cfg["debounce_s"] == 0
    events = record["wall_timeline"]
    origin = next(e["monotonic_s"] for e in events if e["event"] == "source_start")
    checks = [e for e in events if e["event"] == "check_complete"]
    emits = {p["t_sec"]: i for i, p in enumerate(record["predictions"])}
    assert len(emits) == len(record["predictions"])
    assert len(checks) == len(record["gate_trace"])
    ticks = []
    for check, gate in zip(checks, record["gate_trace"]):
        assert check["video_s"] == gate["t_sec"]
        ei = emits.get(gate["t_sec"])
        answer = record["predictions"][ei]["raw"] if ei is not None else None
        structured = None
        if answer:
            if record["task"] in ("cumulative_counting", "dedup_counting"):
                structured = _extract_count(answer)
            elif record["task"] == "realtime_state_monitor":
                structured = _extract_state(answer)
        ticks.append({"arrival_s": check["monotonic_s"] - origin,
                      "p_hit": gate["p_hit"], "emission_index": ei,
                      "task": record["task"], "answer_text": answer,
                      "structured_key": structured})
    assert replay(ticks, POLICIES[0]) == list(range(len(emits)))
    return ticks


def attach_event_identity(root, records, ticks):
    """Align full raw JSON by sample ID AND tick order; reject mixed/retry logs."""
    raw = defaultdict(list)
    banner = re.compile(r"=====\s*\[\d+/\d+\]\s+(\S+)")
    line_pattern = re.compile(r"vid\s+([\d.]+)s\]\s+ctrl\.raw\s+\[([^\]]+)\]\s+(\{.*\})")
    for path in sorted(root.glob("g[0-3].log")):
        sample = None
        with path.open() as stream:
            for line in stream:
                match = banner.search(line)
                if match:
                    sample = match.group(1)
                elif sample in records and (match := line_pattern.search(line)):
                    assert match.group(2) == records[sample]["video_id"]
                    raw[sample].append((float(match.group(1)), json.loads(match.group(3))))
    for k, record in records.items():
        assert len(raw[k]) == len(ticks[k]), f"Incomplete or repeated raw logs for {k}"
        for (vt, diff), gate, tick in zip(raw[k], record["gate_trace"], ticks[k]):
            assert abs(vt - gate["t_sec"]) <= 0.050001, f"Raw/tick alignment failure: {k}"
            event = diff.get("event_time_s")
            tick["event_time_s"] = float(event) if event is not None else None


def score(kept_times, gts):
    matches, fp, fn = match_emits_to_gt(kept_times, gts, 3.0)
    return {"tp": len(matches), "fp": len(fp), "fn": len(fn),
            "emits": len(kept_times), "gt": len(gts)}


def metrics(counts):
    tp, ne, ng = counts["tp"], counts["emits"], counts["gt"]
    return dict(counts, precision=tp / ne if ne else 0.0,
                recall=tp / ng if ng else 0.0,
                time_f1=2 * tp / (ne + ng) if ne + ng else 0.0)


def screen(root):
    records = load_run(root)
    ticks = {k: observations(r) for k, r in records.items()}
    attach_event_identity(root, records, ticks)
    delivered = {k: delivery_times(r) for k, r in records.items()}
    result = {}
    for policy in POLICIES:
        tasks, samples = defaultdict(lambda: defaultdict(int)), []
        for k, r in sorted(records.items()):
            retained = replay(ticks[k], policy)
            c = score([delivered[k][i] for i in retained],
                      [float(g["t_sec"]) for g in r["ground_truth"]])
            for name, value in c.items():
                tasks[r["task"]][name] += value
            samples.append(dict(id=k, retained_prediction_indices=retained, **c))
        pooled = {name: sum(c[name] for c in tasks.values())
                  for name in ("tp", "fp", "fn", "emits", "gt")}
        result[policy.name] = {
            "micro": metrics(pooled),
            "macro_task_time_f1": sum(metrics(c)["time_f1"] for c in tasks.values()) / len(tasks),
            "per_task": {task: metrics(c) for task, c in sorted(tasks.items())},
            "per_sample": samples,
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs=2, type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = {str(root): screen(root) for root in args.roots}
    payload = {"protocol": "Causal suppression-only, fixed-log screen, actual wall delivery times.",
               "caveats": [
                   "Illustrative global constants, not fitted; these 18 selected videos are not held-out validation.",
                   "Both runs use the same videos; do not call one a held-out split.",
                   "Only existing emissions are retained, at unchanged actual delivery times; no new answers.",
                   "Filtering affects no upstream state here; live suppression would change all later trajectories.",
                   "Scores measure timing only; they are NOT new live results or content correctness.",
                   "Availability of p_hit before complete check was not logged; no latency saving is inferred.",
                   "Sample-mean and ema_point variants include the current observation; hold variants integrate only previously available values.",
                   "Event identity uses model-authored onset IDs from aligned ctrl.raw logs, never as backdated delivery timestamps.",
               ],
               "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("Fixed-log causal screen; NOT live re-evaluation. Actual delivery timing, ±3s.")
    print("policy                          run1: emits tp microF1 macroF1    run2: emits tp microF1 macroF1")
    for p in POLICIES:
        values = []
        for r in runs.values():
            c = r[p.name]["micro"]
            values.append(f"{c['emits']:4d} {c['tp']:3d} {c['time_f1']:.4f}  {r[p.name]['macro_task_time_f1']:.4f}")
        print(f"{p.name:31s} " + "           ".join(values))


if __name__ == "__main__":
    main()