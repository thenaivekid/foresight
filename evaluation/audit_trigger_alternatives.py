#!/usr/bin/env python3
"""Comprehensive fixed-log trigger screen, identity audit and content scoring.

No VLM inference. All policies can only suppress existing outputs. Circular
time shifts are statistical references, NOT causal deployable policies.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from causal_trigger_screen import (POLICIES, attach_event_identity, observations,
                                   screen, score)
from repeat_vision_model_only import delivery_times, load_run
from metrics import ContentJudge, JUDGE_TASKS, match_emits_to_gt, score_sample, aggregate


def identity_audit(root, records):
    ticks = {k: observations(r) for k, r in records.items()}
    attach_event_identity(root, records, ticks)
    tasks, sample_stats, collisions = defaultdict(Counter), [], []
    for k, record in records.items():
        seen, stats, lag = set(), Counter(), []
        for gate, tick in zip(record['gate_trace'], ticks[k]):
            if tick['emission_index'] is None:
                continue
            stats['emits'] += 1
            ev = tick.get('event_time_s')
            if ev is None or not math.isfinite(ev) or ev < 0:
                stats['missing_or_invalid'] += 1
                continue
            stats['valid_onset'] += 1
            key = round(ev, 3)
            stats['exact_repeat'] += key in seen
            stats['ahead_of_logged_check'] += ev > gate['t_sec']
            lag.append(gate['t_sec'] - ev)
            seen.add(key)
        tasks[record['task']].update(stats)
        by_emit = {t['emission_index']: t.get('event_time_s') for t in ticks[k]
                   if t['emission_index'] is not None}
        matched_ids = defaultdict(list)
        matches = match_emits_to_gt(delivery_times(record),
                                   [g['t_sec'] for g in record['ground_truth']], 3.0)[0]
        for ei, gi, _ in matches:
            ev = by_emit[ei]
            if ev is not None and math.isfinite(ev) and ev >= 0:
                matched_ids[round(ev, 3)].append(gi)
        for ev, gt_indexes in matched_ids.items():
            if len(gt_indexes) > 1:
                collisions.append({'id': k, 'event_time_s': ev, 'matched_gt_indices': gt_indexes})
        sample_stats.append(dict(id=k, **stats,
                                 ev0_enabled=record['effective_config']['ev0_dedup'],
                                 median_check_minus_onset_s=statistics.median(lag) if lag else None))
    return {'totals': dict(sum(tasks.values(), Counter())),
            'per_task': dict(tasks), 'per_sample': sample_stats,
            'same_id_on_distinct_timing_matches': collisions}


def time_shift_reference(records, times, retained, draws=500):
    """Preserve output count and relative spacing, randomize phase per video."""
    rng = random.Random(1234)
    streams = []
    for item in retained:
        k = item['id']
        actual = [times[k][i] for i in item['retained_prediction_indices']]
        events = records[k]['wall_timeline']
        origin = next(e['monotonic_s'] for e in events if e['event']=='source_start')
        horizon = max(e['monotonic_s'] - origin for e in events) + 1e-6
        gt = [g['t_sec'] for g in records[k]['ground_truth']]
        streams.append((actual, horizon, gt))
    observed = sum(score(t, gt)['tp'] for t, _, gt in streams)
    totals = []
    for _ in range(draws):
        tp = 0
        for actual, horizon, gt in streams:
            shift = rng.uniform(0, horizon)
            tp += score(sorted((t+shift) % horizon for t in actual), gt)['tp']
        totals.append(tp)
    totals.sort()
    return {'observed_tp': observed, 'random_phase_mean_tp': statistics.mean(totals),
            'random_phase_5th_tp': totals[int(.05*(draws-1))],
            'random_phase_95th_tp': totals[int(.95*(draws-1))],
            'draws': draws,
            'note': 'Descriptive count/spacing-preserving null, not a deployable causal gate or adjusted significance test.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots', nargs=2, type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--include-judge', action='store_true')
    ap.add_argument('--max-judge-calls', type=int, default=80)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs, identities, scored_rows, pending = {}, {}, {}, {}
    for root in args.roots:
        records = load_run(root)
        times = {k: delivery_times(r) for k, r in records.items()}
        result = screen(root)
        identities[root.name] = identity_audit(root, records)
        for policy in POLICIES:
            rows = []
            for item in result[policy.name]['per_sample']:
                k = item['id']
                row = dict(records[k], predictions=[
                    dict(records[k]['predictions'][i], t_sec=times[k][i])
                    for i in item['retained_prediction_indices']])
                rows.append(row)
                if row['task'] in JUDGE_TASKS:
                    matches, _, _ = match_emits_to_gt(
                        [p['t_sec'] for p in row['predictions']],
                        [g['t_sec'] for g in row['ground_truth']], 3.0)
                    for ei, gi, _ in matches:
                        triple = (row['question'], row['ground_truth'][gi].get('response',''),
                                  row['predictions'][ei]['raw'])
                        pending[triple] = True
            scored_rows[(root.name, policy.name)] = rows
            if policy.name in ('raw_level_0.5', 'cooldown_5s', 'rising_edge_0.5', 'event_identity_ev0'):
                result[policy.name]['time_shift_reference'] = time_shift_reference(
                    records, times, result[policy.name]['per_sample'])
        runs[root.name] = result
    # First write the timing audit, even if the optional judge is unavailable.
    (args.out/'timing_screen.json').write_text(json.dumps(runs, indent=2)+'\n')
    (args.out/'event_identity_audit.json').write_text(json.dumps(identities, indent=2)+'\n')

    # Read the old cache if available but write ONLY to this experiment's cache.
    judge = ContentJudge(backend='openai')
    judge.CACHE_PATH = str(args.out/'judge_cache.json')
    judge.TRACE_PATH = str(args.out/'judge_trace.jsonl')
    if Path(judge.CACHE_PATH).exists():
        judge._cache.update(json.loads(Path(judge.CACHE_PATH).read_text()))
    judge.offline = True
    calls = 0
    for triple in sorted(pending):
        key = judge._cache_key(*triple)
        if key in judge._cache:
            continue
        if not args.include_judge or calls >= args.max_judge_calls or judge.mode != 'openai':
            continue
        calls += 1
        judge.offline = False
        judge.score(*triple)
        judge.offline = True
    # No API calls during aggregation; missing verdicts remain missing.
    for (run, policy), rows in scored_rows.items():
        runs[run][policy]['content_scoring'] = aggregate([score_sample(r, judge=judge) for r in rows])
    payload = {'kind': 'suppression-only fixed-log analysis, not live counterfactual results',
               'judge_unique_triples': len(pending), 'new_judge_calls': calls,
               'runs': runs, 'identity': identities,
               'caveats': ['Both repeats use the same selected 18 videos, not independent held-out evaluation.',
                           'Changed gates/prompts alter future live history, scheduling and answers.',
                           'Only original actual delivery times are scored; no timestamp backdating.',
                           'Time-only tasks have no evaluated content; free-text tasks require the judge.',
                           'Tolerance and cooldown constants are exploratory, not fitted for publication.']}
    (args.out/'all_results.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# Trigger alternatives: fixed-log screen', '',
             '**Not live re-evaluation.** Same 18 audio=none videos, two async runs, all pruning off.', '',
             '| Policy | R1 emits / TP | R1 micro time F1 | R2 emits / TP | R2 micro time F1 | R1 / R2 micro joint F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for policy in POLICIES:
        a, b = (runs[r.name][policy.name] for r in args.roots)
        ca, cb = a['micro'], b['micro']
        ja, jb = (x['content_scoring']['overall']['joint_f1'] for x in (a,b))
        lines.append(f"| {policy.name} | {ca['emits']} / {ca['tp']} | {ca['time_f1']:.4f} | "
                     f"{cb['emits']} / {cb['tp']} | {cb['time_f1']:.4f} | {ja} / {jb} |")
    lines += ['', '## Identity audit', '']
    for run, info in identities.items():
        lines.append(f"- {run}: {json.dumps(info['totals'])}; ev0 enabled in "
                     f"{sum(s['ev0_enabled'] for s in info['per_sample'])}/18 samples.")
        lines.append(f"  {len(info['same_id_on_distinct_timing_matches'])} onset IDs occurred on emissions "
                     "matched to multiple distinct GT events; timing matches alone do not prove correct identity.")
    lines += ['', '## Rate/spacing-preserving reference (not a causal policy)', '',
              '| Run / policy | Observed GT matches | Random phase mean | Random phase 5–95% |',
              '|---|---:|---:|---:|']
    for run, policies in runs.items():
        for policy, stats in policies.items():
            if 'time_shift_reference' in stats:
                s = stats['time_shift_reference']
                lines.append(f"| {run} / {policy} | {s['observed_tp']} | {s['random_phase_mean_tp']:.2f} | "
                             f"{s['random_phase_5th_tp']}–{s['random_phase_95th_tp']} |")
    lines += ['', '## Caveats', ''] + ['- '+s for s in payload['caveats']]
    lines += [f'- Judge: {len(pending)} unique matched free-text triples; {calls} new calls. Missing joint scores are unjudged, not zero.']
    (args.out/'SCREEN_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()