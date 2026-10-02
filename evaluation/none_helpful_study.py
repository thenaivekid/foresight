#!/usr/bin/env python3
"""Staged vision-only study on none/helpful samples; full run requires validation.

Selection uses metadata only, never predictions or GT event times. The short
pilot is a screening set, not a representative 932-sample benchmark estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from dataset import ALL_TASKS
from repeat_vision_model_only import delivery_times, load_run, source_hashes
from trigger_ablation_profiles import CORE_ARMS, configure_arm

HERE = Path(__file__).resolve().parent


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def choose_samples(rows, previous_video_ids, seed=1234):
    """Match one none + one helpful video per task per phase, globally distinct.

    Validation excludes the 18 previously inspected diagnostic video IDs.
    Bipartite matching avoids starving a small category by greedy selection.
    """
    eligible = [r for r in rows if r.get('audio_dependency') in {'none','helpful'}]
    slots, candidates = [], {}
    for phase in ('calibration','validation'):
        max_duration = 180 if phase == 'calibration' else 300
        for task in sorted(ALL_TASKS):
            for audio in ('none','helpful'):
                slot = (phase, task, audio)
                slots.append(slot)
                choices = [r for r in eligible if r['task']==task and r['audio_dependency']==audio
                           and 0 < float(r['duration']) <= max_duration
                           and (phase != 'validation' or r['video_id'] not in previous_video_ids)]
                def rank(r):
                    digest = hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
                    return (float(r['duration']), digest) if phase == 'calibration' else (digest,)
                candidates[slot] = sorted(choices, key=rank)
                if not choices:
                    raise ValueError(f'No eligible video for {slot}; do not silently drop a task or relax the split')
    assigned, owner = {}, {}

    def assign(slot, visited):
        for record in candidates[slot]:
            video = record['video_id']
            if video in visited:
                continue
            visited.add(video)
            occupant = owner.get(video)
            if occupant is None or assign(occupant, visited):
                owner[video] = slot
                assigned[slot] = record
                return True
        return False

    for slot in sorted(slots, key=lambda s: (len(candidates[s]), s)):
        if not assign(slot, set()):
            raise ValueError('Cannot build the declared video-disjoint pilot; no inference was launched')
    result = {phase: [assigned[s] for s in slots if s[0]==phase]
              for phase in ('calibration','validation')}
    assert len({r['video_id'] for rs in result.values() for r in rs}) == len(slots)
    return result


def write_subset(root, rows, dataset, role):
    ids = [r['id'] for r in rows]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Empty/duplicate subset')
    shards, durations = [[] for _ in range(4)], [0.0]*4
    for r in sorted(rows, key=lambda r: (-float(r['duration']),r['id'])):
        gpu = min(range(4), key=durations.__getitem__)
        shards[gpu].append(r)
        durations[gpu] += float(r['duration'])
    save(root/'benchmark.json', rows)
    for gpu, shard in enumerate(shards):
        save(root/f'shard{gpu}.json', shard)
    save(root/'manifest.json', {'ids':ids, 'n_samples':len(ids), 'dataset':str(dataset),
                               'audio_filter':'none_helpful', 'role':role,
                               'n_unique_videos':len({r['video_id'] for r in rows}),
                               'by_task':dict(Counter(r['task'] for r in rows)),
                               'by_audio_dependency':dict(Counter(r['audio_dependency'] for r in rows)),
                               'shard_duration_s':durations})


def prepare(args):
    if (args.out/'study.json').exists():
        raise FileExistsError('Study already prepared; do not replace frozen sample assignments')
    rows = json.loads(args.benchmark.read_text())
    eligible = [r for r in rows if r.get('audio_dependency') in {'none','helpful'}]
    missing = [r['id'] for r in eligible if not (args.dataset/r['video_path']).is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} videos; cannot claim a full eligible-population run')
    previous = json.loads((args.previous/'benchmark.json').read_text())
    previous_ids = {r['video_id'] for r in previous}
    splits = choose_samples(eligible, previous_ids)
    for phase, selected in splits.items():
        write_subset(args.out/'samples'/phase, selected, args.dataset, phase)
    write_subset(args.out/'samples/full', eligible, args.dataset, 'all_eligible_including_calibration')
    cal = {r['video_id'] for r in splits['calibration']}
    val = {r['video_id'] for r in splits['validation']}
    groups = {r['id']: ('calibration_video' if r['video_id'] in cal else
                        'validation_video' if r['video_id'] in val else
                        'previous_diagnostic_video' if r['video_id'] in previous_ids else
                        'remaining_video') for r in eligible}
    save(args.out/'full_reporting_groups.json', groups)
    from trigger_live_matrix import prepare as prepare_matrix
    prepare_matrix(SimpleNamespace(source=args.out/'samples/calibration',out=args.out/'calibration',arms=CORE_ARMS))
    study = {'audio_filter':'none_helpful', 'vision_only':True, 'pruning':'off',
             'deterministic':False, 'realtime':True, 'threshold':0.5,
             'eligible_samples':len(eligible), 'eligible_video_ids':len({r['video_id'] for r in eligible}),
             'source':str(args.benchmark), 'source_sha256':hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
             'source_hashes':source_hashes(), 'arms':list(CORE_ARMS), 'repeats':2,
             'selection':'18 calibration + 18 validation; one none and one helpful per task; '
                         'metadata-only matching, distinct video IDs across phases; '
                         'calibration prefers short <=180s clips, validation seeded <=300s clips',
             'previous_diagnostic_videos_excluded_from_validation':sorted(previous_ids),
             'full_reporting_group_counts':dict(Counter(groups.values())),
             'promotion_rule':'Select highest mean calibration macro joint F1 across two live repeats. '
                              'Freeze before validation. Validate on two repeats against raw control: '
                              'positive mean macro joint delta, nonnegative joint delta in EACH repeat, '
                              'nonnegative mean macro timing delta, no unjudged matches. '
                              'If the rule fails, STOP; do not tune on validation or launch full.',
             'generalization_limit':'Validation is held out from this experiment selection, not claimed '
                                    'unseen by all historical project experiments. Small pilot is not a '
                                    'confidence interval or proof of a perfect configuration.',
             'status':'prepared; no live selection result or full-run approval yet'}
    save(args.out/'study.json', study)
    print(json.dumps({k:study[k] for k in ('eligible_samples','eligible_video_ids','full_reporting_group_counts','status')},indent=2))


def check_provenance(root):
    study = json.loads((root/'study.json').read_text())
    if study['source_hashes'] != source_hashes():
        raise RuntimeError('Inference code changed after study preparation; create a new study instead of mixing runs')
    return study


def phase_arms(root, phase):
    if phase == 'calibration':
        return json.loads((root/'study.json').read_text())['arms']
    frozen = json.loads((root/'frozen_config.json').read_text())
    return list(dict.fromkeys(['raw',frozen['arm']]))


def evaluate_phase(root, phase, include_judge=False, max_calls=200):
    from metrics import ContentJudge, aggregate, score_sample, match_emits_to_gt, JUDGE_TASKS
    arms = phase_arms(root, phase)
    records, triples = {}, set()
    for repeat in (1,2):
        for arm in arms:
            path = root/phase/f'r{repeat}'/arm
            rows = load_run(path)  # Incomplete inference cannot pick a winner.
            if any('TICK FAILED' in p.read_text() or 'TIMEOUT' in p.read_text()
                   for p in path.glob('g[0-3].log')):
                raise RuntimeError(f'{path}: failed controller ticks/handshakes; not eligible for selection')
            scored = []
            for row in rows.values():
                if row.get('experiment_arm') != arm or row['effective_config']['deterministic']:
                    raise ValueError('Unexpected arm or non-async result')
                times = delivery_times(row)
                p = dict(row,predictions=[dict(e,t_sec=t) for e,t in zip(row['predictions'],times)])
                scored.append(p)
                if p['task'] in JUDGE_TASKS:
                    for ei,gi,_ in match_emits_to_gt(times,[g['t_sec'] for g in p['ground_truth']],3.0)[0]:
                        triples.add((p['question'],p['ground_truth'][gi].get('response',''),p['predictions'][ei]['raw']))
            records[f'r{repeat}/{arm}'] = scored
    judge = ContentJudge(backend='openai')
    judge.CACHE_PATH = str(root/'judge_cache.json')
    judge.TRACE_PATH = str(root/'judge_trace.jsonl')
    if Path(judge.CACHE_PATH).exists():
        judge._cache.update(json.loads(Path(judge.CACHE_PATH).read_text()))
    judge.offline, calls = True, 0
    for triple in sorted(triples):
        if judge._cache_key(*triple) in judge._cache:
            continue
        if include_judge and calls < max_calls and judge.mode == 'openai':
            judge.offline = False
            judge.score(*triple)
            judge.offline = True
            calls += 1
    metrics = {k:aggregate([score_sample(r,judge=judge) for r in rows]) for k,rows in records.items()}
    save(root/f'{phase}_metrics.json',metrics)
    if any(r['overall']['n_unjudged'] for r in metrics.values()):
        raise RuntimeError('Content verdicts remain unjudged; no config selection/full launch is allowed')
    return metrics


def freeze(args):
    study = check_provenance(args.out)
    target = args.out/'frozen_config.json'
    if target.exists():
        raise FileExistsError('Configuration is already frozen; validation cannot be used to select a different arm')
    metrics = evaluate_phase(args.out,'calibration',args.include_judge,args.max_judge_calls)
    def mean(arm,key):
        return sum(metrics[f'r{r}/{arm}']['overall'][key] for r in (1,2))/2
    arm = max(study['arms'],key=lambda a:(mean(a,'macro_joint_f1'),mean(a,'macro_time_f1'),-study['arms'].index(a)))
    from config import AsyncOmniConfig
    from vision_model_only import model_only_config
    cfg = configure_arm(model_only_config(AsyncOmniConfig()),arm)
    save(target, {'arm':arm, 'config':dataclasses.asdict(cfg), 'source_hashes':study['source_hashes'],
                  'calibration_mean_macro_joint_f1':mean(arm,'macro_joint_f1'),
                  'selection_basis':'Only calibration, two fresh live repeats, complete content scoring'})
    from trigger_live_matrix import prepare as prepare_matrix
    prepare_matrix(SimpleNamespace(source=args.out/'samples/validation',out=args.out/'validation',
                                   arms=phase_arms(args.out,'validation')))
    print(f'Frozen {arm}; validation is prepared. No full run permitted yet.',flush=True)


def validate(args):
    check_provenance(args.out)
    frozen = json.loads((args.out/'frozen_config.json').read_text())
    metrics = evaluate_phase(args.out,'validation',args.include_judge,args.max_judge_calls)
    arm = frozen['arm']
    joint = [metrics[f'r{r}/{arm}']['overall']['macro_joint_f1']-
             metrics[f'r{r}/raw']['overall']['macro_joint_f1'] for r in (1,2)]
    timing = [metrics[f'r{r}/{arm}']['overall']['macro_time_f1']-
              metrics[f'r{r}/raw']['overall']['macro_time_f1'] for r in (1,2)]
    passed = sum(joint)>0 and min(joint)>=0 and sum(timing)>=0
    save(args.out/'validation_decision.json', {'arm':arm,'passed':passed,
         'macro_joint_deltas':joint,'macro_time_deltas':timing,
         'status':'Eligible for full descriptive evaluation' if passed else
                  'STOP: no consistently improved validated configuration; do not retune on validation'})
    print('Validation:', 'PASS' if passed else 'FAIL; full run blocked', flush=True)


def full_guard(root):
    """Missing or failed prerequisites always fail closed, never default to raw."""
    check_provenance(root)
    frozen = json.loads((root/'frozen_config.json').read_text())
    decision = json.loads((root/'validation_decision.json').read_text())
    if decision.get('passed') is not True or decision.get('arm') != frozen['arm']:
        raise RuntimeError('Full evaluation requires a frozen, successfully validated configuration')
    return frozen['arm']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','run-calibration','freeze','run-validation','validate','run-full'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repeat',type=int,choices=(1,2),default=1)
    ap.add_argument('--benchmark',type=Path,default=Path('/path/to/work/omnipro_data/benchmark.json'))
    ap.add_argument('--dataset',type=Path,default=Path('/path/to/work/omnipro_data'))
    ap.add_argument('--previous',type=Path,default=HERE/'output_vision_model_only_20260910')
    ap.add_argument('--include-judge',action='store_true')
    ap.add_argument('--max-judge-calls',type=int,default=200)
    args = ap.parse_args()
    if args.action in ('prepare','freeze','validate'):
        return {'prepare':prepare,'freeze':freeze,'validate':validate}[args.action](args)
    check_provenance(args.out)
    from trigger_live_matrix import run as run_matrix, preflight
    if args.action != 'run-full':
        phase = args.action.removeprefix('run-')
        return run_matrix(SimpleNamespace(out=args.out/phase,repeat=args.repeat))
    arm = full_guard(args.out)
    preflight()
    dest = args.out/'full_run'
    if not dest.exists():
        shutil.copytree(args.out/'samples/full',dest)
    env = dict(os.environ,OUT=str(dest),TRIGGER_ARM=arm,REALTIME_DIAGNOSTIC='1',RUN_BUDGET_S='4200')
    with (dest/'launch.log').open('a') as log:
        result = subprocess.run(['timeout','4800','bash',str(HERE.parent/'run_vision_model_only.sbatch')],
                                env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode not in (0,124):
        raise RuntimeError(f'Full-run worker failed ({result.returncode}); inspect logs before continuation')
    try:
        rows = load_run(dest)
    except ValueError:
        print('Full run incomplete; saved complete samples are resumable in another debug job.',flush=True)
    else:
        print(f'Full inference complete: {len(rows)} samples. Score full and reporting groups separately.',flush=True)


if __name__ == '__main__':
    main()