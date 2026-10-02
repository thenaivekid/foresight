#!/usr/bin/env python3
"""Bounded live trigger ablation suite: one repeat per <=90-minute debug job.

Preparation is CPU-only. Execution refuses busy/low-memory login GPUs and never
cancels another workload. Every arm uses fresh workers and fixed source hashes.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO/'foresight'))
from config import AsyncOmniConfig
from vision_model_only import model_only_config
from trigger_ablation_profiles import ARMS, CORE_ARMS, configure_arm
from repeat_vision_model_only import load_run, source_hashes, delivery_times


def prepare(args):
    if (args.out/'experiment.json').exists():
        raise FileExistsError('A prepared experiment exists; choose a new root for a new design')
    args.out.mkdir(parents=True, exist_ok=True)
    ids = json.loads((args.source/'manifest.json').read_text())['ids']
    base = model_only_config(AsyncOmniConfig())
    schedules = {}
    for repeat in (1, 2):
        arms = list(args.arms)
        random.Random(1234+repeat).shuffle(arms)
        schedules[str(repeat)] = arms
        for arm in arms:
            root = args.out/f'r{repeat}'/arm
            root.mkdir(parents=True, exist_ok=False)
            for name in ['manifest.json','benchmark.json']+[f'shard{i}.json' for i in range(4)]:
                shutil.copy2(args.source/name, root/name)
            cfg = configure_arm(base, arm)
            assert not cfg.deterministic and cfg.realtime and cfg.prune_mode=='off'
            assert not cfg.prune_classes and not cfg.plan_classes and not cfg.plan_compact
            assert cfg.plan_fps and cfg.plan_cadence and cfg.plan_question
            assert cfg.hit_threshold == 0.5 and not cfg.task_hit_thresholds
            (root/'expected_profile.json').write_text(json.dumps(dataclasses.asdict(cfg), indent=2)+'\n')
    manifest = {'source': str(args.source), 'ids': ids, 'schedules': schedules,
                'n_arms': len(args.arms), 'n_repeats': 2, 'n_samples_per_arm': len(ids),
                'audio_filter': json.loads((args.source/'manifest.json').read_text()).get('audio_filter','none'),
                'max_parallel_gpu_workers': 4, 'budget_per_repeat_s': 4800,
                'source_hashes': source_hashes(),
                'protocol': 'async, full clips, same RNG seeds/shards, no pruning; report actual wall delivery',
                'status': 'prepared, not executed'}
    (args.out/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Prepared {len(args.arms)} arms × 2 repeats × {len(ids)} samples. No inference launched.', flush=True)


def preflight():
    if not os.environ.get('SLURM_JOB_ID') and os.environ.get('ALLOW_LOGIN_DIAGNOSTIC') != '1':
        raise RuntimeError('Outside SLURM: explicit login diagnostic authorization required')
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    rows = [[float(s.strip()) for s in line.split(',')] for line in text.splitlines()]
    if len(rows) != 4 or any(row[1] < 60000 for row in rows):
        raise RuntimeError('Need a full four-GPU node with >=60,000 MiB free per GPU; other workloads left untouched')
    if not os.environ.get('SLURM_JOB_ID') and any(row[2] > 5 for row in rows):
        raise RuntimeError('Login GPUs are busy; refusing competing inference')


def run(args):
    manifest = json.loads((args.out/'experiment.json').read_text())
    if source_hashes() != manifest['source_hashes']:
        raise RuntimeError('Inference source changed after preparation; prepare a fresh experiment')
    preflight()
    started = time.monotonic()
    for arm in manifest['schedules'][str(args.repeat)]:
        root = args.out/f'r{args.repeat}'/arm
        try:
            load_run(root)
        except ValueError:
            pass
        else:
            print(f'{arm}: already complete', flush=True)
            continue
        remaining = int(manifest['budget_per_repeat_s'] - (time.monotonic()-started))
        if remaining < 600:
            raise RuntimeError('Not enough remaining budget for another arm; completed samples preserved')
        preflight()
        env = dict(os.environ, OUT=str(root), TRIGGER_ARM=arm, REALTIME_DIAGNOSTIC='1', RUN_BUDGET_S='0')
        print(f'Starting repeat {args.repeat}, arm {arm}', flush=True)
        with (root/'launch.log').open('a') as log:
            subprocess.run(['timeout',str(min(1200,remaining)), 'bash',str(REPO/'run_vision_model_only.sbatch')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = load_run(root)
        for row in rows.values():
            assert row['experiment_arm'] == arm
            cfg = row['effective_config']
            assert not cfg['deterministic'] and row['realtime'] and cfg['prune_mode']=='off'
            assert not cfg['prune_classes'] and not cfg['plan_compact']
        if source_hashes() != manifest['source_hashes']:
            raise RuntimeError('Source changed during live experiment')
    report(args)


def report(args):
    from metrics import ContentJudge, aggregate, score_sample
    judge = ContentJudge(backend='openai')
    judge.offline = True
    results = {}
    for repeat in (1,2):
        for root in sorted((args.out/f'r{repeat}').glob('*')):
            if not root.is_dir():
                continue
            try:
                rows = load_run(root)
            except ValueError:
                continue
            scored = []
            for row in rows.values():
                times = delivery_times(row)
                pred = dict(row, predictions=[dict(p, t_sec=t) for p,t in zip(row['predictions'], times)])
                scored.append(score_sample(pred, judge=judge))
            results[f'r{repeat}/{root.name}'] = aggregate(scored)
    (args.out/'live_metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    for name, r in results.items():
        o = r['overall']
        print(name, 'macro_time_f1=',o['macro_time_f1'], 'micro_time_f1=',o['time_f1'],
              'joint_f1=',o['joint_f1'], 'unjudged=',o['n_unjudged'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare','run','report'))
    ap.add_argument('--source', type=Path, default=HERE/'output_vision_model_only_20260910')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repeat', type=int, choices=(1,2), default=1)
    ap.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=CORE_ARMS)
    args = ap.parse_args()
    {'prepare':prepare, 'run':run, 'report':report}[args.action](args)


if __name__ == '__main__':
    main()