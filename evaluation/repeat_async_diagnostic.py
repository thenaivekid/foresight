#!/usr/bin/env python3
"""Two fresh sequential async runs: exact text and delivery-time repeatability.

Same RNG seeds, not deterministic CUDA/lockstep execution. Differences reflect
whole stateful trajectories, including frame/check scheduling and prompts.
"""
import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import shutil
import subprocess

from repeat_vision_model_only import compare, delivery_times, load_run, source_hashes
from realtime_diagnostic import summarize
from metrics import match_emits_to_gt, TIME_ONLY

HERE = Path(__file__).resolve().parent


def report(first, second, out):
    import numpy as np
    aa, bb = load_run(first), load_run(second)
    stats, pairs = [], []
    for key in sorted(aa):
        a, b = aa[key], bb[key]
        assert a['effective_config'] == b['effective_config']
        assert not a['effective_config']['deterministic'] and a['realtime'] and b['realtime']
        at, bt = delivery_times(a), delivery_times(b)
        matches, ua, ub = match_emits_to_gt(at, bt, tolerance=3.0)
        decoded = a['task'] not in TIME_ONLY
        exact = 0
        lexical = []
        for i, j, dt in matches:
            x, y = a['predictions'][i]['raw'], b['predictions'][j]['raw']
            same = x == y
            exact += same
            ratio = SequenceMatcher(None, x, y, autojunk=False).ratio()
            lexical.append(ratio)
            pairs.append({'id': key, 'decoded': decoded, 'run1_delivery_s': at[i],
                          'run2_delivery_s': bt[j], 'abs_time_difference_s': dt,
                          'exact_text': same, 'lexical_similarity': ratio,
                          'run1_text': x, 'run2_text': y})
        gt = [g['t_sec'] for g in a['ground_truth']]
        g1 = match_emits_to_gt(at, gt, 3.0)[0]
        g2 = match_emits_to_gt(bt, gt, 3.0)[0]
        stats.append({'id': key, 'decoded': decoded,
                      'emits_run1': len(at), 'emits_run2': len(bt),
                      'cross_run_pairs_within_3s': len(matches),
                      'unmatched_run1': len(ua), 'unmatched_run2': len(ub),
                      'exact_text_in_pairs': exact,
                      'identical_text_sequence': [p['raw'] for p in a['predictions']] ==
                                                 [p['raw'] for p in b['predictions']],
                      'lexical_mean_in_pairs': float(np.mean(lexical)) if lexical else None,
                      'delivered_gt_matches_run1': len(g1), 'delivered_gt_matches_run2': len(g2)})
    decoded_pairs = [p for p in pairs if p['decoded']]
    dt = [p['abs_time_difference_s'] for p in pairs]
    ne1, ne2 = (sum(s[f'emits_run{i}'] for s in stats) for i in (1, 2))
    result = {'n_samples': len(stats), 'emits_run1': ne1, 'emits_run2': ne2,
              'cross_run_delivery_pairs_within_3s': len(pairs),
              'cross_run_timing_agreement_f1': 2*len(pairs)/(ne1+ne2) if ne1+ne2 else None,
              'paired_delivery_abs_difference_median_s': float(np.median(dt)) if dt else None,
              'paired_delivery_abs_difference_p95_s': float(np.percentile(dt, 95)) if dt else None,
              'decoded_emits_run1': sum(s['emits_run1'] for s in stats if s['decoded']),
              'decoded_emits_run2': sum(s['emits_run2'] for s in stats if s['decoded']),
              'decoded_delivery_pairs': len(decoded_pairs),
              'decoded_pairs_exact_text': sum(p['exact_text'] for p in decoded_pairs),
              'decoded_pairs_lexical_similarity_mean': float(np.mean([p['lexical_similarity'] for p in decoded_pairs])) if decoded_pairs else None,
              'nonempty_decoded_samples_identical_text_sequence': sum(s['identical_text_sequence'] for s in stats if s['decoded'] and (s['emits_run1'] or s['emits_run2'])),
              'nonempty_decoded_samples': sum(1 for s in stats if s['decoded'] and (s['emits_run1'] or s['emits_run2'])),
              'per_sample': stats}
    (out/'async_content_timing.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'paired_answers.json').write_text(json.dumps(pairs, indent=2)+'\n')
    text = ['# Async same-seed repeatability', '',
            'Two fresh sequential runs: deterministic=False, realtime=True, speed=1.0. '
            'Same videos, seeds, shard order, GPUs and configuration. No pruning or fitted gates.', '',
            f"- Emissions: {ne1} / {ne2}",
            f"- Delivery pairs within ±3 s (greedy one-to-one): {len(pairs)}",
            f"- Cross-run delivery agreement F1: {result['cross_run_timing_agreement_f1']}",
            f"- Decoded paired answers exactly equal: {result['decoded_pairs_exact_text']} / {len(decoded_pairs)}",
            f"- Mean lexical similarity of decoded pairs: {result['decoded_pairs_lexical_similarity_mean']}", '',
            '## Interpretation limits',
            '- Text comparison excludes fixed alert text. Exact matches are character-for-character.',
            '- Pairing is based on delivery proximity, not a claim that responses describe the same event.',
            '- Lexical similarity is string similarity, NOT semantic equivalence or content correctness.',
            '- Unmatched emissions remain in the timing agreement denominator; paired text similarity alone is conditional.',
            '- This is a two-run async diagnostic, not a population variance estimate.',
            '- Login-node GPU contention is uncontrolled. Changes cannot be attributed solely to CUDA nondeterminism.',
            '- Input queue/cache synchronization and single-controller decode remain unchanged in this test.', '']
    (out/'ASYNC_REPEATABILITY.md').write_text('\n'.join(text))
    print(json.dumps({k:v for k,v in result.items() if k!='per_sample'}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()
    first, second = args.out/'run1', args.out/'run2'
    if not args.report_only:
        args.out.mkdir(parents=True, exist_ok=False)
        hashes = source_hashes()
        (args.out/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
        for root in (first, second):
            root.mkdir()
            for name in ['manifest.json', 'benchmark.json']+[f'shard{i}.json' for i in range(4)]:
                shutil.copy2(args.source/name, root/name)
            if source_hashes() != hashes:
                raise RuntimeError('Inference source changed; cannot compare controlled runs')
            print(f'Starting {root.name}: async, same seed, fresh processes', flush=True)
            env = dict(os.environ, OUT=str(root), REALTIME_DIAGNOSTIC='1', ALLOW_LOGIN_DIAGNOSTIC='1')
            with (root/'launch.log').open('w') as log:
                subprocess.run(['timeout', '1800', 'bash', str(HERE.parent/'run_vision_model_only.sbatch')],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            load_run(root)
            if source_hashes() != hashes:
                raise RuntimeError('Inference source changed during run')
            summarize(root)
    compare(first, second)
    report(first, second, args.out)


if __name__ == '__main__':
    main()