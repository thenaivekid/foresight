#!/usr/bin/env python3
"""Run and report a 1x wall-clock video simulation, not a physical camera test."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def summarize(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from metrics import match_emits_to_gt

    rows = [json.loads(s) for p in sorted(out.glob('g*/online_pred.jsonl'))
            for s in p.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 18
    records = json.loads((out / 'benchmark.json').read_text())
    duration = {r['id']: float(r['duration']) for r in records}
    assert {r['id'] for r in rows} == set(duration)
    summary, pooled = [], {'frame_ingested': [], 'check_complete': [], 'response_delivered': []}
    fig, axes = plt.subplots(9, 2, figsize=(20, 26))
    for ax, r in zip(axes.flat, sorted(rows, key=lambda r: (r['task'], r['video_id']))):
        assert r['realtime'] and not r['effective_config']['deterministic']
        events = r['wall_timeline']
        origin = next(e['monotonic_s'] for e in events if e['event'] == 'source_start')
        counts, stats = {}, {}
        for name, color, label in [('frame_encoded', '#888888', 'Encoder'),
                                   ('frame_ingested', '#1f77b4', 'Ingestion'),
                                   ('check_complete', '#e67e22', 'Completed check'),
                                   ('response_delivered', '#228833', 'Delivered response')]:
            evs = [e for e in events if e['event'] == name]
            x = [e['monotonic_s'] - origin for e in evs]
            # This is age relative to the logged frame/check timestamp, NOT
            # latency from ground truth or a hardware camera sensor timestamp.
            lag = [t - e['video_s'] for t, e in zip(x, evs)]
            counts[name] = len(evs)
            stats[name] = {'median_s': float(np.median(lag)) if lag else None,
                           'p95_s': float(np.percentile(lag, 95)) if lag else None,
                           'max_s': max(lag) if lag else None}
            if name in pooled:
                pooled[name].extend(lag)
            ax.plot(x, lag, '.', color=color, ms=3, label=label)
        wall_end = next(e['monotonic_s'] - origin for e in events if e['event'] == 'source_end')
        last_activity = max(e['monotonic_s'] - origin for e in events)
        delivery = [e['monotonic_s'] - origin for e in events if e['event'] == 'response_delivered']
        gt = [g['t_sec'] for g in r['ground_truth']]
        mm, fp, fn = match_emits_to_gt(delivery, gt, 3.0)
        item = {'id': r['id'], 'duration_s': duration[r['id']],
                'source_wall_s': wall_end, 'last_activity_wall_s': last_activity,
                'drain_after_clip_s': max(0, last_activity-duration[r['id']]),
                'counts': counts, 'lag': stats,
                'ingested_per_encoded': counts['frame_ingested']/max(1, counts['frame_encoded']),
                'delivered_timing_tp': len(mm), 'delivered_timing_fp': len(fp), 'delivered_timing_fn': len(fn),
                'logged_check_timing': r['timing']}
        summary.append(item)
        ax.axhline(3, color='red', ls='--', lw=0.8)
        ax.axvline(duration[r['id']], color='grey', ls=':', lw=0.8)
        ax.set_title(f"{r['task']} | {r['video_id']}\n"
                     f"{counts['frame_ingested']}/{counts['frame_encoded']} sampled frames ingested · "
                     f"{counts['check_complete']} checks", fontsize=9, loc='left')
        ax.set(xlabel='Wall seconds since source start', ylabel='Age vs logged video time (s)')
        ax.tick_params(labelsize=7)
    fig.suptitle('Real-time camera-like video replay (1×) · 18 audio=none clips\n'
                 'No pruning; asynchronous input/controller; one full model per GPU', fontsize=15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, 0.01, 'Red: 3-second reference. Vertical gray: clip duration. '
             'Camera hardware, changing user requests, and speech output not tested.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'REALTIME_18panels.{ext}', dpi=150)
    plt.close(fig)
    pooled_stats = {k: {'n': len(v), 'median_s': float(np.median(v)) if v else None,
                        'p95_s': float(np.percentile(v, 95)) if v else None,
                        'max_s': max(v) if v else None} for k, v in pooled.items()}
    result = {'pooled_lag': pooled_stats, 'samples': summary,
              'caveats': ['Login-node diagnostic, shared load uncontrolled.',
                          'File source paces forward but can fall behind; lag plot reveals this.',
                          'Frame retention counts sampled/encoded frames, not all original camera frames.',
                          'Check timestamp is sampled before KV snapshot; not an atomic latest-frame timestamp.',
                          'Response is delivered only after full decode, not token-streamed.',
                          'Short clips cannot establish unbounded-stream stability.']}
    (out/'realtime_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(pooled_stats, indent=2), flush=True)
    print(out/'REALTIME_18panels.png', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()
    if not args.report_only:
        args.out.mkdir(parents=True, exist_ok=False)
        for name in ['manifest.json', 'benchmark.json']+[f'shard{i}.json' for i in range(4)]:
            shutil.copy2(args.source/name, args.out/name)
        env = dict(os.environ, OUT=str(args.out), REALTIME_DIAGNOSTIC='1', ALLOW_LOGIN_DIAGNOSTIC='1')
        with (args.out/'launch.log').open('w') as log:
            subprocess.run(['timeout', '1800', 'bash', str(HERE.parent/'run_vision_model_only.sbatch')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize(args.out)


if __name__ == '__main__':
    main()