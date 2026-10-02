#!/usr/bin/env python3
"""Plot fixed-log trigger comparisons; does not perform or claim live inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.root/'all_results.json').read_text())
    runs = data['runs']
    policies = list(runs['run1'])
    y = np.arange(len(policies))
    fig, axes = plt.subplots(1, 2, figsize=(17, 12), sharey=True)
    for ax, metric, title in zip(axes, ['time_f1','joint_f1'],
                                ['Actual-delivery timing F1', 'Actual-delivery + content joint F1']):
        for i, run in enumerate(('run1','run2')):
            vals = [runs[run][p]['content_scoring']['overall'][metric] for p in policies]
            assert all(v is not None for v in vals), 'Unjudged content must not be plotted as zero'
            ax.barh(y+(i-.5)*.36, vals, height=.34, color=('#1f77b4','#e67e22')[i],
                    label=run, alpha=.9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Micro F1 (0–1)')
        ax.set_xlim(0, .5)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels=[p.replace('_',' ') for p in policies], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle('21 causal trigger alternatives · fixed-log suppression-only screen\n'
                 'Two async runs of the SAME 18 audio=none videos · NOT new live results', fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5,.035), frameon=False)
    fig.text(.5,.016,'±3 s one-to-one GT matching at unchanged wall delivery times. '
             'Structured content: exact checks; free text: gpt-5-mini judge. No threshold fitting.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    for ext in ('png','pdf'):
        p=args.root/f'TRIGGER_SCREEN_21_POLICIES.{ext}'
        fig.savefig(p,dpi=160)
        print(p)
    plt.close(fig)


if __name__ == '__main__':
    main()