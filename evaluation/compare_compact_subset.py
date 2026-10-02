#!/usr/bin/env python3
"""Score baseline vs compaction on the identical set of completed videos."""
import argparse
import glob
import json
import os

from metrics import ContentJudge, aggregate, score_sample
from utils import read_jsonl


def load(paths):
    rows = {}
    for path in paths:
        for row in read_jsonl(path):
            rows[row["id"]] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="abl_base/online_pred.jsonl")
    parser.add_argument("--arm_glob", default="abl_compact12k/shard_*/online_pred.jsonl")
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    base = load([os.path.join(here, args.base)])
    arm = load(sorted(glob.glob(os.path.join(here, args.arm_glob))))
    shared = sorted(set(base) & set(arm))

    judge = ContentJudge()
    out = {}
    for name, rows in (("baseline", base), ("compaction", arm)):
        scored = [score_sample(rows[i], tolerance=args.tolerance, judge=judge)
                  for i in shared]
        out[name] = aggregate(scored)["overall"]

    print(f"matched videos: {len(shared)}\n")
    header = f"{'arm':<12}{'time_f1':>9}{'content':>9}{'joint_f1':>10}{'emits':>7}{'matched':>9}"
    print(header)
    for name in ("baseline", "compaction"):
        o = out[name]
        print(f"{name:<12}{o['time_f1']:>9.4f}{o['content_acc']:>9.4f}"
              f"{o['joint_f1']:>10.4f}{o['n_emits']:>7}{o['n_matched']:>9}")
    b, c = out["baseline"], out["compaction"]
    print(f"\n{'delta':<12}{c['time_f1']-b['time_f1']:>+9.4f}"
          f"{c['content_acc']-b['content_acc']:>+9.4f}"
          f"{c['joint_f1']-b['joint_f1']:>+10.4f}")
    json.dump(out, open(os.path.join(here, "compact_subset_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
