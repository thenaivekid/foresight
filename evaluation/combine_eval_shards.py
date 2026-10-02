#!/usr/bin/env python3
"""Combine task-disjoint OmniPro shards and rescore them as one evaluation."""
import argparse
import glob
import json
import os

from metrics import ContentJudge, aggregate, score_sample
from utils import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    rows_by_id = {}
    paths = sorted(glob.glob(os.path.join(args.root, "shard_*", "online_pred.jsonl")))
    for path in paths:
        for row in read_jsonl(path):
            rows_by_id[row["id"]] = row
    if len(rows_by_id) != args.expected:
        raise RuntimeError(
            f"expected {args.expected} unique predictions, found {len(rows_by_id)}")

    rows = list(rows_by_id.values())
    merged_path = os.path.join(args.root, "online_pred.jsonl")
    with open(merged_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")

    judge = ContentJudge()
    result = aggregate([
        score_sample(row, tolerance=args.tolerance, judge=judge) for row in rows
    ])
    result["n"] = len(rows)
    write_json(os.path.join(args.root, "online_metrics.json"), result)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()