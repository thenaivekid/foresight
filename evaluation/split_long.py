"""
split_long.py — split the QUARANTINED long videos (splits_r2/toolong.json) across
the N tasks of the long-video job, skipping anything already predicted.

WHY THIS EXISTS SEPARATELY FROM reshard_remaining.py
reshard_remaining.py deliberately pulls samples longer than one 22-minute job can
chew (cutoff ~= (22*60 - 60)/2.56 ~= 500 video-seconds) into toolong.json, because
evaluate.py walks a shard in FILE ORDER and resumes per sample: one unfinishable
sample at the head of a shard blocks every sample behind it forever. Those samples
need a job with a longer wall clock, which on debug-qos means fewer nodes
(MaxTRESMins node=90 -> 2 nodes x 44 min = 88 node-minutes).

WHY THE SPLIT IS REBUILT EVERY GENERATION, NOT WRITTEN ONCE
Done-ness is read from the predictions themselves (RUN_DIR/**/online_pred.jsonl),
exactly like reshard_remaining.py and exactly like evaluate.py --resume. That has
two consequences we want:
  * samples the main r00..r15 chain happened to finish on its own (5 of the 11 were
    already done when this was written, because an older split still carried them)
    are never re-run;
  * each generation only carries what is genuinely left, so the last generation is
    not 8 workers idling behind one straggler.
evaluate.py appends to online_pred.jsonl only AFTER a sample completes, so a sample
killed mid-flight is simply "not done" and comes back next generation. Moving it to
a different bucket between generations therefore loses nothing.

Balancing is longest-first greedy into the currently-lightest bucket (wall time is
proportional to video duration), then SHORTEST-FIRST inside each bucket so a
wall-clock kill only ever costs the single in-flight sample.

Usage:
    python split_long.py --run-dir RUN_DIR --out splits_long --n 8
    python split_long.py --check           # cover-exactly-once self test, no writes
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def done_ids(run_dir):
    ids = set()
    for p in glob.glob(os.path.join(run_dir, "**", "online_pred.jsonl"),
                       recursive=True):
        with open(p, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    continue          # truncated tail line: treat as not done
    return ids


def bucket(todo, n):
    """Longest-first greedy into the lightest bucket; shortest-first within."""
    buckets = [[] for _ in range(n)]
    load = [0.0] * n
    for e in sorted(todo, key=lambda x: -float(x.get("duration") or 0)):
        k = min(range(n), key=lambda i: load[i])
        buckets[k].append(e)
        load[k] += float(e.get("duration") or 0)
    for b in buckets:
        b.sort(key=lambda x: float(x.get("duration") or 0))
    return buckets, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output")
    ap.add_argument("--source", default="splits_r2/toolong.json")
    ap.add_argument("--out", default="splits_long")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--check", action="store_true",
                    help="verify the split covers every sample exactly once; "
                         "writes nothing")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(here, args.run_dir)
    src = os.path.join(here, args.source)
    out = os.path.join(here, args.out)

    every = json.load(open(src))
    done = done_ids(run_dir)
    todo = [e for e in every if e["id"] not in done]

    if args.check:
        # Cover-exactly-once over the FULL file, not just what is left, so the
        # invariant is tested independently of how much has already run.
        for pool, label in ((every, "all"), (todo, "remaining")):
            buckets, _ = bucket(pool, args.n)
            flat = [e["id"] for b in buckets for e in b]
            src_ids = [e["id"] for e in pool]
            assert len(buckets) == args.n, "wrong bucket count"
            assert len(flat) == len(src_ids), f"{label}: count {len(flat)} != {len(src_ids)}"
            assert sorted(flat) == sorted(src_ids), f"{label}: id set differs"
            assert len(set(flat)) == len(flat), f"{label}: duplicate id"
            print(f"  [check] {label}: {len(src_ids)} samples -> {args.n} buckets "
                  f"sizes={[len(b) for b in buckets]} exactly-once OK")
        print("[check] PASS")
        return

    os.makedirs(out, exist_ok=True)
    buckets, load = bucket(todo, args.n)
    for i, b in enumerate(buckets):
        # every bucket is written, empty ones included: task k must always find
        # its file, and evaluate.py exits cleanly on an empty benchmark.
        with open(os.path.join(out, f"L{i}.json"), "w") as fh:
            json.dump(b, fh)
    print(f"[split_long] corpus {len(every)} | already predicted {len(every) - len(todo)} "
          f"| REMAINING {len(todo)}")
    print(f"[split_long] wrote {args.n} buckets to {out}: "
          f"sizes={[len(b) for b in buckets]} "
          f"video-s={[round(x) for x in load]}")
    print(f"REMAINING={len(todo)}")


if __name__ == "__main__":
    main()
