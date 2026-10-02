"""
reshard_remaining.py — re-split ONLY the unfinished work into N balanced shards.

WHY: the original 16-way split was made against the whole 932-sample corpus. Eight
of those shards are now complete, so re-running that split on 16 GPUs would leave
half of them with nothing to do. This rebuilds the split from what is ACTUALLY
left, so every worker gets real work.

Balancing is by VIDEO DURATION, not sample count: the controller ticks once per
video-second, so duration is what wall time is proportional to (measured 2.56 s of
wall per second of video). Longest-first greedy into the currently-lightest shard
— simple, and good enough that the spread stays under a percent.

Done-ness is read from the predictions themselves (online_pred.jsonl), which is
the same source evaluate.py's --resume uses, so this can never disagree with it.

Usage:
    python reshard_remaining.py --run-dir RUN_DIR --out splits_r2 --n 16
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output")
    ap.add_argument("--splits", default="splits_all")
    ap.add_argument("--out", default="splits_r2")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--job-seconds", type=float, default=22*60,
                    help="wall clock of ONE job (debug: 90 node-min / nodes)")
    ap.add_argument("--startup-seconds", type=float, default=60,
                    help="model load before the first sample starts")
    ap.add_argument("--rate", type=float, default=2.56,
                    help="measured wall seconds per second of video")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(here, args.run_dir)
    src = os.path.join(here, args.splits)
    out = os.path.join(here, args.out)
    os.makedirs(out, exist_ok=True)

    done = done_ids(run_dir)
    every, seen = [], set()
    for f in sorted(glob.glob(os.path.join(src, "*.json"))):
        for e in json.load(open(f)):
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            every.append(e)

    todo = [e for e in every if e["id"] not in done]
    tot = sum(float(e.get("duration") or 0) for e in todo)
    print(f"corpus {len(every)} | done {len(done)} | REMAINING {len(todo)} "
          f"({tot/3600:.1f} video-hours)")
    if not todo:
        print("nothing left to do")
        return

    # Samples too long to EVER finish inside one job window are quarantined.
    # evaluate.py walks a shard in file order, so a single unfinishable sample at
    # the front blocks every sample behind it forever -- the shard makes zero
    # progress no matter how many generations run.
    cutoff = (args.job_seconds - args.startup_seconds) / args.rate
    longs = [e for e in todo if float(e.get("duration") or 0) > cutoff]
    fits = [e for e in todo if float(e.get("duration") or 0) <= cutoff]
    if longs:
        p = os.path.join(out, "toolong.json")
        with open(p, "w") as fh:
            json.dump(sorted(longs, key=lambda x: float(x.get("duration") or 0)), fh)
        print(f"  QUARANTINED {len(longs)} samples > {cutoff:.0f}s "
              f"(need a longer job) -> {p}")

    shards = [[] for _ in range(args.n)]
    load = [0.0] * args.n
    for e in sorted(fits, key=lambda x: -float(x.get("duration") or 0)):
        k = min(range(args.n), key=lambda i: load[i])
        shards[k].append(e)
        load[k] += float(e.get("duration") or 0)

    for i, s in enumerate(shards):
        # SHORTEST FIRST inside each shard. Balancing packs longest-first, which
        # left every shard opening with its own worst case: one wall-clock kill
        # and the generation banked nothing. Ascending order means each job
        # finishes as many whole samples as it can before it is killed, and only
        # the single in-flight sample is ever lost.
        s.sort(key=lambda x: float(x.get("duration") or 0))
        p = os.path.join(out, f"r{i:02d}.json")
        with open(p, "w") as fh:
            json.dump(s, fh)
    lo, hi = min(load), max(load)
    print(f"wrote {args.n} shards to {out}")
    print(f"  per shard: {len(shards[0])}-{len(shards[-1])} samples, "
          f"{lo/3600:.2f}-{hi/3600:.2f} video-hours "
          f"(spread {100*(hi-lo)/(hi or 1):.1f}%)")
    print(f"  projected wall at 2.56x realtime: {hi*2.56/3600:.2f} h per worker")


if __name__ == "__main__":
    main()
