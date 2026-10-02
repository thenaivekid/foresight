"""
make_splits.py — split the OmniPro `audio != required` set into N balanced shards.

Balanced by TOTAL VIDEO SECONDS, not sample count, so every worker finishes at
roughly the same time (cost per sample is ~linear in duration: one controller tick
per video-second).

Samples that share a video_id are kept in the SAME shard — the video is decoded once
per shard, so splitting them would duplicate the most expensive work.

Usage:  python make_splits.py --n 16 --out splits_all
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

DEFAULT_BENCH = "/path/to/work/omnipro_data/benchmark.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=DEFAULT_BENCH)
    ap.add_argument("--n", type=int, default=16, help="number of shards")
    ap.add_argument("--out", default="splits_all")
    ap.add_argument("--audio", default="none,helpful",
                    help="comma-separated audio_dependency values to KEEP")
    ap.add_argument("--max_seconds", type=float, default=600.0,
                    help="only used for the cost estimate, not for filtering")
    args = ap.parse_args()

    keep_audio = set(a.strip() for a in args.audio.split(","))
    data = json.load(open(args.benchmark))
    samples = [e for e in data if e.get("audio_dependency") in keep_audio]

    # group by video: same clip -> same shard (decode it once)
    by_video: dict[str, list] = defaultdict(list)
    for e in samples:
        by_video[e["video_id"]].append(e)

    def dur(e):
        for k in ("duration", "duration_sec", "video_duration"):
            if k in e:
                try:
                    return float(e[k])
                except (TypeError, ValueError):
                    pass
        return 180.0                      # median-ish fallback

    # cost of a video ~ its capped duration (one tick per video-second)
    videos = [(vid, min(dur(items[0]), args.max_seconds), items)
              for vid, items in by_video.items()]
    videos.sort(key=lambda x: -x[1])      # longest first: greedy LPT packing

    shards = [[] for _ in range(args.n)]
    load = [0.0] * args.n
    for vid, cost, items in videos:
        i = load.index(min(load))         # put it on the least-loaded shard
        shards[i].extend(items)
        load[i] += cost

    os.makedirs(args.out, exist_ok=True)
    print(f"{len(samples)} samples / {len(by_video)} videos -> {args.n} shards "
          f"(audio in {sorted(keep_audio)})")
    for i, (sh, ld) in enumerate(zip(shards, load)):
        path = os.path.join(args.out, f"g{i:02d}.json")
        json.dump(sh, open(path, "w"))
        print(f"  g{i:02d}: {len(sh):4d} samples  "
              f"{len({e['video_id'] for e in sh}):4d} videos  "
              f"{ld/3600:5.2f} video-hours")
    spread = (max(load) - min(load)) / max(load) * 100 if max(load) else 0
    print(f"load spread: {spread:.1f}%  (total {sum(load)/3600:.1f} video-hours)")
    assert sum(len(s) for s in shards) == len(samples), "lost samples while sharding"


if __name__ == "__main__":
    main()
