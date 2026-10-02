#!/usr/bin/env python3
"""make_bench_shards.py -- split a benchmark arm into 4 per-GPU shard manifests.

Sharding is by VIDEO, never by question: one streaming pass answers every probe
in a video, so splitting a video across GPUs would re-stream it. Shards are
balanced on total video-seconds (the real cost driver -- durations vary ~10x
across these benchmarks), not on video count.

    python make_bench_shards.py --bench streamingbench --arm sb_po_calib
    python make_bench_shards.py --bench ovobench --arm ovo_far
"""
from __future__ import annotations

import argparse
import math
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SB_DATA = os.environ.get("SB_DATA", "/path/to/work/streamingbench_data")
OVO_DATA = os.environ.get("OVO_DATA", "/path/to/work/ovobench_data")

# arm -> (bench, task filter). section 4.
ARMS = {
    # StreamingBench
    "sb_po_calib":  ("streamingbench", ["Proactive Output"]),
    "sb_po_eval":   ("streamingbench", ["Proactive Output"]),
    "sb_mcq":       ("streamingbench", None),        # all 17 MCQ tasks
    # OVO-Bench
    "ovo_far":      ("ovobench", ["SSR", "CRR", "REC"]),
    "ovo_mcq":      ("ovobench", ["EPM", "ASI", "HLD",
                                  "OCR", "ACR", "ATR", "STU", "FPD", "OJR"]),
}


def _load(bench, tasks):
    if bench == "streamingbench":
        from benchdata import streamingbench as m
        return m.load_groups(os.path.join(SB_DATA, "ann"), SB_DATA, tasks=tasks)
    from benchdata import ovobench as m
    return m.load_groups(os.path.join(OVO_DATA, "ann", "ovo_bench_new.json"),
                         os.path.join(OVO_DATA, "src_videos"), tasks=tasks)


_DURS = None


def cost_s(g):
    """Seconds of video a pass over this group actually decodes.

    NOT simply the last probe timestamp: the pass stops at whichever comes first,
    the last probe or the end of the video. A handful of StreamingBench
    annotations carry corrupt timestamps -- sample_32_Anomaly has a probe at
    18,660 s ("05:11:00") on a 661 s clip, a transposed "00:11:01" -- and costing
    those at face value put a phantom 5 h of work on one shard. That skewed
    sb_mcq to a 17% spread and a 455-minute idle tail, i.e. ~22 GPU-hours of
    three GPUs waiting on one.

    video_durations.json is written by the ffprobe sweep; if a path is missing we
    fall back to the probe time, which is the old (safe, pessimistic) behaviour.
    """
    global _DURS
    if _DURS is None:
        f = os.path.join(HERE, "video_durations.json")
        try:
            _DURS = json.load(open(f))
        except Exception:
            _DURS = {}
    last = max(p["t"] for p in g["probes"])
    d = _DURS.get(g["video_path"])
    return min(last, d) if d else last


def balance(groups, n):
    """Greedy longest-processing-time: sort by cost desc, always append to the
    lightest shard. Within a few percent of optimal for this shape, and it keeps
    the 4 GPUs finishing together so no link wastes its wall on one straggler."""
    shards = [[] for _ in range(n)]
    load = [0.0] * n
    for g in sorted(groups, key=lambda g: -cost_s(g)):
        i = load.index(min(load))
        shards[i].append(g)
        load[i] += cost_s(g)
    return shards, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["streamingbench", "ovobench"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "splits_bench"))
    # PO is the one trajectory-coupled arm, so its calib and eval sets MUST be
    # disjoint: the gate feeds `reported` back into the prompt, so a video used
    # to fit a threshold cannot also report it.
    ap.add_argument("--po_calib_frac", type=float, default=0.30)
    ap.add_argument("--min_calib_frac", type=float, default=0.05,
                    help="hard floor on the calibration split, as a fraction of "
                         "the arm's videos. A threshold fitted on a handful of "
                         "videos is noise, not a calibration; this floor is "
                         "enforced even if --po_calib_frac is set lower.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--require_video", action="store_true", default=True,
                    help="drop groups whose video file is absent (default on)")
    args = ap.parse_args()

    bench, tasks = ARMS.get(args.arm, (args.bench, None))
    if bench is None:
        sys.exit(f"unknown arm {args.arm!r}; pass --bench explicitly")
    groups = _load(bench, tasks)

    if args.require_video:
        have = [g for g in groups if os.path.exists(g["video_path"])]
        if len(have) != len(groups):
            print(f"[shards] {len(groups) - len(have)} groups dropped: video file absent")
        groups = have
    if not groups:
        sys.exit("[shards] nothing to shard")

    if args.arm in ("sb_po_calib", "sb_po_eval"):
        import random
        rnd = random.Random(args.seed)
        vids = sorted(g["video_id"] for g in groups)
        rnd.shuffle(vids)
        k = int(round(args.po_calib_frac * len(vids)))
        # CALIBRATION FLOOR. PO is fitted on calib and reported on eval (section 5.3),
        # so an undersized calib split silently produces a threshold fitted to noise
        # and the eval number inherits it. Never calibrate on less than
        # --min_calib_frac of the arm, and never leave the eval side empty.
        k_min = int(math.ceil(args.min_calib_frac * len(vids)))
        if k < k_min:
            print(f"[shards] CALIB FLOOR: raising calib {k} -> {k_min} videos "
                  f"({args.min_calib_frac:.1%} of {len(vids)})")
            k = k_min
        k = max(1, min(k, len(vids) - 1))
        calib = set(vids[:k])
        keep = calib if args.arm == "sb_po_calib" else (set(vids) - calib)
        groups = [g for g in groups if g["video_id"] in keep]
        print(f"[shards] PO split seed={args.seed}: calib={k}/{len(vids)} videos "
              f"({k / len(vids):.1%}, floor {args.min_calib_frac:.1%}); "
              f"arm={args.arm} keeps {len(groups)}")

    os.makedirs(args.out_dir, exist_ok=True)
    shards, load = balance(groups, args.gpus)
    for i, sh in enumerate(shards):
        path = os.path.join(args.out_dir, f"{args.arm}_g{i}.json")
        with open(path, "w") as fh:
            json.dump(sh, fh)
        print(f"[shards] {path}: {len(sh)} videos, "
              f"{sum(len(g['probes']) for g in sh)} probes, {load[i]/3600:.1f} h video")
    tot = sum(len(g["probes"]) for g in groups)
    print(f"[shards] arm={args.arm} total {len(groups)} videos / {tot} probes / "
          f"{sum(load)/3600:.1f} h video")


if __name__ == "__main__":
    main()
