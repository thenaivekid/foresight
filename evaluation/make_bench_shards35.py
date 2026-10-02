#!/usr/bin/env python3
"""Build a 35% stratified subsample of an arm and re-shard it across N GPUs.

Two things this fixes about just taking a prefix of the existing shards:

1.  STRATIFICATION.  The existing manifests are greedy-packed longest-first, so
    any prefix is a long-video sample: the 182 videos finished on the sb_mcq run
    sat at the 86th duration percentile (815.8 s mean against a 583.5 s set
    mean), and one task (Scene Understanding) had zero coverage.  Here the 35%
    is drawn INSIDE each (dominant task x duration quartile) cell, so task mix
    and length mix both match the full benchmark.

2.  BALANCE.  Shards are filled longest-processing-time-first onto whichever
    shard is currently lightest, which bounds the spread in total streamed
    seconds -- the thing that actually sets each GPU's wall time.  An uneven
    split leaves GPUs idle while one straggler finishes.

Cost model: a sample streams to its LAST probe time (bench_probe.build_cfg sets
max_seconds=last_t+1), so streamed seconds -- not raw file duration -- is what a
shard pays for.  Videos are emitted longest-first within a shard so the tail the
deadline guard may defer is made of the cheapest samples.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SPL = os.path.join(HERE, "splits_bench")
DUR = os.path.join(HERE, "video_durations.json")


_DURS = json.load(open(DUR)) if os.path.exists(DUR) else {}


def streamed_seconds(g):
    """Seconds actually decoded: the pass runs to the last probe time, but
    iter_frames also stops at end-of-file. StreamingBench/OVO carry a handful of
    probe timestamps PAST the end of their video (worst: 18,661 s of probe on a
    661 s file), so taking the probe time alone overstates the cost by up to 28x
    -- which mis-balanced the shards and made the deadline guard defer videos
    that actually finish in minutes."""
    ts = [float(p["t"]) for p in g.get("probes", []) if "t" in p]
    want = (max(ts) + 1.0) if ts else 0.0
    fd = _DURS.get(g.get("video_path"))
    return min(want, fd) if fd else want


def dominant_task(g):
    tasks = [p.get("task") for p in g.get("probes", []) if p.get("task")]
    if tasks:
        return collections.Counter(tasks).most_common(1)[0][0]
    return g.get("task") or (g.get("tasks") or ["?"])[0]


def load_arm(arm):
    files = sorted(glob.glob(os.path.join(SPL, f"{arm}_g*.json")))
    if not files:
        raise SystemExit(f"no manifests matching {arm}_g*.json in {SPL}")
    seen, out = set(), []
    for f in files:
        for g in json.load(open(f)):
            if g["video_id"] in seen:          # shards are disjoint, but be safe
                continue
            seen.add(g["video_id"])
            out.append(g)
    return out


def quartile_edges(vals, k):
    s = sorted(vals)
    return [s[int(len(s) * i / k)] for i in range(1, k)]


def stratum_of(g, edges, durs):
    d = durs.get(g["video_path"]) or streamed_seconds(g)
    q = sum(1 for e in edges if d > e)
    return (dominant_task(g), q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--bins", type=int, default=4, help="duration strata per task")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--suffix", default="35")
    ap.add_argument("--out_dir", default=SPL)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    durs = json.load(open(DUR)) if os.path.exists(DUR) else {}
    groups = load_arm(args.arm)
    dvals = [durs.get(g["video_path"]) or streamed_seconds(g) for g in groups]
    edges = quartile_edges(dvals, args.bins)

    cells = collections.defaultdict(list)
    for g in groups:
        cells[stratum_of(g, edges, durs)].append(g)

    rnd = random.Random(args.seed)
    picked = []
    for key in sorted(cells, key=lambda k: (str(k[0]), k[1])):
        pool = sorted(cells[key], key=lambda g: g["video_id"])   # deterministic
        rnd.shuffle(pool)
        k = max(1, int(round(args.frac * len(pool)))) if pool else 0
        picked.extend(pool[:k])

    # ---- balance: longest-processing-time-first onto the lightest shard -------
    picked.sort(key=streamed_seconds, reverse=True)
    shards = [[] for _ in range(args.gpus)]
    load = [0.0] * args.gpus
    for g in picked:
        i = min(range(args.gpus), key=lambda j: load[j])
        shards[i].append(g)
        load[i] += streamed_seconds(g)

    tot_all = sum(streamed_seconds(g) for g in groups)
    tot_pick = sum(load)
    print(f"arm={args.arm}  videos {len(picked)}/{len(groups)} = "
          f"{100*len(picked)/len(groups):.1f}%   "
          f"streamed {tot_pick/3600:.1f}/{tot_all/3600:.1f} video-h = "
          f"{100*tot_pick/tot_all:.1f}%")
    print(f"  strata: {len(cells)} cells ({args.bins} duration bins x "
          f"{len(set(k[0] for k in cells))} tasks); duration edges "
          f"{[round(e) for e in edges]}s")
    print(f"  shard load (streamed h): min {min(load)/3600:.2f} "
          f"max {max(load)/3600:.2f} spread {100*(max(load)-min(load))/statistics.mean(load):.1f}%")

    # representativeness check against the full arm
    def mix(gs):
        c = collections.Counter(dominant_task(g) for g in gs)
        n = sum(c.values())
        return {k: v / n for k, v in c.items()}
    mf, mp = mix(groups), mix(picked)
    worst = max(((k, abs(mp.get(k, 0) - v)) for k, v in mf.items()), key=lambda z: z[1])
    print(f"  task-mix max deviation from full arm: {worst[0]} {100*worst[1]:.2f} pp")
    dp = [durs.get(g["video_path"]) or streamed_seconds(g) for g in picked]
    print(f"  duration: picked median {statistics.median(dp):.0f}s vs "
          f"full {statistics.median(dvals):.0f}s")

    if args.dry_run:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    for i, sh in enumerate(shards):
        sh.sort(key=streamed_seconds, reverse=True)   # cheap tail for the guard
        p = os.path.join(args.out_dir, f"{args.arm}{args.suffix}_g{i}.json")
        json.dump(sh, open(p, "w"))
        print(f"    wrote {os.path.basename(p)}  videos={len(sh)}  "
              f"streamed={load[i]/3600:.2f} h")


if __name__ == "__main__":
    main()
