#!/usr/bin/env python3
"""bench_eval.py -- run one benchmark arm over one shard. Called per GPU by


    python bench_eval.py --bench streamingbench --arm sb_mcq \
        --manifest splits_bench/sb_mcq_g0.json --data_dir $SB_DATA \
        --out $OUT/sb_mcq/g0 --resume

Appends ONE json line per completed VIDEO to <out>/pred.jsonl. The sbatch's
done-predicates count those lines, and --resume skips video_ids already present,
so a link killed at the wall costs at most the video in flight.

Three interaction modes section 5:
  mcq / yesno / count  -- reactive probe at benchmark timestamps (bench_probe.py)
  po                   -- the one trajectory-coupled arm, run here in the
                          benchmark's own polling loop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils import log                                    # noqa: E402


def done_ids(path):
    out = set()
    if os.path.exists(path):
        with open(path) as fh:
            for ln in fh:
                try:
                    out.add(json.loads(ln)["video_id"])
                except Exception:
                    continue
    return out


def append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_po_video(runner, sample, probes, protocol):
    """Proactive Output: the benchmark's own polling loop, with our gate in place
    of its yes/no string match -- but all of a video's questions merged into ONE
    streaming pass.

    B1 mirrors StreamingBenchProactive.py: poll at 1 Hz from start_time+1 to
    ground_truth+4, first crossing wins, then stop. Note the window is anchored
    on ground truth, so a model structurally cannot fire more than 4 s late;
    that is the benchmark's design, not ours.
    B2 drops that oracle window and polls to the end of the clip, which is what
    OmniPro actually measured. B2 is the honest number, B1 the
    leaderboard-comparable one; neither alone is reportable.

    WHY ONE PASS IS STILL FAITHFUL. The benchmark breaks at the first "yes", but
    a probe is splice-read-erase: it leaves no trace in the cache, so evaluating
    the ticks past the first crossing changes nothing about the earlier ones.
    We therefore evaluate every tick and apply "first crossing wins" offline.
    That costs a little extra compute and buys the COMPLETE p_hit trace per
    question, which is exactly what fit_bench_gates.py needs -- refitting a
    threshold would otherwise require re-running the video.
    """
    thr = float(os.environ.get("BENCH_PO_THRESHOLD", "0.5"))
    b2_horizon = float(os.environ.get("BENCH_PO_B2_HORIZON_S", "600"))

    merged, owners = [], []
    for qi, p in enumerate(probes):
        start, gt = float(p["start_t"]), float(p["gt_t"])
        last = gt + 4.0 if protocol == "b1" else start + b2_horizon
        t = start + 1.0
        while t <= last:
            merged.append({"t": t, "kind": "yesno", "prompt": p["prompt"],
                           "audio_required": False, "_q": qi})
            t += 1.0
    if not merged:
        return []
    res = runner.run(sample, merged, instruction=probes[0]["question"])

    for qi, p in enumerate(probes):
        ticks = [{"t": r["t"], "p_hit": r.get("p_hit")}
                 for r in sorted((x for x in res if x.get("_q") == qi),
                                 key=lambda x: x["t"])]
        answered = None
        for tk in ticks:                      # first crossing wins, as the loop does
            if tk["p_hit"] is not None and tk["p_hit"] >= thr:
                answered = tk["t"]
                break
        owners.append({**{k: v for k, v in p.items() if k != "prompt"},
                       "answered": answered, "threshold": thr,
                       "protocol": protocol, "n_ticks": len(ticks), "ticks": ticks})
    return owners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["streamingbench", "ovobench"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    # ---- deadline guard: never START a video that cannot FINISH ---------------
    # A video killed at the wall is redone from second zero by the next link, so
    # every second spent on it is thrown away. Measured on the sb_mcq arm: 92 of
    # 274 starts (33.6%) were abandoned, burning 31.8% of allocated GPU time.
    # Stopping the shard cleanly instead costs one idle tail and wastes nothing.
    ap.add_argument("--deadline_epoch", type=float, default=0.0,
                    help="unix time this job is killed; 0 disables the guard")
    ap.add_argument("--deadline_margin_s", type=float, default=180.0)
    ap.add_argument("--rate_s_per_video_s", type=float, default=1.2,
                    help="prior for wall-seconds per video-second; refined online")
    ap.add_argument("--max_attempts", type=int, default=2,
                    help="a video started this many times without finishing is "
                         "recorded as too long for one link, so the stage can "
                         "complete instead of the chain retrying it forever")
    args = ap.parse_args()
    if not args.deadline_epoch:
        _end = os.environ.get("SLURM_JOB_END_TIME")
        if _end:
            try: args.deadline_epoch = float(_end)
            except ValueError: pass

    os.makedirs(args.out, exist_ok=True)
    pred = os.path.join(args.out, "pred.jsonl")
    groups = json.load(open(args.manifest))
    if args.limit:
        groups = groups[:args.limit]

    skip = done_ids(pred) if args.resume else set()
    todo = [g for g in groups if g["video_id"] not in skip]
    log(f"arm={args.arm} shard={os.path.basename(args.manifest)} "
                      f"videos={len(groups)} done={len(skip)} todo={len(todo)}", tag="bench")
    if not todo:
        log("nothing to do", tag="bench")
        return

    # the pipeline + model load once for the whole shard
    from foresight_adapter import ForesightRunner
    import bench_probe
    if args.bench == "streamingbench":
        from benchdata import streamingbench as loader
    else:
        from benchdata import ovobench as loader

    adapter = ForesightRunner()   # all model config from foresight/config.py
    runner = bench_probe.ProbeRunner(adapter)
    protocol = os.environ.get("SB_PO_PROTOCOL", "b1")

    rate = args.rate_s_per_video_s      # wall-s per video-s, EMA over this shard
    n_rate = 0
    n_done_here = 0                     # videos completed in THIS link

    # ---- attempt ledger ------------------------------------------------------
    # With a 1 h wall a video costing more than one link can never finish: the
    # deadline guard lets it start (it is first in the link, deferring gains it
    # nothing), it dies at the wall, and the next link starts it again from zero
    # -- forever, and the stage never completes. Count attempts on disk, written
    # BEFORE the run so a SIGKILL at the wall still records it.
    att_path = os.path.join(args.out, "attempts.json")
    try:
        attempts = json.load(open(att_path))
    except Exception:
        attempts = {}

    def bump(vid):
        attempts[vid] = attempts.get(vid, 0) + 1
        tmp = att_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(attempts, fh)
        os.replace(tmp, att_path)
    stopped_early = 0

    _durs = {}
    for _p in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "video_durations.json"),):
        try:
            _durs = json.load(open(_p))
        except Exception:
            pass

    def streamed_seconds(g):
        """Seconds this sample actually decodes. The pass runs to the LAST probe
        time (bench_probe.build_cfg: max_seconds=last_t+1) BUT iter_frames also
        stops at end-of-file, and a few samples carry probe timestamps past the
        end of their video (worst: 18,661 s of probe on a 661 s file). Costing
        them by probe time alone would defer videos that finish in minutes."""
        ts = [float(p["t"]) for p in g.get("probes", []) if "t" in p]
        want = (max(ts) + 1.0) if ts else 0.0
        fd = _durs.get(g.get("video_path"))
        return min(want, fd) if fd else want

    for i, g in enumerate(todo, 1):
        vid = g["video_id"]
        if attempts.get(vid, 0) >= args.max_attempts:
            est = streamed_seconds(g) * rate
            log(f"{vid}: abandoned after {attempts[vid]} attempts "
                f"(~{est:.0f}s needed, longer than one link). Recorded as "
                f"incomplete so the stage can finish -- re-run it with a larger "
                f"WALL if you need it.", tag="bench")
            append(pred, {"video_id": vid, "video_path": g["video_path"],
                          "arm": args.arm, "bench": args.bench,
                          "tasks": g.get("tasks", []),
                          "error": f"exceeds_one_link after {attempts[vid]} attempts",
                          "probes": [], "wall_s": 0.0})
            continue
        if args.deadline_epoch:
            est = streamed_seconds(g) * rate * 1.15        # 15% headroom
            left = args.deadline_epoch - args.deadline_margin_s - time.time()
            # Deferring the FIRST video of a link gains nothing -- it will never
            # have more wall time than it does right now, so a sample longer than
            # one link would be deferred forever and the stage could never
            # complete. (sb_mcq35_g0 holds one 12,451 s sample.) Attempt it; if it
            # dies at the wall the next link retries it with a full link again.
            if est > left and n_done_here == 0:
                log(f"deadline guard: {g['video_id']} needs ~{est:.0f}s and only "
                    f"{left:.0f}s remain, but nothing has run yet this link -- "
                    f"attempting anyway rather than deferring forever.", tag="bench")
            elif est > left:
                stopped_early = len(todo) - i + 1
                log(f"deadline guard: stopping with {stopped_early} video(s) "
                    f"left; next needs ~{est:.0f}s, only {left:.0f}s remain "
                    f"(rate={rate:.2f} wall-s/video-s). They are untouched and "
                    f"the next link starts them clean.", tag="bench")
                break
        bump(vid)
        t0 = time.time()
        sample = loader.to_sample(g)
        try:
            if g["probes"] and g["probes"][0]["kind"] == "po":
                res = run_po_video(runner, sample, g["probes"], protocol)
            else:
                res = runner.run(sample, [dict(p) for p in g["probes"]])
                res = [{k: v for k, v in r.items() if k != "prompt"} for r in res]
            row = {"video_id": g["video_id"], "video_path": g["video_path"],
                   "arm": args.arm, "bench": args.bench, "tasks": g.get("tasks", []),
                   "audio_dependency": g.get("audio_dependency", "none"),
                   "probes": res, "wall_s": round(time.time() - t0, 1)}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # Record the failure as a row so the denominator stays honest and the
            # shard does not stall re-trying one broken video forever.
            row = {"video_id": g["video_id"], "video_path": g["video_path"],
                   "arm": args.arm, "bench": args.bench, "tasks": g.get("tasks", []),
                   "error": f"{type(exc).__name__}: {exc}", "probes": [],
                   "wall_s": round(time.time() - t0, 1)}
        append(pred, row)
        n_done_here += 1
        vs = streamed_seconds(g)
        if vs > 0 and "error" not in row:                  # refine the estimate
            obs = row["wall_s"] / vs
            n_rate += 1
            rate = obs if n_rate == 1 else (0.7 * rate + 0.3 * obs)
        log(f"[{i}/{len(todo)}] {g['video_id']} "
                          f"{row['wall_s']}s probes={len(row['probes'])}"
                          f"{' ERROR' if 'error' in row else ''}", tag="bench")

    log(f"shard finished: {len(todo) - stopped_early}/{len(todo)} attempted, "
        f"{stopped_early} deferred by the deadline guard, "
        f"final rate={rate:.2f} wall-s per video-s", tag="bench")


if __name__ == "__main__":
    main()
