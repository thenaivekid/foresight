"""
make_subset.py — build the bundled `benchmark_mini.json` for the FAST OmniPro eval.

Selects the K SHORTEST videos per task (default from the audio-independent
subset) out of the full OmniPro benchmark.json, verifies each video exists on
disk, and writes a self-contained subset file next to this script. The subset
is what `evaluate.py` scores by default (see utils.BENCHMARK_JSON), so the whole
directory is portable: copy it to another repo, point OMNIPRO_DATASET_DIR at the
videos, and run — no need to ship the full 2,700-sample benchmark.

The written entries keep the ORIGINAL `video_path` (relative, e.g.
"raw_videos/<id>.mp4"), so they resolve against whatever OMNIPRO_DATASET_DIR /
--dataset_dir you use at run time.

Usage:
    python make_subset.py                       # 3 shortest / task, audio=none
    python make_subset.py --per_task 2 --audio none_helpful
    python make_subset.py --tasks instant_event_alert,semantic_condition_alert
    python make_subset.py --full /path/to/benchmark.json --dataset_dir /path/to/dataset
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("SCRATCH", "/path/to/work")

ALL_TASKS = [
    "instant_event_alert", "semantic_condition_alert", "explicit_target_grounding",
    "snapshot_counting", "cumulative_counting", "dedup_counting",
    "realtime_state_monitor", "event_narration", "sequential_step_instruction",
]
AUDIO_SETS = {
    "none": {"none"},
    "helpful": {"helpful"},
    "none_helpful": {"none", "helpful"},
    "all": {"none", "helpful", "required"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=os.environ.get(
        "OMNIPRO_FULL_JSON", os.path.join(SCRATCH, "omni_pro", "dataset", "benchmark.json")),
        help="path to the full OmniPro benchmark.json")
    ap.add_argument("--dataset_dir", default=os.environ.get(
        "OMNIPRO_DATASET_DIR", os.path.join(SCRATCH, "omni_pro", "dataset")),
        help="root the video_path entries are joined onto (for the on-disk check)")
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_mini.json"))
    ap.add_argument("--per_task", type=int, default=3, help="shortest videos to keep per task")
    ap.add_argument("--audio", default="none", choices=list(AUDIO_SETS),
                    help="audio_dependency subset (the system is vision-only; 'none' is safest)")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--require_on_disk", action="store_true", default=True,
                    help="skip samples whose video file is missing")
    ap.add_argument("--no_require_on_disk", dest="require_on_disk", action="store_false")
    args = ap.parse_args()

    keep_audio = AUDIO_SETS[args.audio]
    keep_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    with open(args.full) as f:
        raw = json.load(f)

    by_task: dict[str, list[dict]] = {t: [] for t in keep_tasks}
    for e in raw:
        if e["task"] not in by_task:
            continue
        if e.get("audio_dependency") not in keep_audio:
            continue
        if args.require_on_disk:
            vp = os.path.join(args.dataset_dir, e["video_path"])
            if not os.path.exists(vp):
                continue
        by_task[e["task"]].append(e)

    picked: list[dict] = []
    print(f"{'task':<30}{'kept':>5}  shortest durations (s)")
    print("-" * 70)
    for t in keep_tasks:
        items = sorted(by_task[t], key=lambda e: float(e.get("duration", 0.0)))[:args.per_task]
        picked.extend(items)
        durs = ", ".join(f"{float(i.get('duration',0)):.1f}" for i in items)
        print(f"{t:<30}{len(items):>5}  {durs or '(none available)'}")

    with open(args.out, "w") as f:
        json.dump(picked, f, indent=2)

    total_dur = sum(float(e.get("duration", 0.0)) for e in picked)
    print("-" * 70)
    print(f"wrote {len(picked)} samples ({len(keep_tasks)} tasks x <= {args.per_task}) "
          f"-> {args.out}")
    print(f"total video seconds to process: {total_dur:.0f}s "
          f"(~{total_dur/60:.1f} min of footage at 1 fps)")


if __name__ == "__main__":
    main()
