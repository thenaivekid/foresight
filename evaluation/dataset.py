"""
dataset.py — OmniPro dataloader for the evaluation.

Loads benchmark.json, applies task / audio-dependency / per-task limit filters,
normalises ground truth, resolves absolute video paths, and provides a PyAV
frame iterator at a fixed fps (OmniPro Online mode runs at 1 fps).

IMPORTANT (vision-only caveat): The system ingests NO audio. OmniPro samples are
labelled `audio_dependency` ∈ {none, helpful, required}. `required` samples are
*unsolvable* without audio, so the honest default subset is {none, helpful},
and the headline subset is `none`. Nothing is dropped silently — the loader
reports exactly what it kept.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from utils import (BENCHMARK_JSON, DATASET_DIR, log, mmss_to_sec,
                   parse_ground_truth)

ALL_TASKS = [
    "instant_event_alert", "semantic_condition_alert", "explicit_target_grounding",
    "snapshot_counting", "cumulative_counting", "dedup_counting",
    "realtime_state_monitor", "event_narration", "sequential_step_instruction",
]

# Tasks the system's proactive text gate can meaningfully attempt out of the box.
# (alerts + narration are event/onset + free-text; the others need structured
# count/position/state outputs the writer is not built to emit.)
NATIVE_TASKS = [
    "instant_event_alert", "semantic_condition_alert", "event_narration",
]

# OmniPro online content scoring per task (mirrors metrics/online TASK_CONTENT_KIND).
TASK_CONTENT_KIND = {
    "instant_event_alert": "time_only",
    "semantic_condition_alert": "time_only",
    "explicit_target_grounding": "position",
    "snapshot_counting": "count",
    "cumulative_counting": "count_at_time",
    "dedup_counting": "count_at_time",
    "realtime_state_monitor": "state",
    "event_narration": "gpt_judge",
    "sequential_step_instruction": "gpt_judge",
}


@dataclass
class Sample:
    id: str
    task: str
    video_id: str
    video_path: str          # absolute
    duration: float
    question: str
    event: str
    audio_dependency: str
    ground_truth: list[dict]  # normalised, each has float t_sec

    @property
    def gt_times(self) -> list[float]:
        return [g["t_sec"] for g in self.ground_truth]


def load_samples(
    *,
    tasks: list[str] | None = None,
    audio: str = "none_helpful",      # none | helpful | required | none_helpful | all
    limit_per_task: int | None = None,
    max_duration: float | None = None,
    shortest: bool = False,           # pick the SHORTEST videos per task, not file order
    benchmark_json: str = BENCHMARK_JSON,
    dataset_dir: str = DATASET_DIR,
) -> list[Sample]:
    with open(benchmark_json) as f:
        raw = json.load(f)

    audio_sets = {
        "none": {"none"},
        "helpful": {"helpful"},
        "required": {"required"},
        "none_helpful": {"none", "helpful"},
        "all": {"none", "helpful", "required"},
    }
    keep_audio = audio_sets[audio]
    keep_tasks = set(tasks) if tasks else set(ALL_TASKS)

    # Pass 1: keep everything that passes the task / audio / duration / on-disk
    # filters. We defer the per-task limit so `shortest` can rank first.
    cand: list[Sample] = []
    skipped_audio = skipped_task = skipped_dur = skipped_missing = 0
    for e in raw:
        if e["task"] not in keep_tasks:
            skipped_task += 1
            continue
        if e.get("audio_dependency") not in keep_audio:
            skipped_audio += 1
            continue
        dur = float(e.get("duration", 0.0))
        if max_duration is not None and dur > max_duration:
            skipped_dur += 1
            continue
        vp = os.path.join(dataset_dir, e["video_path"])
        if not os.path.exists(vp):
            skipped_missing += 1
            continue
        cand.append(Sample(
            id=e["id"], task=e["task"], video_id=e["video_id"], video_path=vp,
            duration=dur, question=e.get("question", ""),
            event=e.get("event") or e.get("question", ""),
            audio_dependency=e.get("audio_dependency", "unknown"),
            ground_truth=parse_ground_truth(e["ground_truth"]),
        ))

    # If asked for the shortest, rank each task's candidates by duration so the
    # per-task limit keeps the quickest-to-run clips (fast, ~faithful subset).
    if shortest:
        cand.sort(key=lambda s: s.duration)

    # Pass 2: apply the per-task limit (over the shortest-first order if set).
    per_task_count: dict[str, int] = {}
    out: list[Sample] = []
    for s in cand:
        if limit_per_task is not None:
            c = per_task_count.get(s.task, 0)
            if c >= limit_per_task:
                continue
            per_task_count[s.task] = c + 1
        out.append(s)

    log(f"loaded {len(out)} samples | audio={audio} tasks={sorted(keep_tasks)} "
        f"limit/task={limit_per_task} max_dur={max_duration} shortest={shortest}")
    log(f"  skipped: task={skipped_task} audio={skipped_audio} "
        f"dur>{max_duration}={skipped_dur} missing_video={skipped_missing}")
    from collections import Counter
    log(f"  kept by task: {dict(Counter(s.task for s in out))}")
    if out:
        log(f"  duration range kept: {min(s.duration for s in out):.1f}s .. "
            f"{max(s.duration for s in out):.1f}s")
    return out


def iter_frames(video_path: str, *, fps: float = 1.0, max_seconds: float | None = None):
    """Yield (t_sec, PIL.Image) decoded from the video at ~`fps`.

    Uses PyAV (CPU decode) exactly like foresight's vision_stream.py, so frames
    match what the model would see in the real pipeline.
    """
    import av
    container = av.open(video_path)
    try:
        vstream = container.streams.video[0]
        # PyAV defaults to thread_count=0 = one decode thread per CPU. On a GH200
        # node that is 288 threads per shard x 4 shards on the same node, which
        # oversubscribes the host badly and, where thread creation is capped,
        # fails outright as `avcodec_open2: Resource temporarily unavailable`.
        # Decode is not the bottleneck (the model is), so bound it.
        try:
            vstream.thread_count = int(os.environ.get("OMNIPRO_DECODE_THREADS", "4"))
            vstream.thread_type = os.environ.get("OMNIPRO_DECODE_THREAD_TYPE", "AUTO")
        except Exception:
            pass
        tb = vstream.time_base
        interval = 1.0 / fps if fps > 0 else 0.0
        last_emit = -1e9
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            if max_seconds is not None and t > max_seconds:
                break
            if t - last_emit < interval:
                continue
            last_emit = t
            yield t, frame.to_image()
    finally:
        container.close()
