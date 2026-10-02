"""StreamingBench loader -> evaluation Sample + probe lists.

18 tasks / 900 videos / 4,500 QA. Counts and structure were read out of the
repo's four JSON files, not the paper prose (several published summaries of the
taxonomy disagree with the data; the data wins). section 2.

Audio classification, which drives `Sample.audio_dependency` and therefore the
--audio filter and the --group-by-audio fitter:
  required : the 4 Omni-Source tasks -- unsolvable without the audio track
  helpful  : MCU / ACU -- Contextual tasks, but they ship inside the omni video
             bundle, so the misleading/anomalous cue may be in the audio
  none     : the 10 Real-Time tasks, SQA, PO
"""
from __future__ import annotations

import json
import os
import re

from .prompts import SB_MCQ, SB_MCQ_TAIL, SB_PO_GATE, fmt_options

REAL_TASKS = ["Object Recognition", "Action Recognition", "Text-Rich Understanding",
              "Clips Summarize", "Attribute Recognition", "Spatial Understanding",
              "Counting", "Event Understanding", "Causal Reasoning",
              "Prospective Reasoning"]
OMNI_SOURCE = ["Emotion Recognition", "Scene Understanding",
               "Source Discrimination", "Multimodal Alignment"]
CONTEXTUAL = ["Misleading Context Understanding", "Anomaly Context Understanding",
              "Sequential Question Answering", "Proactive Output"]

AUDIO_DEP = {t: "required" for t in OMNI_SOURCE}
AUDIO_DEP.update({"Misleading Context Understanding": "helpful",
                  "Anomaly Context Understanding": "helpful"})

CATEGORY = {}
CATEGORY.update({t: "real_time" for t in REAL_TASKS})
CATEGORY.update({t: "omni_source" for t in OMNI_SOURCE})
CATEGORY.update({t: "contextual" for t in CONTEXTUAL})

FILES = {"real": "questions_real.json", "omni": "questions_omni.json",
         "sqa": "questions_sqa.json", "proactive": "questions_proactive.json"}


def _secs(t):
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(str(t).split(":"))))


def _flatten(path):
    """questions_sqa.json nests one level deeper than the other three."""
    out = []
    for e in json.load(open(path)):
        out.extend(e if isinstance(e, list) else [e])
    return out


def resolve_video(video_path, data_dir):
    """Map an annotation video_path onto the extracted zip layout.

    The annotations say "./videos/sample_17_proactive.mp4", but every zip roots
    at "sample_N/" and holds the clip as "sample_N/video.mp4". Because all the
    zips use the same sample_N naming, each is extracted into its own directory
    named by the SUFFIX the annotation uses -- so the suffix is exactly the
    lookup key. Scene Understanding carries its range in the suffix
    ("sample_3_Scene_Understanding_26-50.mp4"), which is why the split below
    takes everything after the first "sample_<N>_".
    """
    base = os.path.splitext(os.path.basename(video_path.lstrip("./")))[0]
    m = re.match(r"(sample_\d+)_(.+)$", base)
    if not m:
        return os.path.join(data_dir, video_path.lstrip("./"))
    sample, suffix = m.group(1), m.group(2)
    d = os.path.join(data_dir, "videos", suffix, sample)
    cand = os.path.join(d, "video.mp4")
    if os.path.exists(cand):
        return cand
    # A handful of samples name the clip after the task instead of "video.mp4"
    # (e.g. proactive/sample_45/"Active Output_3.mp4"). Fall back to the single
    # mp4 in the directory rather than reporting the sample as missing.
    if os.path.isdir(d):
        mp4s = sorted(f for f in os.listdir(d) if f.lower().endswith(".mp4"))
        if len(mp4s) == 1:
            return os.path.join(d, mp4s[0])
        if len(mp4s) > 1:
            raise RuntimeError(f"{d} holds {len(mp4s)} mp4s; cannot pick one: {mp4s}")
    return cand                      # keep the expected path for the error message


def load_groups(qdir, data_dir, *, tasks=None, arm=None):
    """Group every question by VIDEO, so one streaming pass answers all of them.

    Returns a list of dicts: {video_path, video_id, task(s), probes:[...]}.
    """
    want = set(tasks) if tasks else None
    groups = {}
    for key, fname in FILES.items():
        path = os.path.join(qdir, fname)
        if not os.path.exists(path):
            continue
        for sub in _flatten(path):
            vp = sub["video_path"]
            for q in sub.get("questions", []):
                task = q.get("task_type")
                if want and task not in want:
                    continue
                g = groups.setdefault(vp, {"video_path": vp, "probes": []})
                if task == "Proactive Output":
                    gt = _secs(q["ground_truth_time_stamp"])
                    g["probes"].append({
                        "task": task, "category": "contextual",
                        # PO polls at 1 Hz; the runner expands this into ticks
                        "kind": "po", "t": float(gt + 4),
                        "start_t": float(_secs(q["time_stamp"])),
                        "gt_t": float(gt),
                        "prompt": SB_PO_GATE.format(
                            question=q["question"], gt_output=q["ground_truth_output"]),
                        "gt_output": q["ground_truth_output"],
                        "question": q["question"],
                        "audio_required": False,
                    })
                else:
                    opts = q.get("options") or []
                    g["probes"].append({
                        "task": task, "category": CATEGORY.get(task, "unknown"),
                        "kind": "mcq", "t": float(_secs(q["time_stamp"])),
                        "prompt": SB_MCQ.format(question=q["question"],
                                                options=fmt_options(opts)) + SB_MCQ_TAIL,
                        "n_options": len(opts) or 4,
                        "answer": q.get("answer"),
                        "question": q["question"],
                        "audio_required": AUDIO_DEP.get(task) == "required",
                    })
    out = []
    for vp, g in sorted(groups.items()):
        g["video_path"] = resolve_video(vp, data_dir)
        g["video_id"] = os.path.splitext(os.path.basename(vp.lstrip("./")))[0]
        g["probes"].sort(key=lambda p: p["t"])
        tset = sorted({p["task"] for p in g["probes"]})
        g["task"] = tset[0] if len(tset) == 1 else "mixed"
        g["tasks"] = tset
        g["audio_dependency"] = ("required"
                                 if any(p["audio_required"] for p in g["probes"])
                                 else "none")
        out.append(g)
    return out


def to_sample(group):
    """Build the evaluation Sample the pipeline expects."""
    from dataset import Sample
    q0 = group["probes"][0]
    return Sample(
        id=group["video_id"], task=group["task"], video_id=group["video_id"],
        video_path=group["video_path"], duration=0.0,
        question=q0.get("question", ""), event=q0.get("question", ""),
        audio_dependency=group["audio_dependency"], ground_truth=[])
