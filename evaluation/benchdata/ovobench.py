"""OVO-Bench loader -> evaluation Sample + probe lists.

644 videos, 1,640 records, 3,035 probe points, 12 tasks in three modes.
Structure read from data/ovo_bench_new.json; scoring semantics from
utils/OVOBenchScore.py. section 3.

THE POINT OF THIS FILE: the FAR tasks (SSR, CRR) carry a human `type` label
in {0,1} at every probe timestamp -- 869 labelled binary decision points for
"do you have enough information to answer yet". That is `have_enough_info` with
a real label, replacing the +-3 s temporal-proximity surrogate the OmniPro
threshold work had to use. REC carries an integer `count` per probe instead.
"""
from __future__ import annotations

import json
import os

from .prompts import OVO_CRR, OVO_MCQ, OVO_REC, OVO_SSR, fmt_options

BACKWARD = ["EPM", "ASI", "HLD"]
REALTIME = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD = ["REC", "SSR", "CRR"]
MODE = {}
MODE.update({t: "backward" for t in BACKWARD})
MODE.update({t: "realtime" for t in REALTIME})
MODE.update({t: "forward" for t in FORWARD})


def load_groups(ann_json, data_dir, *, tasks=None):
    """Group records by VIDEO so one pass serves every probe in it.

    A video can appear under several tasks (CRR reuses 10 MovieNet videos for 48
    questions), so grouping is what turns the official 3,035 runs into 644.
    """
    want = set(tasks) if tasks else None
    groups = {}
    for r in json.load(open(ann_json)):
        task = r["task"]
        if want and task not in want:
            continue
        vp = r["video"]
        g = groups.setdefault(vp, {"video_path": vp, "probes": []})
        mode = MODE.get(task, "unknown")

        if task in ("SSR", "CRR"):
            for i, ti in enumerate(r.get("test_info", [])):
                if task == "SSR":
                    prompt = OVO_SSR.format(step=ti["step"])
                else:
                    prompt = OVO_CRR.format(question=r["question"])
                g["probes"].append({
                    "task": task, "mode": mode, "kind": "yesno",
                    "t": float(ti["realtime"]), "probe_ix": i,
                    "prompt": prompt,
                    # the label: 1 => "Yes" is correct, 0 => "No" is correct
                    "type": int(ti["type"]),
                    "rec_id": r["id"], "audio_required": False,
                })
        elif task == "REC":
            for i, ti in enumerate(r.get("test_info", [])):
                g["probes"].append({
                    "task": task, "mode": mode, "kind": "count",
                    "t": float(ti["realtime"]), "probe_ix": i,
                    # verbatim from utils/OVOBench.py:133 --
                    #   question = "How many times did they " + activity + "?"
                    "prompt": OVO_REC.format(
                        question="How many times did they "
                                 + str(r.get("activity", "")) + "?"),
                    "count": int(ti["count"]),
                    "rec_id": r["id"], "audio_required": False,
                })
        else:
            opts = r.get("options") or []
            gt_ix = r.get("gt")
            g["probes"].append({
                "task": task, "mode": mode, "kind": "mcq",
                "t": float(r["realtime"]),
                "prompt": OVO_MCQ.format(question=r["question"],
                                         options=fmt_options(opts)),
                "n_options": len(opts) or 4,
                # OVOBenchScore does int(ground_truth in response) -- the letter
                "answer": "ABCD"[gt_ix] if isinstance(gt_ix, int) and gt_ix < 4 else None,
                "rec_id": r["id"], "audio_required": False,
            })
    out = []
    for vp, g in sorted(groups.items()):
        g["video_path"] = os.path.join(data_dir, vp)
        g["video_id"] = os.path.splitext(os.path.basename(vp))[0]
        g["probes"].sort(key=lambda p: p["t"])
        tset = sorted({p["task"] for p in g["probes"]})
        g["task"] = tset[0] if len(tset) == 1 else "mixed"
        g["tasks"] = tset
        g["audio_dependency"] = "none"
        out.append(g)
    return out


def to_sample(group):
    from dataset import Sample
    return Sample(
        id=group["video_id"], task=group["task"], video_id=group["video_id"],
        video_path=group["video_path"], duration=0.0,
        question="", event="", audio_dependency="none", ground_truth=[])
