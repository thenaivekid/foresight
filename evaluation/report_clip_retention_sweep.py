#!/usr/bin/env python3
"""Summarize the matched CLIP-QDP retention sweep against abl_base."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
ARMS = [("baseline", "abl_base"), ("clip_50", "abl_clip"),
        ("clip_70", "abl_clip_sweep/f0.70"),
        ("clip_80", "abl_clip_sweep/f0.80"),
        ("clip_90", "abl_clip_sweep/f0.90")]


def keep_rate(directory):
    values = []
    for name in os.listdir(os.path.join(ROOT, directory)):
        if not name.startswith("run_") or not name.endswith(".log"):
            continue
        with open(os.path.join(ROOT, directory, name), errors="replace") as handle:
            for line in handle:
                match = re.search(r"clip_keep_rate\s+n=\d+\s+mean=\s*([\d.]+)", line)
                if match:
                    values.append(float(match.group(1)))
    return sum(values) / len(values) if values else 1.0


print("arm       keep-rate  time-F1  content-acc  joint-F1  status")
for name, directory in ARMS:
    path = os.path.join(ROOT, directory, "online_metrics.json")
    if not os.path.exists(path):
        print(f"{name:<10} {'-':>8} {'-':>8} {'-':>12} {'-':>9}  RUNNING")
        continue
    with open(path) as handle:
        metrics = json.load(handle)["overall"]
    print(f"{name:<10} {keep_rate(directory):>8.3f} {metrics['time_f1']:>8.4f} "
          f"{metrics['content_acc']:>12.4f} {metrics['joint_f1']:>9.4f}  COMPLETE")
