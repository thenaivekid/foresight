"""
dump_scores.py — extract per-tick p_hit into a compact, durable artifact.

WHY: the firing threshold should never require a GPU re-run to change. Every tick
already logs p_hit (the continuous confidence from the logit read), but logs are
big, get rotated, and are awkward to ship. This distils them to one small JSONL
per run so any threshold or gate strategy can be re-derived offline, forever.

    {"task": "...", "video_id": "...", "vt": 12.0, "p_hit": 0.0213, "fire": false}

Feed the result to auc.py / gates.py to re-threshold at zero cost.

Usage:
    python dump_scores.py output_all/g00 [more dirs ...]
    python dump_scores.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_GATE = re.compile(
    r"\|\s*vid\s*([\d.]+)s\]\s+ctrl\.gate\s+\[([^\]]*)\].*?"
    r"fire=(\w+).*?p_hit=([\d.]+)(?:\s+p_more=([\d.]+))?")


def dump(run_dir, out_name="scores.jsonl"):
    paths = sorted(glob.glob(os.path.join(run_dir, "**", "run_*.log"),
                             recursive=True))
    if not paths:
        return 0
    task = os.path.basename(run_dir.rstrip("/")).split("__")[0]
    rows = []
    for p in paths:
        with open(p, errors="replace") as fh:
            for line in fh:
                m = _GATE.search(line)
                if not m:
                    continue
                rows.append({
                    "task": task,
                    "video_id": m.group(2),
                    "vt": float(m.group(1)),
                    "p_hit": float(m.group(4)),
                    "p_more": float(m.group(5)) if m.group(5) else None,
                    "fire": m.group(3) == "True",
                })
    if not rows:
        return 0
    out = os.path.join(run_dir, out_name)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dirs = args.dirs or (sorted(d for d in glob.glob(os.path.join(here, "output_*"))
                                if os.path.isdir(d)) if args.all else [])
    if not dirs:
        ap.error("give run dirs, or --all")
    total = 0
    for d in dirs:
        # a run dir may hold per-shard subdirs; handle both shapes
        subs = [s for s in sorted(glob.glob(os.path.join(d, "*")))
                if os.path.isdir(s) and glob.glob(os.path.join(s, "run_*.log"))]
        for target in (subs or [d]):
            n = dump(target)
            if n:
                print(f"  {target}: {n} ticks -> scores.jsonl")
                total += n
    print(f"total {total} ticks stored")


if __name__ == "__main__":
    main()
