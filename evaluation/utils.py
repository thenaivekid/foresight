"""
utils.py — shared helpers for the OmniPro evaluation harness.

Time parsing, ground-truth normalisation, logging, reproducibility, and a thin
wandb wrapper that reuses the ambient credentials already configured in
~/.netrc (entity taken from $WANDB_ENTITY).
No API key is read or stored here — wandb picks it up from netrc.
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
from typing import Any, Iterable

# ----------------------------------------------------------------------------
# paths (single source of truth)
# ----------------------------------------------------------------------------
SCRATCH = os.environ.get("SCRATCH", "/path/to/work")
_HERE = os.path.dirname(os.path.abspath(__file__))

# --- PORTABLE PATHS -------------------------------------------------------
# This is the *fast* subset duplicate: everything below is env-overridable so
# the same dir runs unchanged in another repo. Defaults are chosen so a bare
# `python evaluate.py` uses the bundled shortest-videos subset and writes into
# a local ./output next to this file (nothing is written to $SCRATCH/eval).
#
#   OMNIPRO_DATASET_DIR    where the video files live (video_path is joined onto it)
#   OMNIPRO_BENCHMARK_JSON the annotation file to score (defaults to the bundled subset)
#   OMNIPRO_OUTPUT_DIR     where predictions/metrics/report land
DATASET_DIR = os.environ.get(
    "OMNIPRO_DATASET_DIR", os.path.join(SCRATCH, "omni_pro", "dataset"))
BENCHMARK_JSON = os.environ.get(
    "OMNIPRO_BENCHMARK_JSON", os.path.join(_HERE, "benchmark_mini.json"))
OUTPUT_DIR = os.environ.get("OMNIPRO_OUTPUT_DIR", os.path.join(_HERE, "output"))
EVAL_ROOT = _HERE
# Location of the core `foresight` package. Override with $FORESIGHT_DIR;
# default = ../foresight next to this file.
_FORESIGHT = os.environ.get("FORESIGHT_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "foresight"))
FORESIGHT_DIR = _FORESIGHT

# wandb defaults — override via environment variables.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "foresight-eval")


# ----------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------
def log(msg: str, *, tag: str = "eval") -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] [{tag}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# time helpers
# ----------------------------------------------------------------------------
def mmss_to_sec(s: str | float | int) -> float:
    """'MM:SS' or 'HH:MM:SS' or a number -> seconds (float)."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if ":" not in s:
        return float(s)
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


# ----------------------------------------------------------------------------
# ground-truth normalisation
# ----------------------------------------------------------------------------
def parse_ground_truth(gt: Any) -> list[dict]:
    """benchmark.json stores ground_truth as a list[dict]; metadata.jsonl stores
    it JSON-stringified. Normalise either into list[dict] and add a float
    `t_sec` to every trigger."""
    if isinstance(gt, str):
        gt = json.loads(gt)
    out = []
    for g in gt:
        g = dict(g)
        if "trigger_time_sec" in g and g["trigger_time_sec"] is not None:
            g["t_sec"] = float(g["trigger_time_sec"])
        elif "trigger_time" in g:
            g["t_sec"] = mmss_to_sec(g["trigger_time"])
        else:
            g["t_sec"] = 0.0
        out.append(g)
    out.sort(key=lambda x: x["t_sec"])
    return out


def gt_times(gt: list[dict]) -> list[float]:
    return [g["t_sec"] for g in gt]


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ----------------------------------------------------------------------------
# wandb wrapper — never touches the API key; relies on ambient netrc login.
# ----------------------------------------------------------------------------
class WandbRun:
    """Best-effort wandb run. If wandb is unavailable or login fails, silently
    degrades to a no-op so the eval still produces local md/json results."""

    def __init__(self, enabled: bool, *, project: str, entity: str | None,
                 name: str, group: str, config: dict, tags: list[str] | None = None):
        self.run = None
        self._enabled = enabled
        if not enabled:
            return
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project, entity=entity, name=name, group=group,
                config=config, tags=tags or [], reinit=True,
                settings=wandb.Settings(init_timeout=30),
            )
            log(f"wandb run started: {self.run.url}", tag="wandb")
        except Exception as e:  # offline node, no creds, etc.
            log(f"wandb disabled ({type(e).__name__}: {e}); logging locally only",
                tag="wandb")
            self.run = None

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is not None:
            try:
                self.wandb.log(data, step=step)
            except Exception:
                pass

    def summary(self, data: dict) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(data)
            except Exception:
                pass

    def log_table(self, key: str, columns: list[str], rows: list[list]) -> None:
        if self.run is not None:
            try:
                tbl = self.wandb.Table(columns=columns, data=rows)
                self.wandb.log({key: tbl})
            except Exception:
                pass

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
