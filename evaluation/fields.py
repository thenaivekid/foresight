"""
fields.py — what did the controller actually emit, per task?

No ground truth, no judge, no GPU. Reads saved run logs and answers one question:
for each task, which fields of the control JSON does the model FILL, how often, and
what did filling them cost in decode steps.

This is the parse + occupancy stage. It does not ablate anything and it does not
claim a field is useless -- it reports how often each field carries information.
A field the model almost never fills is dead by observation; a field it fills every
tick with the same value is dead by constancy. Both are found here, before a single
GPU-hour is spent on an ablation.

Two log lines per tick, emitted by controller.py:

  [  30.1s | vid   1.0s] ctrl.raw  [VID] {"seen":"...","have_enough_info":false}
  [  30.1s | vid   1.0s] ctrl.gate [VID] fps=1.0 level=False ... gen=1.4s ntok=5 ...

`ctrl.raw` WAS truncated at 240 chars by the logger. Measured on a full development run: 100%
of unparsed rows were exactly at the cap -- every one is truncation, none is a model
failure. That matters because truncation is NOT random: it selects the LONGEST
emissions, i.e. exactly the ticks that used the `more` tail. Dropping them would
bias tail-field occupancy toward zero, on `event_narration` (57% truncated) most of
all.

The cap is GONE as of the forced-schema work (controller.py, the `NO CAP` comment
above the `ctrl.raw` log call) and it must stay gone: the forced spine emits
strictly MORE keys per tick, so every emission is longer than the ones that were
being cut, and a re-introduced cap would bias EVERY new field low -- exactly the
fields whose whole point is that they now fill 100% instead of 2-5%. The salvage
path below is kept for the archived logs (produced before the cap was removed), and `foresight/test_plan_schema.py` asserts the cap has not come back.

So truncated rows are SALVAGED, not dropped: the schema walk emits keys in a fixed
order, so every key appearing before the cut is still observed. Such rows count
toward `present` for the keys they show and are excluded only from `informative`
(the value may be cut). `raw_state="truncated"` marks them, and the report prints
the truncation rate per task so no occupancy number is read without it.

Task comes from the sample header the runner prints before each video:

  [online] ===== [1/63] realtime_state_monitor::CR55TVLjTzc::266 gt_times=[...] ...

Usage:
    python fields.py RUN_DIR                      # occupancy table per task
    python fields.py RUN_DIR      --jsonl ticks.jsonl   # dump the tidy table
    python fields.py RUN_DIR      --task dedup_counting --values   # value histograms
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# line grammars
# ---------------------------------------------------------------------------
_HEADER = re.compile(
    r"===== \[\s*\d+\s*/\s*\d+\s*\]\s+([^:]+)::([^:]+)::(\S+)")

_RAW = re.compile(
    r"\|\s*vid\s*([\d.]+)s\]\s+ctrl\.raw\s+\[([^\]]*)\]\s*(.*?)\s*$")

# The `ctrl.gate` tail is parsed as `key=value`, NOT positionally. The head through
# `q=` is fixed and stays a single anchored match; everything after it is optional and
# its membership grows with the spine (`p_hit`/`p_more` are absent on non-schema runs,
# `agree`/`argmax` appear only under verify_logit_read, and the forced-schema work adds
# one confidence per restricted read: p_fps / p_cadence / p_compact / p_replan). A
# positional regex with one optional group per key had to be edited, in order, for every
# new key -- and a key logged in an unexpected position silently vanished. So the tail is
# scanned instead: add a name to _GATE_TAIL and it is parsed, wherever it sits.
#
# q='...' is a repr, so it is matched as a QUOTED span first: a pending question that
# itself contains `count=` or `p_hit=` would otherwise be cut at that point and half of
# it would be parsed as tail keys. The unquoted `.*?` alternative is the fallback for
# older logs and is still anchored on a lookahead for the next KNOWN tail key (or end of
# line) rather than on one hard-coded successor. Same for argmax=, whose value is a repr
# and may itself contain '='.
_GATE_TAIL = {
    # continuous confidences from the restricted reads. p_hit/p_more are today's;
    # the rest arrive with the forced spine (_read_choice / _read_bool per field).
    "p_hit": "f", "p_more": "f",
    "p_fps": "f", "p_cadence": "f", "p_compact": "f", "p_replan": "f",
    # logit-read verification (cfg.verify_logit_read)
    "agree": "b", "argmax": "s",
    "ev0": "b", "dup_ev": "b",
    # Disabled task-conditional fields. Dropped from the line with the
    # fields; kept here so re-enabling them needs no parser change.
    "count": "i", "notes": "i",
    # class gating: the ternary verdicts are logged here in
    # key=value form so the continuous scores survive without a re-run.
    "n_keep": "i", "n_drop": "i", "n_neutral": "i", "retained": "f",
}

_GATE_HEAD = re.compile(
    r"\|\s*vid\s*([\d.]+)s\]\s+ctrl\.gate\s+\[([^\]]*)\]\s+"
    r"fps=([\d.]+)\s+level=(\w+)\s+rise=(\w+)\s+new_occ=(\w+)\s+fire=(\w+)\s+"
    r"next=([\d.]+)s\s+gen=([\d.]+)s\s+ntok=(\d+)\s+q="
    r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|.*?)"
    r"(?=\s+(?:" + "|".join(sorted(_GATE_TAIL)) + r")=|\s*$)")

# one `key=value` in the tail; the value runs to the next known key or end of line.
_GATE_KV = re.compile(
    r"\b(" + "|".join(sorted(_GATE_TAIL)) + r")=(.*?)"
    r"(?=\s+(?:" + "|".join(sorted(_GATE_TAIL)) + r")=|\s*$)")

# a JSON key at the start of a member: `{"k":` or `,"k":`. Used to salvage keys out
# of a truncated line; deliberately anchored on the punctuation so a `"` inside a
# `seen` or `answer` VALUE cannot be mistaken for a key.
_KEY = re.compile(r'[{,]\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:')

# Every key the schema walk can emit, IN WALK ORDER (controller.py:_schema_tick,
# the forced-schema spine). The order is not cosmetic: the walk
# emits keys in exactly this sequence, which is what makes the truncation salvage above
# sound (every key before a cut is still observed) and what makes an out-of-order key a
# signal that the walk did not run as specified.
#
#   seen                 sampled, cfg.seen_mode
#   have_enough_info     bool read
#   event_time_s         sampled, hit only
#   answer               sampled, hit only, DROPPED on time_only tasks
#   fps                  choice read {1,2,3}
#   next_check_s         choice read {0.5,1.0,1.5}
#   compact_now          bool read
#   replan_vision        bool read
#   objects              sampled, replan ticks only
#   keep / ignore        derived from the ternary verdicts, replan ticks only
#   question_for_next    sampled, every tick
#   note / count / phase disabled; behind `more` when enabled
FIELDS = ["seen", "have_enough_info", "event_time_s", "answer",
          "fps", "next_check_s", "compact_now", "replan_vision",
          "objects", "keep", "ignore", "question_for_next",
          "note", "count", "phase"]

# The same field has been spelled two ways across log versions:
# the spine calls the class lists `keep`/`ignore` while older
# emitted JSON writes `keep_classes`/`ignore_classes`, and
# `question_for_next` was logged as `question` in the free-decode tail. Normalise on
# read so one occupancy column means one field regardless of which run wrote the log.
ALIASES = {"question": "question_for_next",
           "keep_classes": "keep", "ignore_classes": "ignore"}


def canon(key):
    return ALIASES.get(key, key)


# These cost sampled tokens; the rest are forced keys or restricted reads (~free).
# NOTE the change: `fps` and `next_check_s` used to be free decodes in the `more`
# tail and are now `_read_choice` reads over a bounded surface-form set, so they cost
# zero decode steps. Archived logs were produced under the old regime;
# the DECODE COST table below reads `ntok` off ctrl.gate, so it stays correct either
# way -- this set is only used to say which fields could have contributed to it.
DECODED = {"seen", "event_time_s", "answer", "objects", "keep", "ignore",
           "question_for_next", "note", "count", "phase"}


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def parse_log(path):
    """Yield one dict per controller tick in one run log.

    ctrl.raw and ctrl.gate are emitted back-to-back for the same tick, raw first.
    They are joined on (video_id, vt) with the pending raw held for exactly one
    gate line, so an interleaved log from another thread cannot mis-pair them.
    """
    task = video = sample = None
    pending = None          # (vid, vt, raw_text) awaiting its gate line
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    with open(path, errors="replace") as fh:
        for line in fh:
            h = _HEADER.search(line)
            if h:
                task, video, sample = h.group(1), h.group(2), h.group(3)
                pending = None
                continue
            r = _RAW.search(line)
            if r:
                pending = (r.group(2), _f(r.group(1)), r.group(3))
                continue
            g = _GATE_HEAD.search(line)
            if not g:
                continue
            tail = {k: v.strip() for k, v in _GATE_KV.findall(line[g.end():])}
            vid, vt = g.group(2), _f(g.group(1))
            raw = pending[2] if (pending and pending[0] == vid
                                 and pending[1] == vt) else None
            pending = None
            row = {
                "run": run, "log": os.path.basename(path),
                "task": task, "video": vid, "sample": sample, "vt": vt,
                "fps": _f(g.group(3)), "level": g.group(4) == "True",
                "rise": g.group(5) == "True", "new_occ": g.group(6) == "True",
                "fire": g.group(7) == "True", "next_s": _f(g.group(8)),
                "gen_s": _f(g.group(9)), "ntok": int(g.group(10)),
                "q": g.group(11).strip("'\""),
                "raw": raw,
            }
            # the tail carries one key per restricted read plus the parked
            # counters; absent keys stay None so "never logged" and "logged as 0"
            # remain distinguishable in the tidy table.
            for k, kind in _GATE_TAIL.items():
                v = tail.get(k)
                if v is None:
                    row[k] = None
                elif kind == "f":
                    row[k] = _f(v)
                elif kind == "i":
                    row[k] = _i(v)
                elif kind == "b":
                    row[k] = v == "True"
                else:
                    row[k] = v.strip("'\"")
            # the emitted JSON itself. NO-DIFF means the model produced nothing
            # parseable; a truncated tail means a logger cap cut it (the 240-char
            # cap is gone -- see the module docstring -- so on any new run this
            # branch firing is itself a finding).
            row["diff"] = None
            row["keys"] = []
            row["raw_state"] = "missing"
            if raw is not None:
                if raw.startswith("NO-DIFF"):
                    row["raw_state"] = "nodiff"
                else:
                    try:
                        d = json.loads(raw)
                        if isinstance(d, dict):
                            row["diff"] = d
                            row["keys"] = list(d)
                            row["raw_state"] = "ok"
                        else:
                            row["raw_state"] = "unparsed"
                    except json.JSONDecodeError:
                        # salvage: keys before the cut are still observed, values
                        # are not trustworthy. See the module docstring.
                        row["keys"] = _KEY.findall(raw)
                        row["raw_state"] = ("truncated" if row["keys"]
                                            else "unparsed")
            yield row


def find_logs(roots):
    out = []
    for r in roots:
        if os.path.isfile(r):
            out.append(r)
        else:
            out += sorted(glob.glob(os.path.join(r, "**", "run_*.log"),
                                    recursive=True))
    return out


# ---------------------------------------------------------------------------
# step 2 -- occupancy
# ---------------------------------------------------------------------------
def occupancy(rows):
    """Per task: how often is each field present, and is its value ever different?

    Two denominators, kept apart on purpose:

    `present[k] / n_keyed` -- how often the model emitted the key at all. Counted
        over ok AND truncated rows, since a key before the cut is still observed.
    `informative[k] / n_ok` -- how often it emitted a value that is not the schema
        default ("" for text, False for the boolean, 0 for count). A key the model
        fills with an empty string is a key it declined to use and must not be
        scored as used. Counted over ok rows only: a truncated value is unknown.
    """
    per = defaultdict(lambda: {
        "ticks": 0, "ok": 0, "truncated": 0, "nodiff": 0, "unparsed": 0,
        "missing": 0, "fires": 0, "hits": 0, "videos": set(),
        "present": Counter(), "informative": Counter(),
        "distinct": defaultdict(set), "ntok": [], "gen": [],
        "ntok_hit": [], "ntok_quiet": [],
    })
    for row in rows:
        t = per[row["task"] or "?"]
        t["ticks"] += 1
        t["videos"].add(row["video"])
        t[row["raw_state"]] += 1
        t["fires"] += bool(row["fire"])
        t["hits"] += bool(row["level"])
        t["ntok"].append(row["ntok"])
        (t["ntok_hit"] if row["level"] else t["ntok_quiet"]).append(row["ntok"])
        if row["gen_s"] is not None:
            t["gen"].append(row["gen_s"])
        for k in row.get("keys") or ():
            t["present"][canon(k)] += 1
        d = row["diff"]
        if not d:
            continue
        for k, v in d.items():
            key = canon(k)
            if isinstance(v, str):
                if v.strip():
                    t["informative"][key] += 1
                    t["distinct"][key].add(v.strip()[:80])
            elif isinstance(v, bool):
                if v:
                    t["informative"][key] += 1
                t["distinct"][key].add(v)
            elif isinstance(v, (list, tuple)):
                # keep / ignore are lists of class names. A non-empty list is a
                # verdict the model actually expressed; an empty one is the
                # schema default and must not be scored as used. Stored as a
                # sorted joined string so it is hashable and so two orderings of
                # the same class set count as ONE distinct value -- otherwise the
                # distinct-value column measures the sampler, not the model.
                if v:
                    t["informative"][key] += 1
                    t["distinct"][key].add(",".join(sorted(map(str, v)))[:80])
            elif v is not None:
                if v != 0:
                    t["informative"][key] += 1
                t["distinct"][key].add(
                    v if isinstance(v, (int, float, str)) else str(v)[:80])
    return per


def _pct(a, b):
    return "-" if not b else f"{100.0 * a / b:5.1f}%"


def _med(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[len(s) // 2]


def report(per, show_values=False, top=8):
    order = sorted(per, key=lambda t: -per[t]["ticks"])
    print(f"\n{'task':<28} {'ticks':>8} {'vids':>5} {'raw ok':>7} {'trunc':>7} "
          f"{'level':>7} {'fire':>7} {'med ntok':>9} {'med gen':>8}")
    print("-" * 96)
    for t in order:
        s = per[t]
        print(f"{t:<28} {s['ticks']:>8} {len(s['videos']):>5} "
              f"{_pct(s['ok'], s['ticks']):>7} "
              f"{_pct(s['truncated'], s['ticks']):>7} "
              f"{_pct(s['hits'], s['ticks']):>7} "
              f"{_pct(s['fires'], s['ticks']):>7} "
              f"{_med(s['ntok']):>9.0f} {_med(s['gen']):>7.2f}s")

    print("\n\nFIELD OCCUPANCY"
          "\n  present      = % of ticks emitting the key   (ok + truncated rows)"
          "\n  informative  = % emitting a NON-DEFAULT value (ok rows only --"
          "\n                 a truncated value is unknown, so on a task with a high"
          "\n                 trunc rate this row is a LOWER BOUND, not a measurement)\n")
    hdr = f"{'task':<28}" + "".join(f"{f[:11]:>13}" for f in FIELDS)
    for t in order:
        s = per[t]
        n_ok, n_keyed = s["ok"], s["ok"] + s["truncated"]
        if not n_ok:
            continue
        print("-" * len(hdr))
        print(hdr)
        row_p = f"{t[:27]:<28}"
        row_i = f"{'  (informative)':<28}"
        row_d = f"{'  (distinct vals)':<28}"
        for f in FIELDS:
            row_p += f"{_pct(s['present'][f], n_keyed):>13}"
            row_i += f"{_pct(s['informative'][f], n_ok):>13}"
            nd = len(s["distinct"][f])
            row_d += f"{(str(nd) if nd else '-'):>13}"
        print(row_p)
        print(row_i)
        print(row_d)

    print("\n\nDECODE COST  --  ntok is sampled tokens; forced keys and logit reads"
          "\n                 cost zero. Quiet vs hit is the schema's two paths.\n")
    print(f"{'task':<28} {'quiet ticks':>12} {'med ntok':>9} "
          f"{'hit ticks':>10} {'med ntok':>9} {'med gen s':>10}")
    print("-" * 88)
    for t in order:
        s = per[t]
        print(f"{t:<28} {len(s['ntok_quiet']):>12} {_med(s['ntok_quiet']):>9.0f} "
              f"{len(s['ntok_hit']):>10} {_med(s['ntok_hit']):>9.0f} "
              f"{_med(s['gen']):>10.2f}")

    if show_values:
        print("\n\nVALUE SAMPLES\n")
        for t in order:
            s = per[t]
            print(f"--- {t} ---")
            for f in FIELDS:
                vals = s["distinct"][f]
                if not vals:
                    continue
                sample = sorted(str(v) for v in vals)[:top]
                print(f"  {f:<20} {len(vals):>6} distinct | "
                      + " | ".join(x[:40] for x in sample))
            print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="output dirs or run_*.log files")
    ap.add_argument("--jsonl", help="dump the tidy per-tick table here")
    ap.add_argument("--task", action="append", help="restrict to task(s)")
    ap.add_argument("--values", action="store_true",
                    help="print distinct-value samples per field")
    a = ap.parse_args(argv)

    logs = find_logs(a.roots)
    if not logs:
        sys.exit(f"no run_*.log under {a.roots}")
    print(f"[fields] {len(logs)} log files", file=sys.stderr)

    keep = set(a.task) if a.task else None
    rows = []
    out = open(a.jsonl, "w") if a.jsonl else None
    n = 0
    try:
        for p in logs:
            for row in parse_log(p):
                if keep and row["task"] not in keep:
                    continue
                n += 1
                if out:
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                rows.append(row)
    finally:
        if out:
            out.close()
    print(f"[fields] {n} ticks parsed"
          + (f" -> {a.jsonl}" if a.jsonl else ""), file=sys.stderr)
    if not rows:
        sys.exit("no ticks parsed -- check the log format")
    report(occupancy(rows), show_values=a.values)


if __name__ == "__main__":
    main()
