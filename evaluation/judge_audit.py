"""
judge_audit.py — read the LLM judge's own words back, for any verdict it made.

WHY: `judge_cache.json` maps sha256(question|gt|pred)[:24] -> 0.0/1.0. That hash is
irreversible and the float is binarised, so the cache alone cannot tell you WHAT
was judged, what the judge said, or why -- you cannot even tell which prediction a
0.0 belongs to. Every content number in the paper rests on those verdicts, so they
have to be inspectable.

`ContentJudge.trace()` therefore appends the whole thing to judge_trace.jsonl: the
full triple in the clear, the raw 1-5 score, the judge's explanation, the model,
the seed, and whether it came from the sync or batch path. This script reads it.

The trace is append-only and is NEVER read by the scorer -- deleting it changes no
metric, it only costs you the ability to check. Re-judging the same triple appends
a second record rather than overwriting, so disagreements between runs stay
visible; `--dupes` finds them.

Usage:
    python judge_audit.py                      # summary + score histogram
    python judge_audit.py --score 1            # every triple the judge scored 1
    python judge_audit.py --verdict 0 --limit 5
    python judge_audit.py --grep "excavator"
    python judge_audit.py --borderline         # scores 2 and 3: where >=3 flips
    python judge_audit.py --dupes              # same triple judged twice
    python judge_audit.py --export audit.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.environ.get("OMNIPRO_JUDGE_TRACE_PATH",
                         os.path.join(HERE, "judge_trace.jsonl"))


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def show(r, width=100):
    sc = r.get("score_raw")
    v = r.get("verdict")
    mark = "CORRECT" if v == 1.0 else "WRONG"
    print(f"\n{'-' * width}")
    print(f"[{r.get('ts','?')}] {r.get('model','?')} score={sc} -> {mark}"
          f"   ({r.get('source','?')}, {r.get('backend','?')})")
    print(f"  Q    : {(r.get('question') or '')[:width]}")
    print(f"  GT   : {(r.get('gt_response') or '')[:width]}")
    print(f"  PRED : {(r.get('pred_text') or '')[:width]}")
    ex = r.get("explanation") or ""
    if ex:
        print(f"  WHY  : {ex[:width * 3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=DEFAULT)
    ap.add_argument("--score", type=int, help="show only this raw 1-5 score")
    ap.add_argument("--verdict", type=float, choices=[0.0, 1.0],
                    help="show only CORRECT (1) or WRONG (0)")
    ap.add_argument("--grep", help="substring match on question/gt/pred/explanation")
    ap.add_argument("--borderline", action="store_true",
                    help="scores 2 and 3 — where the >=3 threshold decides")
    ap.add_argument("--dupes", action="store_true",
                    help="triples judged more than once (and any disagreements)")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--export", help="write all matching rows to a CSV")
    args = ap.parse_args()

    rows = load(args.trace)
    if not rows:
        print(f"no trace at {args.trace}\n"
              "Nothing has been judged yet, or OMNIPRO_JUDGE_TRACE=0 was set.")
        return

    print(f"{args.trace}: {len(rows)} judgements")
    hist = collections.Counter(r.get("score_raw") for r in rows)
    models = collections.Counter(r.get("model") for r in rows)
    srcs = collections.Counter(r.get("source") for r in rows)
    n_ok = sum(1 for r in rows if r.get("verdict") == 1.0)
    print(f"  verdict     : {n_ok} correct / {len(rows) - n_ok} wrong "
          f"({n_ok / len(rows):.1%} content accuracy over judged)")
    print(f"  raw scores  : " + "  ".join(
        f"{k}:{hist[k]}" for k in sorted(hist, key=lambda x: (x is None, x))))
    print(f"  models      : {dict(models)}")
    print(f"  source      : {dict(srcs)}")

    # A verdict is a function of (triple, judge). Two records for one cache_key
    # mean the same triple was judged twice; different verdicts mean the judge is
    # not reproducible, which would undermine every content number.
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[r.get("cache_key")].append(r)
    rejudged = {k: v for k, v in by_key.items() if len(v) > 1}
    disagree = {k: v for k, v in rejudged.items()
                if len({x.get("verdict") for x in v}) > 1}
    print(f"  re-judged   : {len(rejudged)} triples")
    print(f"  DISAGREEING : {len(disagree)}"
          + ("  <-- judge is not reproducible; investigate" if disagree else ""))

    sel = rows
    if args.dupes:
        sel = [r for v in (disagree or rejudged).values() for r in v]
    if args.score is not None:
        sel = [r for r in sel if r.get("score_raw") == args.score]
    if args.verdict is not None:
        sel = [r for r in sel if r.get("verdict") == args.verdict]
    if args.borderline:
        sel = [r for r in sel if r.get("score_raw") in (2, 3)]
    if args.grep:
        g = args.grep.lower()
        sel = [r for r in sel if any(
            g in str(r.get(f, "")).lower()
            for f in ("question", "gt_response", "pred_text", "explanation"))]

    if args.export:
        cols = ["ts", "model", "source", "score_raw", "verdict", "question",
                "gt_response", "pred_text", "explanation", "cache_key"]
        with open(args.export, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sel)
        print(f"\nwrote {len(sel)} rows -> {args.export}")
        return

    print(f"\nshowing {min(len(sel), args.limit)} of {len(sel)} matching")
    for r in sel[:args.limit]:
        show(r)


if __name__ == "__main__":
    main()
