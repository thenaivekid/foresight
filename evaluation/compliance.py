"""
compliance.py — measure INSTRUCTION-FOLLOWING, with no ground truth, no judge, no GPU.

Why this exists: today the only feedback on a prompt is task
F1, which is slow, noisy, and conflates two completely different bugs —

    (a) the model MISUNDERSTOOD the instruction, and
    (b) the model understood it fine but got the VIDEO wrong.

Those need opposite fixes and currently look identical. This probe scores (a) alone,
in seconds, straight off saved run logs.

Use it as a GATE: if compliance is low the prompt was never really read, and tuning
semantics is pointless until it rises. If compliance is high and accuracy is still
low, the prompt WAS understood and the problem is comprehension or the gate.

Usage:
    python compliance.py output_hh_controller [output_etg_controller ...]
    python compliance.py --all
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

# [  29.3s | vid    1.0s] ctrl.raw     [VIDEOID] {json...}
_LINE = re.compile(
    r"\[\s*([\d.]+)s\s*\|\s*vid\s*([\d.]+)s\]\s+ctrl\.raw\s+\[([^\]]*)\]\s*(.*)$")

# Fields the ICL declares CONDITIONAL: "include ONLY when they change from their
# current value". Emitting one whose value is unchanged from the previous tick is a
# direct violation of the instruction.
_CONDITIONAL = ("fps", "next_check_s", "question_for_next")
# Fields the ICL declares UNCONDITIONAL: "ALWAYS start with seen, then have_enough_info".
_REQUIRED = ("seen", "have_enough_info")


def parse_log(path):
    """Yield (video_id, vt, raw_text) for every controller tick in a log file."""
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _LINE.search(line)
            if m:
                yield m.group(3), float(m.group(2)), m.group(4).strip()


def _first_key(raw):
    m = re.search(r'\{\s*"([^"]+)"', raw)
    return m.group(1) if m else None


def score_run(paths, icl_text=""):
    """Compute compliance metrics over every tick in `paths`."""
    n = n_json = n_nodiff = 0
    field_present = Counter()
    order_ok = 0
    cond_emitted = Counter()        # field -> times emitted at all
    cond_redundant = Counter()      # field -> times emitted with an UNCHANGED value
    prev_val = defaultdict(dict)    # video -> field -> last emitted value
    compact = spaced = 0
    ans_words = []
    seen_words = []
    copy_hits = 0
    videos = set()

    # 4-gram index of the ICL's own example text, to catch "NEVER copy the example"
    icl_grams = set()
    if icl_text:
        toks = re.findall(r"[a-z0-9]+", icl_text.lower())
        icl_grams = {tuple(toks[i:i + 4]) for i in range(len(toks) - 3)}

    for path in paths:
        for vid, vt, raw in parse_log(path):
            n += 1
            videos.add(vid)
            if raw.startswith("NO-DIFF"):
                n_nodiff += 1
                continue
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError
            except Exception:
                continue
            n_json += 1

            for f in _REQUIRED:
                if f in obj:
                    field_present[f] += 1
            if _first_key(raw) == "seen":
                order_ok += 1

            # spacing style: the ICL examples are all compact ("fps":1.0)
            if '": ' in raw:
                spaced += 1
            else:
                compact += 1

            for f in _CONDITIONAL:
                if f in obj:
                    cond_emitted[f] += 1
                    if vid in prev_val and prev_val[vid].get(f) == obj[f]:
                        cond_redundant[f] += 1
                    prev_val[vid][f] = obj[f]

            a = str(obj.get("answer") or "")
            if a:
                ans_words.append(len(a.split()))
                if icl_grams:
                    t = re.findall(r"[a-z0-9]+", a.lower())
                    if any(tuple(t[i:i + 4]) in icl_grams for i in range(len(t) - 3)):
                        copy_hits += 1
            s = str(obj.get("seen") or "")
            if s:
                seen_words.append(len(s.split()))

    return dict(ticks=n, videos=len(videos), json_ok=n_json, nodiff=n_nodiff,
                field_present=field_present, order_ok=order_ok,
                cond_emitted=cond_emitted, cond_redundant=cond_redundant,
                compact=compact, spaced=spaced, ans_words=ans_words,
                seen_words=seen_words, copy_hits=copy_hits)


def _pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "    -"


def report(name, r):
    n, j = r["ticks"], r["json_ok"]
    print(f"\n=== {name} ===")
    print(f"  ticks={n}  videos={r['videos']}  parsed={j}  unparseable={n - j - r['nodiff']}  NO-DIFF={r['nodiff']}")
    if not j:
        print("  (no parseable ticks)")
        return
    print(f"  valid JSON                {_pct(j, n)}")
    for f in _REQUIRED:
        print(f"  has '{f}'{' ' * (18 - len(f))}{_pct(r['field_present'][f], j)}")
    print(f"  'seen' emitted FIRST      {_pct(r['order_ok'], j)}")
    print(f"  compact style (as in ICL) {_pct(r['compact'], j)}   "
          f"[spaced: {_pct(r['spaced'], j)}]")

    print("  -- conditional rule: \"include ONLY when they change\" --")
    tot_e = tot_r = 0
    for f in _CONDITIONAL:
        e, red = r["cond_emitted"][f], r["cond_redundant"][f]
        tot_e += e
        tot_r += red
        if e:
            print(f"     {f:18s} emitted {e:5d} ticks ({_pct(e, j)}), "
                  f"REDUNDANT {red:5d} -> compliance {_pct(e - red, e)}")
        else:
            print(f"     {f:18s} never emitted  -> compliance  100.0%")
    print(f"     OVERALL conditional compliance: {_pct(tot_e - tot_r, tot_e)}"
          f"   ({tot_r} redundant emissions of {tot_e})")

    if r["seen_words"]:
        sw = sorted(r["seen_words"])
        print(f"  seen length words         med={sw[len(sw)//2]} max={sw[-1]}")
    if r["ans_words"]:
        aw = sorted(r["ans_words"])
        over = sum(1 for x in aw if x > 25)
        print(f"  answers                   n={len(aw)} med={aw[len(aw)//2]} "
              f"max={aw[-1]}  over-25-words {_pct(over, len(aw))}")
        print(f"  copies ICL example text   {_pct(r['copy_hits'], len(aw))}")


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    icl = ""
    cfg = os.path.join(here, "..", "foresight", "config.py")
    if os.path.exists(cfg):
        icl = open(cfg, errors="replace").read()

    dirs = argv[1:]
    if not dirs or dirs[0] == "--all":
        dirs = sorted(d for d in glob.glob(os.path.join(here, "output_*"))
                      if os.path.isdir(d))
    grand = []
    for d in dirs:
        paths = sorted(glob.glob(os.path.join(here, d, "*.log"))
                       or glob.glob(os.path.join(d, "*.log")))
        if not paths:
            continue
        r = score_run(paths, icl)
        if r["ticks"]:
            report(os.path.basename(d.rstrip("/")), r)
            grand.append(paths)
    if len(grand) > 1:
        report("ALL RUNS COMBINED", score_run([p for g in grand for p in g], icl))


if __name__ == "__main__":
    main(sys.argv)
