#!/usr/bin/env python3
"""bench_status.py -- score every arm from whatever is on disk RIGHT NOW and
print the running table. Safe to run any time, mid-run; touches no GPU.

    python bench_status.py                 # table + refresh RESULTS.json
    python bench_status.py --json          # just the combined JSON
    python bench_status.py --watch 300     # refresh every 5 min

Partial arms are labelled. Shards are balanced and consumed in manifest order,
so a partial number is a real sample -- but it is a SAMPLE, not the final figure,
and the table says so on every row that is not yet complete.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", "/path/to/work/bench_out")
STAMPS = os.path.join(OUT, "STAGE_STAMPS")

# arm -> (bench, shard prefix, repo env var for the shard dir)
ARMS = [
    ("sb_po_calib",   "streamingbench", "sb_po_calib"),
    ("sb_po_eval_b1", "streamingbench", "sb_po_eval"),
    ("sb_po_eval_b2", "streamingbench", "sb_po_eval"),
    ("sb_mcq",        "streamingbench", "sb_mcq"),
    ("ovo_far_probe", "ovobench",       "ovo_far"),
    ("ovo_mcq",       "ovobench",       "ovo_mcq"),
]
STAGES = ["sb_po_calib", "sb_po_fit", "sb_po_eval", "sb_mcq",
          "ovo_far_probe", "ovo_far_fit", "ovo_mcq"]


def expected(prefix):
    n = 0
    for f in glob.glob(os.path.join(HERE, "splits_bench", f"{prefix}_g*.json")):
        try:
            n += len(json.load(open(f)))
        except Exception:
            pass
    return n


def rescore(arm, bench, prefix):
    pred = os.path.join(OUT, arm)
    if not glob.glob(os.path.join(pred, "g*", "pred.jsonl")):
        return None
    out = os.path.join(OUT, f"results_{arm}.json")
    cmd = [sys.executable, os.path.join(HERE, "score_bench.py"),
           "--bench", bench, "--arm", arm, "--pred", pred,
           "--out", out, "--expected", str(expected(prefix))]
    gates = os.path.join(HERE, "fitted_gates_ovo_far.json")
    if arm == "ovo_far_probe" and os.path.exists(gates):
        cmd += ["--gates", gates]
    subprocess.run(cmd, capture_output=True)
    try:
        return json.load(open(out))
    except Exception:
        return None


def pct(x):
    return "--" if x is None else f"{100*x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--watch", type=int, default=0)
    args = ap.parse_args()

    while True:
        combined = {}
        for arm, bench, prefix in ARMS:
            r = rescore(arm, bench, prefix)
            if r:
                combined[arm] = r
        with open(os.path.join(OUT, "RESULTS.json"), "w") as fh:
            json.dump(combined, fh, indent=2)

        if args.json:
            print(json.dumps(combined, indent=2))
        else:
            done = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(STAMPS, "*.done"))}
            print(f"\n=== bench_out  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            print("stages: " + "  ".join(
                ("[x] " if s in done else "[ ] ") + s for s in STAGES))
            if not combined:
                print("\n(no predictions written yet)")
            else:
                print(f"\n{'arm':15s}{'videos':>12s}{'probes':>8s}{'err':>5s}{'state':>10s}  headline")
                for arm, r in combined.items():
                    got, exp = r["n_videos"], r.get("expected_videos", 0)
                    st = "COMPLETE" if r.get("partial") is False else \
                         f"{100*r.get('progress',0):.0f}%"
                    if r["bench"] == "ovobench":
                        m = r.get("modes", {})
                        head = " ".join(
                            f"{k[:3]}={pct(v.get('macro'))}" for k, v in m.items()
                            if v.get("macro") is not None)
                        tot = r.get("total_avg")
                        if tot is not None:
                            head += f"  TOTAL={pct(tot)}"
                        auc = (r.get("have_enough_info_auc") or {}).get("pooled", {})
                        if auc.get("auc_p_hit") is not None:
                            head += f"  | p_hit AUC={auc['auc_p_hit']:.3f} (n={auc['n_pos']}+{auc['n_neg']})"
                    elif "po" in arm:
                        t = r.get("acc_by_tolerance", {})
                        head = " ".join(f"<={k.strip('<=s')}s={pct(v)}"
                                        for k, v in t.items() if v is not None)
                        head += f"  fired={r.get('n_fired')}/{r.get('n')}"
                    else:
                        cats = r.get("categories", {})
                        head = " ".join(f"{k[:9]}={pct(v.get('macro'))}"
                                        for k, v in cats.items())
                    print(f"{arm:15s}{got:>6d}/{exp:<5d}{r['n_probes']:>8d}"
                          f"{r['n_errors']:>5d}{st:>10s}  {head}")
                print("\nAccuracies are percentages. Rows not marked COMPLETE are a "
                      "SAMPLE of finished videos, not the final figure.")
            print(f"\nRESULTS.json -> {os.path.join(OUT, 'RESULTS.json')}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
