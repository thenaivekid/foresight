"""Paired bootstrap over videos for the 3-arm pruning ablation.

n=18 is small. A 1-true-positive swing moves micro-F1 by ~2 points, so the raw
ranking must not be reported as a result until we know whether it survives
resampling. This does a PAIRED bootstrap (resample videos, not samples, keeping
the arms aligned on the same video) and reports the 95% CI of the micro-F1
difference vs baseline.

If a CI straddles 0, the honest statement is "indistinguishable at this sample
size", not "better".
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ARMS = [("baseline", "abl_base"), ("dsh_rank50", "abl_dsh"),
        ("clip_qdp50", "abl_clip")]
B = 10000
RNG = np.random.default_rng(0)


def parse(d):
    out = {}
    dd = os.path.join(ROOT, d)
    for lg in os.listdir(dd):
        if not (lg.startswith("run_") and lg.endswith(".log")):
            continue
        with open(os.path.join(dd, lg), errors="ignore") as f:
            for ln in f:
                m = re.search(r"\[online\]\s+\d+/\d+\s+(\S+)\s+video=(\S+)\s+"
                              r"emits=(\d+)\s+gt=(\d+)\s+tp=(\d+)\s+fp=(\d+)\s+"
                              r"fn=(\d+)", ln)
                if m:
                    key = (m.group(1), m.group(2))
                    out[key] = (int(m.group(5)), int(m.group(6)), int(m.group(7)))
    return out


def micro_f1(triples):
    tp = sum(t[0] for t in triples)
    fp = sum(t[1] for t in triples)
    fn = sum(t[2] for t in triples)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    data = {n: parse(d) for n, d in ARMS}
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    print("paired videos: %d" % len(keys))

    obs = {n: micro_f1([data[n][k] for k in keys]) for n in data}
    print("observed micro time_f1: " +
          ", ".join("%s=%.4f" % (n, v) for n, v in obs.items()))

    idx = np.arange(len(keys))
    boot = defaultdict(list)
    for _ in range(B):
        s = RNG.choice(idx, size=len(idx), replace=True)
        sk = [keys[i] for i in s]
        base = micro_f1([data["baseline"][k] for k in sk])
        for n in data:
            if n == "baseline":
                continue
            boot[n].append(micro_f1([data[n][k] for k in sk]) - base)

    print("\n95%% CI of (arm - baseline) micro time_f1, %d bootstrap resamples:" % B)
    res = {}
    for n, v in boot.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        res[n] = {"delta": round(float(obs[n] - obs["baseline"]), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "p_better": round(float((v > 0).mean()), 4),
                  "verdict": sig}
        print("  %-12s delta=%+.4f  CI=[%+.4f, %+.4f]  P(better)=%.3f  %s" % (
            n, obs[n] - obs["baseline"], lo, hi, (v > 0).mean(), sig))

    with open(os.path.join(ROOT, "ablation_bootstrap.json"), "w") as f:
        json.dump({"observed": obs, "vs_baseline": res,
                   "n_videos": len(keys), "resamples": B}, f, indent=2)
    print("\nwrote ablation_bootstrap.json")


if __name__ == "__main__":
    main()
