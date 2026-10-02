"""Compare the three pruning arms on the shared 18-video ablation split.

Reads each arm's online_metrics.json plus its run log, and reports:
  - overall time_f1 / joint_f1 / content_acc
  - per-task breakdown
  - realised token keep-rate (from the pruner stats logged per run)
  - total wall time and emits

Wall time is reported but must NOT be used as the headline efficiency number:
these arms ran on SHARED login-node GPUs where a neighbour's job changed
per-tick generation time by 73x (2.9s -> 212.9s) on identical code. Keep-rate is
the trustworthy cost measure; wall time is advisory only.
"""
import json
import os
import re
import sys
from collections import defaultdict

ARMS = [("baseline", "abl_base"), ("dsh_rank50", "abl_dsh"), ("clip_qdp50", "abl_clip")]
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_metrics(d):
    p = os.path.join(ROOT, d, "online_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def parse_log(d):
    """Pull per-sample eval lines and pruner keep-rates out of the run log."""
    rows, keeps = [], []
    dd = os.path.join(ROOT, d)
    if not os.path.isdir(dd):
        return rows, keeps
    logs = [x for x in os.listdir(dd) if x.startswith("run_") and x.endswith(".log")]
    for lg in logs:
        with open(os.path.join(dd, lg), errors="ignore") as f:
            for ln in f:
                m = re.search(r"\[online\]\s+\d+/\d+\s+(\S+)\s+video=(\S+)\s+"
                              r"emits=(\d+)\s+gt=(\d+)\s+tp=(\d+)\s+fp=(\d+)\s+"
                              r"fn=(\d+)\s+\(([\d.]+)s\)", ln)
                if m:
                    rows.append({"task": m.group(1), "video": m.group(2),
                                 "emits": int(m.group(3)), "gt": int(m.group(4)),
                                 "tp": int(m.group(5)), "fp": int(m.group(6)),
                                 "fn": int(m.group(7)), "secs": float(m.group(8))})
                k = re.search(r"(?:dsh|clip)_keep_rate\s+n=\d+\s+mean=\s*([\d.]+)", ln)
                if k:
                    keeps.append(float(k.group(1)))
    return rows, keeps


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    print("=" * 78)
    print("PRUNING ABLATION - 18 videos (2 per task), matched ~50%% token budget")
    print("=" * 78)
    summary = {}
    per_task = defaultdict(dict)

    for name, d in ARMS:
        rows, keeps = parse_log(d)
        met = load_metrics(d)
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        summary[name] = {
            "n_done": len(rows),
            "micro_time_f1": round(f1(tp, fp, fn), 4),
            "emits": sum(r["emits"] for r in rows),
            "tp": tp, "fp": fp, "fn": fn,
            "mean_secs_per_video": round(
                sum(r["secs"] for r in rows) / len(rows), 1) if rows else None,
            "keep_rate": round(sum(keeps) / len(keeps), 3) if keeps else 1.0,
            "metrics_json": {k: met[k] for k in
                             ("time_f1", "joint_f1", "content_acc")
                             if met and k in met} if met else None,
        }
        agg = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            a = agg[r["task"]]
            a[0] += r["tp"]; a[1] += r["fp"]; a[2] += r["fn"]
        for t, a in agg.items():
            per_task[t][name] = round(f1(*a), 3)

    print(json.dumps(summary, indent=2))
    print("\n--- per-task time_f1 ---")
    hdr = "%-34s %10s %10s %10s" % ("task", "baseline", "dsh_rank50", "clip_qdp50")
    print(hdr)
    for t in sorted(per_task):
        r = per_task[t]
        print("%-34s %10s %10s %10s" % (
            t, r.get("baseline", "-"), r.get("dsh_rank50", "-"),
            r.get("clip_qdp50", "-")))

    with open(os.path.join(ROOT, "ablation_prune18.json"), "w") as f:
        json.dump({"summary": summary, "per_task": per_task}, f, indent=2)
    print("\nwrote ablation_prune18.json")


if __name__ == "__main__":
    main()
