import json, sys
from metrics import ContentJudge, aggregate, score_sample


def fmt(v, nd=3, dash="UNJUDGED"):
    """Render a metric that may be withheld. metrics.aggregate returns None for
    content_acc/joint_* when any matched emit went unjudged, so a plain :.3f
    would crash — and silently substituting 0.0 would be exactly the kind of
    fake number this change exists to eliminate."""
    return dash if v is None else f"{v:.{nd}f}"

judge = ContentJudge()
runs = {
 "V1 edge": "/path/to/run_v1/evaluation/output_sca_v1edge/online_pred.jsonl",
 "V2 evidence": "/path/to/run_v2/evaluation/output_sca_v2evid/online_pred.jsonl",
 "V3 (+cal+2ex)": "/path/to/run_v2/evaluation/output_sca_v3/online_pred.jsonl",
 "v2best(finegrid)": "/path/to/run_v2/evaluation/output_v2best/online_pred.jsonl",
}
print(f"{'run':<18}{'emits':>6}{'tp':>4}{'fp':>4}{'fn':>4}{'timeP':>7}{'timeR':>7}{'timeF1':>8}{'jointF1':>8}{'contAcc':>8}")
for name, path in runs.items():
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["id"]] = r
    per = [score_sample(r, tolerance=3.0, judge=judge) for r in rows.values()]
    o = aggregate(per)["overall"]
    tp = int(sum(p["tp_time"] for p in per)); fp = int(sum(p["fp"] for p in per)); fn = int(sum(p["fn"] for p in per))
    print(f"{name:<18}{int(o['n_emits']):>6}{tp:>4}{fp:>4}{fn:>4}{o['time_precision']:>7.3f}{o['time_recall']:>7.3f}{o['time_f1']:>8.3f}{fmt(o['joint_f1']):>8}{fmt(o['content_acc']):>8}")
