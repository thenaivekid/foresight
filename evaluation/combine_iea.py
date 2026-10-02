"""Combine the 4 GPU-group metrics of one IEA split-run into a single run metric.
Reconstruction is EXACT: the temporal match is 1-1, so per group
    tpt = round(time_recall*n_gt);  tpc = round(joint_recall*n_gt)
    fp  = n_emits - tpt;            fn  = n_gt - tpt
Micro-aggregate (sum tp/fp/fn) across the 4 groups, then recompute P/R/F1."""
import json, sys, glob

def prf(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f

def combine(run_dir_prefix):
    T=dict(tpt=0,tpc=0,fp=0,fn=0,n_gt=0,n_emits=0,n=0)
    groups=sorted(glob.glob(f"{run_dir_prefix}_g*/online_metrics.json"))
    if not groups:
        return None
    for g in groups:
        o=json.load(open(g))["overall"]
        ngt=o["n_gt"]; ne=o["n_emits"]
        tpt=round(o["time_recall"]*ngt); tpc=round(o["joint_recall"]*ngt)
        fp=ne-tpt; fn=ngt-tpt
        T["tpt"]+=tpt; T["tpc"]+=tpc; T["fp"]+=fp; T["fn"]+=fn
        T["n_gt"]+=ngt; T["n_emits"]+=ne; T["n"]+=o["n_samples"]
    tp_p,tp_r,tp_f=prf(T["tpt"],T["fp"],T["fn"])
    jp,jr,jf=prf(T["tpc"],T["fp"],T["fn"])
    return {"n_samples":T["n"],"n_gt":T["n_gt"],"n_emits":T["n_emits"],
            "time_precision":round(tp_p,4),"time_recall":round(tp_r,4),"time_f1":round(tp_f,4),
            "joint_precision":round(jp,4),"joint_recall":round(jr,4),"joint_f1":round(jf,4),
            "content_acc":round(T["tpc"]/T["tpt"],4) if T["tpt"] else 0.0}

if __name__=="__main__":
    prefix=sys.argv[1]  # e.g. .../output_iea_login/r1
    m=combine(prefix)
    if m is None:
        print(f"NO group metrics found for {prefix}"); sys.exit(1)
    json.dump(m, open(f"{prefix}_combined.json","w"), indent=2)
    print(json.dumps(m, indent=2))
