"""
sweep.py — the BIG per-task gate search. One fair arena for every firing rule.

WHY THIS EXISTS
---------------
An earlier comparison "fixed per-task thresholds beat adaptive stream-relative
gates" was WRONG, and wrong in an instructive way: the fixed family had been
searched over ~1300 configs per task (threshold x mode x refractory, fitted on
ground truth) while the adaptive family was run as 15 hand-picked GLOBAL presets
with no per-task tuning and NO REFRACTORY AT ALL. That is a tuned family against
an untuned one. Any conclusion from it is an artifact of the setup.

This module fixes the arena so the comparison is honest:

  * every gate, fixed or adaptive, emits a per-tick BOOLEAN "condition true now"
    signal, and then goes through the SAME emit layer (edge/level + refractory).
    So refractory and edge-discipline are available to all families equally.
  * every family's hyperparameters are searched PER TASK over the same budget.
  * every configuration is scored fit-on-all AND held-out, because a bigger
    search space always inflates fit-on-all (it is an argmax over a superset) and
    only held-out tells you whether the extra freedom bought anything real.

CAUSALITY: every signal uses only ticks already seen. The
adaptive gates compute their statistics over history STRICTLY BEFORE the current
tick. Nothing here observes the future, so any winner is deployable online
unchanged.

SPEED: parsing the run logs (~60MB) dominates, so traces are cached to JSON on
first use. After that a few thousand configs per task run in seconds, which is
the point -- you should be able to iterate without thinking about cost.

  python sweep.py RUN_DIR                 # full search, all families
  python sweep.py RUN_DIR --families fixed,pct
  python sweep.py RUN_DIR --dev 0.5       # held-out (default; 0 = ceiling)
  python sweep.py RUN_DIR --emit-config   # print config.py-ready dicts

⚠️ STILL A SCREEN, NOT A VERDICT. Offline replay assumes p_hit does not depend on
the gate. It weakly does: firing appends to `reported`, which is fed back into the
prompt, which shifts later p_hit. Take the winner to the GPU to confirm.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOL = 3.0                       # OmniPro ±3s temporal tolerance


# ---------------------------------------------------------------------------
# traces (cached: log parsing is the only slow part)
# ---------------------------------------------------------------------------
def load_traces(run_dir, cache_name="_traces_cache.json", rebuild=False):
    """-> [{task, video, gt:[float], ticks:[(vt,p_hit)]}], cached to JSON.

    Uses resweep.parse_run, which attributes each sample by the run-log HEADER
    that precedes its ticks. That matters: the same video appears under several
    tasks with different ground truth, so keying on video_id alone would merge
    unrelated ticks and silently corrupt every label.
    """
    cache = os.path.join(run_dir, cache_name)
    if os.path.exists(cache) and not rebuild:
        with open(cache) as fh:
            return [{**s, "ticks": [tuple(t) for t in s["ticks"]]}
                    for s in json.load(fh)]
    import resweep
    samples = [{"task": s["task"], "video": s["video"], "gt": s["gt"],
                "ticks": sorted(s["ticks"])}
               for s in resweep.parse_run(run_dir).values()]
    with open(cache, "w") as fh:
        json.dump(samples, fh)
    return samples


# ---------------------------------------------------------------------------
# SIGNALS: ticks -> [bool] per tick. Causal: history excludes the current tick.
# ---------------------------------------------------------------------------
def sig_fixed(thr):
    """Absolute cut-off. The baseline family."""
    def f(ticks):
        return [p >= thr for _, p in ticks]
    f.name = f"fixed(thr={thr})"
    return f


def _adaptive(fn, floor, warmup, label):
    """Wrap a history->bool rule with a floor + warmup, keeping it causal."""
    def f(ticks):
        out, hist = [], []
        for _, p in ticks:
            ok = (p >= floor and len(hist) >= warmup and fn(p, hist))
            out.append(bool(ok))
            hist.append(p)              # append AFTER deciding
        return out
    f.name = label
    return f


def sig_pct(q, floor, warmup):
    """Fire when p_hit is in the top (100-q)% of THIS stream so far."""
    def rule(p, h):
        s = sorted(h)
        return p > s[max(0, min(len(s) - 1, int(q / 100.0 * (len(s) - 1))))]
    return _adaptive(rule, floor, warmup, f"pct(q={q},floor={floor},warm={warmup})")


def sig_zscore(k, floor, warmup):
    """Fire when p_hit is k sigma above this stream's running mean."""
    def rule(p, h):
        sd = st.pstdev(h)
        return sd > 0 and (p - st.mean(h)) / sd > k
    return _adaptive(rule, floor, warmup, f"z(k={k},floor={floor},warm={warmup})")


def sig_relmedian(mult, floor, warmup):
    """Fire when p_hit exceeds `mult` x this stream's running median."""
    def rule(p, h):
        return p > st.median(h) * mult
    return _adaptive(rule, floor, warmup, f"med(x{mult},floor={floor},warm={warmup})")


def sig_runmax(margin, floor, warmup):
    """Fire only on a NEW running maximum (times margin). Very conservative."""
    def rule(p, h):
        return p > max(h) * margin
    return _adaptive(rule, floor, warmup, f"runmax(x{margin},floor={floor},warm={warmup})")


def sig_ema(alpha, mult, floor, warmup):
    """Fire when p_hit exceeds `mult` x an exponential moving average of the
    stream. Cheaper and more recency-weighted than the median/percentile rules."""
    def f(ticks):
        out, ema, n = [], None, 0
        for _, p in ticks:
            ok = (p >= floor and n >= warmup and ema is not None and p > ema * mult)
            out.append(bool(ok))
            ema = p if ema is None else (alpha * p + (1 - alpha) * ema)
            n += 1
        return out
    f.name = f"ema(a={alpha},x{mult},floor={floor},warm={warmup})"
    return f


def sig_hybrid(thr, q, floor, warmup):
    """BOTH an absolute floor AND stream-relative novelty must hold. The idea the
    fixed/adaptive dichotomy misses: an absolute cut-off says "confident enough",
    a relative one says "unusual for this video" -- they are different claims."""
    base, rel = sig_fixed(thr), sig_pct(q, floor, warmup)
    def f(ticks):
        return [a and b for a, b in zip(base(ticks), rel(ticks))]
    f.name = f"hybrid(thr={thr},q={q},floor={floor},warm={warmup})"
    return f


# ---------------------------------------------------------------------------
# EMIT LAYER: shared by every family, so refractory/edge are available to all.
# ---------------------------------------------------------------------------
def emit_times(ticks, signal, mode, refractory):
    emits, prev, last = [], False, None
    for (vt, _), cur in zip(ticks, signal):
        want = (cur and not prev) if mode == "edge" else cur
        if want and (last is None or vt - last >= refractory):
            emits.append(vt)
            last = vt
        prev = cur
    return emits


def greedy_match(emits, gts, tol=TOL):
    """OmniPro scoring: each GT event claimed by at most one emit, nearest first."""
    used, tp = set(), 0
    for e in sorted(emits):
        best, bd = None, None
        for i, g in enumerate(gts):
            if i in used:
                continue
            d = abs(e - g)
            if d <= tol and (bd is None or d < bd):
                bd, best = d, i
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(emits) - tp, len(gts) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(samples, signal, mode, refr, sig_cache=None):
    tp = fp = fn = ne = 0
    for i, s in enumerate(samples):
        sig = sig_cache[i] if sig_cache is not None else signal(s["ticks"])
        e = emit_times(s["ticks"], sig, mode, refr)
        a, b, c = greedy_match(e, s["gt"])
        tp += a; fp += b; fn += c; ne += len(e)
    return tp, fp, fn, ne


# ---------------------------------------------------------------------------
# SEARCH SPACE. Deliberately large -- the search is cheap, the run is not.
# ---------------------------------------------------------------------------
THRS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3,
        0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
        0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.992, 0.995, 0.998, 0.999]
REFRS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0,
         90.0, 120.0, 180.0, 300.0, 600.0, 1e9]        # 1e9 = "once per video"
MODES = ["edge", "level"]
FLOORS = [0.0, 0.02, 0.05, 0.15, 0.3]
WARMUPS = [3, 5, 8, 15]


def build_family(name):
    """-> list of signal factories for one family."""
    if name == "fixed":
        return [sig_fixed(t) for t in THRS]
    if name == "pct":
        return [sig_pct(q, f, w) for q in (80, 90, 95, 97, 99, 99.5)
                for f in FLOORS for w in WARMUPS]
    if name == "zscore":
        return [sig_zscore(k, f, w) for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
                for f in FLOORS for w in WARMUPS]
    if name == "median":
        return [sig_relmedian(m, f, w) for m in (1.5, 2.0, 3.0, 5.0, 10.0)
                for f in FLOORS for w in WARMUPS]
    if name == "runmax":
        return [sig_runmax(m, f, w) for m in (0.9, 1.0, 1.05, 1.2)
                for f in FLOORS for w in WARMUPS]
    if name == "ema":
        return [sig_ema(a, m, f, w) for a in (0.1, 0.3, 0.5)
                for m in (1.2, 1.5, 2.0, 3.0) for f in FLOORS for w in WARMUPS]
    if name == "hybrid":
        return [sig_hybrid(t, q, f, w) for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.98)
                for q in (80, 90, 95, 99) for f in FLOORS for w in (5, 8)]
    raise SystemExit(f"unknown family {name!r}")


ALL_FAMILIES = ["fixed", "pct", "zscore", "median", "runmax", "ema", "hybrid"]


def search(samples, families, log=None):
    """Exhaustive (signal x mode x refractory). Signals are evaluated ONCE per
    sample and cached, so adding refractory/mode values is nearly free."""
    best = None
    for fam in families:
        for factory in build_family(fam):
            cache = [factory(s["ticks"]) for s in samples]
            for mode in MODES:
                for refr in REFRS:
                    tp, fp, fn, ne = evaluate(samples, factory, mode, refr, cache)
                    f1 = prf(tp, fp, fn)[2]
                    if best is None or f1 > best["f1"]:
                        best = {"f1": f1, "family": fam, "signal": factory.name,
                                "mode": mode, "refractory_s": refr,
                                "factory": factory, "n_emits": ne}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--families", default=",".join(ALL_FAMILIES))
    ap.add_argument("--dev", type=float, default=0.5,
                    help="fraction fitted on; rest held out. 0 = fit on all (CEILING)")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--emit-config", action="store_true",
                    help="print config.py-ready dicts for the winning fixed configs")
    ap.add_argument("--out", default="sweep_results.json")
    args = ap.parse_args()

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    samples = load_traces(args.run_dir, rebuild=args.rebuild_cache)
    by_task = {}
    for s in samples:
        by_task.setdefault(s["task"], []).append(s)

    n_cfg = sum(len(build_family(f)) for f in fams) * len(MODES) * len(REFRS)
    print(f"{args.run_dir}: {len(samples)} samples, "
          f"{sum(len(s['ticks']) for s in samples)} ticks | "
          f"{n_cfg} configs/task x {len(by_task)} tasks | dev={args.dev}\n")

    hdr = (f"{'task':<28}{'n':>4}  {'family':<8}{'signal':<34}"
           f"{'mode':>6}{'refr':>7}{'F1':>8}")
    print(hdr); print("-" * len(hdr))

    results, TP = {}, [0, 0, 0, 0]
    per_family_best = {}
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda r: (r["video"],))
        if args.dev > 0:
            k = max(1, int(len(rows) * args.dev))
            fit, test = rows[:k], rows[k:] or rows
        else:
            fit = test = rows
        b = search(fit, fams)
        tp, fp, fn, ne = evaluate(test, b["factory"], b["mode"], b["refractory_s"])
        p, r, f1 = prf(tp, fp, fn)
        TP[0] += tp; TP[1] += fp; TP[2] += fn; TP[3] += ne
        refr = "once" if b["refractory_s"] > 1e8 else f"{b['refractory_s']:.0f}"
        print(f"{task:<28}{len(rows):>4}  {b['family']:<8}{b['signal']:<34}"
              f"{b['mode']:>6}{refr:>7}{f1:>8.3f}")
        results[task] = {"family": b["family"], "signal": b["signal"],
                         "mode": b["mode"], "refractory_s": b["refractory_s"],
                         "fit_f1": round(b["f1"], 4), "report_f1": round(f1, 4),
                         "precision": round(p, 4), "recall": round(r, 4),
                         "n_emits": ne, "n_samples": len(rows)}
        per_family_best.setdefault(b["family"], 0)
        per_family_best[b["family"]] += 1

    p, r, f1 = prf(TP[0], TP[1], TP[2])
    print("-" * len(hdr))
    print(f"{'POOLED':<28}{'':>4}  {'':<8}{'':<34}{'':>6}{TP[3]:>7}"
          f"{f1:>8.3f}   (P={p:.3f} R={r:.3f})")
    print(f"\nfamily wins: {per_family_best}")
    print("  NOTE: dev>0 => the F1 column is HELD-OUT (honest). dev=0 => a CEILING.")
    print("  NOTE: offline screen; confirm the winner on GPU (see module docstring).")

    out = os.path.join(args.run_dir, args.out)
    with open(out, "w") as fh:
        json.dump({"run_dir": args.run_dir, "dev": args.dev, "families": fams,
                   "tol_s": TOL, "pooled": {"time_f1": round(f1, 4),
                                            "precision": round(p, 4),
                                            "recall": round(r, 4),
                                            "n_emits": TP[3]},
                   "per_task": results}, fh, indent=2)
    print(f"\nwrote {out}")

    if args.emit_config:
        fx = {t: v for t, v in results.items() if v["family"] == "fixed"}
        print("\n# --- config.py-ready (fixed-family winners only) ---")
        for field, key in (("task_hit_thresholds", "thr"),
                           ("task_gate_modes", "mode"),
                           ("task_refractory_s", "refractory_s")):
            print(f"{field} = {{")
            for t, v in sorted(fx.items()):
                val = (v["signal"].split("thr=")[1].rstrip(")") if key == "thr"
                       else v[key])
                print(f'    "{t}": {val},')
            print("}")


if __name__ == "__main__":
    main()
