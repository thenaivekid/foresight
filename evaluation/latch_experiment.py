#!/usr/bin/env python3
"""KV-latch experiment: does p_hit latch, and what carries the persistence?

The harness tests falsifiable predictions about the latch mechanisms.

Three actions, none of which reads video:

  prepare  build the two sample sets and one working directory per
           (set, arm, repeat). Metadata-only selection: task, audio_dependency,
           duration, video_id, sample id and the NUMBER of ground-truth events.
           Ground-truth TIMES, answers and predictions are never consulted.
  status   report completion per run directory.
  analyze  read saved gate_trace/wall_timeline records, compute the latch
           statistics and write panels + a report.

Inference itself is delegated to vision_model_only.py run, unchanged, so the
arms, profile hashes and guard rails are exactly the audited ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

BENCHMARK = "/path/to/work/omnipro_data/benchmark.json"
DATASET = "/path/to/work/omnipro_data"

ARMS = ("raw", "strict_edge", "unreported_raw", "seen_after", "seen_off", "fix")
REPEATS = (1, 2)
SETS = ("iea_none", "multievent")
THR = 0.5
TOL = 3.0
N_PHASE = 500


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def n_gt(record) -> int:
    return len(record["ground_truth"])


def duration(record) -> float:
    return float(record["duration"])


def bin_pack(records, n_bins=4):
    """Greedy longest-processing-time packing by duration, as in vision_model_only."""
    bins, load = [[] for _ in range(n_bins)], [0.0] * n_bins
    for r in sorted(records, key=lambda r: -duration(r)):
        g = min(range(n_bins), key=lambda i: load[i])
        bins[g].append(r)
        load[g] += duration(r)
    return bins, load


def match_emits_to_gt(emit_times, gt_times, tolerance=TOL):
    """One-to-one greedy match, smallest |dt| first. Same rule as evaluation/metrics.py."""
    pairs = []
    for i, et in enumerate(emit_times):
        for j, gt in enumerate(gt_times):
            dt = abs(et - gt)
            if dt <= tolerance:
                pairs.append((dt, i, j))
    pairs.sort(key=lambda x: x[0])
    used_e, used_g, matches = set(), set(), []
    for dt, i, j in pairs:
        if i in used_e or j in used_g:
            continue
        used_e.add(i)
        used_g.add(j)
        matches.append((i, j, dt))
    return (matches,
            [i for i in range(len(emit_times)) if i not in used_e],
            [j for j in range(len(gt_times)) if j not in used_g])


def delivery_times(record):
    """ACTUAL wall delivery time, seconds since source_start.

    In an async run predictions[].t_sec is only the controller's sampled video
    timestamp, not when the user would have heard anything. Falls back to it
    only when no wall timeline was captured (lockstep runs).
    """
    timeline = record.get("wall_timeline") or []
    origin = next((e["monotonic_s"] for e in timeline if e["event"] == "source_start"), None)
    if origin is not None:
        out = sorted(e["monotonic_s"] - origin
                     for e in timeline if e["event"] == "response_delivered")
        if out:
            return out, "wall_delivery"
    return sorted(float(p["t_sec"]) for p in record.get("predictions", [])), "controller_tick"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def select_sets(raw, dataset):
    """Metadata-only selection. GT times / answers are never read, only GT COUNT."""
    def eligible(task, deps):
        return [r for r in raw
                if r["task"] == task
                and r.get("audio_dependency") in deps
                and (Path(dataset) / r["video_path"]).is_file()]

    # --- Set A: the ENTIRE instant_event_alert audio=none population -------
    set_a = sorted(eligible("instant_event_alert", {"none"}), key=lambda r: (duration(r), r["id"]))

    # --- Set B: vision-only, multi-event, alert family ---------------------
    sca_none = eligible("semantic_condition_alert", {"none"})
    iea_none = eligible("instant_event_alert", {"none"})
    iea_help = eligible("instant_event_alert", {"helpful"})

    key = lambda r: (duration(r), r["id"])  # noqa: E731  deterministic, no GT involved
    picked, used = [], set()

    def take(records, want_multi):
        for r in sorted(records, key=key):
            if r["id"] in used:
                continue
            if want_multi and n_gt(r) < 2:
                continue
            if not want_multi and n_gt(r) >= 2:
                continue
            picked.append(r)
            used.add(r["id"])

    take(sca_none, True)       # 9
    take(iea_none, True)       # 2
    take(iea_help, True)       # 7
    set_b = list(picked)
    if len(set_b) < 20:        # top up with the longest single-GT SCA/none clips
        filler = sorted((r for r in sca_none if r["id"] not in used),
                        key=lambda r: (-duration(r), r["id"]))
        set_b += filler[:20 - len(set_b)]
    set_b = set_b[:20]

    return {"iea_none": (set_a, "none"), "multievent": (set_b, "none_helpful")}


def prepare(args):
    source = Path(args.benchmark)
    raw = json.loads(source.read_text())
    src_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    root = args.out
    arms = tuple(args.arms) if args.arms else ARMS
    extending = (root / "experiment.json").exists()
    if extending:
        # Adding arms to a live experiment is allowed; silently rewriting an
        # existing arm's inputs is not -- that would invalidate saved runs.
        print(f"experiment.json exists: extending with arms {arms} "
              f"(existing arm directories are never rewritten)")

    sets = select_sets(raw, args.dataset)
    summary = {}

    for name, (records, audio) in sets.items():
        if not records:
            raise ValueError(f"Set {name} is empty")
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Set {name} has duplicate sample ids")
        for r in records:
            if not (Path(args.dataset) / r["video_path"]).is_file():
                raise FileNotFoundError(r["video_path"])
            if r["audio_dependency"] == "required":
                raise ValueError("audio=required must never enter a vision-only study")

        shards, load = bin_pack(records)
        manifest = {
            "name": f"kv_latch_{name}",
            "selection": ("Entire instant_event_alert audio=none population; no sampling."
                          if name == "iea_none" else
                          "Vision-only alert family, multi-event first: all multi-event "
                          "semantic_condition_alert/none, then multi-event instant_event_alert/none, "
                          "then multi-event instant_event_alert/helpful, topped up to 20 with the "
                          "longest single-GT semantic_condition_alert/none. Ordered by duration; "
                          "only task, audio_dependency, duration, video_id, sample id and the NUMBER "
                          "of ground-truth events were read."),
            "source": str(source), "source_sha256": src_sha, "dataset": args.dataset,
            "ids": ids, "audio_filter": audio,
            "shard_duration_s": load,
            "n_samples": len(records),
            "n_unique_videos": len({r["video_id"] for r in records}),
            "n_multi_event": sum(1 for r in records if n_gt(r) >= 2),
            "total_duration_s": sum(duration(r) for r in records),
            "interpretation": ("Screening diagnostic on a small vision-only subset. Not a benchmark "
                               "estimate, not a powered significance test, and not OmniPro overall."),
        }

        master = root / name
        if not (master / "manifest.json").exists():
            write_json(master / "manifest.json", manifest)
            write_json(master / "benchmark.json", records)
            for g, part in enumerate(shards):
                write_json(master / f"shard{g}.json", part)
        else:
            # The master must be byte-identical or the arms are not comparable.
            old = json.loads((master / "manifest.json").read_text())
            if old["ids"] != manifest["ids"] or old["source_sha256"] != src_sha:
                raise ValueError(f"{master}/manifest.json differs from the regenerated "
                                 "selection; refusing to mix incomparable sample sets")

        created = 0
        for arm in arms:
            for rep in REPEATS:
                run_dir = master / arm / f"r{rep}"
                if (run_dir / "manifest.json").exists():
                    continue
                write_json(run_dir / "manifest.json", manifest)
                write_json(run_dir / "benchmark.json", records)
                for g, part in enumerate(shards):
                    write_json(run_dir / f"shard{g}.json", part)
                created += 1

        summary[name] = {
            "n": len(records), "audio_filter": audio,
            "n_multi_event": manifest["n_multi_event"],
            "total_duration_s": round(manifest["total_duration_s"], 1),
            "shard_duration_s": [round(x, 1) for x in load],
            "by_task_audio": _tally(records),
            "gt_histogram": _hist(records),
        }
        print(f"\n### set {name}  n={len(records)}  audio={audio}  "
              f"multi-event={manifest['n_multi_event']}  "
              f"total={manifest['total_duration_s']:.0f}s "
              f"({manifest['total_duration_s']/60:.1f} min)  "
              f"new run dirs created: {created}")
        if not extending:
            print(f"{'task':<28}{'audio':<9}{'video_id':<14}{'dur_s':>8}{'nGT':>5}")
            for r in records:
                print(f"{r['task']:<28}{r['audio_dependency']:<9}{r['video_id']:<14}"
                      f"{duration(r):>8.1f}{n_gt(r):>5}")
        print(f"shard duration totals {[round(x) for x in load]}  "
              f"critical path ~{max(load)*1.06/60:.1f} min at 1.06x realtime")

    known = set(arms)
    if extending:
        known |= set(json.loads((root / "experiment.json").read_text()).get("arms", []))
    write_json(root / "experiment.json", {
        "sets": summary, "arms": sorted(known), "repeats": list(REPEATS),
        "execution": {"deterministic": False, "realtime": True, "speed": 1.0,
                      "hit_threshold": THR, "tolerance_s": TOL,
                      "pruning": "off", "compaction": "off", "ev0_dedup": False,
                      "debounce_s": 0.0},
        "arm_contrast": {
            "raw": "control: level gate, original prompt, seen_mode=before",
            "strict_edge": "gate only; prompt byte-identical to raw -> isolates report feedback",
            "unreported_raw": "prompt only; gate identical to raw -> isolates prompt semantics",
            "seen_after": "seen_mode=after; level is read BEFORE the scene description is "
                          "generated -> isolates within-tick seen priming, keeps the "
                          "perception step",
            "seen_off": "seen_mode=off; no description generated at all -> tests whether the "
                        "perception step is needed, not just its order",
            "fix": "combined candidate: seen_mode=after + unreported prompt + strict_edge gate",
        },
        "repeat_meaning": ("Seeds are IDENTICAL across repeats (seed=0, writer_seed=3407). "
                           "Repeats measure run-to-run variance of the asynchronous system "
                           "itself, not seed variance."),
        "source_sha256": src_sha,
        "n_runs": len(sets) * len(known) * len(REPEATS),
    })
    print(f"\nExperiment now covers arms: {sorted(known)}")


def _tally(records):
    out = {}
    for r in records:
        out.setdefault(f"{r['task']}/{r['audio_dependency']}", 0)
        out[f"{r['task']}/{r['audio_dependency']}"] += 1
    return dict(sorted(out.items()))


def _hist(records):
    out = {}
    for r in records:
        out[str(n_gt(r))] = out.get(str(n_gt(r)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def iter_runs(root):
    for name in SETS:
        for arm in ARMS:
            for rep in REPEATS:
                d = root / name / arm / f"r{rep}"
                if d.is_dir():
                    yield name, arm, rep, d


def load_run(run_dir):
    recs = []
    for path in sorted(run_dir.glob("g*/online_pred.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def status(args):
    print(f"{'set':<12}{'arm':<16}{'rep':>4}{'done':>7}{'/':^3}{'want':<6}{'state'}")
    for name, arm, rep, d in iter_runs(args.out):
        want = len(json.loads((d / "manifest.json").read_text())["ids"])
        recs = load_run(d)
        state = "COMPLETE" if len(recs) == want else ("empty" if not recs else "partial")
        print(f"{name:<12}{arm:<16}{rep:>4}{len(recs):>7}{'/':^3}{want:<6}{state}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def latch_stats(record):
    """Per-clip latch statistics. Reads only saved observations."""
    trace = record.get("gate_trace") or []
    if not trace:
        return None
    ts = [t["t_sec"] for t in trace]
    ps = [t["p_hit"] for t in trace]
    above = [p >= THR for p in ps]
    gts = sorted(float(g["t_sec"]) for g in record["ground_truth"])
    emits, emit_src = delivery_times(record)
    matches, unmatched_e, unmatched_g = match_emits_to_gt(emits, gts)
    matched_gt = {j for _, j, _ in matches}

    out = {
        "id": record["id"], "task": record["task"], "video_id": record["video_id"],
        "audio_dependency": record["audio_dependency"],
        "n_gt": len(gts), "n_ticks": len(ts), "n_emits": len(emits),
        "emit_time_source": emit_src,
        "tp": len(matches), "fp": len(unmatched_e), "fn": len(unmatched_g),
        "precision": len(matches) / len(emits) if emits else float("nan"),
        "recall": len(matches) / len(gts) if gts else float("nan"),
        "gt_ordinal_hit": [1 if j in matched_gt else 0 for j in range(len(gts))],
        "mean_p_hit_all": st.mean(ps),
        "duty_cycle_all": sum(above) / len(above),
    }

    # --- statistics defined on the whole trace, latched or not ------------
    # two-state transition rates
    stay = [above[i + 1] for i in range(len(above) - 1) if above[i]]
    rise = [above[i + 1] for i in range(len(above) - 1) if not above[i]]
    out["p_stay"] = sum(stay) / len(stay) if stay else None
    out["p_rise"] = sum(rise) / len(rise) if rise else None

    # longest run above threshold, in ticks and in video seconds
    best = cur = 0
    best_s = cur_s = 0.0
    for i, a in enumerate(above):
        if a:
            cur += 1
            cur_s = ts[i] - ts[i - cur + 1] if cur > 1 else 0.0
            best, best_s = max(best, cur), max(best_s, cur_s)
        else:
            cur, cur_s = 0, 0.0
    out["longest_above_ticks"] = best
    out["longest_above_s"] = best_s

    # rate-matched random-phase reference: keep the emission count and the within
    # stream spacing, destroy the phase. Offline statistical reference only --
    # not a causal trigger and not a multiple-comparison-corrected significance test.
    # The full draw vector is kept so the JOINT reference can be formed per draw at
    # aggregate level; summing per-clip percentiles would be invalid (the per-clip
    # maxima do not co-occur) and would make the reference far too loose.
    if emits and gts:
        span = max(max(ts), max(gts), max(emits)) + 1.0
        rng = random.Random(1234)
        ref = []
        for _ in range(N_PHASE):
            shift = rng.uniform(0, span)
            shifted = sorted((e + shift) % span for e in emits)
            ref.append(len(match_emits_to_gt(shifted, gts)[0]))
        out["_phase_draws"] = ref
        out["phase_ref_mean"] = st.mean(ref)
    else:
        out["_phase_draws"] = None
        out["phase_ref_mean"] = None

    # --- latch-specific statistics ----------------------------------------
    fi = next((i for i, a in enumerate(above) if a), None)
    out["latched"] = fi is not None
    if fi is None:
        out.update({"t_first_crossing": None, "pre_mean": None, "post_mean": None,
                    "pre_duty": None, "post_duty": None, "ratio": None,
                    "crossing_minus_gt1": None, "crossing_class": "never"})
        return out

    pre, post = ps[:fi], ps[fi + 1:]
    out["t_first_crossing"] = ts[fi]
    out["pre_mean"] = st.mean(pre) if pre else None
    out["post_mean"] = st.mean(post) if post else None
    out["pre_duty"] = (sum(1 for p in pre if p >= THR) / len(pre)) if pre else None
    out["post_duty"] = (sum(1 for p in post if p >= THR) / len(post)) if post else None
    out["ratio"] = (out["post_mean"] / out["pre_mean"]
                    if out["pre_mean"] and out["post_mean"] is not None else None)
    if gts:
        d = ts[fi] - gts[0]
        out["crossing_minus_gt1"] = d
        out["crossing_class"] = "early" if d < -TOL else ("late" if d > TOL else "on_gt1")
    else:
        out["crossing_minus_gt1"] = None
        out["crossing_class"] = "no_gt"
    return out


def aggregate(rows):
    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None
                and isinstance(r[key], (int, float)) and not math.isnan(r[key])]
        return st.mean(vals) if vals else None

    cls = [r["crossing_class"] for r in rows]
    ordinal = {}
    for r in rows:
        for k, hit in enumerate(r["gt_ordinal_hit"]):
            ordinal.setdefault(k, []).append(hit)

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    # JOINT random-phase reference: sum matched counts across clips WITHIN each
    # draw, then take the distribution of those sums. This is the correct null for
    # the run-level total; summing per-clip percentiles is not.
    draws = [r["_phase_draws"] for r in rows if r.get("_phase_draws")]
    if draws:
        joint = [sum(d[i] for d in draws) for i in range(min(len(d) for d in draws))]
        joint_sorted = sorted(joint)
        phase_mean = st.mean(joint)
        phase_p95 = joint_sorted[int(0.95 * (len(joint_sorted) - 1))]
        # one-sided Monte-Carlo p-value with the standard +1 correction
        p_value = (sum(1 for v in joint if v >= tp) + 1) / (len(joint) + 1)
    else:
        phase_mean = phase_p95 = p_value = None

    return {
        "n_clips": len(rows),
        "n_latched": sum(1 for r in rows if r["latched"]),
        "pre_mean": m("pre_mean"), "post_mean": m("post_mean"),
        "ratio": (m("post_mean") / m("pre_mean")) if m("pre_mean") else None,
        "pre_duty": m("pre_duty"), "post_duty": m("post_duty"),
        "p_stay": m("p_stay"), "p_rise": m("p_rise"),
        "odds_ratio": _odds(m("p_stay"), m("p_rise")),
        "longest_above_s": m("longest_above_s"),
        "crossing_early": cls.count("early"), "crossing_on_gt1": cls.count("on_gt1"),
        "crossing_late": cls.count("late"), "crossing_never": cls.count("never"),
        "mean_crossing_minus_gt1": m("crossing_minus_gt1"),
        "micro_tp": tp, "micro_fp": fp, "micro_fn": fn,
        "micro_precision": prec, "micro_recall": rec,
        "micro_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "n_emits": sum(r["n_emits"] for r in rows),
        "phase_ref_joint_mean": phase_mean,
        "phase_ref_joint_p95": phase_p95,
        "phase_ref_p_value": p_value,
        "gt_ordinal_recall": {f"GT#{k+1}": [sum(v), len(v), sum(v) / len(v)]
                              for k, v in sorted(ordinal.items()) if len(v) >= 3},
    }


def _odds(a, b):
    if not a or not b or a >= 1 or b >= 1:
        return None
    return (a / (1 - a)) / (b / (1 - b))


def analyze(args):
    root = args.out
    runs, per_run = {}, {}
    for name, arm, rep, d in iter_runs(root):
        recs = load_run(d)
        if not recs:
            continue
        rows = [s for s in (latch_stats(r) for r in recs) if s]
        if not rows:
            continue
        runs[(name, arm, rep)] = rows
        per_run[f"{name}/{arm}/r{rep}"] = aggregate(rows)

    if not runs:
        print("No completed runs found. Nothing analysed, nothing claimed.")
        return

    write_json(root / "latch_metrics.json",
               {"per_run": per_run,
                "per_clip": {f"{k[0]}/{k[1]}/r{k[2]}":
                             [{kk: vv for kk, vv in row.items() if kk != "_phase_draws"}
                              for row in v]
                             for k, v in runs.items()}})

    lines = ["# KV-latch experiment — measured results", "",
             "Offline analysis of saved `gate_trace` / `wall_timeline` records. "
             "No video was re-read and no model was run. Threshold 0.5 (not fitted), "
             "±3 s one-to-one greedy matching at **actual wall delivery time**.", ""]

    for name in SETS:
        keys = [k for k in runs if k[0] == name]
        if not keys:
            continue
        lines += [f"## Set `{name}`", "",
                  "| arm | rep | clips | latched | mean pre | mean post | ratio | "
                  "pre duty | post duty | P(a\\|a) | P(a\\|b) | OR | early/on/late |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for arm in ARMS:
            for rep in REPEATS:
                a = per_run.get(f"{name}/{arm}/r{rep}")
                if not a:
                    continue
                f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"  # noqa: E731
                lines.append(
                    f"| `{arm}` | {rep} | {a['n_clips']} | {a['n_latched']} | "
                    f"{f(a['pre_mean'],4)} | {f(a['post_mean'])} | "
                    f"{f(a['ratio'],1)}× | {f(a['pre_duty'])} | {f(a['post_duty'])} | "
                    f"{f(a['p_stay'])} | {f(a['p_rise'])} | {f(a['odds_ratio'],1)} | "
                    f"{a['crossing_early']}/{a['crossing_on_gt1']}/{a['crossing_late']} |")
        lines += ["", "Timing at actual wall delivery, against a rate-matched "
                  "circular-shift null (500 joint draws; emission count and within-stream "
                  "spacing preserved, phase destroyed):", "",
                  "| arm | rep | emits | TP | FP | FN | precision | recall | F1 | "
                  "null mean | null p95 | p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            for rep in REPEATS:
                a = per_run.get(f"{name}/{arm}/r{rep}")
                if not a:
                    continue
                pm, p95, pv = (a["phase_ref_joint_mean"], a["phase_ref_joint_p95"],
                               a["phase_ref_p_value"])
                lines.append(
                    f"| `{arm}` | {rep} | {a['n_emits']} | {a['micro_tp']} | {a['micro_fp']} | "
                    f"{a['micro_fn']} | {a['micro_precision']:.3f} | {a['micro_recall']:.3f} | "
                    f"{a['micro_f1']:.3f} | "
                    f"{'—' if pm is None else f'{pm:.1f}'} | "
                    f"{'—' if p95 is None else f'{p95:.0f}'} | "
                    f"{'—' if pv is None else f'{pv:.3f}'} |")
        lines += ["", "Per-GT-ordinal recall:", "",
                  "| arm | rep | " + " | ".join(f"GT#{i}" for i in range(1, 8)) + " |",
                  "|---|---:|" + "---:|" * 7]
        for arm in ARMS:
            for rep in REPEATS:
                a = per_run.get(f"{name}/{arm}/r{rep}")
                if not a:
                    continue
                cells = []
                for i in range(1, 8):
                    v = a["gt_ordinal_recall"].get(f"GT#{i}")
                    cells.append("—" if not v else f"{v[2]:.2f} ({v[0]}/{v[1]})")
                lines.append(f"| `{arm}` | {rep} | " + " | ".join(cells) + " |")
        lines.append("")

    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {root/'RESULTS.md'} and {root/'latch_metrics.json'}")

    if not args.no_plots:
        make_plots(root, runs)


def make_plots(root, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="#1f77b4", marker=".", ms=4, label="p_hit (dots = actual observations)"),
        Line2D([], [], color="red", ls="--", label="Ground-truth event"),
        Line2D([], [], color="#ff7f0e", ls="-", lw=2, label="First threshold crossing (latch)"),
        Line2D([], [], color="#2ca02c", marker="v", ls="none", label="Actual wall delivery"),
        Line2D([], [], color="#166534", marker="$✓$", ms=11, ls="none",
               label="Correct timing (±3 s, one-to-one)"),
        Line2D([], [], color="grey", ls=":", label="Decision boundary (0.5; not fitted)"),
    ]

    def panel(ax, record, stats):
        trace = record["gate_trace"]
        ts = [t["t_sec"] for t in trace]
        ps = [t["p_hit"] for t in trace]
        gts = sorted(float(g["t_sec"]) for g in record["ground_truth"])
        emits, _ = delivery_times(record)
        matches, _, _ = match_emits_to_gt(emits, gts)
        ax.plot(ts, ps, color="#1f77b4", marker=".", ms=2.5, lw=0.65)
        for gt in gts:
            ax.axvline(gt, color="red", ls="--", lw=1, alpha=0.8)
        if stats["t_first_crossing"] is not None:
            ax.axvline(stats["t_first_crossing"], color="#ff7f0e", lw=2, alpha=0.9)
            ax.axvspan(stats["t_first_crossing"], max(ts + gts + emits) + 1,
                       color="#ff7f0e", alpha=0.06)
        ax.plot(emits, [1.04] * len(emits), "v", color="#2ca02c", ms=5)
        ax.plot([emits[i] for i, _, _ in matches], [1.15] * len(matches),
                marker="$✓$", ls="none", ms=10, color="#166534")
        ax.axhline(THR, color="grey", ls=":", lw=1)
        ax.set(xlim=(0, max(ts + gts + emits) + 1), ylim=(-0.02, 1.24),
               xlabel="Video time (s)", ylabel="p_hit", yticks=[0, 0.5, 1])
        dl = ("" if stats["crossing_minus_gt1"] is None
              else f" · latch-GT1 {stats['crossing_minus_gt1']:+.0f}s")
        duty = "n/a" if stats["post_duty"] is None else f"{stats['post_duty']:.2f}"
        ax.set_title(f"{record['task']} | {record['video_id']} | audio={record['audio_dependency']}\n"
                     f"{len(gts)} GT · {len(emits)} emits · {len(matches)} matched · "
                     f"post-latch duty {duty}{dl}", loc="left", fontsize=9)
        ax.tick_params(labelsize=8)

    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)
    for (name, arm, rep), rows in sorted(runs.items()):
        recs = {r["id"]: r for r in load_run(root / name / arm / f"r{rep}")}
        rows = sorted(rows, key=lambda s: (s["task"], s["video_id"]))
        n = len(rows)
        ncol = 2
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 3 * nrow), squeeze=False)
        for ax, s in zip(axes.flat, rows):
            panel(ax, recs[s["id"]], s)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"KV latch · set={name} · arm={arm} · repeat {rep} · "
                     f"non-deterministic async · no pruning · threshold 0.5 (not fitted)",
                     fontsize=15, y=0.997)
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
                   bbox_to_anchor=(0.5, 0.004))
        fig.tight_layout(rect=(0, 0.045, 1, 0.985))
        for ext in ("png", "pdf"):
            p = out_dir / f"{name}__{arm}__r{rep}.{ext}"
            fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  figure {out_dir}/{name}__{arm}__r{rep}.png  ({n} panels)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "status", "analyze"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--benchmark", default=BENCHMARK)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="prepare: which arms to create directories for (default: all)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    {"prepare": prepare, "status": status, "analyze": analyze}[args.action](args)


if __name__ == "__main__":
    main()
