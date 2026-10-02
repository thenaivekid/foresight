"""
judge_offline.py — complete the content half of a finished eval, without a GPU.

WHY THIS EXISTS
---------------
The GPU eval and the LLM judge have nothing in common except the text between
them, and they fail for unrelated reasons: the eval needs four A100s for a day,
the judge needs API quota. Coupling them meant a rate-limited free-tier judge
silently degraded a 24-hour run (it fell back to word overlap and reported the
result as if it were judged). They are now separate.

The eval writes every prediction's TEXT to online_pred.jsonl and marks content
UNJUDGED when the judge is unreachable. This script picks that up later:

  1. collect every (question, gt_response, pred_text) triple that a matched emit
     on a JUDGE_TASK would need,
  2. judge only the ones missing from the cache, with retry + exponential backoff
     so a 429/503 costs patience instead of a wrong number,
  3. write the verdicts into the shared judge cache,
  4. re-run metrics.aggregate so content_acc / joint_f1 become real numbers.

Resumable: everything already judged is a cache hit, so an interrupted pass costs
nothing. Safe to run repeatedly, and safe to run while the GPU eval is going --
the cache merges rather than overwrites.

BACKENDS
--------
--backend openai (default) uses OpenAI chat.completions with Structured Outputs,
model $OPENAI_JUDGE_MODEL (default gpt-5-mini). The paper's judge is Gemini-3-Flash
(--backend gemini), but we have no Gemini quota, so it returns None for everything.
Both backends write into the same cache; OpenAI verdicts are namespaced by model so
the two can never collide (see ContentJudge._cache_key).

Baselines must be judged with the SAME backend and model as the system, or the
comparison is between two judges rather than two systems.

Usage:
    python judge_offline.py RUN_DIR --dry-run   # how many calls needed?
    python judge_offline.py RUN_DIR             # judge, then rescore
    python judge_offline.py RUN_DIR --max-calls 200   # respect a daily quota
    python judge_offline.py RUN_DIR --rescore-only    # no API, just recompute

    # 50% cheaper, 24h SLA -- fine, since judging is decoupled from the GPU eval:
    python judge_offline.py RUN_DIR --batch submit    # prints a batch id
    python judge_offline.py RUN_DIR --batch fetch     # merge + rescore
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics

TOL = 3.0


def load_preds(run_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "**", "online_pred.jsonl"),
                              recursive=True)):
        with open(p, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["id"]] = r
    return list(out.values())


def needed_triples(records, tolerance=TOL):
    """Exactly the judge calls metrics.score_sample would make, in the same order.

    Mirrors score_sample's matching so we never judge a pair that scoring will not
    ask about -- on a free tier, every wasted call is a real cost.
    """
    want = []
    seen = set()
    for r in records:
        if r["task"] not in metrics.JUDGE_TASKS:
            continue
        emits = r.get("predictions", [])
        gts = r["ground_truth"]
        matches, _, _ = metrics.match_emits_to_gt(
            [float(e["t_sec"]) for e in emits],
            [float(g["t_sec"]) for g in gts], tolerance)
        for ei, gj, _dt in matches:
            t = (r.get("question", ""), gts[gj].get("response", ""),
                 emits[ei].get("raw", ""))
            if t not in seen:
                seen.add(t)
                want.append(t)
    return want


def judge_all(triples, judge, max_calls=None, base_sleep=2.0, max_retries=6):
    """Judge the uncached triples. Returns (n_new, n_failed, stopped_early)."""
    todo = [t for t in triples
            if judge._cache_key(*t) not in judge._cache]
    print(f"{len(triples)} triples needed | {len(triples) - len(todo)} already "
          f"cached | {len(todo)} to judge")
    if max_calls is not None:
        todo = todo[:max_calls]
        print(f"  limited to {len(todo)} calls this pass (--max-calls)")

    new = failed = 0
    for i, (q, gt, pred) in enumerate(todo, 1):
        delay = base_sleep
        for attempt in range(1, max_retries + 1):
            # score() returns None on any failure and never invents a verdict, so
            # a None here genuinely means "still unjudged".
            before = judge.n_unjudged
            v = judge.score(q, gt, pred)
            if v is not None:
                new += 1
                break
            if judge.n_unjudged > before and attempt < max_retries:
                # jitter so parallel passes do not synchronise their retries
                s = delay * (1 + random.random() * 0.3)
                print(f"  [{i}/{len(todo)}] retry {attempt}/{max_retries} "
                      f"in {s:.1f}s", flush=True)
                time.sleep(s)
                delay = min(delay * 2, 120.0)
            elif attempt >= max_retries:
                failed += 1
        if i % 25 == 0:
            print(f"  judged {i}/{len(todo)} (new={new} failed={failed})",
                  flush=True)
    return new, failed, len(todo) < len([t for t in triples
                                         if judge._cache_key(*t) not in judge._cache])


# ---------------------------------------------------------------------------
# Batch API (OpenAI only): half price, 24h SLA.
# submit writes the uncached triples out with a custom_id each, remembers the
# triples next to those ids, and prints the batch id; fetch matches results back
# by custom_id and merges them into the shared cache. The state file carries the
# triples themselves, so a fetch works in a brand-new process, on another node,
# days later, with nothing but the run dir.
# ---------------------------------------------------------------------------
BATCH_DIRNAME = "judge_batch"


def _batch_dir(run_dir):
    d = os.path.join(run_dir, BATCH_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _require_openai(judge):
    if judge.mode != "openai" or judge._openai is None:
        print("--batch requires --backend openai with OPENAI_API_KEY set "
              "(the Batch API is OpenAI's).")
        return None
    return judge._openai


def batch_submit(run_dir, triples, judge, max_calls=None):
    client = _require_openai(judge)
    if client is None:
        return
    todo = [t for t in triples if judge._cache_key(*t) not in judge._cache]
    print(f"{len(triples)} triples needed | {len(triples) - len(todo)} cached | "
          f"{len(todo)} to submit")
    if max_calls is not None:
        todo = todo[:max_calls]
        print(f"  limited to {len(todo)} requests (--max-calls)")
    if not todo:
        print("nothing to submit — every triple is already judged.")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bdir = _batch_dir(run_dir)
    inp = os.path.join(bdir, f"requests-{stamp}.jsonl")
    reqs = []
    with open(inp, "w") as fh:
        for i, (q, gt, pred) in enumerate(todo):
            cid = f"omnipro-{stamp}-{i:06d}"
            fh.write(json.dumps({
                "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                # identical body to the sync path, so a batch verdict and a sync
                # verdict for one triple are the same computation
                "body": metrics.openai_judge_request(
                    judge._model, q, gt, pred,
                    seed=judge._seed if judge._use_seed else None),
            }) + "\n")
            reqs.append({"custom_id": cid, "cache_key": judge._cache_key(q, gt, pred),
                         "question": q, "gt": gt, "pred": pred})

    up = client.files.create(file=open(inp, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"purpose": "omnipro-content-judge", "run_dir": run_dir})

    state = os.path.join(bdir, f"batch-{batch.id}.json")
    with open(state, "w") as fh:
        json.dump({"batch_id": batch.id, "model": judge._model,
                   "seed": judge._seed if judge._use_seed else None,
                   "created": stamp, "run_dir": os.path.abspath(run_dir),
                   "input_file": os.path.abspath(inp), "input_file_id": up.id,
                   "n_requests": len(reqs), "requests": reqs}, fh, indent=2)
    print(f"\nsubmitted batch {batch.id} ({len(reqs)} requests, status="
          f"{batch.status})\n  state: {state}\n"
          f"  fetch with: python judge_offline.py {run_dir} --batch fetch")
    return batch.id


def _latest_state(run_dir, batch_id=None):
    bdir = _batch_dir(run_dir)
    if batch_id:
        p = os.path.join(bdir, f"batch-{batch_id}.json")
        return p if os.path.exists(p) else None
    cands = sorted(glob.glob(os.path.join(bdir, "batch-*.json")),
                   key=os.path.getmtime)
    return cands[-1] if cands else None


def batch_fetch(run_dir, judge, batch_id=None, wait=0.0):
    """Retrieve one batch, merge its verdicts into the cache. Returns n_merged."""
    client = _require_openai(judge)
    if client is None:
        return 0
    sp = _latest_state(run_dir, batch_id)
    if sp is None:
        print(f"no batch state found in {os.path.join(run_dir, BATCH_DIRNAME)}; "
              f"run --batch submit first.")
        return 0
    with open(sp) as fh:
        state = json.load(fh)
    print(f"batch {state['batch_id']} ({state['n_requests']} requests, "
          f"model={state['model']}) from {sp}")
    if state["model"] != judge._model:
        print(f"  WARNING: submitted with model={state['model']} but this judge is "
              f"{judge._model}; verdicts land under the submitting model's key.")

    # bounded poll: never spin on the API forever
    deadline = time.time() + max(0.0, wait)
    while True:
        b = client.batches.retrieve(state["batch_id"])
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        rc = getattr(b, "request_counts", None)
        print(f"  status={b.status}"
              + (f" ({rc.completed}/{rc.total} done)" if rc else ""), flush=True)
        if time.time() >= deadline:
            print("  not finished yet — re-run --batch fetch later "
                  "(state is on disk; nothing is lost).")
            return 0
        time.sleep(min(30.0, max(1.0, deadline - time.time())))

    if b.status != "completed":
        print(f"  batch ended as {b.status}; no verdicts merged (still UNJUDGED).")
        return 0

    by_cid = {r["custom_id"]: r for r in state["requests"]}
    merged = failed = 0
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cid = row["custom_id"]
                req = by_cid.get(cid)
                if req is None:
                    failed += 1
                    print(f"  unknown custom_id {cid} (not in state file)")
                    continue
                resp = row.get("response") or {}
                if row.get("error") or resp.get("status_code") != 200:
                    failed += 1
                    print(f"  {cid}: error {row.get('error') or resp.get('status_code')}")
                    continue
                msg = resp["body"]["choices"][0]["message"]
                if msg.get("refusal"):
                    failed += 1
                    print(f"  {cid}: model refused")
                    continue
                _payload = json.loads(msg["content"])
                sc = int(_payload["score"])
            except Exception as e:
                failed += 1
                print(f"  unparseable batch line ({type(e).__name__}: {str(e)[:100]})")
                continue
            # recompute the key from the stored triple rather than trusting the
            # one saved at submit time, so the cache stays correct even if the
            # key scheme changed in between
            _key = judge._cache_key(req["question"], req["gt"], req["pred"])
            _verdict = 1.0 if sc >= 3 else 0.0
            judge._cache_put(_key, _verdict)
            # Same audit trail as the sync path -- a batch verdict must be just as
            # inspectable as one made interactively.
            judge.trace(_key, req["question"], req["gt"], req["pred"], _verdict,
                        {"score_raw": sc,
                         "explanation": _payload.get("explanation", ""),
                         "model": state.get("model"), "seed": state.get("seed"),
                         "batch_id": state.get("batch_id")},
                        source="batch")
            merged += 1
    if b.error_file_id:
        errs = client.files.content(b.error_file_id).text.strip().splitlines()
        failed += len(errs)
        for e in errs[:5]:
            print(f"  error line: {e[:200]}")

    state["fetched"] = time.strftime("%Y%m%d-%H%M%S")
    state["merged"] = merged
    state["failed"] = failed
    with open(sp, "w") as fh:
        json.dump(state, fh, indent=2)
    print(f"merged {merged} verdicts | {failed} failed (left UNJUDGED, never guessed)")
    return merged


def report(records, judge, label):
    # READ-ONLY BY CONSTRUCTION. score_sample() calls the judge, so before this
    # guard existed a report made a live API call for every uncached triple --
    # i.e. --dry-run judged the whole run, and `--max-calls 3` judged 3 in the
    # counted loop and then the remaining 469 here. Anything missing from the
    # cache at report time is UNJUDGED, which is exactly what the table says.
    judge.offline = True
    agg = metrics.aggregate(
        [metrics.score_sample(r, TOL, judge) for r in records])
    o = agg["overall"]

    def f(v, nd=3):
        return "UNJUDGED" if v is None else f"{v:.{nd}f}"

    print(f"\n=== {label} ===")
    hdr = (f"{'task':<28}{'n':>4}{'GT':>6}{'emits':>7}{'tF1':>7}"
           f"{'cover':>8}{'cAcc':>10}{'jF1':>10}")
    print(hdr); print("-" * len(hdr))
    for t, b in sorted(agg["per_task"].items()):
        print(f"{t:<28}{b['n_samples']:>4}{b['n_gt']:>6}{b['n_emits']:>7}"
              f"{b['time_f1']:>7.3f}{b['content_coverage']:>8.1%}"
              f"{f(b['content_acc']):>10}{f(b['joint_f1']):>10}")
    print("-" * len(hdr))
    print(f"{'OVERALL':<28}{o['n_samples']:>4}{o['n_gt']:>6}{o['n_emits']:>7}"
          f"{o['time_f1']:>7.3f}{o['content_coverage']:>8.1%}"
          f"{f(o['content_acc']):>10}{f(o['joint_f1']):>10}")
    if not o["content_complete"]:
        print(f"\n{o['n_unjudged']}/{o['n_matched']} matched emits still UNJUDGED "
              f"— content_acc/joint_f1 withheld on purpose. Lower bounds: "
              f"cAcc>={o['content_acc_lb']:.3f} jF1>={o['joint_f1_lb']:.3f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many judge calls are needed, make none")
    ap.add_argument("--rescore-only", action="store_true",
                    help="recompute from the cache only; never call the API")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="stop after this many new judgements (daily quota)")
    ap.add_argument("--backend", choices=["openai", "gemini"], default="openai",
                    help="judge backend (default openai; the paper's gemini needs "
                         "quota we do not have)")
    ap.add_argument("--batch", choices=["submit", "fetch"], default=None,
                    help="use the OpenAI Batch API (50%% cheaper, 24h SLA)")
    ap.add_argument("--batch-id", default=None,
                    help="fetch this batch id (default: newest in the run dir)")
    ap.add_argument("--batch-wait", type=float, default=0.0,
                    help="seconds to poll while fetching before giving up (bounded)")
    ap.add_argument("--out", default="online_metrics_judged.json")
    args = ap.parse_args()

    records = load_preds(args.run_dir)
    triples = needed_triples(records)
    judge = metrics.ContentJudge(backend=args.backend)
    print(f"{args.run_dir}: {len(records)} samples | backend={args.backend} "
          f"mode={judge.mode} model={judge._model} | "
          f"cache={len(judge._cache)} verdicts")

    if args.dry_run:
        todo = [t for t in triples if judge._cache_key(*t) not in judge._cache]
        print(f"\n{len(triples)} triples needed, {len(todo)} NOT cached "
              f"-> {len(todo)} API calls required")
        report(records, judge, "current state (no calls made)")
        return

    if args.batch == "submit":
        if judge.mode == "unavailable":
            print("\nNo judge configured. Set OPENAI_API_KEY (or GEMINI_API_KEY) "
                  "in your environment.")
            return
        batch_submit(args.run_dir, triples, judge, args.max_calls)
        return                      # results arrive later; nothing to rescore yet

    if not args.rescore_only:
        if judge.mode == "unavailable":
            print("\nNo judge configured. Set OPENAI_API_KEY (or GEMINI_API_KEY) "
                  "in your environment.")
            return
        if args.batch == "fetch":
            batch_fetch(args.run_dir, judge, args.batch_id, args.batch_wait)
        else:
            # the openai backend already retries 429/5xx internally with backoff,
            # so the outer loop must not multiply that into 30 calls per triple
            new, failed, _ = judge_all(
                triples, judge, args.max_calls,
                max_retries=1 if judge.mode == "openai" else 6)
            print(f"\njudged {new} new | {failed} still failing after retries")

    agg = report(records, judge, f"after judging ({args.run_dir})")
    out = os.path.join(args.run_dir, args.out)
    with open(out, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
