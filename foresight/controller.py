"""
controller.py — the PURE-GENERATIVE proactivity controller.

Replaces the fixed-cadence input/output gates with a single agentic loop. The
controller IS the writer: because it reads the shared KV cache (via an MVCC
snapshot) it already knows what's happening, so it acts as the orchestrator.

Each cycle:
  1. wait until video time reaches the self-scheduled next-check point;
  2. snapshot the shared cache and generate ONE control JSON, e.g.
       {"seen": "...", "have_enough_info": true, "event_time_s": 41,
        "answer": "The target event just occurred.", "fps": 3,
        "next_check_s": 1.0, "question_for_next": "did the door close?"}
  3. apply it: steer encoder fps (input gate); if have_enough_info (and not a
     repeat of an onset already reported), emit the answer to the user (output
     gate + writer in one); schedule the next check at vt + next_check_s.
  4. loop — go back to reading the incoming video until the next check.

Over-firing is minimized by the `ev0` onset-identity dedup + the model's chosen
cadence, plus a local last-answer guard. It never writes back to the primary cache
(single-writer invariant preserved; "no writer cache").
"""
import importlib.util
import json
import math
import os
import re
import time

import torch

from util import log
from event_identity import already_reported, event_key, identity_prompt

# COMPACTION ADMISSION GATE — DEFENSIVE IMPORT.
# `compact_now` is a REQUEST from the model; code decides. The decision lives in
# compaction.py as a pure, GPU-free `admit(occupancy, novelty, p_compact, vt, cfg)
# -> (admitted, reason)` so it can be unit-tested without a run. The import is
# guarded because the field is emitted (and its confidence logged) whether or not
# the consumer has landed — but note that `cfg.plan_compact` defaults to False, so
# on the default path we neither ask for it nor act on it: no field is written
# ahead of its reader.
try:
    from .compaction import admit as _compact_admit
except ImportError:                     # noqa: BLE001 - also covers a flat import
    try:
        from compaction import admit as _compact_admit
    except ImportError:
        _compact_admit = None

# =============================================================================
# Persistent task memory: `count` and `phase`
# =============================================================================
# `count` -- the running total for the counting tasks. Those tasks are scored on
#   an integer the model must carry ACROSS ticks; the accumulator is fed back
#   into the prompt so the model counts incrementally instead of re-deriving the
#   total from `reported` text on every tick.
#
# `phase` -- the current-state label for realtime_state_monitor. That task asks
#   "tell me when the state CHANGES", which is undefined without a memory of the
#   state last reported; `phase` is that memory and is likewise fed back.
#
# Both are task-conditional schema slots (see _plan_fields) and each has a live
# consumer in the prompt. Design rule: a field is only written if something
# reads it.
# =============================================================================


def _sample(logits, prev_ids, cfg, gen=None):
    """Sample one token id using the controller preset: repetition_penalty +
    presence_penalty over already-generated tokens, then temperature / top-k /
    top-p. cfg.writer_greedy -> pure argmax. `gen` is a seeded torch.Generator
    for reproducible sampling (cfg.writer_seed)."""
    logits = logits.clone()

    # penalties over already-generated tokens — VECTORIZED (a few kernels) instead
    # of a Python loop of per-index scalar writes, so it stays cheap on the GPU.
    if prev_ids and (cfg.writer_repetition_penalty != 1.0 or cfg.writer_presence_penalty != 0.0):
        idx = torch.tensor(sorted(set(prev_ids)), device=logits.device, dtype=torch.long)
        if cfg.writer_repetition_penalty != 1.0:
            v = logits[idx]
            logits[idx] = torch.where(v > 0, v / cfg.writer_repetition_penalty,
                                      v * cfg.writer_repetition_penalty)
        if cfg.writer_presence_penalty != 0.0:
            logits[idx] -= cfg.writer_presence_penalty

    if cfg.writer_greedy or not cfg.writer_temperature or cfg.writer_temperature <= 0:
        return int(torch.argmax(logits).item())     # GPU argmax; only the id crosses

    logits = logits / cfg.writer_temperature

    if cfg.writer_top_k and cfg.writer_top_k > 0:
        k = min(cfg.writer_top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    if cfg.writer_top_p and 0 < cfg.writer_top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        drop = (cum - sp) > cfg.writer_top_p
        sp[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, 1, generator=gen).item())


def _cache_to(cache, device):
    """Move a cloned cache's K/V tensors onto `device` (the controller's GPU).
    Handles transformers 5.x (`.layers`) and 4.x (`.key_cache`) layouts."""
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is None:
                continue
            layer.keys = layer.keys.to(device, non_blocking=True)
            layer.values = layer.values.to(device, non_blocking=True)
    else:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].to(device, non_blocking=True)
            cache.value_cache[i] = cache.value_cache[i].to(device, non_blocking=True)
    return cache


def _extract_json(text):
    """Pull the first flat {...} object out of the model's text; {} on failure."""
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _clamp(x, lo, hi, default):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


def _single_tok(tok, s):
    """Token id for a surface form that MUST be one token; None otherwise."""
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _read_bool(logits, true_id, false_id):
    """P(true) restricted to {true,false} at the CURRENT position — a logit read,
    not a decode. Costs zero extra forward passes: these logits were already
    produced by the forward that force-fed the key. Returns a CONTINUOUS
    confidence, which is what lets the tuned Schmitt/hysteresis gate apply here at
    all (a decoded 'true'/'false' token gives only a hard bit)."""
    lt, lf = float(logits[true_id]), float(logits[false_id])
    m = max(lt, lf)
    et, ef = math.exp(lt - m), math.exp(lf - m)
    return et / (et + ef)


def _read_choice(logits, ids, values):
    """Softmax restricted to a small set of SINGLE-TOKEN surface forms; returns
    (value, p, dist). Same cost as _read_bool: zero forwards beyond the forced key.

    This is the generalisation of _read_bool from {true,false} to any small choice
    set, and it is what promotes `fps` and `next_check_s` out of the `more` tail.
    Both were already _clamp-ed to bounded ranges (1..3 fps, 0.2..1.5 s), so a free
    decode of them never bought anything a restricted read cannot — and the read
    turns them from "~60-token tail, 2-3% reachable" into "100% of ticks, zero
    decodes, plus a calibrated confidence" (`p_fps` / `p_cadence`, free telemetry).

    Numerics deliberately mirror _read_bool exactly: pull the handful of logits we
    care about across as python floats, subtract the max for stability, normalise
    over the choice set only. No full-vocabulary softmax is ever materialised."""
    ls = [float(logits[i]) for i in ids]
    m = max(ls)
    es = [math.exp(x - m) for x in ls]
    z = sum(es)
    ps = [e / z for e in es]
    k = max(range(len(ps)), key=ps.__getitem__)
    return values[k], ps[k], {v: p for v, p in zip(values, ps)}


def _choice_ids(tok, values, fmts):
    """Token ids + surface forms for a choice set, or (None, None) if NO candidate
    formatting makes every option a single token.

    ALL-OR-NOTHING on purpose. If one option of {1,2,3} split into two tokens, a
    restricted read over the survivors would put probability zero on it silently —
    the model could never ask for that value again and nothing in the log would say
    so. `fmts` is tried in order so a splitting surface form can be swapped for an
    equivalent one ("1.0" -> "1") before we fall back to a decode; the caller is
    responsible for warning LOUDLY when this returns None (never silently)."""
    for fmt in fmts:
        texts = [fmt(v) for v in values]
        ids = [_single_tok(tok, t) for t in texts]
        if all(i is not None for i in ids):
            return ids, texts
    return None, None


def _choice_slot(step, b, cfg, gen, logits, values, spec, cap):
    """Fill a bounded-choice value slot. Returns (value, p, dist, pend, n_dec, logits).

    Fast path (`spec` present): a restricted logit read — ZERO decodes — and `pend`
    is the winning surface form, which the caller prefixes to the next forced key so
    the JSON stays well formed by construction (the same trick the `true`/`false`
    literal already uses).

    Fallback (`spec` is None because the tokenizer splits a surface form): a capped
    decode, snapped to the nearest legal choice. It costs tokens and it is loud at
    startup, but it keeps the field in the spine rather than dropping it."""
    if spec is not None:
        ids, texts = spec
        v, p, dist = _read_choice(logits, ids, values)
        return v, p, dist, texts[list(values).index(v)], 0, logits
    txt, tids, logits = _decode_until(step, b, logits, cfg, gen, ',"}', cap)
    try:
        x = float(txt.strip())
    except (TypeError, ValueError):
        return None, None, None, "", len(tids), logits
    v = min(values, key=lambda c: abs(float(c) - x))
    return v, None, None, "", len(tids), logits


# ---------------------------------------------------------------------------
# per-task schema
# ---------------------------------------------------------------------------
# Every key the walk can force, IN WALK ORDER. `objects`/`keep`/`ignore` are
# sub-slots of `replan_vision` and are listed with it rather than here.
_PLAN_FIELDS = ("seen", "have_enough_info", "event_time_s", "answer", "fps",
                "next_check_s", "compact_now", "replan_vision", "question_for_next",
                "count", "phase")


def _time_only_tasks():
    """The set of tasks whose content is never scored, READ from the scorer.

    metrics.py:TIME_ONLY is derived from TASK_CONTENT_KIND, which is itself copied
    from the OmniPro reference scorer. Re-listing the two task names here would
    create a second copy that can drift — and a drifted copy would silently delete
    the `answer` field from a task that IS content-scored, which is the one failure
    mode of this whole change that zeroes a task without a crash. So it is imported,
    three ways (package / flat / by path, since foresight is run both as a
    package and with its own directory on sys.path), and returns None — bypass
    DISABLED, answer kept — if none of them work."""
    try:
        from evaluation.metrics import TIME_ONLY
        return set(TIME_ONLY)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        from metrics import TIME_ONLY                   # evaluation/ on sys.path
        return set(TIME_ONLY)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "evaluation", "metrics.py")
        spec = importlib.util.spec_from_file_location("_omnipro_metrics", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return set(mod.TIME_ONLY)
    except Exception:                                   # noqa: BLE001
        return None


def _task_name(cfg):
    """The current sample's task name, or "" if it cannot be established.

    cfg.task is the direct channel. The eval adapter does not set it yet (it injects
    only question/event/video_path), but it DOES select the per-task ICL prompt out
    of cfg.task_controller_prompts — so the prompt identity recovers the name with no
    second table to maintain and no edit to the adapter."""
    t = (getattr(cfg, "task", "") or "").strip()
    if t:
        return t
    prompt = getattr(cfg, "controller_prompt", None)
    for name, text in (getattr(cfg, "task_controller_prompts", None) or {}).items():
        if text == prompt:
            return name
    return ""


def _plan_fields(cfg, task, time_only_tasks):
    """WHITELIST the fields this task's schema walk may force; returns (set, time_only).

    A whitelist, never a blacklist: a field
    added to the spine later must not leak into a task by default. The unknown-task
    path is deliberately the PERMISSIVE one — an unrecognised task keeps `answer`,
    because dropping it there would silently zero a content-scored task, whereas
    keeping it merely wastes tokens on a time-only one."""
    time_only = bool(cfg.time_only_bypass and time_only_tasks is not None
                     and task in time_only_tasks)
    f = {"seen", "have_enough_info", "event_time_s"}
    if not time_only:
        f.add("answer")
    if cfg.plan_fps:
        f.add("fps")
    if cfg.plan_cadence:
        f.add("next_check_s")
    if cfg.plan_compact:
        f.add("compact_now")
    if cfg.plan_classes:
        f.add("replan_vision")
    if cfg.plan_question:
        f.add("question_for_next")
    if cfg.plan_count and task in ("snapshot_counting", "cumulative_counting",
                                   "dedup_counting"):
        f.add("count")
    if cfg.plan_phase and task == "realtime_state_monitor":
        f.add("phase")
    return frozenset(f), time_only


def _decode_until(step, b, logits, cfg, gen, stop_chars, max_tokens):
    """Sample a VALUE slot until a stop char appears (or the cap is hit).

    The stop token is deliberately NOT fed back into the cache — the caller
    force-feeds the next literal (which begins with that same delimiter) instead,
    so the JSON stays well-formed by construction."""
    ids, text = [], ""
    for _ in range(max_tokens):
        logits[b.eos_id] = float("-inf")
        tid = _sample(logits, ids, cfg, gen)
        piece = b.tok.decode([tid])
        stops = [c for c in stop_chars if c in piece]
        if stops:
            text += piece[:min(piece.index(c) for c in stops)]
            return text, ids, logits
        ids.append(tid)
        text += piece
        logits = step(b.embed_token(tid))
    return text, ids, logits


def _schema_tick(b, cfg, gen, step, prompt, ids_bool, ids_choice=None,
                 fields=None, ev_seen=None, progress=None):
    """SCHEMA-WALKED DECODE — the fix for "the diff never actually diffs".

    Instead of handing the model an open brace and hoping it obeys prose rules
    ("omit fps unless it changed" — measured 0/15 compliance), the CODE walks a
    fixed skeleton and the model only fills value slots. The spine, with its
    per-tick decode cost:

        force  {"seen":"                 4 tokens, ONE forward (prefill is parallel)
        SAMPLE <seen text>               <=12 decodes, cfg.seen_mode
        force  ","have_enough_info":
        READ   P(true)                   0 decodes, continuous confidence
        [if hit] force ,"event_time_s":  SAMPLE <=4
                 [ev0: same onset already reported -> STOP HERE, no answer]
                 force ,"answer":"       SAMPLE <=32   (whitelist: not on time_only)
        force  ,"fps":                   READ CHOICE {1,2,3}          0 decodes
        force  ,"next_check_s":          READ CHOICE {0.5,1.0,1.5}    0 decodes
        force  ,"compact_now":           READ bool                    0 decodes
                force  ,"replan_vision":         READ bool                    0 decodes
                    [if replan] ,"keep":" / ,"ignore":"   SAMPLE, replan ticks only
        force  ,"question_for_next":"    SAMPLE <=8    <- the ONLY costly addition

    Prefill of k tokens costs ONE forward; decoding k tokens costs k forwards. So
    every forced token is ~free and every restricted read is FREE — which is why
    four figure fields could be promoted out of the `more` tail (where they filled
    2.08% / 3.13% of 199,909 ticks, or did not exist at all) for zero decode cost.
    `question_for_next` is the one addition that is genuinely paid for in tokens
    (~8/tick); it has its own flag, cfg.plan_question, precisely so its cost can be
    measured against its benefit on its own (ablation arm L1_noQ).

    `fields` is the per-task WHITELIST from _plan_fields(); `ev_seen` is the set of
    onsets already reported (the ev0 dedup key). Returns (diff, meta)."""
    true_id, false_id = ids_bool
    fields = _PLAN_FIELDS if fields is None else fields
    ids_choice = ids_choice or {}
    diff, meta = {}, {}
    n_dec = 0

    # ---- seen: forced key, sampled value (look BEFORE judging, every tick) ----
    # SEEN_MODE is an ablation: does the hit read actually
    # NEED `seen` decoded first, or does only the ANSWER need it?
    #   "before" -- current: describe the scene, THEN read the level (~1.3s/tick)
    #   "off"    -- read the level immediately, no decode at all (~0.15s/tick)
    #   "after"  -- read the level FIRST, then describe. Separates "does the
    #               perception step help?" from "does its ORDER matter?"
    # `seen` took F1 from 0.0 to 0.255 when it was introduced, so this is not a
    # refactor to be assumed safe — it is measured.
    if cfg.seen_mode == "before":
        logits = step(b.embed_text(prompt + '{"seen":"'))
        seen, sids, logits = _decode_until(step, b, logits, cfg, gen,
                                           '"', cfg.schema_max_seen_tokens)
        n_dec += len(sids)
        diff["seen"] = seen.strip()
        logits = step(b.embed_text('","have_enough_info":'))
    else:
        logits = step(b.embed_text(prompt + '{"have_enough_info":'))
    p_hit = _read_bool(logits, true_id, false_id)
    hit = p_hit >= cfg.hit_threshold
    diff["have_enough_info"] = hit
    meta["p_hit"] = p_hit
    if progress is not None:
        progress("bool_ready")
    if cfg.verify_logit_read:
        # VERIFICATION: what would a FREE decode have produced here?
        # If the unrestricted argmax is not a boolean at all, the logit read is
        # imposing structure the model did not intend — we must know that before
        # trusting this path. Logged every tick, costs nothing.
        top = int(torch.argmax(logits).item())
        meta["argmax_tok"] = b.tok.decode([top])
        meta["argmax_is_bool"] = top in (true_id, false_id)
        meta["argmax_agrees"] = (top == (true_id if hit else false_id))

    lit = "true" if hit else "false"

    # seen_mode="after": the level is already read; NOW describe the scene. If
    # this scores like "before", the perception step helps by existing; if it
    # scores like "off", the ORDER is what mattered.
    if cfg.seen_mode == "after":
        logits = step(b.embed_text(lit + ',"seen":"'))
        seen, sids, logits = _decode_until(step, b, logits, cfg, gen,
                                           '"', cfg.schema_max_seen_tokens)
        n_dec += len(sids)
        diff["seen"] = seen.strip()
        lit = '"'                       # we are mid-string; close it, not a bool

    # ---- hot path only: onset time + answer -----------------------------------
    # `pend` is the literal that must be written into the cache before the NEXT
    # forced key: either the value of a slot we only READ (so it was never fed
    # back), or the quote that closes a string slot we sampled. It is always
    # prefixed to the next key's forced literal, so it rides the same forward and
    # costs nothing, and the emitted JSON stays well formed by construction.
    pend = lit
    if hit:
        logits = step(b.embed_text(pend + ',"event_time_s":'))
        t_txt, tids, logits = _decode_until(step, b, logits, cfg, gen,
                                            ',"}', cfg.schema_max_int_tokens)
        n_dec += len(tids)
        pend = ""
        ev = None
        try:
            ev = float(t_txt.strip())
            if event_key(ev) is not None:
                diff["event_time_s"] = ev
            else:
                ev = None
        except (TypeError, ValueError):
            pass
        meta["event_time_valid"] = ev is not None
        if progress is not None:
            progress("event_id_ready")
        # ---- ev0 dedup: onset IDENTITY, not word overlap --------
        # "If the model reports an event time it has already spoken about, it is
        # the same occurrence — stay quiet." Measured +0.053 macro time-F1 with NO
        # free parameter, and it beats a plain refractory timer at a LARGER emission
        # budget, so `event_time_s` is carrying real information rather than any
        # dedup helping. Crucially it fires HERE, before the ~32-token answer
        # decode, which is both the token saving (636,096 answer tokens never
        # decoded in replay) and what makes the answer-free time_only path possible.
        # ⚠️ CAVEAT for the writeup: that number is an OFFLINE replay
        # (retime.py --dedup). Suppressing an emission changes `reported`, hence the
        # next prompt, hence every later tick — a real run must confirm it.
        dup = bool(cfg.ev0_dedup and already_reported(
            ev, ev_seen, getattr(cfg, "event_dedup_window_s", 0.0)))
        meta["dup_ev"] = dup
        if "answer" in fields and not dup:
            logits = step(b.embed_text(',"answer":"'))
            ans, aids, logits = _decode_until(step, b, logits, cfg, gen,
                                              '"', cfg.schema_max_answer_tokens)
            n_dec += len(aids)
            diff["answer"] = ans.strip()
            pend = '"'
            if progress is not None:
                progress("answer_ready")

    # ---- the PLAN proper: four restricted reads, zero decodes -----------------
    # Everything in this block used to live behind `more` (or nowhere at all).
    # Each field is gated on the per-task whitelist, which is gated on its own
    # cfg.plan_* flag, so every one of them is independently ablatable — that is
    # the purpose of the ablation matrix.
    if "fps" in fields:
        logits = step(b.embed_text(pend + ',"fps":'))
        v, p, dist, pend, nd, logits = _choice_slot(
            step, b, cfg, gen, logits, cfg.fps_choices, ids_choice.get("fps"),
            cfg.schema_max_int_tokens)
        n_dec += nd
        if v is not None:
            diff["fps"] = v
            meta["p_fps"] = p
            meta["dist_fps"] = dist
    if "next_check_s" in fields:
        logits = step(b.embed_text(pend + ',"next_check_s":'))
        v, p, dist, pend, nd, logits = _choice_slot(
            step, b, cfg, gen, logits, cfg.cadence_choices,
            ids_choice.get("cadence"), cfg.schema_max_int_tokens)
        n_dec += nd
        if v is not None:
            diff["next_check_s"] = v
            meta["p_cadence"] = p
            meta["dist_cadence"] = dist
    # compact_now / replan_vision are REQUESTS, not commands. The controller reads
    # the confidence here and the CODE decides (admission gate / class pruner);
    # both default OFF at the config, since a field should not be acted on until
    # its consumer is in place.
    # 0.5 is the plain argmax of the restricted read — no new threshold knob, since
    # the continuous p_* is what the consumers actually take.
    if "compact_now" in fields:
        logits = step(b.embed_text(pend + ',"compact_now":'))
        p = _read_bool(logits, true_id, false_id)
        meta["p_compact"] = p
        want = p >= 0.5
        diff["compact_now"] = want
        pend = "true" if want else "false"
    if "replan_vision" in fields:
        logits = step(b.embed_text(pend + ',"replan_vision":'))
        p = _read_bool(logits, true_id, false_id)
        meta["p_replan"] = p
        want = p >= 0.5
        diff["replan_vision"] = want
        pend = "true" if want else "false"
        if want and cfg.plan_classes:
            # The encoder-side pruner consumes these lists. Decode them only on
            # replan ticks so class control adds no steady-state decode cost.
            for key, cap in (("keep", getattr(cfg, "schema_max_class_tokens", 3)),
                             ("ignore", getattr(cfg, "schema_max_class_tokens", 3))):
                logits = step(b.embed_text(pend + f',"{key}":"'))
                txt, cids, logits = _decode_until(step, b, logits, cfg, gen, '"', cap)
                n_dec += len(cids)
                diff[key] = [w.strip() for w in txt.split(",") if w.strip()]
                pend = '"'
    # ---- question_for_next: the one addition with a real token cost -----------
    # ~8 SAMPLED tokens on EVERY tick (not just hit ticks) — the deferred-question
    # mechanism exists precisely to be set while quiet, and at 4.85% fill it has
    # never had a fair test. Its consumer is live (spliced into the next prompt).
    if "question_for_next" in fields:
        logits = step(b.embed_text(pend + ',"question_for_next":"'))
        q, qids, logits = _decode_until(step, b, logits, cfg, gen,
                                        '"', cfg.schema_max_question_tokens)
        n_dec += len(qids)
        diff["question_for_next"] = q.strip()
        pend = '"'

    # ---- count: capped integer decode for counting tasks -----------------
    if "count" in fields:
        logits = step(b.embed_text(pend + ',"count":'))
        cnt_txt, cnt_ids, logits = _decode_until(step, b, logits, cfg, gen,
                                                  ',"}', cfg.schema_max_count_tokens)
        n_dec += len(cnt_ids)
        pend = ""
        try:
            diff["count"] = int(cnt_txt.strip())
        except (TypeError, ValueError):
            pass

    # ---- phase: short string decode for realtime_state_monitor -----------
    if "phase" in fields:
        logits = step(b.embed_text(pend + ',"phase":"'))
        ph_txt, ph_ids, logits = _decode_until(step, b, logits, cfg, gen,
                                                '"', cfg.schema_max_phase_tokens)
        n_dec += len(ph_ids)
        diff["phase"] = ph_txt.strip()
        pend = '"'

    meta["n_decode"] = n_dec
    return diff, meta


def _word_sim(a, b):
    """Jaccard word overlap in [0,1]; rewordings of the SAME occurrence score high
    ('reports 80 dead' vs 'now reports 80 dead'), different occurrences score low
    ('match date August 14th' vs 'ticket costs and purchase website')."""
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _clean_class_plan(keep, ignore):
    def clean(values):
        out = []
        for value in values or []:
            label = str(value).strip().lower()
            if not label or label in {"-", "none", "null", "n/a"} or label in out:
                continue
            out.append(label)
        return out

    keep = clean(keep)
    return keep, [label for label in clean(ignore) if label not in keep]


def controller_thread(cfg, mgr, ctrl, clock, stop, prof=None, evaluator=None, wb=None,
                      feed_done=None):
    b = wb if wb is not None else mgr.b
    cross_gpu = b.device != mgr.b.device
    # generator on the backend's device: logits now stay on GPU, so multinomial
    # sampling (non-greedy path) needs a matching-device generator.
    try:
        gen = torch.Generator(device=b.device).manual_seed(int(cfg.writer_seed))
    except (RuntimeError, TypeError):
        gen = torch.Generator().manual_seed(int(cfg.writer_seed))
    log("controller", 0.0, "model-scheduled proactivity ON (pure-generative control loop)")
    _tick_failures = 0          # consecutive failed ticks
    _MAX_TICK_FAILURES = 5      # give up on the SAMPLE, not the shard

    # ---- snapshot vs in-place (cfg.controller_cache_mode) ---------------------
    # "inplace" removes the per-tick deep copy of the KV cache. It is only sound
    # when nothing else can mutate the primary while we generate, and when the
    # primary is on OUR device. Both conditions are checked here and a failure
    # DOWNGRADES LOUDLY -- a silent fallback would make a latency/memory result
    # unattributable, and a silent non-fallback would corrupt the cache.
    use_inplace = (cfg.controller_cache_mode == "inplace")
    if use_inplace and not cfg.deterministic:
        log("controller", 0.0, "WARNING: controller_cache_mode=inplace requires "
                               "deterministic=True (lockstep) -- a free-running "
                               "ingester would append inside the borrow. Falling "
                               "back to snapshot.")
        use_inplace = False
    if use_inplace and cross_gpu:
        log("controller", 0.0, "WARNING: controller_cache_mode=inplace cannot cross "
                               "GPUs (the primary lives on the manager's device). "
                               "Falling back to snapshot.")
        use_inplace = False
    log("controller", 0.0, f"cache_mode={'inplace' if use_inplace else 'snapshot'}")

    next_check_vt = cfg.probe_min_s           # skip the empty-cache tick at t=0
    reported = []                              # conversation history: (vt, answer) already emitted
    pending_q = ""                             # question_for_next: what to verify on the next tick
    # DIFF-MERGE: keep a persistent control config; the model emits only the fields
    # that CHANGE each tick (a compact JSON diff), so it rarely decodes the string
    # fields -> much less latency. Transient fields (have_enough_info/answer/seen/
    # event_time_s) RESET to default every tick and must be re-asserted.
    #
    # `count` and `phase` are persistent task accumulators (see the module-level
    # note above); unlike the transient fields they carry across ticks.
    state = {"fps": cfg.encoder_idle_fps, "next_check_s": cfg.probe_default_s,
             "have_enough_info": False, "answer": "",
             "question_for_next": "", "seen": "", "event_time_s": None,
             "compact_now": False, "replan_vision": False,
             "keep": [], "ignore": [],
             "count": 0, "phase": ""}          # persistent task memory
    # Dedup is handled by `ev0` (onset identity), below.
    seen_trace = []                            # (vt, seen) — the perception trace;
                                               # consecutive duplicates collapsed
    reported_ev = set()                        # ev0 key: onsets already spoken about
    # true/false must be SINGLE tokens for the logit read; verified for Qwen3-VL
    # ('true'->1866, 'false'->3849). Fall back to free decode if a tokenizer splits them.
    ids_bool = (_single_tok(b.tok, "true"), _single_tok(b.tok, "false"))
    use_schema = (cfg.decode_mode == "schema" and None not in ids_bool)
    if cfg.decode_mode == "schema" and not use_schema:
        log("controller", 0.0, "WARNING: true/false are not single tokens -> "
                               "falling back to free decode")
    log("controller", 0.0, f"decode_mode={'schema' if use_schema else 'free'}")

    # ---- the restricted-read choice sets --------------------------------------
    # The blocking check, done at runtime rather than trusted: every surface form
    # in a choice set must be ONE token, exactly the way ids_bool verifies
    # true/false. If a form splits, we do NOT silently continue — that would put a
    # hidden zero on one option — we say so and drop to a capped decode for that
    # field. `1.0` is offered before `1` for cadence so the JSON keeps its float
    # spelling when the tokenizer allows it.
    ids_choice = {}
    if cfg.plan_fps:
        _i, _t = _choice_ids(b.tok, cfg.fps_choices, (lambda v: str(int(v)),))
        ids_choice["fps"] = (_i, _t) if _i else None
    if cfg.plan_cadence:
        _i, _t = _choice_ids(b.tok, cfg.cadence_choices,
                             (lambda v: f"{v:.1f}", lambda v: f"{v:g}"))
        ids_choice["cadence"] = (_i, _t) if _i else None
    for _name, _vals in (("fps", cfg.fps_choices), ("cadence", cfg.cadence_choices)):
        if _name in ids_choice and ids_choice[_name] is None:
            log("controller", 0.0,
                f"WARNING: {_name} choices {tuple(_vals)} are NOT single tokens for "
                f"this tokenizer -> falling back to a capped decode for that field "
                f"(costs tokens; p_{_name} telemetry will be absent)")
        elif _name in ids_choice:
            log("controller", 0.0,
                f"choice_read {_name}: {ids_choice[_name][1]} -> {ids_choice[_name][0]}")

    # ---- the per-task schema --------------------------------------------------
    _time_only_set = _time_only_tasks()
    if _time_only_set is None and cfg.time_only_bypass:
        log("controller", 0.0, "WARNING: could not import metrics.TIME_ONLY -> the "
                               "time-only answer bypass is DISABLED (answer kept on "
                               "every task). Never guess this membership: dropping "
                               "`answer` from a content-scored task zeroes it.")
    task = _task_name(cfg)
    plan_fields, time_only = _plan_fields(cfg, task, _time_only_set)
    if not task:
        log("controller", 0.0, "WARNING: task name unknown (cfg.task unset and the "
                               "controller prompt matches no per-task ICL) -> using "
                               "the FULL field set; `answer` is kept.")
    log("controller", 0.0, f"task={task or '?'} time_only={time_only} "
                           f"schema_fields={sorted(plan_fields)}")
    # LEVEL -> EDGE in CODE (semantic Schmitt gate): the model reports whether the
    # condition is satisfied NOW (a LEVEL it can judge from the cache snapshot);
    # firing happens only on the RISING edge (false -> true across ticks), so a
    # condition that stays on screen cannot re-fire. Within a true stretch, a fire
    # is also allowed when the answer describes a clearly DIFFERENT occurrence
    # (low word overlap with the last fired answer) — e.g. date poster then ticket
    # prices with no gap in between.
    prev_level = False
    armed = True             # hysteresis gate (cfg.gate_strategy): ready to fire
    last_fire_vt = -1e9      # for debounce + timed re-arm
    ema_value, ema_time, ema_armed = None, None, True
    clock.set_next_check(next_check_vt)        # LOCKSTEP: tell the ingester when the
                                               # first tick is due (it waits on this)

    while True:
        vt = clock.get()
        due = vt >= next_check_vt
        # Exit: stop requested AND nothing due. In lockstep the ingester holds the
        # stream while a tick is due, so we must keep servicing ticks during the
        # drain and only leave once the feed is fully done (feed_done set by the
        # ingester). Without feed_done (standalone/async), plain stop suffices.
        if stop.is_set() and not due and (feed_done is None or feed_done.is_set()):
            break
        if not due:                            # not time to check yet — keep reading
            time.sleep(0.005 if cfg.deterministic else 0.05)
            continue

        t0 = time.time()
        # ---- READ THE SHARED CACHE ------------------------------------------
        # Two paths, identical arithmetic (see cfg.controller_cache_mode):
        #   inplace  -> generate on the PRIMARY at (next_pos, len), erase after.
        #   snapshot -> deep-copy the primary and generate on the copy.
        # Both start the generation at the same pos_start/phys_start over the same
        # key/value prefix, so the logits are the same numbers; only the copy is
        # skipped. The try/finally below is load-bearing: if a tick raises, the
        # borrow MUST still be erased or the primary keeps the controller's own
        # prompt tokens and every later frame is conditioned on them.
        if use_inplace:
            pos, phys = mgr.borrow_begin("ctrl.borrow")
            cache = mgr.cache
        else:
            cache, pos, phys = mgr.snapshot_clone()
            if cross_gpu:
                cache = _cache_to(cache, b.device)
        phys0 = phys        # cache occupancy BEFORE our splice — the compaction
                            # admission gate's pressure term (step() mutates phys)
        try:

            def step(embeds):
                nonlocal pos, phys, cache
                logits, cache = b.forward(embeds, cache, pos_start=pos, phys_start=phys)
                if use_inplace:
                    # forward() may hand back a different cache object; the manager
                    # must keep pointing at the one that actually grew, or
                    # borrow_end() would truncate a stale object and leave the
                    # controller's tokens in the primary.
                    mgr.cache = cache
                pos += embeds.shape[1]
                phys += embeds.shape[1]
                return logits

            # Build the controller prompt: task ICL + (optional) deferred check +
            # CONVERSATION HISTORY of what has already been reported, so the model can
            # recognise repeats and only fire on genuinely new details. (It used to be
            # asked for a `new_event` boolean here; that field never existed in the
            # schema walk and is gone — the dedup that actually runs is `ev0` on
            # event_time_s, in code.)
            # With icl_in_sink the ICL was already seeded into the pinned sink by the
            # ingester, so DON'T splice it again here — re-splicing would both duplicate
            # it and re-insert the 1400-token wall between the newest frame and the
            # generation point, which is exactly what we are removing.
            prompt = "" if cfg.icl_in_sink else cfg.controller_prompt.rstrip()
            if cfg.plan_compact:
                prompt += ("\ncompact_now is true only when the cache needs to retain a "
                           "long stream; otherwise false. This request is destructive, "
                           "so choose true only with strong evidence.\n")
            if cfg.plan_classes:
                prompt += ("\nreplan_vision is true only when attention should change. "
                           "When true, keep and ignore are short comma-separated class "
                           "lists for the vision encoder; otherwise leave them empty.\n")
            if pending_q:
                prompt += (f"\nYou previously asked yourself: '{pending_q}'. "
                           f"Judge it now from the MOST RECENT frames.")
            # timestamped history: a LATER onset is a new event, not a repeat of these
            # ---- MEMORY: the writer's own trace, fed back in ----
            # Two halves, both timestamped and append-only:
            #   WHAT I SAW  -- the `seen` trace. Without it `seen` is reset and discarded
            #                  every tick, so the controller has no record of what it
            #                  has already looked at. Consecutive duplicates are
            #                  collapsed (the log showed the same scene repeated 3-4
            #                  ticks running), so this reads as a scene-CHANGE history
            #                  and stays cheap.
            #   WHAT I SAID -- `reported`, code-owned. The model must never be able to
            #                  write this: if it could, it could hallucinate having
            #                  already answered and dedup would silently fail open.
            # Both are bounded rings -> per-tick prompt cost is O(1), not O(stream).
            # This is what makes dedup and accumulation possible at all.
            # FROZEN-PERCEPTION FAILURE MODE (measured). Feeding the `seen` trace
            # back here froze the perception channel: 20 of 58 videos emitted exactly ONE
            # byte-identical scene description for the WHOLE video (578 ticks, 100%), and
            # 31 of 58 emitted <=2 distinct descriptions ever. Failure-mode breakdown over
            # 87 GT triggers: PERCEPTION 81.6%, JUDGMENT 3.4%. A null model using only the
            # timestamp — no pixels at all — BEAT p_hit on 3 of 4 tasks.
            #
            # Mechanism: the last `seen` line sits a few tokens before the `{"seen":"` slot
            # the model must now fill. Under greedy decode with a 12-token cap, copying the
            # adjacent line is the cheapest continuation — and once copied, "repeats
            # collapsed" makes the trace shorter and the copy even more attractive. A
            # self-reinforcing induction loop. The memory feature caused it.
            #
            # Default OFF. To isolate one variable, leave `now_anchor` off as well.
            if cfg.seen_trace_in_prompt:
                if seen_trace:
                    trace = "".join(f"  @{svt:.0f}s {s}\n" for svt, s in seen_trace)
                else:
                    trace = "  (nothing yet)\n"
                prompt += ("\n\nWHAT YOU HAVE SEEN so far (your own observations, newest last; "
                           "repeats collapsed):\n" + trace)
            convo = "".join(f"  @{rvt:.0f}s {a}\n" for rvt, a in reported) or "  (nothing yet)\n"
            prompt += ("\nWHAT YOU HAVE ALREADY TOLD THE USER (past occurrences with their "
                       "times; a fresh onset at a later time is a NEW event — it is only a "
                       "repeat while the SAME occurrence is still on screen):\n" + convo)
            if getattr(cfg, "event_identity_in_prompt", False):
                prompt += identity_prompt(reported_ev)
            # MEMORY: count accumulator for counting tasks
            if "count" in plan_fields:
                prompt += f"\nYour running count from previous ticks: {state['count']}\n"
            # MEMORY: phase for realtime_state_monitor
            if "phase" in plan_fields and state.get("phase"):
                prompt += f"\nYour last recorded state (phase): {state['phase']}\n"

            # Optional PRESENT anchor — the second candidate cause of frozen perception:
            # nothing in the prompt says what time it is NOW or that `seen` must describe
            # the LATEST frame rather than any frame in the cache. Kept OFF by default so
            # it is tested as its own variable, not confounded with seen_trace_in_prompt.
            if cfg.now_anchor:
                prompt += (f"\n\nIt is now {vt:.0f}s. Describe ONLY what is on screen in the "
                           f"MOST RECENT frame, not what you saw earlier.")
            prompt += "\nNow emit ONLY your control JSON for the current stream:\n"

            # PRIME the decoder with an open brace: Qwen3-VL is an instruct model and,
            # spliced as raw text onto the cache (no assistant-turn markers), it would
            # otherwise emit EOS immediately at the splice point. Starting mid-object
            # forces it to complete the JSON. We reconstruct raw = "{" + generated.
            meta = {}
            t_dec0 = time.time()
            if use_schema:
                # SCHEMA WALK: code forces every key/punctuation, model fills only the
                # value slots, booleans come from a logit read. The model can no longer
                # emit fps/next_check_s on the default path -> the diff is a diff.
                #
                # RESILIENCE: these runs share GPUs with other users, and under
                # memory pressure a single tick can raise a transient CUDA/cuDNN
                # fault (observed live: cuDNN's fused MHA graph returning
                # is_good()==false at ~75k KV tokens). Letting that propagate
                # kills the controller THREAD, which silently zeroes the whole
                # sample: the encoder/ingester keep feeding a listener that will
                # never tick again. Swallow the tick instead, drop the cached
                # workspace, and carry on -- one lost tick is a far smaller error
                # than one lost video.
                try:
                    diff, meta = _schema_tick(b, cfg, gen, step, prompt, ids_bool,
                                              ids_choice=ids_choice, fields=plan_fields,
                                              ev_seen=reported_ev,
                                              progress=(lambda event: prof.mark(event, vt))
                                              if prof is not None else None)
                except RuntimeError as _e:
                    _tick_failures += 1
                    log("controller", vt_now,
                        "TICK FAILED (%d/%d) %s: %s" % (
                            _tick_failures, _MAX_TICK_FAILURES,
                            type(_e).__name__, str(_e).split(chr(10))[0][:160]))
                    try:
                        import torch as _t
                        _t.cuda.empty_cache()
                    except Exception:
                        pass
                    if _tick_failures >= _MAX_TICK_FAILURES:
                        log("controller", vt_now,
                            "too many consecutive tick failures — stopping this "
                            "sample so the shard moves on")
                        raise
                    continue
                _tick_failures = 0
                ids = [None] * meta.get("n_decode", 0)   # count only, for telemetry
                raw = json.dumps(diff, separators=(",", ":"))
            else:
                logits = step(b.embed_text(prompt + "{"))
                ids = []
                for _ in range(cfg.controller_max_tokens):
                    # MASK EOS: as an instruct model spliced raw onto the cache, Qwen often
                    # samples the end token as the very first token (-> empty output). We
                    # stop on the closing "}" ourselves, so EOS is never wanted here.
                    logits[b.eos_id] = float("-inf")
                    tok_id = _sample(logits, ids, cfg, gen)
                    ids.append(tok_id)
                    if "}" in b.tok.decode([tok_id]):     # first close -> flat object done
                        break
                    logits = step(b.embed_token(tok_id))
                raw = "{" + b.decode(ids)
                diff = _extract_json(raw)             # the model's DIFF (partial dict); {} = no change
            decode_s = time.time() - t_dec0
            if prof is not None:
                prof.observe("ctrl_decode_s", decode_s)
                if ids:
                    prof.observe("ctrl_decode_ms_per_tok", 1000 * decode_s / len(ids))
                if "p_hit" in meta:
                    prof.observe("ctrl_p_hit", meta["p_hit"])
                    prof.incr("ctrl_argmax_agrees" if meta.get("argmax_agrees")
                              else "ctrl_argmax_disagrees")
                # free telemetry from the restricted reads: these are the calibrated
                # confidences the promoted figure fields now come with (they did not
                # exist while the fields lived behind `more`).
                for _k, _stat in (("p_fps", "ctrl_p_fps"), ("p_cadence", "ctrl_p_cadence"),
                                  ("p_compact", "ctrl_p_compact"), ("p_replan", "ctrl_p_replan")):
                    if meta.get(_k) is not None:
                        prof.observe(_stat, meta[_k])
                if meta.get("dup_ev"):
                    prof.incr("ctrl_ev0_suppressed")
            gen_s = time.time() - t0

            # ---- apply the DIFF onto the persistent config ----
            # reset the transient fields first (they only hold this tick), then merge
            # whatever the model re-stated; persistent fields (fps/next_check_s/question)
            # survive.
            state["have_enough_info"] = False
            state["answer"] = ""
            state["seen"] = ""
            state["event_time_s"] = None
            state["compact_now"] = False
            state["replan_vision"] = False
            for k, v in diff.items():
                key = "question_for_next" if k == "question" else k   # accept legacy key
                # `count` and `phase` live in `state`, so the merge below picks
                # them up automatically.
                if key in state:
                    state[key] = v

            fps = _clamp(state["fps"], cfg.encoder_idle_fps, cfg.encoder_focus_fps,
                         cfg.encoder_idle_fps)
            ctrl.set_fps(fps)
            # Push the plan's class lists to the encoder-side pruner (no-op
            # unless cfg.prune_classes and a ClassPrunerState were wired in).
            _ps = getattr(ctrl, "_pruner_state", None)
            if _ps is not None:
                _keep, _ign = _clean_class_plan(state.get("keep", []),
                                                 state.get("ignore", []))
                if _keep or _ign:
                    _ps.set_classes(_keep, _ign)
            nxt = _clamp(state["next_check_s"], cfg.probe_min_s, cfg.probe_max_s,
                         cfg.probe_default_s)
            next_check_vt = vt + nxt

            level = bool(state["have_enough_info"])   # "condition satisfied NOW"
            answer = (state["answer"] or "").strip()
            pending_q = (state.get("question_for_next") or "").strip()
            dup_ev = bool(meta.get("dup_ev"))         # ev0: this onset was already spoken
            # EMISSION TEXT on time_only tasks. The scorer
            # never reads the content on instant_event_alert / semantic_condition_alert
            # — a time-matched emit is correct regardless of what it says — so a fixed
            # constant CANNOT change the score, and it buys back the ~32-token answer
            # decode on 39-49% of those tasks' ticks. cfg.event is the sample's own
            # monitored condition, which keeps the log readable.
            emit_text = answer
            if time_only:
                emit_text = ((cfg.event or cfg.instruction or "").strip()
                             or "[time-only alert: content not scored]")

            # MEMORY: record what we just saw. Collapse consecutive duplicates so the
            # trace is a scene-CHANGE history rather than one line per tick, then keep
            # only the last `seen_trace_ring` entries so the prompt cost stays O(1).
            seen_now = (state["seen"] or "").strip()
            novelty = 1.0 - (_word_sim(seen_now, seen_trace[-1][1])
                             if (seen_now and seen_trace) else 0.0)
            if seen_now and (not seen_trace or seen_trace[-1][1] != seen_now):
                seen_trace.append((vt, seen_now))
                del seen_trace[:-cfg.seen_trace_ring]

            # ---- COMPACTION: the model REQUESTS, code DECIDES -------------------
            # compact_now is a request; admit() is the pure, GPU-free admission gate
            # (occupancy pressure + novelty + the model's confidence). It is called
            # on EVERY tick the field is in the schema, including ticks where the
            # model did not ask, so the ledger of reasons is complete — the refusal
            # counts are themselves the reportable result if the strict gate turns
            # out never to admit. Nothing here touches the
            # cache: the ingester remains the single writer.
            # Occupancy is taken from `phys0`, the length BEFORE this tick's splice,
            # rather than mgr.occupancy(): in inplace mode the primary currently also
            # holds our own borrowed prompt tokens, which would inflate the pressure
            # signal by exactly the amount we are about to erase.
            if cfg.plan_compact and "p_compact" in meta:
                occ = min(1.0, phys0 / float(cfg.kv_budget)) if cfg.kv_budget else 0.0
                if _compact_admit is None:
                    log("ctrl.compact", vt, f"[{cfg.video_id or '?'}] REFUSED "
                                            f"reason=no_consumer occ={occ:.3f}")
                    if prof is not None:
                        prof.incr("ctrl_compact_refused")
                else:
                    ok, why = _compact_admit(occ, novelty, meta["p_compact"], vt, cfg)
                    if ok:
                        mgr.request_compaction(vt, occ, novelty, meta["p_compact"])
                    if ok or bool(state.get("compact_now")):
                        log("ctrl.compact", vt,
                            f"[{cfg.video_id or '?'}] {'ADMITTED' if ok else 'REFUSED'} "
                            f"reason={why} occ={occ:.3f} novelty={novelty:.3f} "
                            f"p_compact={meta['p_compact']:.3f}")
                    if prof is not None:
                        prof.incr("ctrl_compact_admitted" if ok
                                  else f"ctrl_compact_refused_{why}")

            # ---- FIRE DECISION -------------------------------------------------
            # "edge"       : rising edge of the boolean level (original).
            # "hysteresis" : Schmitt gate on the CONTINUOUS p_hit — only possible now
            #   that the logit read gives a real number instead of a bit. This is the
            #  tuned "hyst2b" from the probe-gate arm), which lifted
            #   joint_f1 0.149 -> 0.206 there. Aimed at PRECISION: measured 0.112, i.e.
            #   89% of emits were false positives (206 emits for 84 GT).
            #     fire   when armed AND p_hit >= high AND debounce elapsed
            #     re-arm when p_hit < low  OR  rearm_s elapsed since the last fire
            # A single threshold cannot do this: it re-fires on every jitter across the
            # line, which is what the raw level did.
            #
            # `has_text` REPLACES the bare `bool(answer)` conjunct. On a
            # time_only task there is no answer to be had, so requiring one would make
            # the task emit NOTHING AT ALL — the silent-zero trap. There the gate is
            # p_hit + arming + debounce alone, which is exactly what the scorer scores.
            has_text = bool(answer) or time_only
            rising = level and not prev_level
            # `distinct` stays word-overlap based, and stays OFF for time_only tasks:
            # there the texts are a constant, so every pair would look identical and
            # the "different occurrence" escape could never open. ev0 is the dedup on
            # that path (and it is the earlier, cheaper one on every path).
            distinct = (not time_only and level and prev_level and answer and reported
                        and _word_sim(answer, reported[-1][1]) < cfg.distinct_sim_thr)
            p_hit = meta.get("p_hit")
            if cfg.gate_strategy == "hysteresis" and p_hit is not None:
                if not armed and (p_hit < cfg.gate_low_thr
                                  or (cfg.gate_rearm_s > 0
                                      and (vt - last_fire_vt) >= cfg.gate_rearm_s)):
                    armed = True
                fire = has_text and armed and p_hit >= cfg.gate_high_thr \
                    and (vt - last_fire_vt) > cfg.debounce_s
                if fire and dup_ev:
                    fire = False          # ev0 veto, explicit rather than implied
                if fire:
                    armed = False
            elif cfg.gate_strategy == "level":
                fire = has_text and not dup_ev and level \
                    and (vt - last_fire_vt) > cfg.debounce_s
            elif cfg.gate_strategy == "strict_edge":
                fire = has_text and not dup_ev and rising \
                    and (vt - last_fire_vt) > cfg.debounce_s
            elif cfg.gate_strategy == "ema_edge" and p_hit is not None:
                now = time.monotonic()
                tau = cfg.trigger_ema_tau_s
                if not math.isfinite(tau) or tau <= 0:
                    raise ValueError("trigger_ema_tau_s must be finite and positive")
                alpha = 1.0 if ema_time is None else -math.expm1(-(now - ema_time) / tau)
                ema_value = p_hit if ema_value is None else ema_value + alpha * (p_hit - ema_value)
                ema_time = now
                if ema_value < cfg.hit_threshold:
                    ema_armed = True
                fire = has_text and not dup_ev and ema_armed and ema_value >= cfg.hit_threshold \
                    and (vt - last_fire_vt) > cfg.debounce_s
                if fire:
                    ema_armed = False
            else:
                fire = has_text and not dup_ev and (rising or distinct) \
                    and (vt - last_fire_vt) > cfg.debounce_s
            if fire:
                last_fire_vt = vt

            # log EVERY tick's raw diff so all responses are inspectable in the log file
            #
            # NO CAP. A `[:240]` truncation here would silently cut ~8.5% of ticks
            # (16,995 of 199,909 on a full development run -- 100% of them landing EXACTLY at
            # the cap, which is how we know it was truncation and not the model).
            # The cut was selective: it ate the LONGEST emissions, and the longest
            # emissions are precisely the ticks that used the `more` tail
            # (fps/next_check_s/note/count/phase), so every tail-field measurement
            # taken off these logs was biased low -- 57% of event_narration ticks,
            # 20% of sequential_step_instruction. The SYSTEM was never affected: the
            # full diff is applied to `state` above regardless. Only our ability to
            # see it afterwards was clipped.
            #
            # Newlines are ESCAPED, not dropped. Every consumer of these logs
            # (fields.py, auc.py, compliance.py) reads one tick per line, so a raw
            # newline from the free-decode path would silently split one record into
            # two. Escaping keeps the log both lossless and one-line-per-tick. Only
            # literal CR/LF are touched: in schema mode `raw` is json.dumps output,
            # which already encodes newlines as the two characters \ and n, so this
            # leaves schema-mode lines byte-for-byte parseable by json.loads.
            vid = cfg.video_id or "?"
            _raw = raw.strip() if diff else f"NO-DIFF raw={raw!r}"
            log("ctrl.raw", vt,
                f"[{vid}] " + _raw.replace("\r", "\\r").replace("\n", "\\n"))
            # p_hit is the CONTINUOUS confidence from the logit read; agree= is the
            # verification that a free decode would have produced the same boolean.
            extra = ""
            if meta:
                extra = f" p_hit={meta.get('p_hit', float('nan')):.3f}"
                # one confidence per restricted read — the free telemetry the
                # promoted figure fields now come with. Emitted as `key=value` in the
                # tail, which fields.py scans by NAME (_GATE_TAIL), so the order here
                # is free to change and a missing key just means that field is off.
                for _k in ("p_fps", "p_cadence", "p_compact", "p_replan"):
                    if meta.get(_k) is not None:
                        extra += f" {_k}={meta[_k]:.3f}"
                if cfg.verify_logit_read and "argmax_agrees" in meta:
                    extra += (f" agree={meta['argmax_agrees']}"
                              f" argmax={meta['argmax_tok']!r}")
            # NOTE: fields.py and auc.py parse ctrl.gate by key, so any field added to
            # this line must keep `key=value` form.
            log("ctrl.gate", vt, f"[{vid}] fps={fps:.1f} level={level} rise={rising} "
                                 f"new_occ={distinct} fire={fire} next={nxt:.1f}s "
                                 f"gen={gen_s:.1f}s ntok={len(ids)} q={pending_q!r}"
                                 f" ev0={cfg.ev0_dedup} dup_ev={dup_ev}"
                                 f"{extra}")

            # Preserve full-precision observations for diagnostic plots; unlike
            # the text log these are not rounded to three decimal places.
            if p_hit is not None and evaluator is not None and hasattr(evaluator, "record_gate"):
                evaluator.record_gate(vt, p_hit, cfg.hit_threshold)
            if prof is not None:
                prof.mark("check_complete", vt)

            if fire:
                t_rec = vt
                ev = state.get("event_time_s")
                reported.append((t_rec, emit_text))
                # ev0 memory. Keyed on the onset the MODEL reported, not on the
                # clamped t_rec: the rule is "an event time it has already spoken
                # about", and clamping would fuse two genuinely different onsets that
                # both fell outside the 10 s window onto the same key.
                key = event_key(ev)
                if key is not None:
                    reported_ev.add(key)
                if evaluator is not None:
                    evaluator.record_trigger(t_rec, 1.0)
                    evaluator.record_write(t_rec, emit_text, gen_s)
                if prof is not None:
                    prof.mark("response_delivered", t_rec)
                log("CONTROLLER", vt, f"[{vid}] \U0001F4E2 @{t_rec:.1f}s  {emit_text!r}")

            # latch the level ONLY when backed by an answer: a bare true (no answer)
            # must not swallow the edge — the next answered tick can still fire.
            # On time_only tasks there IS no answer by construction, so the
            # latch runs off the level alone; without this the edge/hysteresis gate
            # degenerates — prev_level could never become True, every tick would look
            # like a rising edge, and the "condition still on screen" suppression the
            # whole gate is built on would be gone.
            if not level:
                prev_level = False
            elif answer or time_only:
                prev_level = True
            if prof is not None:
                prof.observe("controller_gen_s", gen_s)
                prof.observe("controller_tokens", len(ids))
            # RESTORE BEFORE RELEASE. set_next_check() frees the waiting ingester,
            # and the ingester's first act is mgr.ingest() -- so the primary must
            # already be restored at that instant. Publishing first left a window a
            # few statements wide in which the ingester wrote into a STILL-BORROWED
            # cache: measured 3 and 13 RuntimeErrors across two 18-sample arms. Each one killed the ingester THREAD, so the video stopped
            # being fed and the rest of that sample went silent -- one arm lost two
            # whole tasks and half its emits. The guard turned what would have been a
            # silent KV-cache corruption into a loud crash; this ordering is what
            # makes the "no concurrent writer in lockstep" premise actually true.
            if use_inplace:
                mgr.borrow_end()
            clock.set_next_check(next_check_vt)
        finally:
            # erase the controller's splice from the primary. Idempotent (so the
            # release above is not repeated work) and a no-op in snapshot mode. This
            # stays for the EXCEPTION path, where the tick never reached the release.
            if use_inplace:
                mgr.borrow_end()

    log("controller", 0.0, "controller stopped")
