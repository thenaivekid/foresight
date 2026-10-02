"""
metrics.py — OmniPro metrics, self-contained.

Mirrors the official OmniPro online scorer (repo: metrics/online/scorer.py):
  - greedy 1-to-1 temporal matching of emissions to ground-truth triggers
    within ±tolerance seconds (closest pair first);
  - time-only Precision / Recall / F1;
  - content scoring per task (time_only | count | count_at_time | position |
    state | gpt_judge);
  - content-gated Joint Precision / Recall / F1;
  - micro aggregation across samples (sum tp/fp/fn, then recompute).

Content judging for the free-text tasks (event_narration,
sequential_step_instruction) uses an LLM judge — OpenAI with Structured Outputs
(default, $OPENAI_API_KEY) or the paper's Gemini ($GEMINI_API_KEY) — following
OmniPro's protocol: score 1-5, correct if >=3. See ContentJudge for backends.

THERE IS NO FALLBACK JUDGE. If the judge cannot be reached, the verdict is None
(UNJUDGED) and `content_acc` / `joint_*` are reported as None rather than
estimated — see ContentJudge. Lower bounds remain available as `content_acc_lb` /
`joint_f1_lb`, named so they cannot be mistaken for the real thing, alongside
`n_judged` / `n_unjudged` / `content_coverage`. Timing metrics never involve the
judge and are always exact.

Predictions are stored with their text in online_pred.jsonl, so an unjudged run
can be completed later, without a GPU, via judge_offline.py.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from collections import defaultdict

# CANONICAL TASK -> CONTENT-SCORING KIND.
# Copied from the OmniPro reference implementation (metrics/online/scorer.py,
# TASK_CONTENT_KIND) so our numbers are comparable to the published baselines.
# Verified against the paper (S3.2.1): the LLM judge is used ONLY for the two
# open-ended generation tasks; everything else is exact match on structured
# output, and the two alert tasks do not score content at all.
#
# WE PREVIOUSLY HAD semantic_condition_alert IN THE JUDGE SET. That was wrong:
# upstream scores it time_only, so a time-matched SCA emit is correct regardless
# of what it says. Judging it made our SCA joint-F1 strictly lower than the same
# system would score under the benchmark's own scorer.
TASK_CONTENT_KIND = {
    # Alert tasks — content is not evaluated; any time-matched emit is correct.
    "instant_event_alert":         "time_only",
    "semantic_condition_alert":    "time_only",
    # Structured tasks — rule-based exact matching.
    "explicit_target_grounding":   "position",
    "snapshot_counting":           "count",
    "cumulative_counting":         "count",
    "dedup_counting":              "count",
    "realtime_state_monitor":      "state",
    # Free-text narration tasks — need the LLM judge.
    "event_narration":             "gpt_judge",
    "sequential_step_instruction": "gpt_judge",
}

TIME_ONLY = {t for t, k in TASK_CONTENT_KIND.items() if k == "time_only"}
COUNT_TASKS = {t for t, k in TASK_CONTENT_KIND.items() if k == "count"}
POSITION_TASKS = {t for t, k in TASK_CONTENT_KIND.items() if k == "position"}
STATE_TASKS = {t for t, k in TASK_CONTENT_KIND.items() if k == "state"}
JUDGE_TASKS = {t for t, k in TASK_CONTENT_KIND.items() if k == "gpt_judge"}

# ORDER MATTERS: _extract_position returns the FIRST substring hit, so every
# compound name must be tested before the bare "center" it contains. With the old
# alphabetical-ish order, "center-right" and "bottom-center" both matched "center"
# first and scored as the wrong cell -> 2 of the 9 cells were unwinnable, and both
# occur in the saved ETG ground truth. Sorted longest-first so this cannot regress.
_POSITIONS = sorted(
    ["top-left", "top-center", "top-right", "center-left", "center",
     "center-right", "bottom-left", "bottom-center", "bottom-right"],
    key=len, reverse=True)


# ---------------------------------------------------------------------------
# temporal matching (greedy, closest-first)
# ---------------------------------------------------------------------------
def match_emits_to_gt(emit_times: list[float], gt_times: list[float],
                      tolerance: float = 3.0):
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
        used_e.add(i); used_g.add(j)
        matches.append((i, j, dt))
    unmatched_e = [i for i in range(len(emit_times)) if i not in used_e]
    unmatched_g = [j for j in range(len(gt_times)) if j not in used_g]
    return matches, unmatched_e, unmatched_g


# ---------------------------------------------------------------------------
# content parsing / scoring
# ---------------------------------------------------------------------------
def _extract_count(text: str):
    """First integer in the emit, AFTER removing clock times.

    Mirrors utils/online_parser.py::_extract_integer upstream. The timestamp
    strip is load-bearing: our writer often opens with "At 01:23, ..." and the
    old version returned 1 (from "01") instead of the real count, marking a
    correct answer wrong. Only affects the three counting tasks.
    """
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text or "")
    m = re.search(r"-?\d+", cleaned)
    return int(m.group(0)) if m else None


def _extract_position(text: str):
    """Byte-for-byte upstream (utils/online_parser.py::_extract_region):
    prefer an explicit `Position: <region>` anchor, else longest-match scan over
    the nine HYPHENATED labels.

    We used to also accept the space-separated form ("bottom center" ->
    bottom-center). Upstream does not: it scans only hyphenated names, so
    "bottom center" matches the bare "center" and scores as a different cell.
    Our leniency inflated explicit_target_grounding relative to the benchmark's
    own scorer, so it is gone. Also note upstream returns None for "top left"
    (no bare "left" region), where we used to return top-left.
    """
    t = (text or "").lower()
    m = re.search(r"position\s*:\s*([\w\-]+)", t)
    if m:
        cand = m.group(1).strip()
        if cand in _POSITIONS:
            return cand
    for p in _POSITIONS:                     # already sorted longest-first
        if p in t:
            return p
    return None


def _extract_state(text: str, states: list[str] | None = None):
    """The WHOLE emit, lowercased and unquoted — upstream's rule.

    Mirrors utils/online_parser.py: `out["state"] = payload.strip().strip("'\\"").lower()`,
    which scorer.py then compares to gt["state_to"] by EXACT equality.

    The `states` argument is ignored and kept only for call compatibility. It used
    to drive a substring search ("is the GT state name mentioned anywhere?"), which
    is far more lenient than upstream: a rambling emit that happened to contain the
    target word scored correct. That inflated our realtime_state_monitor content
    accuracy relative to the benchmark's own scorer, so it is gone.
    """
    return (text or "").strip().strip("'\"").lower() or None


# NOTE: a lexical-overlap fallback used to live here and was used whenever the
# LLM judge failed. It was removed on purpose. Word overlap does not
# measure whether a response describes the right event, and because the failure
# was swallowed silently, a whole 24h run reported lexical scores under the label
# "google-genai judge active". A missing verdict must stay MISSING -- see
# ContentJudge.score, which returns None rather than inventing a number.


_GENAI_JUDGE_PROMPT = (
    "You are evaluating whether a model's response correctly describes an event in a "
    "video, compared to the ground-truth answer.\n\n"
    "Question/Instruction: {question}\n"
    "Ground-truth answer: {gt}\n"
    "Model response: {pred}\n\n"
    "Score 1-5:\n"
    "- 5: Perfect - same event/information, accurate\n"
    "- 4: Good - mostly correct, minor differences\n"
    "- 3: Acceptable - right event, notable inaccuracies\n"
    "- 2: Poor - partially relevant but significantly wrong/vague\n"
    "- 1: Wrong - irrelevant or wrong event\n\n"
    'Respond with ONLY a JSON object: {{"score": <int 1-5>, "explanation": "<brief>"}}')


# Structured Outputs schema for the OpenAI backend. With strict=True the model
# CANNOT return anything but a valid object of this shape, so "unparseable judge
# output" stops being a failure mode -- there is no regex, and therefore nothing
# to silently mis-parse into a wrong verdict.
OPENAI_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "explanation": {"type": "string"},
    },
    "required": ["score", "explanation"],
    "additionalProperties": False,
}


def openai_judge_request(model: str, question: str, gt: str, pred: str,
                         seed: int | None = None) -> dict:
    """The EXACT chat.completions body used by both the sync and the Batch path.

    Shared on purpose: a batch verdict and a sync verdict for the same triple must
    be the same computation, otherwise the cache would mix two different judges
    under one key."""
    body = {
        "model": model,
        "messages": [{"role": "user",
                      "content": _GENAI_JUDGE_PROMPT.format(
                          question=question, gt=gt, pred=pred)}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "omnipro_judge_verdict",
                                            "strict": True,
                                            "schema": OPENAI_JUDGE_SCHEMA}},
    }
    if seed is not None:
        body["seed"] = seed
    return body


def _parse_judge_score(text: str):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return int(json.loads(m.group()).get("score", 0))
    except Exception:
        m2 = re.search(r'"?score"?\s*[:=]\s*(\d)', text)
        return int(m2.group(1)) if m2 else None


class ContentJudge:
    """LLM-as-judge for free-text content. Correct if score >= 3 (paper).

    Backends (`backend=` or $OMNIPRO_JUDGE_BACKEND):
      "auto" (default)  (1) OpenAI ($OPENAI_API_KEY, model $OPENAI_JUDGE_MODEL);
                        (2) google-genai SDK (bare GEMINI_API_KEY, no base URL);
                        (3) the OmniPro REST llm_judge (needs GEMINI_API_BASE).
      "openai"          OpenAI only.
      "gemini"          Gemini only (google-genai SDK or REST llm_judge).

    An explicitly requested backend NEVER silently downgrades to another one: if it
    cannot be constructed, mode stays "unavailable" and every verdict is UNJUDGED.
    Mixing judges inside one number is exactly the kind of silent corruption this
    module exists to prevent.

    THERE IS NO FALLBACK. If neither judge can be reached, or a call fails (quota,
    503, parse error), `score` returns None — an explicit UNJUDGED marker that
    propagates all the way to the reported metrics, which then refuse to publish a
    content number. This is deliberate: the previous lexical fallback made a failed
    judge indistinguishable from a real verdict, and a full run silently reported
    word-overlap scores. Unjudged predictions stay on disk in online_pred.jsonl and
    can be judged later by judge_offline.py once quota allows.

    REPRODUCIBLE: the Gemini call is pinned with a fixed `seed`, and every verdict
    is persisted to a JSON cache keyed by (question, gt, pred) — re-scoring the
    same predictions always returns the same joint-F1 and costs zero API calls."""

    CACHE_PATH = os.environ.get(
        "OMNIPRO_JUDGE_CACHE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_cache.json"))
    # Append-only audit log. Never read by the scorer; see ContentJudge.trace.
    TRACE_PATH = os.environ.get(
        "OMNIPRO_JUDGE_TRACE_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_trace.jsonl"))

    def __init__(self, backend: str | None = None):
        self.mode = "unavailable"
        self.n_judged = 0           # verdicts actually obtained (incl. cache hits)
        self.n_unjudged = 0         # calls that could not produce a verdict
        self._last_detail = None    # raw judge payload for the most recent call
        self._warned = False
        self._judge = None
        self._genai = None
        self._openai = None
        self._model = None
        self._use_seed = True
        # When True, score() answers from the cache only and never calls the API.
        # Set it for pure scoring passes (see judge_offline.report).
        self.offline = False
        self._seed = int(os.environ.get("OMNIPRO_JUDGE_SEED", "1234"))
        self._max_retries = int(os.environ.get("OMNIPRO_JUDGE_RETRIES", "5"))
        # Prefix mixed into the cache key; see _cache_key.
        self._key_ns = ""
        self._cache = {}
        try:
            with open(self.CACHE_PATH) as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}
        backend = (backend or os.environ.get("OMNIPRO_JUDGE_BACKEND")
                   or "auto").lower()
        self.backend = backend

        if backend == "openai":
            self._init_openai()
            return                  # no cross-backend fallback (see class docstring)

        if backend == "gemini":
            self._init_gemini()
            if self.mode == "unavailable":
                print("[judge] NO JUDGE AVAILABLE \u2014 every content verdict will be "
                      "UNJUDGED and no content_acc/joint_f1 will be reported. "
                      "Timing metrics are unaffected. Re-score later with "
                      "judge_offline.py.", flush=True)
            return                  # no cross-backend fallback

        # backend == "auto": prefer OpenAI when OPENAI_API_KEY is present
        if os.environ.get("OPENAI_API_KEY"):
            if self._init_openai():
                return

        # Gemini fallback for auto (no OPENAI_API_KEY, or OpenAI init failed)
        if self._init_gemini():
            return

        if self.mode == "unavailable":
            print("[judge] NO JUDGE AVAILABLE \u2014 every content verdict will be "
                  "UNJUDGED and no content_acc/joint_f1 will be reported. "
                  "Timing metrics are unaffected. Re-score later with "
                  "judge_offline.py.", flush=True)

    def _init_openai(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[judge] backend=openai requested but OPENAI_API_KEY is unset — "
                  "every verdict will be UNJUDGED. Set OPENAI_API_KEY (or GEMINI_API_KEY) in your environment.",
                  flush=True)
            return False
        try:
            from openai import OpenAI
            self._openai = OpenAI()     # reads OPENAI_API_KEY from the environment
            self._model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5-mini")
            # WHY THE MODEL IS IN THE CACHE KEY (and why only for OpenAI):
            # a verdict is a function of (triple, judge), not of the triple alone,
            # so two judges must not share a slot. The 42 pre-existing entries were
            # all written by the Gemini/REST judge under the bare
            # sha256(q|gt|pred) key (the deleted lexical fallback never wrote to
            # the cache -- checked), so Gemini keeps the unprefixed namespace and
            # those entries stay valid and free. OpenAI verdicts get their own
            # "openai:<model>" namespace, which also means switching
            # OPENAI_JUDGE_MODEL re-judges from scratch instead of quietly serving
            # gpt-5-mini verdicts as if another model had produced them.
            self._key_ns = f"openai:{self._model}\x1f"
            self.mode = "openai"
            print(f"[judge] OpenAI judge active: model={self._model} "
                  f"(structured outputs, seed={self._seed})", flush=True)
            return True
        except Exception as e:
            print(f"[judge] OpenAI judge unavailable ({type(e).__name__}: {e}) — "
                  f"verdicts will be UNJUDGED", flush=True)
            return False

    def _init_gemini(self) -> bool:
        """Try google-genai SDK, then REST llm_judge. Returns True on success."""
        key = os.environ.get("GEMINI_API_KEY")

        # (1) google-genai SDK with the key passed explicitly
        if key:
            try:
                from google import genai
                self._genai = genai.Client(api_key=key)
                self._model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
                self.mode = "genai"
                print(f"[judge] google-genai judge active: model={self._model}", flush=True)
                return True
            except Exception as e:
                print(f"[judge] genai SDK unavailable ({type(e).__name__}: {e}); "
                      f"trying REST judge", flush=True)

        # (2) fallback: OmniPro's REST llm_judge (loaded by path to avoid the
        # metrics-package shadowing; needs GEMINI_API_BASE/OPENAI_API_BASE set).
        if os.environ.get("OPENAI_API_KEY") or key:
            try:
                import importlib.util
                from utils import SCRATCH
                judge_path = os.environ.get(
                    "OMNIPRO_LLM_JUDGE",
                    os.path.join(SCRATCH, "omni_pro", "repo", "metrics", "llm_judge.py"))
                spec = importlib.util.spec_from_file_location("omnipro_llm_judge", judge_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._judge = mod.LLMJudge()
                self.mode = "llm"
                print(f"[judge] REST LLM judge active: provider={self._judge.provider} "
                      f"model={self._judge.model} base={self._judge.api_base}", flush=True)
                return True
            except Exception as e:
                print(f"[judge] REST judge unavailable ({type(e).__name__}: {e})",
                      flush=True)

        return False

    def _cache_key(self, question: str, gt: str, pred: str) -> str:
        import hashlib
        return hashlib.sha256(
            f"{self._key_ns}{question}\x1f{gt}\x1f{pred}".encode()).hexdigest()[:24]

    def _cache_put(self, key: str, score: float):
        """Merge-then-write. Four eval processes share this file; the old version
        dumped its own dict straight over the top, so the last writer erased the
        other three (24h of judging left 42 entries). Re-read, merge, and rename
        a temp file into place so a reader never sees a half-written cache."""
        self._cache[key] = score
        try:
            merged = {}
            try:
                with open(self.CACHE_PATH) as f:
                    merged = json.load(f)
            except Exception:
                merged = {}
            merged.update(self._cache)
            self._cache = merged
            tmp = f"{self.CACHE_PATH}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(merged, f)
            os.replace(tmp, self.CACHE_PATH)
        except Exception:
            pass

    def trace(self, key, question, gt, pred, verdict, detail=None, source="sync"):
        """Append the FULL judgement to judge_trace.jsonl so it can be audited.

        WHY THIS IS SEPARATE FROM THE CACHE. The cache is a lookup table: it maps
        sha256(question|gt|pred)[:24] -> 0.0/1.0. That hash is irreversible and the
        float is binarised, so from the cache alone you cannot recover WHAT was
        judged, WHAT the judge said, or WHY -- you cannot even tell which
        prediction a 0.0 belongs to. For a number that goes in a paper, that is not
        good enough: a reviewer (or you, in three months) must be able to pull up
        any verdict and read the judge's reasoning next to the text it judged.

        So the trace stores the whole triple in the clear, the raw 1-5 score, the
        judge's explanation, the model and seed, and where it came from. It is
        append-only and never read by the scorer -- deleting it changes no metric,
        it only costs you the ability to check.

        Line-buffered O_APPEND writes of one JSON object are atomic enough for the
        concurrent writers we have (same reason the eval's own jsonl appends are
        safe); the trace is a log, not a source of truth.
        """
        if os.environ.get("OMNIPRO_JUDGE_TRACE", "1") == "0":
            return
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cache_key": key,
            "backend": self.mode,
            "source": source,
            "verdict": verdict,                 # 1.0 / 0.0, what the metric uses
            "question": question,
            "gt_response": gt,
            "pred_text": pred,
        }
        rec.update(detail or {})
        try:
            with open(self.TRACE_PATH, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass                                 # auditing must never break scoring

    def _fail(self, why: str):
        """Record an unjudged call, loudly the first time."""
        self.n_unjudged += 1
        if not self._warned:
            self._warned = True
            print(f"[judge] UNJUDGED: {why}. Content metrics will be withheld "
                  f"(not estimated). Further failures are counted silently.",
                  flush=True)
        return None

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """Only transient server-side conditions are worth waiting for. A 400/401
        is a bug or a bad key and retrying it just burns the budget."""
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(status, int):
            return status == 408 or status == 429 or status >= 500
        return type(exc).__name__ in ("RateLimitError", "APIConnectionError",
                                      "APITimeoutError", "InternalServerError",
                                      "APIError")

    def _openai_verdict(self, question: str, gt: str, pred: str):
        """One judged triple -> int score 1-5, or None if no verdict was obtained.

        Bounded: at most self._max_retries requests, exponential backoff with
        jitter on 429/5xx only. Exhausting them returns None (UNJUDGED) — never a
        guess."""
        why = "unknown"
        delay = 2.0
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                body = openai_judge_request(
                    self._model, question, gt, pred,
                    seed=self._seed if self._use_seed else None)
                r = self._openai.chat.completions.create(**body)
                msg = r.choices[0].message
                if getattr(msg, "refusal", None):
                    why = f"model refused: {str(msg.refusal)[:80]}"
                    break                                   # not retryable
                payload = json.loads(msg.content)
                # Keep the judge's own words so the verdict can be audited later.
                # The cache stores a bare 0.0/1.0 under an irreversible hash, so
                # without this there is no way to ask "why was this marked wrong?"
                self._last_detail = {
                    "score_raw": int(payload["score"]),
                    "explanation": payload.get("explanation", ""),
                    "model": self._model,
                    "seed": self._seed if self._use_seed else None,
                    "attempts": attempt,
                }
                return int(payload["score"])
            except Exception as e:
                why = f"{type(e).__name__}: {str(e)[:160]}"
                # Some models reject `seed`. Drop it once and retry -- the cache is
                # what actually makes re-scoring reproducible; seed is a bonus.
                if (self._use_seed and "seed" in str(e).lower()
                        and getattr(e, "status_code", None) == 400):
                    self._use_seed = False
                    attempt -= 1
                    print("[judge] model rejected `seed`; retrying without it "
                          "(cache still guarantees stable re-scores)", flush=True)
                    continue
                if not self._retryable(e) or attempt >= self._max_retries:
                    break
                s = delay * (1 + random.random() * 0.3)
                print(f"[judge] transient ({why}); retry {attempt}/"
                      f"{self._max_retries} in {s:.1f}s", flush=True)
                time.sleep(s)
                delay = min(delay * 2, 120.0)
        self._fail(f"openai judge failed ({why})")
        return None

    def score(self, question: str, gt: str, pred: str):
        """Correctness in [0,1] (1.0 if judged score >= 3, per the paper), or None
        if no verdict could be obtained. None is NOT a failure to be papered over:
        callers must propagate it so the metric is withheld rather than guessed."""
        k = self._cache_key(question, gt, pred)
        if k in self._cache:                        # persisted verdict -> stable re-scores
            self.n_judged += 1
            return float(self._cache[k])
        if self.offline:
            # A scoring pass is a READ of the cache. Judging happens in an explicit,
            # counted, budgeted loop -- never as a side effect of computing a table.
            # Without this, --dry-run / --rescore-only / a --max-calls 3 run's final
            # report each silently judged the entire run through the API.
            self.n_unjudged += 1
            return None
        if self.mode == "openai" and self._openai is not None:
            sc = self._openai_verdict(question, gt, pred)
            if sc is None:
                return None                     # _fail() already counted + logged
            out = 1.0 if sc >= 3 else 0.0       # paper: score>=3 is correct
            self._cache_put(k, out)
            self.trace(k, question, gt, pred, out, self._last_detail)
            self.n_judged += 1
            return out
        if self.mode == "genai" and self._genai is not None:
            try:
                prompt = _GENAI_JUDGE_PROMPT.format(question=question, gt=gt, pred=pred)
                # pin `seed` so the judge itself is reproducible (per-user choice:
                # seed rather than temperature=0)
                try:
                    from google.genai import types
                    gcfg = types.GenerateContentConfig(seed=self._seed)
                except Exception:
                    gcfg = None
                r = self._genai.models.generate_content(
                    model=self._model, contents=prompt,
                    **({"config": gcfg} if gcfg is not None else {}))
                sc = _parse_judge_score(getattr(r, "text", "") or "")
                if sc is not None:
                    out = 1.0 if sc >= 3 else 0.0   # paper: score>=3 is correct
                    self._cache_put(k, out)
                    self.trace(k, question, gt, pred, out,
                               {"score_raw": sc, "model": self._model,
                                "seed": self._seed,
                                "explanation": (getattr(r, "text", "") or "")[:500]})
                    self.n_judged += 1
                    return out
                return self._fail("genai returned an unparseable score")
            except Exception as e:
                return self._fail(f"genai call failed ({type(e).__name__}: "
                                  f"{str(e)[:120]})")
        if self.mode == "llm" and self._judge is not None:
            try:
                r = self._judge.judge(question, gt, pred)
                out = 1.0 if int(r.get("score", 0)) >= 3 else 0.0   # paper: >=3
                self._cache_put(k, out)
                self.n_judged += 1
                return out
            except Exception as e:
                return self._fail(f"REST judge call failed ({type(e).__name__}: "
                                  f"{str(e)[:120]})")
        return self._fail("no judge configured (set OPENAI_API_KEY, or "
                          "GEMINI_API_KEY for the paper's judge)")


def _content_correct(task: str, emit_raw: str, gt_item: dict,
                     question: str, judge: ContentJudge):
    """True / False / None. None means UNJUDGED — the LLM judge could not be
    reached — and must never be coerced to a bool. Only JUDGE_TASKS can return it;
    the count/position/state tasks are decided by deterministic extraction and are
    always available offline."""
    kind = TASK_CONTENT_KIND.get(task, "gpt_judge")

    # time_only: content is not part of the metric at all (IEA, SCA).
    if kind == "time_only":
        return True

    if kind == "gpt_judge":
        s = judge.score(question, gt_item.get("response", ""), emit_raw)
        return None if s is None else s >= 0.5

    # Structured kinds. Upstream marks an emit it cannot parse as WRONG (not
    # unscorable) — an unparseable count is a failure to follow the output
    # format, which is exactly what the benchmark is testing.
    if not (emit_raw or "").strip():
        return False

    if kind == "position":
        pp = _extract_position(emit_raw)
        gp = (gt_item.get("position") or "").lower().strip()
        return bool(pp and gp and pp == gp)

    if kind == "count":
        pc = _extract_count(emit_raw)
        gc = gt_item.get("count")
        if pc is None or gc is None:
            return False
        return int(pc) == int(gc)

    if kind == "state":
        ps = _extract_state(emit_raw)
        gs = (gt_item.get("state_to") or gt_item.get("state") or "").lower().strip()
        return bool(ps and gs and ps == gs)

    return False


# ---------------------------------------------------------------------------
# per-sample + aggregate
# ---------------------------------------------------------------------------
def score_sample(pred: dict, tolerance: float = 3.0,
                 judge: ContentJudge | None = None) -> dict:
    judge = judge or ContentJudge()
    task = pred["task"]
    emits = pred.get("predictions", [])
    gts = pred["ground_truth"]
    emit_times = [float(e["t_sec"]) for e in emits]
    gtt = [float(g["t_sec"]) for g in gts]

    matches, un_e, un_g = match_emits_to_gt(emit_times, gtt, tolerance)
    tp_time = len(matches)
    fp = len(un_e)
    fn = len(un_g)

    tp_content = 0
    n_unjudged = 0
    for ei, gj, _dt in matches:
        ok = _content_correct(task, emits[ei].get("raw", ""), gts[gj],
                              pred.get("question", ""), judge)
        if ok is None:
            n_unjudged += 1          # withheld, not counted as right OR wrong
        elif ok:
            tp_content += 1

    return {"id": pred["id"], "task": task,
            "tp_time": tp_time, "tp_content": tp_content,
            "n_unjudged": n_unjudged, "fp": fp, "fn": fn,
            "n_emits": len(emits), "n_gt": len(gtt)}


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def aggregate(per_sample: list[dict]) -> dict:
    by_task: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    overall = defaultdict(float)
    for s in per_sample:
        for k in ("tp_time", "tp_content", "n_unjudged", "fp", "fn",
                  "n_emits", "n_gt"):
            by_task[s["task"]][k] += s.get(k, 0)
            overall[k] += s.get(k, 0)
        by_task[s["task"]]["n"] += 1
        overall["n"] += 1

    def block(d):
        tpt, tpc, fp, fn = d["tp_time"], d["tp_content"], d["fp"], d["fn"]
        unj = d["n_unjudged"]
        tp_p, tp_r, tp_f = _prf(tpt, fp, fn)

        # JOINT = content-gated P/R/F1, and the denominators are NOT the ones _prf
        # would give. The paper (S3.2.1) and the reference scorer both define a
        # response as valid only if it is BOTH within +-tol AND content-correct:
        #     precision = valid / ALL RESPONSES        = tp_content / n_emits
        #     recall    = valid / ALL GT TRIGGERS      = tp_content / n_gt
        # and since tp_time + fp == n_emits and tp_time + fn == n_gt, upstream
        # writes it as tp_content / (tp_time + fp) and tp_content / (tp_time + fn).
        #
        # We previously used _prf(tpc, fp, fn) = tpc/(tpc+fp), whose denominator is
        # n_emits - (tp_time - tp_content) -- it QUIETLY DROPS every emit that
        # matched in time but failed on content, instead of counting it as a false
        # positive. That inflated joint-F1 by 2-18%, and worst when content was
        # worst (1.18x at cAcc 0.10), i.e. it flattered us exactly where we were
        # weakest. Timing metrics were never affected.
        jp = tpc / (tpt + fp) if (tpt + fp) else 0.0
        jr = tpc / (tpt + fn) if (tpt + fn) else 0.0
        jf = 2 * jp * jr / (jp + jr) if (jp + jr) else 0.0
        judged = tpt - unj                      # matches with a real verdict

        # WITHHELD, NOT ZERO. With unjudged matches, tp_content is only a lower
        # bound, so joint_* and content_acc are not the quantities they claim to
        # be. Publishing a number here is what let a lexical fallback masquerade
        # as a judged result, so we publish None and keep the lower bound under a
        # name that says what it is. Timing metrics are always exact.
        complete = (unj == 0)
        out = {"n_samples": int(d["n"]), "n_gt": int(d["n_gt"]),
               "n_emits": int(d["n_emits"]),
               "time_precision": round(tp_p, 4), "time_recall": round(tp_r, 4),
               "time_f1": round(tp_f, 4),
               "n_matched": int(tpt), "n_judged": int(judged),
               "n_unjudged": int(unj),
               "content_complete": complete,
               "content_coverage": round(judged / tpt, 4) if tpt else 0.0,
               "joint_f1_lb": round(jf, 4),
               "content_acc_lb": round(tpc / tpt, 4) if tpt else 0.0}
        if complete:
            out.update({"joint_precision": round(jp, 4),
                        "joint_recall": round(jr, 4),
                        "joint_f1": round(jf, 4),
                        "content_acc": round(tpc / tpt, 4) if tpt else 0.0})
        else:
            out.update({"joint_precision": None, "joint_recall": None,
                        "joint_f1": None, "content_acc": None})
        return out

    per_task = {t: block(d) for t, d in sorted(by_task.items())}
    ov = block(overall)

    # THE HEADLINE NUMBER IS A MACRO AVERAGE OVER TASKS, NOT A MICRO POOL.
    # scripts/compute_online_metrics.py::_macro_overall averages each task's
    # time_f1 / content_accuracy / joint_f1 with equal weight. Our `overall` block
    # pools tp/fp/fn across every sample, which weights a task by how many emits
    # and ground-truth events it happens to have -- dedup_counting alone is a third
    # of our emits, so a micro overall is largely dedup's score wearing the name
    # "OVERALL". Both are reported; `macro_*` is the one comparable to the paper's
    # table, `overall` stays micro for continuity with our own history.
    # Tasks whose metric is withheld (None) are EXCLUDED from the macro mean, and
    # the count is reported so a partial average can never be mistaken for a full
    # one -- same policy as upstream.
    def _macro(key):
        vals = [b[key] for b in per_task.values() if b.get(key) is not None]
        return (round(sum(vals) / len(vals), 4) if vals else None), len(vals)

    mt, nt = _macro("time_f1")
    mc, nc = _macro("content_acc")
    mj, nj = _macro("joint_f1")
    mc_lb, _ = _macro("content_acc_lb")
    mj_lb, _ = _macro("joint_f1_lb")
    ov.update({
        "num_tasks": len(per_task),
        "macro_time_f1": mt, "macro_time_f1_n": nt,
        "macro_content_accuracy": mc, "content_scored_task_count": nc,
        "macro_joint_f1": mj, "joint_task_count": nj,
        "macro_content_acc_lb": mc_lb, "macro_joint_f1_lb": mj_lb,
    })
    return {"overall": ov, "per_task": per_task}


# ---------------------------------------------------------------------------
# probe-mode metrics (OmniPro GT-probe), computed from probe records
# ---------------------------------------------------------------------------
# A probe record (one per ground-truth trigger):
#   {task, question, pre_share, post_share, post_text, gt_item}
# - paired/pre/post temporal accuracy from the yes/no probe shares vs threshold
#   (pre expects NO, post expects YES) — the temporal-awareness axis, all tasks;
# - post content accuracy via the task-appropriate parse/judge — the content
#   axis (for alerts this coincides with the yes/no post).
def probe_metrics(records: list[dict], *, threshold: float = 0.5,
                  judge: ContentJudge | None = None) -> dict:
    judge = judge or ContentJudge()
    by_task: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    overall = defaultdict(float)

    for r in records:
        task = r["task"]
        pre_ok = float(r["pre_share"] < threshold)
        post_ok = float(r["post_share"] >= threshold)
        paired_ok = float(pre_ok and post_ok)
        raw_ok = _content_correct(task, r.get("post_text", ""),
                                  r["gt_item"], r.get("question", ""), judge)
        for d in (by_task[task], overall):
            d["pre_ok"] += pre_ok; d["post_ok"] += post_ok
            d["paired_ok"] += paired_ok; d["n"] += 1
            if raw_ok is None:
                d["content_unjudged"] += 1
            else:
                d["content_ok"] += float(raw_ok)
                d["content_n"] += 1

    def block(d):
        n = d["n"] or 1
        pre_a, post_a = d["pre_ok"] / n, d["post_ok"] / n
        f1 = 2 * pre_a * post_a / (pre_a + post_a) if (pre_a + post_a) else 0.0
        unj = int(d["content_unjudged"])
        cn = int(d["content_n"])
        return {"n": int(d["n"]),
                "paired_accuracy": round(d["paired_ok"] / n, 4),
                "pre_accuracy": round(pre_a, 4), "post_accuracy": round(post_a, 4),
                "pre_post_f1": round(f1, 4),
                "n_judged": cn, "n_unjudged": unj,
                "content_complete": unj == 0,
                # withheld unless every probe got a real verdict (see aggregate)
                "content_accuracy": (round(d["content_ok"] / cn, 4)
                                     if unj == 0 and cn else None)}

    return {"overall": block(overall),
            "per_task": {t: block(d) for t, d in sorted(by_task.items())}}
