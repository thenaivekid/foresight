"""
test_plan_schema.py — static and log-based verification of the forced decode schema.

WHAT THIS FILE IS FOR
---------------------
The forced schema promotes six figure fields into the forced decode spine
(`fps`, `next_check_s`, `question_for_next`, `compact_now`, `replan_vision`, and the
class lists). Every one of them is a WRITE. The recurring failure mode this guards
against:

    "On every one of them we shipped the WRITE and never shipped the READ."

`note` (0 of 199,909 ticks filled), `count` (write-only) and `phase` (write-only) are
three fields that exist in the schema, cost tokens, and are read by nobody. The point of
this file is that adding six more cannot happen silently. Every assertion here checks one
verification requirement, and the ones that need a GPU are marked, not faked.

WHY IT IS TORCH-FREE
--------------------
`controller.py` imports torch and cannot be imported on a laptop, and the interesting
failure (a field in the spine with no consumer) is a STATIC property of the source — it
does not need a forward pass to detect. So every check here is one of:

  * a structural check against `evaluation/fields.py` and `evaluation/metrics.py`
    (importable, pure-python),
  * an `ast` / source-text inspection of `controller.py`, `config.py`,
    `vision_stream.py`, `compaction.py` (parsed, never imported),
  * a log-analysis assertion run against a synthetic fixture log, whose real use is to
    be pointed at an actual run directory on a GPU machine.

The last group is the important one: `assert_full_occupancy()` and
`assert_answer_decodes_per_task()` below are the run-level tools
("assert it in the run, do not eyeball the log"), and the fixture tests exist to prove
the tools themselves work before anyone trusts their verdict on a 200k-tick run.

MID-EDIT TOLERANCE
------------------
`controller.py`, `config.py`, `vision_stream.py` and `compaction.py` may evolve
independently of this file. Anything not yet present is reported as a SKIP with the
exact reason, never as a spurious failure — but anything present and WRONG is a failure.
A field that has entered the spine while its consumer is still absent is precisely the
bug this file exists to catch, so that case fails loudly by design.

Usage:
    python -m pytest foresight/test_plan_schema.py -v
    python foresight/test_plan_schema.py <run_dir>   # run the log assertions   
                                                         # against a real run
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_OMNIPRO = os.path.join(_ROOT, "evaluation")
if _OMNIPRO not in sys.path:
    sys.path.insert(0, _OMNIPRO)

import fields as fields_mod          # noqa: E402  evaluation/fields.py
import metrics as metrics_mod        # noqa: E402  evaluation/metrics.py

CONTROLLER = os.path.join(_HERE, "controller.py")
CONFIG = os.path.join(_HERE, "config.py")
VISION_STREAM = os.path.join(_HERE, "vision_stream.py")
COMPACTION = os.path.join(_HERE, "compaction.py")


# ---------------------------------------------------------------------------
# the spine, as specified
# ---------------------------------------------------------------------------
# Order matters twice over. (1) `fields.py` salvages keys out of a truncated
# `ctrl.raw` row by relying on the walk emitting keys in a fixed order. (2) an
# out-of-order key in a real log means the walk did not run as specified, which is a
# different bug from the model declining to fill a field. This list is the SPEC; the
# code is advisory, because the code is being written right now.
SPINE_ORDER = [
    "seen", "have_enough_info", "event_time_s", "answer",
    "fps", "next_check_s", "compact_now", "replan_vision",
    "keep", "ignore", "question_for_next",
    "count", "phase",
]

# Conditioning: which ticks is a field even supposed to appear on? An occupancy
# number read without this is meaningless — `answer` at 63% is not a model that
# declined 37% of the time, it is a model that was quiet 37% of the time.
UNCONDITIONAL = {"seen", "have_enough_info", "fps", "next_check_s",
                 "compact_now", "replan_vision", "question_for_next"}
HIT_GATED = {"event_time_s", "answer"}
REPLAN_GATED = {"keep", "ignore"}

# Historical parser fields only. Count/phase are now wired into the spine and
# prompt memory; the older parked design is no longer tested.
PARKED = {"note", "objects"}


# ---------------------------------------------------------------------------
# the consumer table, encoded as data
# ---------------------------------------------------------------------------
# "A field goes into the spine only when its consumer is merged and has a test."
#
# `probe` is a (file, regex) pair naming the CALL SITE that reads the field. It is a
# static probe because we cannot run the model here — but a consumer that is not
# textually present in any source file is definitively absent, which is the direction
# that matters. `status` is the plan's claim; the tests check the claim against the
# tree rather than trusting it.
class Consumer:
    def __init__(self, field, what, status, probes, note=""):
        self.field, self.what, self.status = field, what, status
        self.probes, self.note = probes, note

    def resolve(self):
        """(exists, detail). Missing source files count as a missing consumer.

        Probed against COMMENT-STRIPPED source: these files document the defect they
        fixed by quoting the old call site, so a raw-text grep finds the consumer in a
        comment and reports a dead read as live."""
        misses = []
        for path, pattern in self.probes:
            src = _code_only(path)
            if src is None:
                misses.append(f"{os.path.basename(path)} does not exist")
            elif not re.search(pattern, src):
                misses.append(f"{os.path.basename(path)} has no /{pattern}/")
            else:
                continue
        if misses:
            return False, "; ".join(misses)
        return True, "; ".join(f"{os.path.basename(p)} matches /{r}/"
                               for p, r in self.probes)


# status: "live" = shipped and read; "wip" = being merged right now; "absent" = no
# consumer exists, so the field MUST NOT be enabled in the spine.
CONSUMERS = {
    "seen": Consumer(
        "seen", "in-cache perception step read by the p_hit logit read", "live",
        [(CONTROLLER, r"cfg\.seen_mode")],
        "not a plan field; the Vision-Encoder -> LLM edge of the figure"),
    "have_enough_info": Consumer(
        "have_enough_info", "the fire gate", "live",
        [(CONTROLLER, r"p_hit\s*>=\s*cfg\.gate_high_thr")]),
    "event_time_s": Consumer(
        "event_time_s", "the emitted timestamp (+ ev0 dedup)", "live",
        [(CONTROLLER, r'state\.get\(\s*["\']event_time_s["\']')]),
    "answer": Consumer(
        "answer", "emission / `reported` (+ time-only bypass)", "live",
        [(CONTROLLER, r"reported\.append\(")]),
    "next_check_s": Consumer(
        "next_check_s", "next_check_vt -> clock.set_next_check", "live",
        [(CONTROLLER, r"set_next_check\(")]),
    "question_for_next": Consumer(
        "question_for_next", "spliced into the next tick's prompt", "live",
        [(CONTROLLER, r"pending_q")]),
    "fps": Consumer(
        "fps", "ctrl.set_fps() -> encoder cadence (vision_stream.py)", "wip",
        [(VISION_STREAM, r"ctrl\.get_fps\(\)")],
        "The read exists but is dead — `cfg.fps if cfg.deterministic "
        "else ctrl.get_fps()` never takes the else branch, because every eval sets "
        "deterministic=True. Checked separately in test_fps_consumer_is_not_dead."),
    "compact_now": Consumer(
        "compact_now", "compaction.admit() admission gate", "wip",
        [(COMPACTION, r"def\s+admit\s*\("),
         (CONTROLLER, r"\bcompaction\b|\badmit\s*\(")]),
    "replan_vision": Consumer(
        "replan_vision", "gate on the class-verdict block + the pruner", "absent",
        [(VISION_STREAM, r"prune_tokens|prune_frame")],
        "The gate is only meaningful once the pruner it "
        "gates exists."),
    "keep": Consumer(
        "keep", "vision pruner (visionzip prune_tokens, swapped scoring)", "absent",
        [(VISION_STREAM, r"prune_tokens|prune_frame")]),
    "ignore": Consumer(
        "ignore", "vision pruner (visionzip prune_tokens, swapped scoring)", "absent",
        [(VISION_STREAM, r"prune_tokens|prune_frame")]),
    "count": Consumer(
        "count", "prompt accumulator", "live",
        [(CONTROLLER, r"Your running count from previous ticks")]),
    "phase": Consumer(
        "phase", "previous state in the next prompt", "live",
        [(CONTROLLER, r"Your last recorded state")]),
    "objects": Consumer(
        "objects", "legacy class list, not in the current schema", "absent",
        [(CONTROLLER, r"state\[['\"]objects['\"]\]")]),
    "note": Consumer(
        "note", "the append-only thought trace", "absent",
        [(CONTROLLER, r"notes_ring|note_ring")],
        "0 of 199,909 ticks filled in a full evaluation run. Parked."),
}


# ---------------------------------------------------------------------------
# the per-task whitelist
# ---------------------------------------------------------------------------
def task_whitelist(task, *, plan_compact=False, plan_classes=False,
                   plan_question=True):
    """The exact key set the spine may emit for `task`. A WHITELIST, never a blacklist.

    Build the schema from what
    a task is allowed to have, so a field added later cannot leak into a task silently.
    A blacklist inverts the failure mode — a new field is in every task until someone
    remembers to exclude it, which is exactly how `count` ended up being emitted on
    16.16% of ALL ticks including tasks that never count anything.

    Time-only membership is READ from `metrics.py:TIME_ONLY`, not re-listed. Two copies
    of that set drift, and the drift is silent: the day someone adds a third alert task,
    a re-listed copy keeps decoding a 32-token `answer` that the scorer never reads.
    """
    fs = ["seen", "have_enough_info", "event_time_s"]
    if task not in metrics_mod.TIME_ONLY:
        fs.append("answer")
    fs += ["fps", "next_check_s"]
    if plan_compact:
        fs.append("compact_now")
    if plan_classes:
        fs += ["replan_vision", "keep", "ignore"]
    if plan_question:
        fs.append("question_for_next")
    if task in ("snapshot_counting", "cumulative_counting", "dedup_counting"):
        fs.append("count")
    if task == "realtime_state_monitor":
        fs.append("phase")
    return [f for f in SPINE_ORDER if f in set(fs)]


ALL_TASKS = sorted(metrics_mod.TASK_CONTENT_KIND)


# ---------------------------------------------------------------------------
# source inspection (never import: controller.py needs torch and a GPU)
# ---------------------------------------------------------------------------
def _read(path):
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def _code_only(path):
    """Source with comments and docstrings removed, for probes that must not match prose.

    Every one of these files documents the defect it fixed by quoting the old line
    verbatim (`# Was: interval = 1.0 / (cfg.fps if cfg.deterministic else ...)`), which
    is exactly the pattern the dead-consumer probes look for. Matching a comment would
    report a fixed defect as live — the same class of error as reporting a live one as
    fixed, and just as expensive.
    """
    import io
    import tokenize
    src = _read(path)
    if src is None:
        return None
    out, prev_end, prev_tok = [], (1, 0), tokenize.INDENT
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                                  # mid-edit; fall back to raw text
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_tok in (
                tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL):
            prev_end = tok.end
            continue                                # a docstring
        if tok.start[0] > prev_end[0]:
            out.append("\n" * (tok.start[0] - prev_end[0]))
            out.append(" " * tok.start[1])
        elif tok.start[1] > prev_end[1]:
            out.append(" " * (tok.start[1] - prev_end[1]))
        out.append(tok.string)
        prev_end, prev_tok = tok.end, tok.type
    return "".join(out)


def _function_source(path, name):
    """Source text of one top-level (or nested) function, via ast. None if absent.

    Text-slicing by `ast` rather than `inspect`, because importing the module would
    pull in torch. Robust to concurrent edits of the file body; only
    a syntax error can defeat it, and that is reported as a skip.
    """
    src = _read(path)
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError as e:                       # someone is mid-save
        pytest.skip(f"{os.path.basename(path)} does not parse right now ({e}); "
                    f"it may be mid-edit. Re-run.")
    lines = src.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return "".join(lines[node.lineno - 1:node.end_lineno])
    return None


def spine_fields_in_code():
    """Which spine keys can `_schema_tick` emit at all, capability-wise?

    Detected by looking for the QUOTED key (`"fps"`), which covers both halves of how
    the walk writes a field: the forced literal (`',"fps":'`) and the diff assignment
    (`diff["fps"] = ...`). Deliberately not an import — see the module docstring.

    CAPABILITY, not enablement: the walk gates every optional field on the per-task
    whitelist, so a key appearing here says only that the code KNOWS how to emit it.
    `enabled_spine_fields()` is the set that actually reaches a run.
    """
    body = _function_source(CONTROLLER, "_schema_tick")
    if body is None:
        return None
    return {f for f in SPINE_ORDER + sorted(PARKED) if f'"{f}"' in body}


# Fields whose whitelist membership is not decided field-by-field in the whitelist
# function: the class block is one unit gated by one flag (
# `replan_vision` is the gate, `objects`/`keep`/`ignore` are decoded behind it).
_FLAG_FALLBACK = {"keep": "plan_classes", "ignore": "plan_classes"}


def _whitelist_guards():
    """{field: flag or None} read out of controller.py's whitelist function.

    `None` means the field is added unconditionally (or under a non-config guard such
    as `if not time_only`), i.e. it ships in every run. A field with a flag ships only
    if that flag's DEFAULT is True — which is what makes the default config, not the
    ablation matrix, the thing that has to be safe.
    """
    src = _read(CONTROLLER)
    if src is None:
        return None
    m = re.search(r"def\s+(_?plan_fields|_?task_(?:whitelist|fields)|_?schema_fields)"
                  r"\s*\(", src)
    if not m:
        return None
    body = _function_source(CONTROLLER, m.group(1))
    try:
        tree = ast.parse(body.lstrip() if body else "")
    except SyntaxError:
        return None

    def _added(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                    and n.value in set(SPINE_ORDER) | PARKED:
                out.add(n.value)
        return out

    guards = {f: None for f in _added(tree)}
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        flag = None
        t = n.test
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
            flag = t.attr
        elif isinstance(t, ast.BoolOp):
            flags = [n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id == "cfg"]
            flag = flags[0] if len(flags) == 1 else None
        if flag is None:
            continue
        for f in _added(ast.Module(body=n.body, type_ignores=[])):
            guards[f] = flag
    return guards


def enabled_spine_fields():
    """The fields a DEFAULT run actually forces: capable in the walk AND flag-enabled.

    A capable-but-flagged-off field is the correct state for anything whose consumer is
    still being written — that is what `plan_compact=False` / `plan_classes=False` are
    for. A missing flag is treated as ENABLED on purpose: an unrecognised gate must fail
    towards "this is live, prove its consumer exists", never towards silence.
    """
    capable = spine_fields_in_code()
    if capable is None:
        return None
    guards = _whitelist_guards() or {}
    defaults = config_defaults()
    out = set()
    for f in capable:
        if f in PARKED:
            continue                       # behind `more`, never forced
        flag = guards.get(f) or _FLAG_FALLBACK.get(f)
        if flag is None or defaults.get(flag, True):
            out.add(f)
    return out


def config_defaults():
    """{field: default} for the config dataclass, read with `ast`, no import.

    `config.py` imports `prompts.py`, and is being edited concurrently; parsing the
    literal defaults is both cheaper and immune to a half-written import.
    """
    src = _read(CONFIG)
    if src is None:
        return {}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.skip(f"config.py does not parse right now ({e}); re-run.")
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                try:
                    out[stmt.target.id] = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError, TypeError):
                    pass                          # field(default_factory=...) etc.
    return out


# ---------------------------------------------------------------------------
# fields.py knows every spine key
# ---------------------------------------------------------------------------
def test_fields_list_covers_every_whitelisted_field():
    """`fields.py:FIELDS` must contain every field any task's whitelist can hold.

    If it does not, the next occupancy table reports a brand-new field as ABSENT and
    the run looks like a model failure instead of a tooling gap. Such gaps are "cheap to
    forget, expensive to notice late."
    """
    known = set(fields_mod.FIELDS)
    wanted = set()
    for t in ALL_TASKS:
        wanted |= set(task_whitelist(t, plan_compact=True, plan_classes=True))
    missing = sorted(wanted - known)
    assert not missing, (
        f"evaluation/fields.py:FIELDS is missing {missing}. Every occupancy number "
        f"for those fields would read 0% no matter what the controller emitted.")


def test_fields_list_is_in_spine_order():
    """FIELDS order must equal the spine's walk order (parked fields trail).

    The truncation salvage in `fields.py` is only sound because the walk emits keys in
    a fixed order; the report prints columns in FIELDS order, so a mismatch here means
    the occupancy table is printed in an order that does not correspond to how the keys
    were produced, and a partially-cut row cannot be reasoned about.
    """
    got = [f for f in fields_mod.FIELDS if f in set(SPINE_ORDER)]
    assert got == SPINE_ORDER, (
        f"fields.py:FIELDS spine order {got} != spec spine {SPINE_ORDER}")
    tail = [f for f in fields_mod.FIELDS if f not in set(SPINE_ORDER)]
    assert set(tail) <= PARKED, f"unexpected non-spine keys in FIELDS: {tail}"


def test_gate_parser_knows_every_restricted_read_confidence():
    """The `ctrl.gate` parse must accept one confidence per restricted read.

    `_read_choice`/`_read_bool` give a continuous confidence for free at every forced
    position. Those are logged on `ctrl.gate` in `key=value` form, and they
    are the ONLY record of the model's uncertainty on the new fields — the JSON carries
    the decision, not the distribution. A parser that drops them throws away the
    calibration data with no error message.
    """
    tail = set(getattr(fields_mod, "_GATE_TAIL", {}))
    for k in ["p_hit", "p_more", "p_fps", "p_cadence", "p_compact", "p_replan"]:
        assert k in tail, (
            f"fields.py cannot parse `{k}=` off ctrl.gate; the confidence for that "
            f"read is silently discarded from every future occupancy run.")
    for k in ["count", "notes"]:
        assert k in tail, (
            f"fields.py lost the parked `{k}=` gate key. controller.py's own comment "
            f"says re-enabling it must keep key=value form and keep this parser "
            f"working; dropping it here breaks re-enabling that field later.")


def test_gate_parser_knows_every_confidence_the_controller_actually_emits():
    """Cross-check the parser against the SOURCE, not against this file's own list.

    The previous test asserts the keys the plan specifies; this one asserts the keys
    the controller emits. They are different failure modes: the plan can be right and
    the code can name a read `p_cadence` where the parser expects `p_next_check`, and
    the only symptom would be a telemetry column that is empty forever.
    """
    code = _code_only(CONTROLLER)
    if code is None:
        pytest.skip("controller.py is missing")
    emitted = {n for n in re.findall(r"[\"']((?:p_)[a-z][a-z0-9_]*)[\"']", code)}
    emitted |= {n for n in re.findall(r"\s((?:p_)[a-z][a-z0-9_]*)=\{", code)}
    tail = set(getattr(fields_mod, "_GATE_TAIL", {}))
    unknown = sorted(emitted - tail)
    assert not unknown, (
        f"controller.py names confidence(s) {unknown} that fields.py's ctrl.gate "
        f"parser does not know. They will be logged and never read — add them to "
        f"fields.py:_GATE_TAIL.")


def test_gate_parser_round_trips_a_full_spine_gate_line():
    """End-to-end proof of the parse, including a `q=` repr containing `=` and spaces.

    The pending question is free text spliced into the next prompt. If it happens to
    contain `count=3` and the parser is anchored positionally, half the question is
    read as tail keys and `count` acquires a value the model never emitted.
    """
    log = _synthetic_log([_tick(vt=1.0, hit=True, answer="two cars",
                               q="does count=3 hold?", extra_tail=True)],
                         task="dedup_counting")
    rows = list(_parse_text(log))
    assert len(rows) == 1
    r = rows[0]
    assert r["q"] == "does count=3 hold?"
    assert r["p_fps"] == pytest.approx(0.70)
    assert r["p_cadence"] == pytest.approx(0.61)
    assert r["p_compact"] == pytest.approx(0.02)
    assert r["p_replan"] == pytest.approx(0.55)


def test_fields_aliases_normalise_the_two_spellings_of_the_class_lists():
    """`keep`/`ignore` and `keep_classes`/`ignore_classes`
    are the same field. Both
    spellings will appear in logs; two occupancy columns for one field would halve both.
    """
    canon = getattr(fields_mod, "canon", None)
    assert canon is not None, "fields.py lost its alias normaliser"
    assert canon("keep_classes") == "keep"
    assert canon("ignore_classes") == "ignore"
    assert canon("question") == "question_for_next"


def test_ev0_tail_does_not_contaminate_pending_question():
    text = _synthetic_log([_tick(vt=1.0, hit=True, q="is this the same event?")])
    text = text.replace(" p_hit=", " ev0=True dup_ev=False p_hit=")
    row = list(_parse_text(text))[0]
    assert row["q"] == "is this the same event?"
    assert row["ev0"] is True
    assert row["dup_ev"] is False
    assert row["p_hit"] == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# per-task whitelist correctness
# ---------------------------------------------------------------------------
def test_time_only_membership_is_read_from_metrics_not_relisted():
    """The whitelist must derive time-only membership from `metrics.py`.

    Proven by mutation rather than by inspection: temporarily declare a normally
    content-scored task time-only and assert the whitelist follows. If the whitelist
    carried its own copy of the set, `answer` would survive the mutation — which is
    exactly the drift to avoid ("read it, do not re-list it").
    """
    victim = "event_narration"
    assert victim not in metrics_mod.TIME_ONLY
    assert "answer" in task_whitelist(victim)
    original = metrics_mod.TIME_ONLY
    try:
        metrics_mod.TIME_ONLY = set(original) | {victim}
        assert "answer" not in task_whitelist(victim), (
            "task_whitelist() does not read metrics.py:TIME_ONLY — it holds a second "
            "copy, and the two will drift the first time a task changes kind.")
    finally:
        metrics_mod.TIME_ONLY = original


def test_time_only_tasks_drop_answer_and_keep_timing():
    """IEA and SCA keep every timing field and lose `answer` entirely.

    Not "emit an empty answer" — no answer decode at all, which is the ~32 tokens on
    39-49% of ticks the change is for.
    """
    for t in sorted(metrics_mod.TIME_ONLY):
        wl = task_whitelist(t)
        assert "answer" not in wl, f"{t} is time_only but its whitelist keeps `answer`"
        assert "event_time_s" in wl, f"{t} lost its timestamp; the task is timing-only"
        assert "have_enough_info" in wl, f"{t} lost the fire gate"
    for t in ALL_TASKS:
        if t in metrics_mod.TIME_ONLY:
            continue
        assert "answer" in task_whitelist(t), f"{t} is content-scored but has no answer"


def test_whitelist_is_a_whitelist_a_new_field_cannot_leak_in():
    """A field added to the spine spec must not appear in any task until it is declared.

    The blacklist failure mode: a new key is in every task from the moment it exists,
    and nobody notices until an occupancy table shows it filling on tasks that have no
    use for it. Simulated by appending an undeclared field to the spine order.
    """
    global SPINE_ORDER
    original = SPINE_ORDER
    try:
        SPINE_ORDER = original + ["zzz_untested_field"]
        for t in ALL_TASKS:
            wl = task_whitelist(t, plan_compact=True, plan_classes=True)
            assert "zzz_untested_field" not in wl, (
                f"an undeclared field leaked into {t}'s schema. The whitelist has "
                f"become a blacklist — it must be built from an explicit whitelist.")
    finally:
        SPINE_ORDER = original


def test_count_and_phase_are_task_conditional_with_live_consumers():
    """The current forced accumulators must stay scoped to their own tasks."""
    for t in ALL_TASKS:
        wl = set(task_whitelist(t, plan_compact=True, plan_classes=True))
        assert ("count" in wl) == (t in {"snapshot_counting", "cumulative_counting", "dedup_counting"})
        assert ("phase" in wl) == (t == "realtime_state_monitor")
        assert not (wl & PARKED), (
            f"{t}'s whitelist contains parked field(s) {sorted(wl & PARKED)}; their "
            f"consumers do not exist.")
    assert CONSUMERS["count"].resolve()[0]
    assert CONSUMERS["phase"].resolve()[0]


def test_whitelist_matches_the_implementation_if_one_exists():
    """If the controller has grown its own whitelist, it must agree with this one.

    Skipped while that code is being written — but the moment a `task_whitelist` /
    `_task_fields` function appears in controller.py, the two definitions are two
    copies and this test is what keeps them equal.
    """
    src = _read(CONTROLLER) or ""
    m = re.search(r"def\s+(_?plan_fields|_?task_(?:whitelist|fields)|_?schema_fields)"
                  r"\s*\(", src)
    if not m:
        pytest.skip("controller.py has no per-task whitelist function yet "
                    "This test "
                    "activates automatically once it lands.")
    # The whitelist may take the time-only set as an argument rather than importing it
    # (controller.py:_plan_fields does). Either is fine; a HARD-CODED list of task
    # names is not, and that is what this looks for.
    code = _code_only(CONTROLLER) or src
    assert "TIME_ONLY" in code or "TASK_CONTENT_KIND" in code, (
        f"controller.py:{m.group(1)} never touches metrics.py's task-kind table; "
        f"time-only membership is being re-listed and the two copies will drift "
        f"('read it, do not re-list it').")
    guards = _whitelist_guards()
    assert guards is not None, f"could not read controller.py:{m.group(1)}"
    # Every field this test file believes is whitelistable must be decided there too,
    # or one of the two definitions has grown a field the other has never heard of.
    ours = set()
    for t in ALL_TASKS:
        ours |= set(task_whitelist(t, plan_compact=True, plan_classes=True))
    theirs = set(guards) | {f for f in _FLAG_FALLBACK if f in
                            (spine_fields_in_code() or set())}
    assert ours == theirs, (
        f"the test's whitelist and controller.py:{m.group(1)} disagree: only here "
        f"{sorted(ours - theirs)}, only there {sorted(theirs - ours)}")


# ---------------------------------------------------------------------------
# NO WRITE WITHOUT A READ.  The most important test in this file.
# ---------------------------------------------------------------------------
def test_consumer_table_names_a_consumer_for_every_spine_field():
    """Every field in the spine (and every parked one) is named in the consumer table.

    A field with no ROW here is worse than a field with a missing consumer: nobody has
    even asked the question.
    """
    for f in SPINE_ORDER + sorted(PARKED):
        assert f in CONSUMERS, (
            f"`{f}` is in the schema and has no entry in the consumer table. "
            f"Name its reader or take it out of the spine.")


def test_no_write_without_a_read():
    """For every field the CODE emits, its consumer must exist in the tree.

        "On every one of them we shipped the WRITE and never shipped the READ."
                                                    — on note/count/phase

    This is the concrete form of that lesson. `_schema_tick` is read statically; for
    each key it emits, the consumer's call site is looked for in the file that is
    supposed to contain it. A field that is emitted with no reader fails here, with the
    name of the file that should have contained the read.
    """
    enabled = enabled_spine_fields()
    if enabled is None:
        pytest.skip("controller.py has no _schema_tick right now (it may be "
                    "mid-rewrite). Re-run once it is back.")
    problems = []
    for f in sorted(enabled):
        c = CONSUMERS.get(f)
        if c is None:
            problems.append(f"{f}: forced by the spine, not in the consumer table")
            continue
        ok, detail = c.resolve()
        if not ok:
            problems.append(
                f"{f}: forced in the default spine but its consumer ({c.what}) is "
                f"ABSENT — {detail}."
                + (f" {c.note}" if c.note else "")
                + f" Either merge the consumer or drop `{f}` from the spine: "
                  f"'a field goes into the spine only when its consumer is merged and "
                  f"has a test'.")
    assert not problems, "write-without-a-read:\n  " + "\n  ".join(problems)


def test_fields_without_consumers_are_not_in_the_spine():
    """The other direction: anything the consumer table marks `absent` must not be emitted.

    `keep`/`ignore`/`objects`/`replan_vision` stay OFF until the pruner lands and the
    cosine-alignment gate passes; `note` stays parked behind `more`.
    A parked field appearing in the forced walk is a silent promotion.
    """
    enabled = enabled_spine_fields()
    if enabled is None:
        pytest.skip("controller.py has no _schema_tick right now; re-run.")
    bad = sorted(f for f in enabled
                 if CONSUMERS[f].status == "absent" and not CONSUMERS[f].resolve()[0])
    assert not bad, (
        f"fields with NO consumer are forced in the spine: {bad}. Each one is a new "
        f"`note`: tokens spent every tick, read by nobody.")


def test_capable_but_consumerless_fields_are_behind_a_flag_that_defaults_off():
    """A field the walk can emit but whose consumer is absent must be gated, off.

    `_schema_tick` knowing how to emit `keep`/`ignore` is not a defect — that code has
    to exist before the pruner can be measured. Shipping it ON is. This is the
    difference between "written, off, waiting for its reader" and a fourth `note`.
    """
    capable = spine_fields_in_code()
    if capable is None:
        pytest.skip("controller.py has no _schema_tick right now; re-run.")
    guards = _whitelist_guards() or {}
    defaults = config_defaults()
    problems = []
    for f in sorted(capable - PARKED):
        if CONSUMERS[f].resolve()[0]:
            continue
        flag = guards.get(f) or _FLAG_FALLBACK.get(f)
        if flag is None:
            problems.append(f"`{f}` has no consumer and no cfg flag gating it — it is "
                            f"forced unconditionally")
        elif flag not in defaults:
            problems.append(f"`{f}` is gated on cfg.{flag}, which config.py does not "
                            f"define; the gate cannot be checked")
        elif defaults[flag]:
            problems.append(f"`{f}` has no consumer and cfg.{flag} defaults True")
    assert not problems, ("consumer-less fields are not safely parked:\n  "
                          + "\n  ".join(problems))


def test_fps_consumer_is_not_dead():
    """`ctrl.get_fps()` is called but never reached.

        interval = 1.0 / (cfg.fps if cfg.deterministic else ctrl.get_fps())

    `cfg.deterministic` is True for every eval, so the encoder walks at a constant
    1.0 fps regardless of what the plan asked for, and the steered input gate is
    — on the benchmark — a constant. A grep for `ctrl.get_fps()` passes on this line,
    which is exactly why it needs its own test: the call site exists, the READ does not.
    """
    src = _code_only(VISION_STREAM)
    if src is None:
        pytest.skip("vision_stream.py is missing")
    dead = re.search(r"cfg\.fps\s+if\s+cfg\.deterministic\s+else\s+ctrl\.get_fps\(\)",
                     src)
    if dead and "fps" in (enabled_spine_fields() or set()):
        pytest.fail(
            "`fps` is in the forced spine (100% fill) while vision_stream.py still "
            "reads `cfg.fps if cfg.deterministic else ctrl.get_fps()` — every eval "
            "sets deterministic=True, so the plan's fps is discarded. The fix "
            "is `1.0 / (ctrl.get_fps() if cfg.plan_fps else cfg.encoder_idle_fps)`, "
            "which keeps lockstep because under lockstep the controller's fps is a "
            "step function of video time and is identical every run.")
    if dead:
        pytest.skip("the dead-fps branch is still present but `fps` is not in the "
                    "spine yet; this becomes a failure the moment it is promoted "
                    "(both halves should land together).")
    assert "ctrl.get_fps()" in src, (
        "vision_stream.py no longer reads ctrl.get_fps() at all — the fps consumer "
        "went from dead to absent.")


def test_compaction_admission_is_a_pure_testable_function_if_it_exists():
    """`compact_now` is a REQUEST; code decides, and refusals are a result.

    The admission gate is specified as pure and unit-testable without a GPU. If it has
    landed, check the shape the plan asks for — a reason returned alongside the verdict,
    so a refusal can be counted and reported rather than silently swallowed.
    """
    src = _read(COMPACTION)
    if src is None:
        pytest.skip("foresight/compaction.py does not exist yet. "
                    "`compact_now` must stay out of the spine "
                    "until it does — enforced by test_no_write_without_a_read.")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.skip(f"compaction.py does not parse right now ({e}); re-run.")
    admit = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "admit"), None)
    assert admit is not None, "compaction.py exists but has no admit()"
    args = [a.arg for a in admit.args.args if a.arg != "self"]
    assert args, "compaction.admit() takes no arguments; it cannot be a pressure gate"
    body = _function_source(COMPACTION, "admit") or ""
    assert re.search(r"return\s+.*,", body), (
        "compaction.admit() must return (bool, reason) — refusals must "
        "to be logged and REPORTED AS A RESULT, which needs a reason string.")


# ---------------------------------------------------------------------------
# the ctrl.raw log cap
# ---------------------------------------------------------------------------
def test_ctrl_raw_log_has_no_length_cap():
    """The 240-char cap must stay removed.

    It cut 8.5% of ticks in a full evaluation run (16,995 of 199,909, 100% of them landing
    exactly at the cap), and it cut them SELECTIVELY: the longest emissions, which are
    exactly the ticks that used the `more` tail. The forced spine makes every emission
    longer than the ones that were being cut, so a re-introduced cap would bias every
    new field's occupancy low — and the bias would look like the model declining.
    """
    src = _read(CONTROLLER)
    if src is None:
        pytest.skip("controller.py is missing")
    m = re.search(r"^.*log\(\s*[\"']ctrl\.raw[\"'].*(?:\n.*?)??\)\s*$",
                  src, re.MULTILINE)
    assert m, "no `log(\"ctrl.raw\", ...)` call found in controller.py"
    call = m.group(0)
    caps = re.findall(r"\[\s*:\s*\d+\s*\]", call)
    assert not caps, (
        f"a length cap {caps} is back on the ctrl.raw log line: {call.strip()!r}. "
        f"Every occupancy number taken off the next run would be biased low, "
        f"selectively on the longest emissions.")
    # the surrounding block is also checked, since the cap used to be applied to the
    # local before the call (`_raw = raw.strip()[:240]`).
    block = src[max(0, m.start() - 1200):m.end()]
    stale = re.findall(r"raw\.strip\(\)\s*\[\s*:\s*(\d+)\s*\]", block)
    assert not stale, f"`raw.strip()[:{stale[0]}]` is back above the ctrl.raw log call"


# ---------------------------------------------------------------------------
# the run assertions.  These are the tools, not just tests.
# ---------------------------------------------------------------------------
class OccupancyError(AssertionError):
    pass


def load_rows(root, tasks=None):
    """Every parsed tick under a run directory (or a single run_*.log)."""
    logs = fields_mod.find_logs([root])
    if not logs:
        raise OccupancyError(f"no run_*.log under {root!r}")
    rows = []
    for p in logs:
        for row in fields_mod.parse_log(p):
            if tasks and row["task"] not in tasks:
                continue
            rows.append(row)
    if not rows:
        raise OccupancyError(f"no ticks parsed under {root!r} — log format changed?")
    return rows


def assert_full_occupancy(root, *, plan_compact=False, plan_classes=False,
                          plan_question=True, min_rate=1.0, min_ticks=1):
    """Occupancy must be ~100% for every field in a task's whitelist.

    "A forced field with a fill rate below 100% means the walk did not run, not that
    the model declined." That is the whole reason to force a field: the decision to
    emit it is the CODE's, so anything under 100% is a control-flow bug — an exception
    swallowed mid-walk, a task taking a branch it should not, a key spelled differently
    by the walk than by the parser.

    Denominators are conditioned, because a conditional field measured against all
    ticks is guaranteed to look broken:
      * unconditional fields   -> every parsed tick (ok + truncated: a key before a
                                  cut is still observed, per fields.py)
      * event_time_s / answer  -> hit ticks only (`have_enough_info` true)
      * objects / keep/ignore  -> replan ticks only (`replan_vision` true)

    Returns {task: {field: (present, denominator)}} so a caller can report as well as
    assert. Raises OccupancyError listing every violation at once — a run is expensive,
    so surface all of them, not the first.
    """
    rows = load_rows(root)
    per, bad = {}, []
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"] or "?", []).append(r)
    for task, trs in sorted(by_task.items()):
        wl = task_whitelist(task, plan_compact=plan_compact,
                            plan_classes=plan_classes, plan_question=plan_question)
        keyed = [r for r in trs if r["raw_state"] in ("ok", "truncated")]
        if len(keyed) < min_ticks:
            bad.append(f"{task}: only {len(keyed)} parseable ticks "
                       f"(< min_ticks={min_ticks}) — the run did not produce a log "
                       f"this assertion can read")
            continue
        hit = [r for r in keyed if r["level"]]
        replan = [r for r in keyed if (r["diff"] or {}).get("replan_vision") is True]
        per[task] = {}
        for f in wl:
            if f in HIT_GATED:
                pool, cond = hit, "hit ticks"
            elif f in REPLAN_GATED:
                pool, cond = replan, "replan ticks"
            else:
                pool, cond = keyed, "all ticks"
            n = sum(1 for r in pool if f in (r["keys"] or ()))
            per[task][f] = (n, len(pool))
            if not pool:
                continue                       # nothing to divide by; not a violation
            rate = n / len(pool)
            if rate < min_rate:
                bad.append(
                    f"{task}.{f}: {100 * rate:.2f}% of {len(pool)} {cond} "
                    f"({n} present). A forced field below 100% means the schema walk "
                    f"did not run for those ticks, not that the model declined.")
    if bad:
        raise OccupancyError(
            f"forced-field occupancy violations under {root!r}:\n  "
            + "\n  ".join(bad))
    return per


def assert_answer_decodes_per_task(root):
    """IEA/SCA must log ZERO `answer` decodes; every other task non-zero.

    The time-only bypass is three coupled changes — drop the `bool(answer)`
    conjunct from the fire gate, emit a fixed string, and switch dedup to `ev0`. Get it
    half-right and the task does not error, it goes SILENT: no answer means no fire
    means no emissions, and the score drops to zero for a reason no metric names. This
    assertion is the tripwire, and its other half (a non-zero count everywhere else)
    catches the opposite mistake — a whitelist that dropped `answer` from every task.

    Returns {task: n_answer_ticks}.
    """
    rows = load_rows(root)
    counts, ticks = {}, {}
    for r in rows:
        t = r["task"] or "?"
        ticks[t] = ticks.get(t, 0) + 1
        counts[t] = counts.get(t, 0) + int("answer" in (r["keys"] or ()))
    bad = []
    for t, n in sorted(counts.items()):
        if t in metrics_mod.TIME_ONLY:
            if n:
                bad.append(f"{t} is time_only but decoded `answer` on {n}/{ticks[t]} "
                           f"ticks — the per-task whitelist did not take, and ~32 "
                           f"tokens/tick are being spent on text the scorer never "
                           f"reads")
        elif t != "?" and n == 0:
            bad.append(f"{t} decoded `answer` on 0/{ticks[t]} ticks — content is "
                       f"scored for this task, so it will score 0. Check the "
                       f"whitelist did not drop `answer` for everyone")
    if bad:
        raise OccupancyError("per-task answer-decode assertion failed:\n  "
                             + "\n  ".join(bad))
    return counts


# ---------------------------------------------------------------------------
# synthetic log fixtures — the tools above, exercised without a GPU
# ---------------------------------------------------------------------------
# Written into a directory NEXT TO this file (never a system temp dir) and removed
# afterwards, so a fixture left behind by a crashed run is visible in the tree rather
# than accumulating somewhere nobody looks.
_FIXTURE_DIR = os.path.join(_HERE, "_fixtures_plan_schema")


def _tick(vt, *, hit, answer="an answer", q="", extra_tail=False, fields_on=None,
          truncate_at=None, replan=False):
    """One (ctrl.raw, ctrl.gate) pair, in the forced spine's key order.

    `truncate_at` cuts the raw JSON: a negative value trims the tail so every KEY
    survives and only the last value is lost, which is the case fields.py salvages.
    """
    on = set(SPINE_ORDER if fields_on is None else fields_on)
    d = []
    if "seen" in on:
        d.append(('seen', '"a scene"'))
    if "have_enough_info" in on:
        d.append(('have_enough_info', "true" if hit else "false"))
    if hit and "event_time_s" in on:
        d.append(('event_time_s', f"{max(0.0, vt - 0.5):.1f}"))
    if hit and answer is not None and "answer" in on:
        d.append(('answer', f'"{answer}"'))
    if "fps" in on:
        d.append(('fps', "2"))
    if "next_check_s" in on:
        d.append(('next_check_s', "1.0"))
    if "compact_now" in on:
        d.append(('compact_now', "false"))
    if "replan_vision" in on:
        d.append(('replan_vision', "true" if replan else "false"))
    if replan:
        for k, v in (("objects", '"car, road"'), ("keep", '["car"]'),
                     ("ignore", '["sky"]')):
            if k in on:
                d.append((k, v))
    if "question_for_next" in on:
        d.append(('question_for_next', f'"{q or "what next?"}"'))
    if "count" in on:
        d.append(('count', "1"))
    if "phase" in on:
        d.append(('phase', '"test state"'))
    raw = "{" + ",".join(f'"{k}":{v}' for k, v in d) + "}"
    if truncate_at is not None:
        raw = raw[:truncate_at]
    tail = " p_hit=%.3f p_more=0.100" % (0.91 if hit else 0.10)
    if extra_tail:
        tail += " p_fps=0.700 p_cadence=0.610 p_compact=0.020 p_replan=0.550"
    gate = (f"fps=2.0 level={hit} rise={hit} new_occ=False fire={hit} "
            f"next=1.0s gen=1.4s ntok=25 q={q!r}{tail}")
    return (f"[  10.0s | vid {vt:5.1f}s] ctrl.raw  [VID1] {raw}\n"
            f"[  10.0s | vid {vt:5.1f}s] ctrl.gate [VID1] {gate}\n")


def _synthetic_log(ticks, task="dedup_counting", video="VID1", sample="7"):
    head = (f"[   0.0s | vid   0.0s] [online] ===== [1/1] {task}::{video}::{sample} "
            f"gt_times=[1.0] q='x'\n")
    return head + "".join(ticks)


def _parse_text(text, name="run_synth.log"):
    """Parse a log held in memory by staging it on disk where fields.py expects it."""
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    path = os.path.join(_FIXTURE_DIR, name)
    with open(path, "w") as fh:
        fh.write(text)
    return list(fields_mod.parse_log(path))


@pytest.fixture
def run_dir():
    """A throwaway run directory under this package; removed after each test."""
    d = os.path.join(_FIXTURE_DIR, "run_fixture")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(_FIXTURE_DIR, ignore_errors=True)


def _write_run(run_dir, task, ticks, name=None):
    with open(os.path.join(run_dir, name or f"run_{task}.log"), "w") as fh:
        fh.write(_synthetic_log(ticks, task=task))


def test_occupancy_helper_passes_on_a_correct_forced_run(run_dir):
    """The happy path: every unconditional field on every tick, `answer` on hit ticks."""
    _write_run(run_dir, "dedup_counting",
               [_tick(vt=float(i), hit=(i % 2 == 0)) for i in range(1, 11)])
    per = assert_full_occupancy(run_dir)
    assert per["dedup_counting"]["fps"][0] == 10
    assert per["dedup_counting"]["answer"][0] == 5      # hit ticks only


def test_occupancy_helper_catches_a_field_the_walk_skipped(run_dir):
    """A field forced by the spec but missing from 20% of ticks must be a hard error.

    This is the failure the assertion exists for: not a model that declined, a walk
    that did not run. `fps` fills 100% on 8 ticks and is absent on 2.
    """
    ticks = [_tick(vt=float(i), hit=False) for i in range(1, 9)]
    ticks += [_tick(vt=float(i), hit=False,
                    fields_on=set(SPINE_ORDER) - {"fps"}) for i in (9, 10)]
    _write_run(run_dir, "dedup_counting", ticks)
    with pytest.raises(OccupancyError) as e:
        assert_full_occupancy(run_dir)
    assert "dedup_counting.fps" in str(e.value)
    assert "80.00%" in str(e.value)


def test_occupancy_helper_counts_truncated_rows_as_present(run_dir):
    """A cut row still proves the keys before the cut were emitted (fields.py salvage).

    If the cap ever comes back, this is what stops the occupancy assertion from
    reporting a tooling artefact as a model failure — while
    test_ctrl_raw_log_has_no_length_cap stops the cap itself.
    """
    ticks = [_tick(vt=float(i), hit=False) for i in range(1, 9)]
    ticks += [_tick(vt=float(i), hit=False, truncate_at=-4) for i in (9, 10)]
    _write_run(run_dir, "dedup_counting", ticks)
    rows = load_rows(run_dir)
    assert sum(r["raw_state"] == "truncated" for r in rows) == 2
    per = assert_full_occupancy(run_dir)
    assert per["dedup_counting"]["seen"] == (10, 10)


def test_answer_decode_assertion_accepts_a_correct_per_task_run(run_dir):
    """IEA logs no `answer` at all; a counting task logs it on every hit tick."""
    _write_run(run_dir, "instant_event_alert",
               [_tick(vt=float(i), hit=True, answer=None) for i in range(1, 6)])
    _write_run(run_dir, "dedup_counting",
               [_tick(vt=float(i), hit=True) for i in range(1, 6)])
    counts = assert_answer_decodes_per_task(run_dir)
    assert counts["instant_event_alert"] == 0
    assert counts["dedup_counting"] == 5


def test_answer_decode_assertion_catches_answer_on_a_time_only_task(run_dir):
    """The whitelist failing to take on IEA — 32 wasted tokens on ~48% of ticks."""
    _write_run(run_dir, "instant_event_alert",
               [_tick(vt=float(i), hit=True) for i in range(1, 6)])
    with pytest.raises(OccupancyError) as e:
        assert_answer_decodes_per_task(run_dir)
    assert "instant_event_alert" in str(e.value)
    assert "time_only" in str(e.value)


def test_answer_decode_assertion_catches_a_silenced_content_task(run_dir):
    """The opposite mistake: the bypass applied to everyone, so a scored task goes mute.

    This is the half-applied time-only change — no answer means no fire means no emission,
    and the score goes to zero without any metric naming the cause.
    """
    _write_run(run_dir, "dedup_counting",
               [_tick(vt=float(i), hit=True, answer=None) for i in range(1, 6)])
    with pytest.raises(OccupancyError) as e:
        assert_answer_decodes_per_task(run_dir)
    assert "dedup_counting" in str(e.value)
    assert "0/5" in str(e.value)


# ---------------------------------------------------------------------------
# config-flag safety — the defaults cannot ship a write-without-a-read
# ---------------------------------------------------------------------------
def test_default_config_disables_every_field_without_a_consumer():
    """Defaults must leave every consumer-less field OFF.

    The ablation knobs (`plan_compact`, `plan_classes`, ...) are how a field is turned
    on for a measurement. Their DEFAULT is what every ordinary run gets, and a default
    of True on a field whose consumer does not exist ships the write-without-a-read to
    everyone at once.
    """
    defaults = config_defaults()
    if not defaults:
        pytest.skip("config.py could not be read")
    # flag names as they appear across the ablation and class-gating configs
    guarded = {
        "compact_now": ["plan_compact", "plan_compaction"],
        "keep": ["plan_classes", "plan_keep_classes"],
        "ignore": ["plan_classes", "plan_ignore_classes"],
        "replan_vision": ["plan_classes", "plan_replan_vision"],
        "objects": ["plan_classes", "plan_objects"],
    }
    checked, problems = [], []
    for fieldname, names in guarded.items():
        if CONSUMERS[fieldname].resolve()[0]:
            continue                       # consumer exists; the flag may default on
        for n in names:
            if n not in defaults:
                continue
            checked.append(n)
            if defaults[n]:
                problems.append(
                    f"cfg.{n}={defaults[n]!r} by default, but `{fieldname}`'s "
                    f"consumer ({CONSUMERS[fieldname].what}) does not exist yet: "
                    f"{CONSUMERS[fieldname].resolve()[1]}. Default it False until the "
                    f"reader is merged.")
    assert not problems, "unsafe config defaults:\n  " + "\n  ".join(problems)
    if not checked:
        pytest.skip("none of the plan_* ablation flags exist in config.py yet "
                    "; this test activates when they land.")


def test_live_async_is_default_and_lockstep_handshake_is_conditional():
    """Contract: async by default, lockstep opt-in."""
    defaults = config_defaults()
    if "deterministic" not in defaults:
        pytest.skip("config.py has no `deterministic` field right now")
    assert defaults["deterministic"] is False
    assert defaults["realtime"] is True
    ingester = _code_only(os.path.join(_HERE, "input_ingester.py"))
    assert "if cfg.deterministic and clock is not None" in ingester
    assert "if cfg.deterministic and clock is not None" in (_code_only(VISION_STREAM) or "")


# ---------------------------------------------------------------------------
# determinism.  Needs the model; documented, not faked.
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="needs a GPU and two full runs; procedure documented in the "
                         "test body. Run it on a GPU machine, not here.")
def test_determinism_same_arm_twice_is_bit_identical():
    """Determinism gate — the exact procedure, for whoever runs it.

    NOT runnable here: it needs the model, a GPU, and two complete passes over a shard.
    Faking it with a stub would be worse than skipping it, because a green stub is
    indistinguishable from a real pass in CI output.

    Procedure (GPU machine):

        cd foresight
        OMNIPRO_ARM=A00_all ./run.sh --shard 0 --out runs/det_a
        OMNIPRO_ARM=A00_all ./run.sh --shard 0 --out runs/det_b
        # strip wall-clock fields, then compare the decode spine byte for byte:
        for f in ctrl.raw ctrl.gate; do
          diff <(grep -o "$f .*" runs/det_a/run_*.log | sed 's/gen=[0-9.]*s//') \\
               <(grep -o "$f .*" runs/det_b/run_*.log | sed 's/gen=[0-9.]*s//')
        done
  python evaluation/ runs/det_a runs/det_b

    Expected: zero differing lines. `vt`, `p_hit`, every forced key and every sampled
    token must match; only wall-clock timings may differ.

    Two things to know before interpreting a failure:

      * `writer_greedy=False` (config.py) is a LIVE SUSPECT for non-reproducibility —
        an earlier null control, in which the same mode run twice differed on
        17 of 25 ticks, which made an entire A/B undecidable. If this gate fails, set
        `writer_greedy=True` and re-run before concluding anything about the schema.
      * This path is SINGLE-PROCESS and lockstep (`deterministic=True`), so it *should*
        hold. If it does not, that is a finding in its own right and it blocks every
        ablation arm — an arm cannot be compared against a
        baseline whose noise floor is the size of the effect.
    """


@pytest.mark.skip(reason="needs the tokenizer (blocking check); run on a "
                         "GPU machine with the model checkout available.")
def test_choice_surface_forms_are_single_tokens():
    """Blocking check, for whoever has the tokenizer.

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model_path)
        for s in ["1", "2", "3", "0.5", "1.0", "1.5", "keep", "drop", "neutral"]:
            ids = tok.encode(s, add_special_tokens=False)
            assert len(ids) == 1, (s, ids)

    `_read_choice` reads a softmax restricted to SINGLE-TOKEN surface forms. If a form
    splits, the read is over the first piece of a multi-token string and the resulting
    probability is meaningless — and it would look like a working field. The fallback
    (a capped 2-token decode for that field) must be taken LOUDLY, the way controller.py
    already refuses `inplace` and logs the downgrade.
    """


# ---------------------------------------------------------------------------
# CLI — point the run assertions at a real run
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="a run output dir, or one run_*.log")
    ap.add_argument("--plan-compact", action="store_true")
    ap.add_argument("--plan-classes", action="store_true")
    ap.add_argument("--no-question", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    try:
        per = assert_full_occupancy(a.root, plan_compact=a.plan_compact,
                                    plan_classes=a.plan_classes,
                                    plan_question=not a.no_question)
        for t in sorted(per):
            print(f"[occupancy] {t}: " + "  ".join(
                f"{f}={n}/{d}" for f, (n, d) in per[t].items()))
        print("[occupancy] OK — every whitelisted field at 100% of its eligible ticks")
    except OccupancyError as e:
        print(f"[occupancy] FAIL\n{e}", file=sys.stderr)
        rc = 1
    try:
        counts = assert_answer_decodes_per_task(a.root)
        print("[answer] OK — " + "  ".join(f"{t}={n}" for t, n in sorted(counts.items())))
    except OccupancyError as e:
        print(f"[answer] FAIL\n{e}", file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
