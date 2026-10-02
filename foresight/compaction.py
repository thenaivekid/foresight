"""
compaction.py — the ADMISSION GATE for the plan's `compact_now` field.

`compact_now` is a **request from the model; the CODE decides.** That asymmetry is
the whole design: the controller reads one bool off the logits (free, zero decode
steps) and this module — pure, torch-free, GPU-free, unit-testable — decides
whether the request is honored, and if not, *why*, with a stable machine-readable
reason string.

Why so little machinery: KV cache management is outside the scope of this work,
so this is the defensible minimum: emit the field (it is ~free), wire a STRICT
admission gate, and **report the refusal counts** — a real result either way. A gate
that refuses 98% of requests, with the refusals broken down by reason, is a
finding; a gate that rubber-stamps them is a liability on a frozen model whose
cache is the only memory the system has.

The model's request is NECESSARY BUT NOT SUFFICIENT. Four independent conditions
must also hold, in this fixed evaluation order (order is part of the contract:
the refusal histogram is only comparable across arms if the same tick always
attributes to the same reason):

    disabled -> bad_input -> budget_too_large_noop -> not_requested
    -> low_confidence -> below_pressure -> insufficient_novelty -> cooldown

Determinism: every input is a float derived from state at <= vt, and the only
retained state is the video time of the last admission. No wall clock, no RNG.
Two runs over the same tick sequence produce the same decisions (INVARIANT:
lockstep determinism).

INVARIANT 3 (single writer) is untouched: this module decides, it does not act.
The actual compaction, whenever it is implemented, must still be performed by the
ingester — the sole writer of the primary KV cache — never by the controller.

---------------------------------------------------------------------------
TWO HARD PREREQUISITES, NEITHER OF WHICH THIS FILE MAY PERFORM
---------------------------------------------------------------------------
1. **`kv_budget` must go 262144 -> 32768** (`config.py`, owned elsewhere).
   At 256K the pressure gate is UNREACHABLE on this subset: every eval runs
   `max_seconds=300` at ~1 fps and ~180 vision tokens/frame, i.e. ~55k tokens,
   against a 0.85 * 262144 = ~223k trigger. Every compaction arm would be a
   silent no-op and the ablation would report "no effect" for a mechanism that
   never ran. `admit()` DETECTS this configuration and returns
   `budget_too_large_noop`, so the situation shows up as a named row in the
   refusal counts instead of as a plausible-looking zero.
2. **Position re-basing must be enabled first.** Compaction
   and eviction both make physical cache length diverge from the logical RoPE
   clock. `manager.py:_evict_locked` now refuses to continue silently past that
   divergence (`on_evict="raise"` by default) and carries an opt-in re-basing
   implementation; read that code before turning any of this on.
"""

_DEF = {
    # Master switch. OFF by default: a default-config run must reproduce today's
    # behaviour exactly, and today nothing compacts. `cfg.plan_compact` is the
    # name config.py uses; `cfg.plan_compaction` is accepted as an alias so the
    # two halves of this change cannot disagree silently.
    "plan_compact": False,
    # Occupancy (phys_len / kv_budget) below which the cache is simply not under
    # pressure. 0.85 leaves ~15% of the window as headroom for the controller's
    # own borrow (a tick appends its prompt + decode to the primary and truncates
    # it back — manager.borrow_begin), so admission never fires *because of* a
    # transient the reader itself created.
    "compact_occupancy_thr": 0.85,
    # P(compact_now=true) from the restricted logit read. 0.70, not 0.50: the
    # 0.50 crossing is "the model leans yes", and `have_enough_info` (the one
    # calibrated bool we have measured) needed a high threshold before its
    # positives were worth acting on. Compaction is destructive and irreversible;
    # it should cost more confidence than an emission does.
    "compact_p_thr": 0.70,
    # Video-time cooldown. Compaction changes the cache the next tick reads, so
    # back-to-back compactions feed on their own output — the thrash mode. 30 s
    # is ~30 ticks at the 1 fps grid: long enough that the effect of one
    # compaction is observable in the emissions before another is allowed.
    "compact_cooldown_s": 30.0,
    # Novelty ceiling. Compaction discards information; it is defensible when the
    # recent stream is REDUNDANT and indefensible when it is carrying new content.
    # 0.60 on the caller's [0,1] novelty scale = "clearly more novel than not".
    "compact_novelty_max": 0.60,
    # Only used to detect prerequisite (1). Measured: ~55k tokens per 300 s clip
    # (manager.py's own 144 KB/token / ~8 GB figure), i.e. ~180 tokens/frame at
    # 1 fps, including the timestamp text tokens.
    "compact_tokens_per_frame": 180.0,
}

# Video-time of the last ADMITTED compaction. Module state, not wall-clock state:
# it is a function of the tick sequence alone, so it cannot make a run irreproducible.
_last_admit_vt = None


def reset(last_vt=None):
    """Clear (or preset) the cooldown state. Call once per video/run — a new clip
    restarts video time at 0 and must not inherit the previous clip's cooldown."""
    global _last_admit_vt
    _last_admit_vt = last_vt


def last_admit_vt():
    return _last_admit_vt


def _get(cfg, name):
    val = getattr(cfg, name, _DEF[name]) if cfg is not None else _DEF[name]
    return _DEF[name] if val is None else val


def _enabled(cfg):
    if cfg is None:
        return False
    for name in ("plan_compact", "plan_compaction"):
        val = getattr(cfg, name, None)
        if val is not None:
            return bool(val)
    return False


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _pressure_reachable(cfg):
    """False when the configured kv_budget is so large that occupancy can never
    reach the threshold on this benchmark — prerequisite (1) not done."""
    budget = _get_raw(cfg, "kv_budget", 262144)
    if not budget or budget <= 0:
        return True                       # unknown budget: do not claim a no-op
    max_seconds = _get_raw(cfg, "max_seconds", 300.0)
    fps = _get_raw(cfg, "fps", 1.0)
    # An fps-steered run can only ever encode MORE frames than the base rate, so
    # using the base fps here is the conservative direction: it can only make us
    # under-claim reachability, never falsely report a no-op.
    est_tokens = max_seconds * max(fps, 0.0) * _get(cfg, "compact_tokens_per_frame")
    return est_tokens >= _get(cfg, "compact_occupancy_thr") * budget


def _get_raw(cfg, name, default):
    val = getattr(cfg, name, default) if cfg is not None else default
    return default if val is None else val


def admit(occupancy: float, novelty: float, p_compact: float, vt: float, cfg=None):
    """Decide whether the model's `compact_now` request is honored.

    Args (all plain floats — no tensors, no cache handles, no GPU):
      occupancy: physical cache length / kv_budget, in [0, 1]. See
                 `KVCacheManager.occupancy()`.
      novelty:   [0, 1], how new the recent stream is; HIGH means "do not throw
                 this away". Any monotone caller-side estimator is acceptable as
                 long as one arm uses one estimator.
      p_compact: P(compact_now = true) from the restricted logit read, in [0, 1].
      vt:        video time of this tick, seconds. Video time, never wall clock.

    Returns (admitted, reason). `reason` is one of the stable strings:

        "admitted"               request honored; cooldown starts at `vt`
        "disabled"               cfg.plan_compact is False (the default)
        "bad_input"              a non-finite / out-of-range argument
        "budget_too_large_noop"  kv_budget so large the pressure gate is
                                 unreachable on this subset -> every compaction
                                 arm would be a silent no-op (prerequisite 1)
        "not_requested"          the model did not ask (p_compact < 0.5)
        "low_confidence"         asked, but below cfg.compact_p_thr
        "below_pressure"         cache not under pressure yet
        "insufficient_novelty"   the novelty condition is not satisfied: the
                                 recent stream is TOO novel to discard safely
        "cooldown"               < cfg.compact_cooldown_s of video time since the
                                 last admitted compaction

    Admission records `vt` for the cooldown; refusals record nothing. Call
    `reset()` between videos.
    """
    global _last_admit_vt
    if not _enabled(cfg):
        return False, "disabled"
    if not (_finite(occupancy) and _finite(novelty) and _finite(p_compact) and _finite(vt)):
        return False, "bad_input"
    if not (0.0 <= occupancy <= 1.0) or not (0.0 <= novelty <= 1.0) or not (0.0 <= p_compact <= 1.0):
        return False, "bad_input"
    if not _pressure_reachable(cfg):
        return False, "budget_too_large_noop"
    if p_compact < 0.5:
        return False, "not_requested"
    if p_compact < _get(cfg, "compact_p_thr"):
        return False, "low_confidence"
    if occupancy < _get(cfg, "compact_occupancy_thr"):
        return False, "below_pressure"
    if novelty > _get(cfg, "compact_novelty_max"):
        return False, "insufficient_novelty"
    if _last_admit_vt is not None and (vt - _last_admit_vt) < _get(cfg, "compact_cooldown_s"):
        return False, "cooldown"
    _last_admit_vt = vt
    return True, "admitted"


#: Every reason `admit()` can return. Iterate this to build the refusal table so
#: a reason that is never hit still appears as an explicit zero.
REASONS = ("admitted", "disabled", "bad_input", "budget_too_large_noop",
           "not_requested", "low_confidence", "below_pressure",
           "insufficient_novelty", "cooldown")


class RefusalCounter:
    """Tally of `admit()` outcomes — the reportable result of the gate.

    Kept here rather than in the controller so the numbers in the paper come from
    the same file as the policy that produced them."""

    def __init__(self):
        self.counts = {r: 0 for r in REASONS}

    def note(self, reason):
        self.counts[reason] = self.counts.get(reason, 0) + 1

    def admit(self, occupancy, novelty, p_compact, vt, cfg=None):
        ok, reason = admit(occupancy, novelty, p_compact, vt, cfg)
        self.note(reason)
        return ok, reason

    def summary(self):
        total = sum(self.counts.values())
        parts = " ".join(f"{r}={self.counts[r]}" for r in REASONS if self.counts[r])
        return f"compaction: {total} decisions [{parts or 'none'}]"
