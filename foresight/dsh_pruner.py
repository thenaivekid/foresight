"""
dsh_pruner.py — temporal-novelty pruning via Dynamically-Smoothed History (DSH).

Implements the query-agnostic temporal-novelty gate from the QueryStream paper
(ICLR 2026) and TimeChat-Online: each patch position maintains an exponentially
smoothed history vector, and a patch is kept only when it deviates from its own
history.

Two selection rules (cfg.dsh_rule):
  "rank"      (DEFAULT) — keep the ceil(dsh_keep_frac * N) most-novel positions
              per frame.  Scale-free: works identically whether mean cosine is
              0.82 (dynamic video) or 0.999 (near-static).  Motivated by the
              empirical finding that absolute cosine scale is video-dependent in
  Qwen3-VL's embedding space (, ).
  "threshold" — keep positions with cos < dsh_tau_temp (the original design).
              Retained for ablation; NOT recommended for production.

Key properties:
  - Pure functions, no global state, unit-testable on CPU with fake tensors.
  - Fail-safe: first frame always kept in full (seeds the history).
  - History always updated from the FULL frame (all patches, pre-pruning), so
    the temporal baseline tracks reality, not the pruned view.
  - Variable N handled explicitly: when N changes between frames (Qwen3-VL
    dynamic resolution), the history is reset and the frame is kept in full.
  - Deterministic: ties broken by ascending token index.
  - Dependency-light: torch only, no new packages.

Config fields (all read via getattr with defaults):
  dsh_rule             "rank"   selection rule ("rank" or "threshold")
  dsh_keep_frac        0.30     fraction of tokens to keep in rank mode
  dsh_tau_temp         0.90     cosine threshold (threshold mode only)
  dsh_alpha            0.10     EMA smoothing coefficient
  dsh_retention_floor  0.30     minimum fraction to keep (threshold mode only)
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Shared state: history vectors for the DSH temporal filter
# ---------------------------------------------------------------------------

class DSHState:
    """Thread-safe container for the per-patch smoothed history vectors.

    The encoder thread is the sole writer/reader in practice, but the lock
    allows safe reset from the controller thread if needed (e.g. on scene cut).
    Mirrors ``ClassPrunerState``'s pointer-swap pattern.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v_dsh: Optional[torch.Tensor] = None   # [N, H] float32
        self._N: int = 0                               # token count of current history

    # -- read / write (encoder thread) ------------------------------------

    def get(self) -> Tuple[Optional[torch.Tensor], int]:
        """Return (v_dsh_copy_or_None, N). None means no history yet."""
        with self._lock:
            if self._v_dsh is None:
                return None, 0
            return self._v_dsh.clone(), self._N

    def update(self, v_dsh_new: torch.Tensor, N: int) -> None:
        """Replace the history with *v_dsh_new* ``[N, H]``."""
        with self._lock:
            self._v_dsh = v_dsh_new.clone()
            self._N = N

    # -- control (controller / reset) --------------------------------------

    def reset(self) -> None:
        """Clear the history. Next frame will be treated as first frame."""
        with self._lock:
            self._v_dsh = None
            self._N = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _unit_normalise(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation with zero-norm guard.  Input: [N, H]."""
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return x / norms


def _compute_cos_sim(
    tokens: torch.Tensor,
    v_dsh_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-position cosine similarity and normalised vectors.

    Args:
        tokens:     [N, H] float32 current frame.
        v_dsh_prev: [N, H] float32 history (may not be unit-norm after EMA).

    Returns:
        (cos_sim [N], tokens_normed [N,H], history_normed [N,H])
    """
    tokens_normed = _unit_normalise(tokens)
    history_normed = _unit_normalise(v_dsh_prev)
    cos_sim = (tokens_normed * history_normed).sum(dim=1).clamp(-1.0, 1.0)
    return cos_sim, tokens_normed, history_normed


def _select_topk_novel(
    cos_sim: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select the k positions with the LOWEST cosine (most novel).

    Deterministic tie-break: among equal cosines, lower index wins.

    Args:
        cos_sim: [N] cosine similarities.
        k:       number of positions to keep (must be >= 1).

    Returns:
        Sorted 1-D int64 tensor of kept indices (ascending, spatial order).
    """
    N = cos_sim.shape[0]
    k = min(k, N)
    # Composite key: cos_sim + tiny index-proportional offset for tie-break.
    # With cosines in [-1, 1] and indices < 1e6, 1e-9 per index is safe.
    key = cos_sim + torch.arange(N, dtype=cos_sim.dtype, device=cos_sim.device) * 1e-9
    _, order = key.sort()          # ascending = most novel first
    selected = order[:k]
    selected, _ = selected.sort()  # restore spatial order
    return selected


# ---------------------------------------------------------------------------
# Legacy: threshold-mode keep mask (original design, for ablation)
# ---------------------------------------------------------------------------

def dsh_keep_mask(
    embeds: torch.Tensor,
    state: DSHState,
    tau_temp: float = 0.90,
    alpha: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute which tokens to keep based on temporal novelty (threshold mode).

    Args:
        embeds:    ``[1, N, H]`` post-merger visual tokens for the current frame.
        state:     ``DSHState`` holding the smoothed history.
        tau_temp:  cosine similarity threshold. A token whose similarity to its
                   history is **below** this is considered novel (kept).
        alpha:     EMA coefficient for the history update.

    Returns:
        ``(keep_idx, stats)`` where *keep_idx* is a 1-D int64 tensor of token
        indices to keep (sorted ascending), and *stats* dict.

    Side-effect: updates *state* with the new EMA history.
    """
    tokens = embeds[0].float()
    N, H = tokens.shape

    v_dsh_prev, prev_N = state.get()

    history_reset = False

    if v_dsh_prev is None or prev_N != N:
        history_reset = True
        keep_idx = torch.arange(N, dtype=torch.long)
        mean_cos = 1.0
        v_dsh_new = _unit_normalise(tokens)
        state.update(v_dsh_new, N)
        stats = {
            "keep_rate": 1.0, "n_in": N, "n_out": N,
            "mean_cos": mean_cos, "history_reset": history_reset,
            "rule": "threshold",
        }
        return keep_idx, stats

    cos_sim, tokens_normed, history_normed = _compute_cos_sim(tokens, v_dsh_prev)
    mean_cos = cos_sim.mean().item()

    novel_mask = cos_sim < tau_temp
    keep_idx = torch.where(novel_mask)[0]

    v_dsh_new = alpha * tokens_normed + (1.0 - alpha) * history_normed
    state.update(v_dsh_new, N)

    stats = {
        "keep_rate": keep_idx.shape[0] / N if N > 0 else 1.0,
        "n_in": N, "n_out": int(keep_idx.shape[0]),
        "mean_cos": mean_cos, "history_reset": history_reset,
        "rule": "threshold",
    }
    return keep_idx, stats


# ---------------------------------------------------------------------------
# Retention floor enforcement (threshold mode only)
# ---------------------------------------------------------------------------

def _enforce_floor(
    keep_idx: torch.Tensor,
    cos_sim: torch.Tensor,
    N: int,
    retention_floor: float,
) -> torch.Tensor:
    """Ensure at least ceil(floor * N) tokens are kept.

    When the novelty test keeps fewer than the floor, fill the quota with the
    most-novel tokens (lowest cosine to history).  Ties broken by ascending
    index for determinism.
    """
    min_keep = max(1, math.ceil(retention_floor * N))
    if keep_idx.shape[0] >= min_keep:
        return keep_idx
    return _select_topk_novel(cos_sim, min_keep)


# ---------------------------------------------------------------------------
# Entry point — full per-frame pipeline
# ---------------------------------------------------------------------------

def prune_frame_dsh(
    embeds: torch.Tensor,
    state: DSHState,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Score and prune one frame of vision tokens by temporal novelty.

    Reads configuration from *cfg* via ``getattr`` with defaults:

    +------------------------+---------+-----------------------------------------+
    | cfg field              | default | meaning                                 |
    +------------------------+---------+-----------------------------------------+
    | dsh_rule               | "rank"  | "rank" (scale-free) or "threshold"      |
    | dsh_keep_frac          | 0.30    | fraction to keep (rank mode)            |
    | dsh_tau_temp           | 0.90    | cosine threshold (threshold mode only)  |
    | dsh_alpha              | 0.10    | EMA smoothing coefficient               |
    | dsh_retention_floor    | 0.30    | min fraction to keep (threshold only)   |
    +------------------------+---------+-----------------------------------------+

    Args:
        embeds: ``[1, N, H]`` from ``backend.embed_frame``.
        state:  ``DSHState`` instance (persistent across frames).
        cfg:    config object (e.g. ``AsyncOmniConfig``).

    Returns:
        ``(pruned_embeds, stats)`` where *stats* contains at least
        ``n_in``, ``n_out``, ``keep_rate``, ``mean_cos``, ``rule``,
        ``history_reset``.
    """
    rule = getattr(cfg, "dsh_rule", "rank")
    alpha = getattr(cfg, "dsh_alpha", 0.10)

    tokens = embeds[0].float()
    N, H = tokens.shape

    v_dsh_prev, prev_N = state.get()

    history_reset = False

    # --- First frame or N changed: keep everything, seed history -----------
    if v_dsh_prev is None or prev_N != N:
        history_reset = True
        v_dsh_new = _unit_normalise(tokens)
        state.update(v_dsh_new, N)
        stats = {
            "n_in": N, "n_out": N, "keep_rate": 1.0,
            "mean_cos": 1.0, "rule": rule, "history_reset": True,
        }
        return embeds, stats

    # --- Compute per-position cosine similarity ----------------------------
    cos_sim, tokens_normed, history_normed = _compute_cos_sim(tokens, v_dsh_prev)
    mean_cos = cos_sim.mean().item()

    # --- Selection ---------------------------------------------------------
    if rule == "rank":
        keep_frac = getattr(cfg, "dsh_keep_frac", 0.30)
        k = max(1, math.ceil(keep_frac * N))
        keep_idx = _select_topk_novel(cos_sim, k)
    elif rule == "threshold":
        tau_temp = getattr(cfg, "dsh_tau_temp", 0.90)
        retention_floor = getattr(cfg, "dsh_retention_floor", 0.30)
        novel_mask = cos_sim < tau_temp
        keep_idx = torch.where(novel_mask)[0]
        keep_idx = _enforce_floor(keep_idx, cos_sim, N, retention_floor)
    else:
        raise ValueError(f"dsh_rule must be 'rank' or 'threshold', got {rule!r}")

    # --- Always update history from full frame -----------------------------
    v_dsh_new = alpha * tokens_normed + (1.0 - alpha) * history_normed
    state.update(v_dsh_new, N)

    # --- Prune and return --------------------------------------------------
    pruned = embeds[:, keep_idx, :]
    n_out = int(keep_idx.shape[0])
    stats = {
        "n_in": N, "n_out": n_out,
        "keep_rate": n_out / N if N > 0 else 1.0,
        "mean_cos": mean_cos, "rule": rule, "history_reset": history_reset,
    }
    return pruned, stats
