"""
class_pruner.py — semantic-class-based pruning of encoded vision tokens.

Implements the APPLY stage of class-based pruning: scores each post-merger
visual token against controller-provided keep/ignore class lists using cosine
similarity in the LM input space, and retains the top tokens by score.

Key properties:
  - Pure functions, no global state, unit-testable on CPU with fake tensors.
  - Fail-safe: empty keep list -> no-op (never prune without positive guidance).
  - Retention floor: a frame is never emptied; at least ``retention_floor``
    fraction of tokens always survive.
  - Deterministic: ties broken by token index).
  - Class embeddings cached and recomputed only when the class lists change.

Gate: the cosine-alignment probe must PASS before this
is enabled in production. The probe measures whether the cosine alignment has
discriminative signal; this module assumes it does.

Invariants respected:
  INVARIANT 1 — uses only current frame + cached class names, no look-ahead.
  INVARIANT 2 — one matmul per frame (~200x3584x8 FLOPs); does not block.
  INVARIANT 4 — no gradients, no training; frozen embedding cosine only.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Shared state: controller (writer) <-> encoder (reader)
# ---------------------------------------------------------------------------

class ClassPrunerState:
    """Thread-safe container for the current keep/ignore class lists.

    The controller calls ``set_classes`` on replan ticks.
    The encoder calls ``get_classes`` every frame.  Reads and writes are
    lock-protected pointer swaps — neither side ever blocks the other for more
    than a pointer update, mirroring ``EncoderControl.set_fps/get_fps``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keep: List[str] = []
        self._ignore: List[str] = []
        self._version: int = 0

    def set_classes(self, keep: List[str], ignore: List[str]) -> None:
        """Called by the controller when replan_vision fires."""
        with self._lock:
            self._keep = list(keep)
            self._ignore = list(ignore)
            self._version += 1

    def get_classes(self) -> Tuple[List[str], List[str], int]:
        """Called by the encoder each frame. Returns (keep, ignore, version)."""
        with self._lock:
            return list(self._keep), list(self._ignore), self._version


# ---------------------------------------------------------------------------
# Class embedding cache
# ---------------------------------------------------------------------------

def embed_classes(
    embed_text_fn: Callable[[str], torch.Tensor],
    class_names: List[str],
    cache: Dict[str, torch.Tensor],
    template: str = "{}",
) -> Dict[str, torch.Tensor]:
    """Compute (or retrieve) unit-norm, mean-pooled embeddings for class names.

    Mean-pooling is the honest choice for multi-token nouns (see
  ````, ``class_embeddings``).  Normalisation and the mean
    are computed in float32 regardless of model dtype, because bf16 has ~3
    significant decimal digits and the margins here are small.

    Only computes embeddings for names not already in *cache*.
    """
    for name in class_names:
        if name not in cache:
            raw = embed_text_fn(template.format(name))[0].float()   # [L, H]
            v = raw.mean(dim=0)                                      # [H]
            cache[name] = v / (v.norm() + 1e-12)
    return {n: cache[n] for n in class_names if n in cache}


# ---------------------------------------------------------------------------
# Scoring — class-score formula
# ---------------------------------------------------------------------------

def compute_scores(
    vision_tokens: torch.Tensor,
    keep_embs: Optional[torch.Tensor],
    ignore_embs: Optional[torch.Tensor],
    ignore_lambda: float = 0.5,
) -> torch.Tensor:
    """Per-token class-relevance score.

    ``score[i] = max_k cos(v_i, e_keep[k]) - lambda * max_j cos(v_i, e_ign[j])``

    Args:
        vision_tokens: ``[N, H]`` post-merger visual tokens in LM input space.
        keep_embs:     ``[K, H]`` unit-norm keep-class embeddings, or ``None``.
        ignore_embs:   ``[J, H]`` unit-norm ignore-class embeddings, or ``None``.
        ignore_lambda: weight on the ignore (suppression) term.

    Returns:
        ``[N]`` float32 score tensor.
    """
    N = vision_tokens.shape[0]
    device = vision_tokens.device
    vf = vision_tokens.float()
    norms = vf.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vn = vf / norms

    if keep_embs is not None and keep_embs.shape[0] > 0:
        cos_keep = vn @ keep_embs.t()                        # [N, K]
        keep_max = cos_keep.max(dim=1).values                # [N]
    else:
        keep_max = torch.zeros(N, dtype=torch.float32, device=device)

    if ignore_embs is not None and ignore_embs.shape[0] > 0:
        cos_ign = vn @ ignore_embs.t()                       # [N, J]
        ign_max = cos_ign.max(dim=1).values                  # [N]
    else:
        ign_max = torch.zeros(N, dtype=torch.float32, device=device)

    return keep_max - ignore_lambda * ign_max


# ---------------------------------------------------------------------------
# Selection — deterministic top-k with retention floor
# ---------------------------------------------------------------------------

def select_tokens(
    embeds: torch.Tensor,
    scores: torch.Tensor,
    retention_floor: float = 0.30,
) -> Tuple[torch.Tensor, List[int]]:
    """Select top tokens by score, respecting the retention floor.

    Deterministic tie-breaking by ``(-score, index)`` — see
  ```` ``topk_indices``.

    Args:
        embeds: ``[1, N, H]`` vision embeddings.
        scores: ``[N]`` per-token scores from ``compute_scores``.
        retention_floor: minimum fraction of tokens to keep.

    Returns:
        ``(pruned_embeds, kept_indices)`` where ``pruned_embeds`` is
        ``[1, M, H]`` with ``M >= ceil(retention_floor * N)`` and
        ``kept_indices`` is a sorted list of original token positions
        (preserves spatial order).
    """
    N = scores.shape[0]
    min_keep = max(1, math.ceil(retention_floor * N))

    score_list = scores.tolist()
    order = sorted(range(N), key=lambda i: (-score_list[i], i))

    n_keep = max(min_keep, 1)
    kept = sorted(order[:n_keep])

    pruned = embeds[:, kept, :]
    return pruned, kept


# ---------------------------------------------------------------------------
# Entry point — full per-frame pipeline
# ---------------------------------------------------------------------------

def prune_frame(
    embeds: torch.Tensor,
    keep_classes: List[str],
    ignore_classes: List[str],
    embed_text_fn: Callable[[str], torch.Tensor],
    embed_cache: Dict[str, torch.Tensor],
    retention_floor: float = 0.30,
    ignore_lambda: float = 0.5,
    class_template: str = "{}",
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Score and prune one frame of vision tokens.

    **FAIL-SAFE**: if *keep_classes* is empty, returns *embeds* unchanged.
    An empty keep list means the controller has not named anything to preserve,
    so the pruner has no positive signal to select on — pruning without one
    would discard tokens based only on the ignore term, which is more aggressive
    than intended and not what the figure describes.

    Args:
        embeds:          ``[1, N, H]`` from ``backend.embed_frame``.
        keep_classes:    class names to preserve (from the controller plan).
        ignore_classes:  class names to suppress.
        embed_text_fn:   ``backend.embed_text`` callable.
        embed_cache:     persistent ``{name: [H] tensor}`` dict, updated in place.
        retention_floor: minimum fraction of tokens to keep.
        ignore_lambda:   weight on the ignore term in scoring.
        class_template:  text template for embedding class names.

    Returns:
        ``(pruned_embeds, stats_or_None)``.  *stats* is ``None`` on a no-op.
    """
    if not keep_classes:
        return embeds, None

    N = embeds.shape[1]

    all_names = sorted(set(keep_classes) | set(ignore_classes))
    class_embs = embed_classes(embed_text_fn, all_names, embed_cache,
                               class_template)

    keep_vecs = [class_embs[c] for c in keep_classes if c in class_embs]
    ign_vecs  = [class_embs[c] for c in ignore_classes if c in class_embs]

    if not keep_vecs:
        return embeds, None

    keep_mat = torch.stack(keep_vecs)
    ign_mat  = torch.stack(ign_vecs) if ign_vecs else None

    scores = compute_scores(embeds[0], keep_mat, ign_mat, ignore_lambda)
    pruned, kept_idx = select_tokens(embeds, scores, retention_floor)

    n_kept = len(kept_idx)
    stats = {
        "n_original": N,
        "n_kept": n_kept,
        "retention": n_kept / N if N > 0 else 1.0,
        "keep_classes": list(keep_classes),
        "ignore_classes": list(ignore_classes),
    }
    return pruned, stats
