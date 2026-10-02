"""
clip_pruner.py — CLIP-space semantic pruning of Qwen visual tokens (QDP Route 1).

Implements the *semantic half* of QueryStream Sec. 3.2: score CLIP ViT-L/14
per-patch features against the CLIP text embedding of the user query, then
apply a **frame-adaptive** mean-threshold mask to the Qwen post-merger visual
tokens via bilinear resampling of the CLIP score map.

    M_sem(t,i) = 1[ sim(q, v_i) > mean_j sim(q, v_j) ]

Key properties (mirrors class_pruner.py house style):
  - Pure functions, no global state (all state in CLIPScorerState), unit-testable.
  - Fail-safe: empty/blank query -> no-op (never prune without a query).
  - Retention floor: at least ``retention_floor`` fraction of tokens always survive.
  - Deterministic: ties broken by ascending token index; indices returned sorted.
  - CLIP lazy-loaded once as a module-level singleton, eval mode, torch.no_grad().

Grid mismatch handling (THE CENTRAL TECHNICAL PROBLEM):
  CLIP ViT-L/14: 16x16 = 256 patches at 224x224.
  Qwen3-VL post-merger: dynamic grid (gh, gw) depending on frame resolution.
  Recovery: ``image_grid_thw`` from Qwen's image processor gives the pre-merger
  grid; post-merger grid is ``(h // merge_size, w // merge_size)`` where
  ``merge_size = 2``. See backend.py:130-140 (``embed_frame``).
  We resample CLIP's 16x16 score map to (gh, gw) via bilinear interpolation,
  then flatten row-major to get one score per Qwen token.

Aspect-ratio treatment:
  CLIP's default processor center-crops to 224x224 (losing edges on non-square
  images). To preserve spatial correspondence with Qwen (which preserves AR),
  we pre-resize the PIL image to 224x224 *without cropping* (stretch-to-fill)
  before CLIP forward. This introduces mild AR distortion in CLIP features but
  ensures that CLIP patch (r,c) and Qwen token at grid position (r',c') refer
  to the same image region after bilinear resampling. Residual misalignment is
  bounded by 1/(2*min(gh,gw)) pixels — sub-patch for typical grids.

Cost model:
  CLIP ViT-L/14 forward: ~4ms on GH200 GPU, ~80-120ms on CPU.
  Qwen ViT forward: ~15-25ms on GH200 GPU (at max_pixels=200704).
  CLIP adds ~15-25% overhead on GPU — affordable at 1 fps.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lazy singleton CLIP loader
# ---------------------------------------------------------------------------

_clip_lock = threading.Lock()
_clip_instances: Dict[str, "CLIPScorerState"] = {}


def _get_clip_state(
    model_id: str = "openai/clip-vit-large-patch14",
    device: str = "cpu",
) -> "CLIPScorerState":
    """Return a CLIPScorerState for the given model, loading it at most once."""
    key = f"{model_id}@{device}"
    if key not in _clip_instances:
        with _clip_lock:
            if key not in _clip_instances:
                _clip_instances[key] = CLIPScorerState(model_id, device)
    return _clip_instances[key]


class CLIPScorerState:
    """Holds CLIP model/processor handles and the query-embedding cache.

    Thread-safe for reads (the cache is append-only after construction).
    """

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14",
                 device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor

        hf_home = os.environ.get("HF_HOME", None)
        t0 = time.time()
        try:
            self.model = CLIPModel.from_pretrained(
                model_id, local_files_only=True).eval().to(device)
        except Exception as e:
            raise RuntimeError(
                f"[clip_pruner] CLIP model '{model_id}' not found in local "
                f"HF cache (HF_HOME={hf_home}). Cannot download at runtime. "
                f"Original error: {e}"
            ) from e
        self.processor = CLIPProcessor.from_pretrained(
            model_id, local_files_only=True)
        self.device = device
        self.model_id = model_id
        load_s = time.time() - t0
        cache_dir = hf_home or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        print(f"[clip_pruner] CLIP loaded: {model_id} on {device} "
              f"in {load_s:.2f}s (cache: {cache_dir})", flush=True)

        # Patch grid dimensions
        vc = self.model.config.vision_config
        self.patch_size = vc.patch_size          # 14
        self.image_size = vc.image_size          # 224
        self.grid_side = self.image_size // self.patch_size  # 16
        self.n_patches = self.grid_side ** 2     # 256

        # Projection layer references (robust across transformers versions)
        self.text_projection = self.model.text_projection
        self.visual_projection = self.model.visual_projection

        # Query embedding cache: query_text -> [proj_dim] unit-norm tensor
        self._query_cache: Dict[str, torch.Tensor] = {}
        self._cache_lock = threading.Lock()

    def get_query_embedding(self, query: str) -> torch.Tensor:
        """Return the CLIP text embedding for ``query``, cached."""
        if query in self._query_cache:
            return self._query_cache[query]
        with self._cache_lock:
            if query in self._query_cache:
                return self._query_cache[query]
            inputs = self.processor(text=[query], return_tensors="pt",
                                    padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                text_out = self.model.text_model(**inputs)
                pooled = text_out[1]                     # pooler_output [1, hidden]
                projected = self.text_projection(pooled)  # [1, proj_dim]
            emb = projected[0].float()
            emb = emb / (emb.norm() + 1e-12)
            self._query_cache[query] = emb
            return emb


# ---------------------------------------------------------------------------
# CLIP patch scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_patch_scores(
    img,  # PIL.Image
    query: str,
    state: CLIPScorerState,
) -> torch.Tensor:
    """Compute per-patch cosine similarity between CLIP visual patches and query.

    Returns: scores tensor of shape (grid_side, grid_side) = (16, 16).

    Aspect-ratio mitigation: the input image is resized to 224x224 WITHOUT
    cropping (stretch-to-fill) so that spatial positions in the CLIP patch grid
    correspond to the same image regions as Qwen's grid.
    """
    from PIL import Image

    # Pre-resize to 224x224 without cropping (stretch-to-fill)
    img_resized = img.resize(
        (state.image_size, state.image_size), Image.BILINEAR)

    # Process through CLIP (resize/crop are no-ops since already 224x224)
    inputs = state.processor(images=img_resized, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(state.device)

    # Get per-patch features from CLIP vision model
    vis_out = state.model.vision_model(pixel_values=pixel_values)
    # last_hidden_state: [1, 257, 1024] — CLS + 256 patches
    patch_features = vis_out.last_hidden_state[0, 1:, :]  # [256, 1024]

    # Project patches to CLIP joint space via the visual projection layer
    patch_proj = state.visual_projection(patch_features).float()  # [256, proj_dim]
    patch_proj = patch_proj / (patch_proj.norm(dim=1, keepdim=True) + 1e-12)

    # Get (cached) query embedding
    q_emb = state.get_query_embedding(query)  # [proj_dim]

    # Cosine similarity (both already unit-norm)
    scores = patch_proj @ q_emb  # [256]

    # Reshape to spatial grid
    gs = state.grid_side  # 16
    scores_hw = scores.reshape(gs, gs)
    return scores_hw


# ---------------------------------------------------------------------------
# Grid mismatch: CLIP (16,16) -> Qwen (gh, gw)
# ---------------------------------------------------------------------------

def map_scores_to_qwen(
    scores_hw: torch.Tensor,
    gh: int,
    gw: int,
) -> torch.Tensor:
    """Resample CLIP's (16,16) score map to Qwen's (gh,gw) grid, flatten row-major.

    Uses bilinear interpolation (align_corners=True so corner patches correspond
    exactly, which is the correct semantic for grid-aligned features).

    Args:
        scores_hw: CLIP score map, shape (H_clip, W_clip) — typically (16, 16).
        gh, gw: Qwen post-merger grid dimensions.

    Returns:
        scores_N: shape (gh*gw,), one score per Qwen visual token in row-major order.
    """
    inp = scores_hw.unsqueeze(0).unsqueeze(0).float()  # [1, 1, 16, 16]
    out = F.interpolate(inp, size=(gh, gw), mode="bilinear",
                        align_corners=True)  # [1, 1, gh, gw]
    return out.reshape(-1)  # [gh*gw]


# ---------------------------------------------------------------------------
# Infer Qwen's (gh, gw) from N and image aspect ratio
# ---------------------------------------------------------------------------

def infer_qwen_grid(N: int, img_w: int, img_h: int) -> Tuple[int, int]:
    """Best-effort recovery of Qwen's post-merger grid (gh, gw) from token count
    and image dimensions.

    Qwen3-VL's image processor resizes the image to fit within max_pixels with
    both dimensions multiples of patch_size*merge_size = 32, preserving aspect
    ratio. So gh/gw ≈ img_h/img_w.

    We find the factorisation of N that minimises aspect-ratio distortion:
        argmin_{(a,b): a*b=N} |b/a - img_w/img_h|

    Falls back to the closest-to-square factorisation if N is prime.

    NOTE: This is an APPROXIMATION. The exact grid depends on the Qwen image
    processor's rounding/snapping logic. For production use, pass the true grid
    from the processor output via the ``grid`` parameter of ``prune_frame_clip``.
    """
    if N <= 0:
        return (0, 0)
    if N == 1:
        return (1, 1)

    target_ratio = img_w / max(img_h, 1)  # w/h ratio

    best = (1, N)
    best_err = abs(N / 1 - target_ratio)

    for a in range(1, int(math.isqrt(N)) + 1):
        if N % a == 0:
            b = N // a
            # (a, b) means gh=a, gw=b
            err_ab = abs(b / a - target_ratio)
            if err_ab < best_err:
                best_err = err_ab
                best = (a, b)
            # (b, a) means gh=b, gw=a
            err_ba = abs(a / b - target_ratio)
            if err_ba < best_err:
                best_err = err_ba
                best = (b, a)

    return best  # (gh, gw)


# ---------------------------------------------------------------------------
# Frame-adaptive mean threshold + retention floor
# ---------------------------------------------------------------------------

def qdp_keep_mask(
    scores_N: torch.Tensor,
    retention_floor: float = 0.30,
) -> Tuple[List[int], Dict[str, Any]]:
    """Apply frame-adaptive mean threshold, enforce retention floor.

    QueryStream Sec. 3.2 semantic mask:
        keep_i = 1[ score_i > mean(scores) ]

    If fewer than ``ceil(retention_floor * N)`` tokens survive the threshold,
    fill up to the floor with the highest-scoring tokens (ties broken by
    ascending index).

    Args:
        scores_N: per-token scores, shape (N,).
        retention_floor: minimum fraction of tokens to keep (default 0.30).

    Returns:
        (keep_idx, stats) where keep_idx is a sorted list of token indices
        and stats is a dict with diagnostic fields.
    """
    N = scores_N.shape[0]
    min_keep = max(1, math.ceil(retention_floor * N))

    score_list = scores_N.tolist()
    mean_score = sum(score_list) / N

    # Frame-adaptive threshold: keep patches above the frame mean
    above_mean = [i for i in range(N) if score_list[i] > mean_score]

    if len(above_mean) >= min_keep:
        keep_idx = sorted(above_mean)
    else:
        # Fill to retention floor with highest-scoring tokens
        order = sorted(range(N), key=lambda i: (-score_list[i], i))
        keep_idx = sorted(order[:min_keep])

    stats = {
        "n_in": N,
        "n_out": len(keep_idx),
        "keep_rate": len(keep_idx) / N if N > 0 else 1.0,
        "mean_score": mean_score,
        "threshold": mean_score,
        "above_mean_count": len(above_mean),
        "floor_applied": len(above_mean) < min_keep,
    }
    return keep_idx, stats


# ---------------------------------------------------------------------------
# Entry point — full per-frame pipeline
# ---------------------------------------------------------------------------

def prune_frame_clip(
    embeds: torch.Tensor,
    img,  # PIL.Image
    query: str,
    state: CLIPScorerState,
    cfg,
    grid: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Score and prune one frame of Qwen visual tokens using CLIP semantic scores.

    **FAIL-SAFE**: if ``query`` is empty/blank, returns ``embeds`` unchanged (no-op).

    Args:
        embeds: ``[1, N, H]`` from ``backend.embed_frame``.
        img:    the original PIL image (same frame, before Qwen processing).
        query:  the user query string (cached in state after first use).
        state:  CLIPScorerState instance (holds model + query cache).
        cfg:    config object. Reads (via getattr with defaults):
                  - ``clip_retention_floor`` (float, default 0.30)
                  - ``clip_model_id`` (str, default "openai/clip-vit-large-patch14")
        grid:   ``(gh, gw)`` — Qwen's post-merger grid for this frame.
                Recover from ``image_grid_thw`` (backend.py:130-133):
                    feat = processor.image_processor(images=[img], ...)
                    t, h, w = feat["image_grid_thw"][0].tolist()
                    grid = (h // merge_size, w // merge_size)
                If None, inferred from N and image aspect ratio (approximate!).

    Returns:
        ``(pruned_embeds, stats_or_None)``.  stats is None on a no-op.
    """
    # Fail-safe: blank/empty query -> no-op
    if not query or not query.strip():
        return embeds, None

    N = embeds.shape[1]
    retention_floor = getattr(cfg, "clip_retention_floor", 0.30)

    # Determine Qwen grid (gh, gw)
    if grid is not None:
        gh, gw = grid
    else:
        img_w, img_h = img.size  # PIL: (width, height)
        gh, gw = infer_qwen_grid(N, img_w, img_h)
        if gh * gw != N:
            print(f"[clip_pruner] WARNING: cannot factor N={N} into a valid "
                  f"grid for image {img_w}x{img_h}. Skipping pruning.",
                  flush=True)
            return embeds, None

    assert gh * gw == N, (
        f"Grid mismatch: gh={gh} * gw={gw} = {gh*gw} != N={N}")

    # 1. CLIP patch scores (16x16)
    scores_hw = clip_patch_scores(img, query, state)

    # 2. Resample to Qwen grid
    scores_N = map_scores_to_qwen(scores_hw, gh, gw)

    # 3. Apply frame-adaptive threshold + retention floor
    keep_idx, mask_stats = qdp_keep_mask(scores_N, retention_floor)

    # 4. Select surviving tokens
    pruned = embeds[:, keep_idx, :]  # [1, M, H]

    stats = {
        "keep_rate": mask_stats["keep_rate"],
        "n_in": N,
        "n_out": len(keep_idx),
        "grid": (gh, gw),
        "grid_inferred": grid is None,
        "mean_score": mask_stats["mean_score"],
        "floor_applied": mask_stats["floor_applied"],
    }
    return pruned, stats
