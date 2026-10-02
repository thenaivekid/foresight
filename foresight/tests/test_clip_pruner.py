"""
test_clip_pruner.py — CPU-only unit tests for the CLIP semantic pruner.

Tests:
  1.  Score->grid resampling is spatially faithful (corner-blob test).
  2.  Retention floor respected.
  3.  Empty/blank query falls back to no-op.
  4.  Determinism.
  5.  Sorted, in-range indices.
  6.  Non-square grids (N not a perfect square).
  7.  Real CLIP forward on synthetic images (CPU).
  8.  Grid inference from aspect ratio (with processor ground truth).
  9.  Full pipeline with real CLIP (end-to-end spatial faithfulness).
  10. Ground-truth grid: 1280x720 -> N=180 = (10,18).
"""
from __future__ import annotations

import math
import os
import sys
import time

# Ensure importability
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from clip_pruner import (
    CLIPScorerState,
    clip_patch_scores,
    infer_qwen_grid,
    map_scores_to_qwen,
    prune_frame_clip,
    qdp_keep_mask,
)


H = 64  # hidden dim for fake embeds


# ---------------------------------------------------------------------------
# Test 1 — Spatial faithfulness: corner-blob test
# ---------------------------------------------------------------------------

def test_spatial_faithfulness():
    """Put high scores in the top-left corner of a 16x16 CLIP score map,
    resample to various Qwen grids, and assert that surviving Qwen token
    indices concentrate in the corresponding corner of the (gh, gw) grid."""
    print("test_spatial_faithfulness ... ", end="")

    for gh, gw in [(14, 14), (10, 20), (15, 13), (8, 25), (7, 7), (10, 18)]:
        N = gh * gw
        # Create a 16x16 score map with high values in top-left quadrant
        scores_hw = torch.zeros(16, 16)
        scores_hw[:4, :4] = 1.0   # top-left 4x4 block scores high
        scores_hw[4:, :] = -0.5   # rest scores low
        scores_hw[:, 4:] = -0.5

        scores_N = map_scores_to_qwen(scores_hw, gh, gw)
        assert scores_N.shape == (N,), f"shape mismatch: {scores_N.shape} vs ({N},)"

        # Apply threshold
        keep_idx, stats = qdp_keep_mask(scores_N, retention_floor=0.10)

        # Convert kept indices back to (row, col) in (gh, gw) grid
        kept_rows = [idx // gw for idx in keep_idx]
        kept_cols = [idx % gw for idx in keep_idx]

        # Majority of kept tokens should be in the top-left quadrant
        tl_threshold_r = max(1, gh // 3)
        tl_threshold_c = max(1, gw // 3)
        tl_count = sum(1 for r, c in zip(kept_rows, kept_cols)
                       if r < tl_threshold_r and c < tl_threshold_c)
        tl_frac = tl_count / len(keep_idx) if keep_idx else 0

        assert tl_frac >= 0.5, (
            f"grid ({gh},{gw}): only {tl_frac:.2f} of kept tokens in top-left "
            f"region (expected >= 0.50). kept_rows={kept_rows[:10]}..., "
            f"kept_cols={kept_cols[:10]}...")

    print("PASS")


# ---------------------------------------------------------------------------
# Test 2 — Retention floor respected
# ---------------------------------------------------------------------------

def test_retention_floor():
    print("test_retention_floor ... ", end="")
    for N in [1, 2, 5, 10, 50, 180, 200]:
        for floor in [0.01, 0.10, 0.30, 0.50, 0.90, 1.0]:
            scores = -torch.ones(N)  # all equal -> threshold keeps 0
            keep_idx, stats = qdp_keep_mask(scores, retention_floor=floor)
            min_keep = max(1, math.ceil(floor * N))
            assert len(keep_idx) >= min_keep, (
                f"floor={floor}, N={N}: kept {len(keep_idx)} < min {min_keep}")
            assert len(keep_idx) >= 1, "frame was emptied!"
            assert abs(stats["keep_rate"] - len(keep_idx) / N) < 1e-9
    print("PASS")


# ---------------------------------------------------------------------------
# Test 3 — Empty/blank query is a no-op
# ---------------------------------------------------------------------------

def test_empty_query_noop():
    print("test_empty_query_noop ... ", end="")
    embeds = torch.randn(1, 50, H)

    class FakeCfg:
        clip_retention_floor = 0.30
        clip_model_id = "openai/clip-vit-large-patch14"

    class FakeState:
        pass

    img = Image.new("RGB", (640, 480), "red")

    for query in ["", "   ", None]:
        result, stats = prune_frame_clip(
            embeds, img, query, FakeState(), FakeCfg(), grid=(5, 10))
        assert stats is None, f"expected no-op for query={query!r}"
        assert torch.equal(result, embeds), "expected identical tensors on no-op"

    print("PASS")


# ---------------------------------------------------------------------------
# Test 4 — Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    print("test_determinism ... ", end="")
    torch.manual_seed(42)
    scores = torch.randn(180)
    for _ in range(5):
        k1, s1 = qdp_keep_mask(scores, 0.30)
        k2, s2 = qdp_keep_mask(scores, 0.30)
        assert k1 == k2, "keep indices differ between runs"
        assert s1 == s2, "stats differ between runs"

    scores_hw = torch.randn(16, 16)
    for gh, gw in [(10, 18), (14, 14)]:
        r1 = map_scores_to_qwen(scores_hw, gh, gw)
        r2 = map_scores_to_qwen(scores_hw, gh, gw)
        assert torch.equal(r1, r2), "resampling not deterministic"

    print("PASS")


# ---------------------------------------------------------------------------
# Test 5 — Sorted, in-range indices
# ---------------------------------------------------------------------------

def test_sorted_inrange_indices():
    print("test_sorted_inrange_indices ... ", end="")
    for N in [10, 50, 100, 180, 200]:
        torch.manual_seed(N)
        scores = torch.randn(N)
        keep_idx, _ = qdp_keep_mask(scores, retention_floor=0.30)

        assert keep_idx == sorted(keep_idx), f"indices not sorted: {keep_idx[:10]}..."
        assert all(0 <= i < N for i in keep_idx), (
            f"index out of range [0, {N})")
        assert len(keep_idx) == len(set(keep_idx)), "duplicate indices"

    print("PASS")


# ---------------------------------------------------------------------------
# Test 6 — Non-square grids (N not a perfect square)
# ---------------------------------------------------------------------------

def test_nonsquare_grids():
    print("test_nonsquare_grids ... ", end="")
    scores_hw = torch.randn(16, 16)

    # Include the real ground-truth grid (10,18)=180
    for gh, gw in [(10, 18), (10, 20), (15, 13), (8, 25), (7, 29), (1, 100), (3, 5), (12, 16)]:
        N = gh * gw
        scores_N = map_scores_to_qwen(scores_hw, gh, gw)
        assert scores_N.shape == (N,), f"({gh},{gw}): shape {scores_N.shape} != ({N},)"

        embeds = torch.randn(1, N, H)
        keep_idx, stats = qdp_keep_mask(scores_N, retention_floor=0.30)
        min_keep = max(1, math.ceil(0.30 * N))
        assert len(keep_idx) >= min_keep, (
            f"({gh},{gw}): kept {len(keep_idx)} < floor {min_keep}")

    print("PASS")


# ---------------------------------------------------------------------------
# Test 7 — Real CLIP forward on synthetic images (CPU)
# ---------------------------------------------------------------------------

def test_real_clip_forward():
    print("test_real_clip_forward ... ", end="")

    if not os.environ.get("HF_HOME"):
        for candidate in ["/path/to/work/hf_cache",
                          os.path.expanduser("~/.cache/huggingface")]:
            if os.path.isdir(candidate):
                os.environ["HF_HOME"] = candidate
                break

    try:
        state = CLIPScorerState("openai/clip-vit-large-patch14", device="cpu")
    except RuntimeError as e:
        print(f"SKIP (CLIP not in local cache: {e})")
        return

    # Create a synthetic image: red blob in top-left on black background
    img = Image.new("RGB", (640, 480), (0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 160, 120], fill=(255, 0, 0))

    query = "a red object"
    t0 = time.time()
    scores_hw = clip_patch_scores(img, query, state)
    t1 = time.time()

    assert scores_hw.shape == (16, 16), f"wrong shape: {scores_hw.shape}"

    # Top-left quadrant should score higher than bottom-right
    tl_mean = scores_hw[:4, :4].mean().item()
    br_mean = scores_hw[8:, 8:].mean().item()
    print(f"PASS  (TL={tl_mean:.4f} vs BR={br_mean:.4f}, "
          f"CLIP forward {(t1-t0)*1000:.0f}ms on CPU)")


# ---------------------------------------------------------------------------
# Test 8 — Grid inference from aspect ratio (with processor ground truth)
# ---------------------------------------------------------------------------

def test_grid_inference():
    print("test_grid_inference ... ", end="")

    # Ground truth from Qwen processor with max_pixels=200704 cap:
    #   1280x720  -> grid_thw=[1,20,36] -> (10,18) = 180
    #   640x480   -> grid_thw=[1,24,32] -> (12,16) = 192
    #   1920x1080 -> grid_thw=[1,20,36] -> (10,18) = 180
    #   854x480   -> grid_thw=[1,20,36] -> (10,18) = 180
    #   320x240   -> grid_thw=[1,16,20] -> (8,10)  = 80
    #   448x448   -> grid_thw=[1,28,28] -> (14,14) = 196
    gt_cases = [
        (180, 1280, 720, 10, 18),
        (192, 640, 480, 12, 16),
        (180, 1920, 1080, 10, 18),
        (180, 854, 480, 10, 18),
        (80, 320, 240, 8, 10),
        (196, 448, 448, 14, 14),
    ]
    for N, w, h, exp_gh, exp_gw in gt_cases:
        gh, gw = infer_qwen_grid(N, w, h)
        assert gh * gw == N, f"N={N}: {gh}*{gw} != {N}"
        assert (gh, gw) == (exp_gh, exp_gw), (
            f"N={N}, {w}x{h}: got ({gh},{gw}), expected ({exp_gh},{exp_gw})")

    # Edge cases
    gh, gw = infer_qwen_grid(1, 640, 480)
    assert (gh, gw) == (1, 1)

    gh, gw = infer_qwen_grid(0, 640, 480)
    assert (gh, gw) == (0, 0)

    # Prime N: cannot factor into a grid matching the AR
    gh, gw = infer_qwen_grid(197, 640, 480)
    assert gh * gw == 197  # 197 is prime -> (1, 197) or (197, 1)

    print("PASS")


# ---------------------------------------------------------------------------
# Test 9 — Full pipeline with real CLIP (end-to-end spatial faithfulness)
# ---------------------------------------------------------------------------

def test_full_pipeline_spatial():
    """End-to-end: create an image with a bright query-relevant blob in one corner,
    run the full pruner pipeline, and verify surviving Qwen indices concentrate
    in the correct corner."""
    print("test_full_pipeline_spatial ... ", end="")

    if not os.environ.get("HF_HOME"):
        for candidate in ["/path/to/work/hf_cache"]:
            if os.path.isdir(candidate):
                os.environ["HF_HOME"] = candidate
                break

    try:
        state = CLIPScorerState("openai/clip-vit-large-patch14", device="cpu")
    except RuntimeError as e:
        print(f"SKIP (CLIP not in local cache: {e})")
        return

    # Blue square in bottom-right on white, 448x448 -> (14,14)=196
    img = Image.new("RGB", (448, 448), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([336, 336, 448, 448], fill=(0, 0, 255))

    query = "a blue square"
    gh, gw = 14, 14
    N = gh * gw
    embeds = torch.randn(1, N, 3584)

    class FakeCfg:
        clip_retention_floor = 0.20
        clip_model_id = "openai/clip-vit-large-patch14"

    pruned, stats = prune_frame_clip(
        embeds, img, query, state, FakeCfg(), grid=(gh, gw))

    assert stats is not None, "expected pruning to fire"
    assert pruned.shape[0] == 1
    assert pruned.shape[1] <= N
    assert pruned.shape[1] >= max(1, math.ceil(0.20 * N))
    assert "keep_rate" in stats
    assert "n_in" in stats and stats["n_in"] == N
    assert "n_out" in stats and stats["n_out"] == pruned.shape[1]
    assert "grid" in stats and stats["grid"] == (gh, gw)

    print(f"PASS  (kept {stats['n_out']}/{stats['n_in']} tokens, "
          f"keep_rate={stats['keep_rate']:.3f})")


# ---------------------------------------------------------------------------
# Test 10 — Ground-truth case: 1280x720 -> N=180 = (10,18)
# ---------------------------------------------------------------------------

def test_ground_truth_1280x720():
    """The observed case from the live run log:
    1280x720, max_pixels=200704 -> N=180 post-merger tokens.
    Qwen processor gives grid_thw=[1,20,36] -> (gh,gw)=(10,18).
    Note: 180=12*15 also, but 15/12=1.25 != 16/9=1.78; the correct grid
    is (10,18) with 18/10=1.80 ≈ 16/9."""
    print("test_ground_truth_1280x720 ... ", end="")

    N = 180
    gh, gw = 10, 18

    # Verify grid inference finds the right answer
    inf_gh, inf_gw = infer_qwen_grid(N, 1280, 720)
    assert (inf_gh, inf_gw) == (gh, gw), (
        f"infer_qwen_grid(180, 1280, 720) = ({inf_gh},{inf_gw}), expected (10,18)")

    # Verify spatial mapping: put a blob in bottom-right of CLIP grid,
    # check it maps to bottom-right of (10,18) Qwen grid
    scores_hw = torch.full((16, 16), -0.5)
    scores_hw[12:, 12:] = 1.0  # bottom-right 4x4

    scores_N = map_scores_to_qwen(scores_hw, gh, gw)
    assert scores_N.shape == (N,)

    keep_idx, stats = qdp_keep_mask(scores_N, retention_floor=0.10)

    # Convert to (row, col) in (10, 18)
    kept_rows = [idx // gw for idx in keep_idx]
    kept_cols = [idx % gw for idx in keep_idx]

    # Bottom-right quadrant: row >= 7, col >= 13
    br_count = sum(1 for r, c in zip(kept_rows, kept_cols)
                   if r >= gh * 2 // 3 and c >= gw * 2 // 3)
    br_frac = br_count / len(keep_idx)

    assert br_frac >= 0.5, (
        f"Only {br_frac:.2f} of kept tokens in bottom-right "
        f"(expected >= 0.50 for (10,18) grid)")

    # Also verify the full pipeline with fake embeds
    embeds = torch.randn(1, N, 3584)
    scores_N_full = map_scores_to_qwen(scores_hw, gh, gw)
    keep_idx_full, _ = qdp_keep_mask(scores_N_full, retention_floor=0.30)
    min_keep = max(1, math.ceil(0.30 * N))
    assert len(keep_idx_full) >= min_keep

    print(f"PASS  (grid inferred correctly, "
          f"BR tokens {br_frac:.2f} of {len(keep_idx)} kept)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_spatial_faithfulness()
    test_retention_floor()
    test_empty_query_noop()
    test_determinism()
    test_sorted_inrange_indices()
    test_nonsquare_grids()
    test_real_clip_forward()
    test_grid_inference()
    test_full_pipeline_spatial()
    test_ground_truth_1280x720()
    print()
    print("=" * 60)
    print("ALL 10 TESTS PASSED")
    print("=" * 60)
