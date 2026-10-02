"""
test_dsh_pruner.py — CPU-only unit tests for the DSH temporal-novelty pruner.

Tests with small fake tensors (H=32, N=20): no model, no GPU required.
Covers:
  1–10: Original tests (threshold mode regression + shared invariants).
  11: Rank mode yields exact keep_rate on static AND random sequences.
  12: Rank mode selects exactly the perturbed positions.
  13: Rank mode is invariant to global cosine shift (scale-free).
  14: Determinism in rank mode.
  15: Variable-N reset in rank mode.
"""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dsh_pruner import (
    DSHState,
    dsh_keep_mask,
    prune_frame_dsh,
    _unit_normalise,
    _enforce_floor,
    _select_topk_novel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H = 32   # small hidden dim for tests


def _make_frame(N, seed=42):
    """[1, N, H] with diverse directions, seeded."""
    torch.manual_seed(seed)
    return torch.randn(1, N, H)


class _FakeCfg:
    """Minimal config stub with getattr defaults."""
    pass


# ===================================================================
# ORIGINAL TESTS 1–10 (threshold mode where applicable)
# ===================================================================

# ---------------------------------------------------------------------------
# Test 1 — first frame keeps all tokens
# ---------------------------------------------------------------------------

def test_first_frame_keeps_all():
    print("test_first_frame_keeps_all ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()
    # Default rule is "rank" — first-frame behaviour is rule-independent
    embeds = _make_frame(20, seed=1)
    out, stats = prune_frame_dsh(embeds, state, cfg)
    assert out.shape == embeds.shape, f"shape mismatch: {out.shape} vs {embeds.shape}"
    assert torch.equal(out, embeds), "first frame should be returned unchanged"
    assert stats["keep_rate"] == 1.0
    assert stats["n_in"] == 20
    assert stats["n_out"] == 20
    assert stats["history_reset"] is True
    assert "rule" in stats
    print(f"PASS  (kept {stats['n_out']}/{stats['n_in']}, rule={stats['rule']})")


# ---------------------------------------------------------------------------
# Test 2 — static (repeated) frame prunes heavily on later ticks (threshold)
# ---------------------------------------------------------------------------

def test_static_frame_prunes():
    print("test_static_frame_prunes ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()
    cfg.dsh_rule = "threshold"
    cfg.dsh_tau_temp = 0.90
    cfg.dsh_alpha = 0.10
    cfg.dsh_retention_floor = 0.30

    N = 20
    frame = _make_frame(N, seed=7)

    prune_frame_dsh(frame.clone(), state, cfg)

    for _ in range(5):
        out, stats = prune_frame_dsh(frame.clone(), state, cfg)

    assert stats["keep_rate"] <= 0.50, (
        f"static stream should prune heavily, got keep_rate={stats['keep_rate']:.3f}")
    min_keep = max(1, math.ceil(0.30 * N))
    assert stats["n_out"] >= min_keep, (
        f"retention floor violated: kept {stats['n_out']} < min {min_keep}")
    assert stats["mean_cos"] > 0.85, (
        f"mean_cos should be high for static scene, got {stats['mean_cos']:.4f}")
    assert stats["rule"] == "threshold"
    print(f"PASS  (keep_rate={stats['keep_rate']:.3f}, mean_cos={stats['mean_cos']:.4f})")


# ---------------------------------------------------------------------------
# Test 3 — fully novel frame keeps ~everything (threshold)
# ---------------------------------------------------------------------------

def test_novel_frame_keeps_all():
    print("test_novel_frame_keeps_all ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()
    cfg.dsh_rule = "threshold"
    cfg.dsh_tau_temp = 0.90
    cfg.dsh_alpha = 0.10

    N = 20
    frame1 = _make_frame(N, seed=10)
    prune_frame_dsh(frame1, state, cfg)

    frame2 = _make_frame(N, seed=999)
    out, stats = prune_frame_dsh(frame2, state, cfg)

    assert stats["keep_rate"] >= 0.90, (
        f"novel frame should keep most tokens, got keep_rate={stats['keep_rate']:.3f}")
    assert stats["mean_cos"] < 0.50, (
        f"mean_cos should be low for novel frame, got {stats['mean_cos']:.4f}")
    print(f"PASS  (keep_rate={stats['keep_rate']:.3f}, mean_cos={stats['mean_cos']:.4f})")


# ---------------------------------------------------------------------------
# Test 4 — retention floor under pathological all-identical stream (threshold)
# ---------------------------------------------------------------------------

def test_retention_floor():
    print("test_retention_floor ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()
    cfg.dsh_rule = "threshold"
    cfg.dsh_tau_temp = 0.90
    cfg.dsh_alpha = 0.50
    cfg.dsh_retention_floor = 0.50

    N = 20
    frame = torch.ones(1, N, H)

    for i in range(10):
        out, stats = prune_frame_dsh(frame.clone(), state, cfg)

    min_keep = max(1, math.ceil(0.50 * N))
    assert stats["n_out"] >= min_keep, (
        f"retention floor violated: kept {stats['n_out']} < min {min_keep}")
    print(f"PASS  (kept {stats['n_out']}/{N}, floor={min_keep})")


# ---------------------------------------------------------------------------
# Test 5 — N changing between frames does not crash, resets history
# ---------------------------------------------------------------------------

def test_variable_N():
    print("test_variable_N ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()
    cfg.dsh_rule = "threshold"

    frame1 = _make_frame(20, seed=1)
    out1, stats1 = prune_frame_dsh(frame1, state, cfg)
    assert stats1["history_reset"] is True
    assert stats1["n_in"] == 20

    frame2 = _make_frame(20, seed=2)
    out2, stats2 = prune_frame_dsh(frame2, state, cfg)
    assert stats2["history_reset"] is False

    frame3 = _make_frame(30, seed=3)
    out3, stats3 = prune_frame_dsh(frame3, state, cfg)
    assert stats3["history_reset"] is True, "N change should trigger history reset"
    assert stats3["n_in"] == 30
    assert stats3["n_out"] == 30, "reset frame should keep all tokens"
    assert out3.shape == frame3.shape

    frame4 = _make_frame(30, seed=4)
    out4, stats4 = prune_frame_dsh(frame4, state, cfg)
    assert stats4["history_reset"] is False

    frame5 = _make_frame(20, seed=5)
    out5, stats5 = prune_frame_dsh(frame5, state, cfg)
    assert stats5["history_reset"] is True
    assert stats5["n_out"] == 20
    print("PASS  (N=20->20->30->30->20, resets at each N change)")


# ---------------------------------------------------------------------------
# Test 6 — determinism: same inputs -> identical outputs (threshold)
# ---------------------------------------------------------------------------

def test_determinism():
    print("test_determinism ... ", end="")
    cfg = _FakeCfg()
    cfg.dsh_rule = "threshold"
    cfg.dsh_tau_temp = 0.90
    cfg.dsh_alpha = 0.10

    N = 20
    frames = [_make_frame(N, seed=i) for i in range(5)]

    state_a = DSHState()
    results_a = []
    for f in frames:
        out, stats = prune_frame_dsh(f.clone(), state_a, cfg)
        results_a.append((out.clone(), dict(stats)))

    state_b = DSHState()
    results_b = []
    for f in frames:
        out, stats = prune_frame_dsh(f.clone(), state_b, cfg)
        results_b.append((out.clone(), dict(stats)))

    for i in range(len(frames)):
        assert torch.equal(results_a[i][0], results_b[i][0]), (
            f"frame {i}: pruned tensors differ")
        assert results_a[i][1] == results_b[i][1], (
            f"frame {i}: stats differ")
    print("PASS  (5 frames, two runs identical)")


# ---------------------------------------------------------------------------
# Test 7 — returned indices sorted and within range (threshold via dsh_keep_mask)
# ---------------------------------------------------------------------------

def test_indices_sorted_in_range():
    print("test_indices_sorted_in_range ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()

    N = 20
    frames = [_make_frame(N, seed=i) for i in range(8)]

    for i, f in enumerate(frames):
        keep_idx, stats = dsh_keep_mask(f, state, tau_temp=0.90, alpha=0.10)
        idx_list = keep_idx.tolist()

        assert idx_list == sorted(idx_list), (
            f"frame {i}: indices not sorted: {idx_list}")
        if idx_list:
            assert min(idx_list) >= 0, f"frame {i}: negative index"
            assert max(idx_list) < N, f"frame {i}: index {max(idx_list)} >= N={N}"
        assert len(idx_list) == len(set(idx_list)), (
            f"frame {i}: duplicate indices")

    print("PASS  (8 frames, all indices valid)")


# ---------------------------------------------------------------------------
# Test 8 — dsh_keep_mask and prune_frame_dsh agree
# ---------------------------------------------------------------------------

def test_mask_vs_prune_agree():
    print("test_mask_vs_prune_agree ... ", end="")
    cfg = _FakeCfg()
    state = DSHState()
    N = 20
    frame = _make_frame(N, seed=42)

    out, stats = prune_frame_dsh(frame.clone(), state, cfg)
    assert out.shape[0] == 1
    assert out.shape[2] == H
    assert out.shape[1] == stats["n_out"]
    assert stats["n_out"] <= stats["n_in"]
    assert 0.0 <= stats["keep_rate"] <= 1.0
    assert "rule" in stats
    assert "mean_cos" in stats
    print("PASS")


# ---------------------------------------------------------------------------
# Test 9 — DSHState reset works
# ---------------------------------------------------------------------------

def test_state_reset():
    print("test_state_reset ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()

    frame = _make_frame(20, seed=1)
    prune_frame_dsh(frame, state, cfg)

    v, n = state.get()
    assert v is not None and n == 20

    state.reset()
    v, n = state.get()
    assert v is None and n == 0

    out, stats = prune_frame_dsh(frame.clone(), state, cfg)
    assert stats["history_reset"] is True
    assert stats["keep_rate"] == 1.0
    print("PASS")


# ---------------------------------------------------------------------------
# Test 10 — unit normalisation handles zero-norm rows
# ---------------------------------------------------------------------------

def test_zero_norm_safety():
    print("test_zero_norm_safety ... ", end="")
    x = torch.zeros(5, H)
    x[0] = 1.0
    normed = _unit_normalise(x)
    assert not torch.isnan(normed).any(), "NaN from zero-norm row"
    assert not torch.isinf(normed).any(), "Inf from zero-norm row"
    assert normed[1].abs().max() < 1e-6
    print("PASS")


# ===================================================================
# NEW RANK-MODE TESTS 11–15
# ===================================================================

# ---------------------------------------------------------------------------
# Test 11 — rank mode yields exact keep_rate on static AND random sequences
# ---------------------------------------------------------------------------

def test_rank_exact_keep_rate():
    print("test_rank_exact_keep_rate ... ", end="")
    N = 50
    for keep_frac in [0.10, 0.30, 0.50, 0.70]:
        expected_k = max(1, math.ceil(keep_frac * N))

        # --- static stream ---
        state = DSHState()
        cfg = _FakeCfg()
        cfg.dsh_rule = "rank"
        cfg.dsh_keep_frac = keep_frac
        cfg.dsh_alpha = 0.10
        frame = _make_frame(N, seed=7)
        prune_frame_dsh(frame.clone(), state, cfg)  # first frame (keeps all)
        for _ in range(5):
            out, stats = prune_frame_dsh(frame.clone(), state, cfg)
        assert stats["n_out"] == expected_k, (
            f"static frac={keep_frac}: got n_out={stats['n_out']}, want {expected_k}")

        # --- random stream ---
        state2 = DSHState()
        for i in range(8):
            f = _make_frame(N, seed=100 + i)
            out2, stats2 = prune_frame_dsh(f, state2, cfg)
        # Last frame (not first) should also have exactly expected_k
        assert stats2["n_out"] == expected_k, (
            f"random frac={keep_frac}: got n_out={stats2['n_out']}, want {expected_k}")

    print("PASS  (exact keep count for 4 fractions, static & random)")


# ---------------------------------------------------------------------------
# Test 12 — rank mode selects exactly the perturbed positions
# ---------------------------------------------------------------------------

def test_rank_selects_perturbed():
    print("test_rank_selects_perturbed ... ", end="")
    N = 20
    k = 5  # perturb exactly 5 positions
    keep_frac = k / N  # 0.25

    state = DSHState()
    cfg = _FakeCfg()
    cfg.dsh_rule = "rank"
    cfg.dsh_keep_frac = keep_frac
    cfg.dsh_alpha = 0.10

    # Seed history with a base frame
    torch.manual_seed(42)
    base = torch.randn(1, N, H)
    prune_frame_dsh(base.clone(), state, cfg)

    # Build a frame identical to base except at 5 specific positions
    perturbed_positions = [2, 7, 11, 15, 18]
    frame2 = base.clone()
    torch.manual_seed(999)
    for p in perturbed_positions:
        frame2[0, p] = torch.randn(H) * 10  # large perturbation

    out, stats = prune_frame_dsh(frame2, state, cfg)
    expected_k = max(1, math.ceil(keep_frac * N))
    kept = []
    # Recover kept indices by matching output tokens to frame2
    for j in range(out.shape[1]):
        for idx in range(N):
            if torch.equal(out[0, j], frame2[0, idx]):
                kept.append(idx)
                break

    assert len(kept) == expected_k, f"expected {expected_k} kept, got {len(kept)}"
    for p in perturbed_positions:
        assert p in kept, f"perturbed position {p} not in kept set {kept}"
    print(f"PASS  (perturbed {perturbed_positions} all selected, kept={kept})")


# ---------------------------------------------------------------------------
# Test 13 — rank mode is invariant to global cosine shift (scale-free)
# ---------------------------------------------------------------------------

def test_rank_scale_invariant():
    print("test_rank_scale_invariant ... ", end="")
    N = 30

    # Generate a fixed cosine similarity vector (simulate the comparison result)
    torch.manual_seed(77)
    cos_sim = torch.rand(N) * 0.5 + 0.5  # uniform in [0.5, 1.0]

    k = 10
    baseline = _select_topk_novel(cos_sim, k)

    # Affine transform: scale + offset (simulates a different video's cosine range)
    for scale, offset in [(0.01, 0.99), (2.0, -0.5), (0.001, 0.999)]:
        shifted = cos_sim * scale + offset
        result = _select_topk_novel(shifted, k)
        assert torch.equal(baseline, result), (
            f"scale={scale}, offset={offset}: rank order changed!\n"
            f"  baseline={baseline.tolist()}\n  shifted ={result.tolist()}")

    print("PASS  (3 affine transforms, identical selections)")


# ---------------------------------------------------------------------------
# Test 14 — determinism in rank mode
# ---------------------------------------------------------------------------

def test_rank_determinism():
    print("test_rank_determinism ... ", end="")
    cfg = _FakeCfg()
    cfg.dsh_rule = "rank"
    cfg.dsh_keep_frac = 0.30
    cfg.dsh_alpha = 0.10

    N = 20
    frames = [_make_frame(N, seed=i) for i in range(5)]

    state_a = DSHState()
    results_a = []
    for f in frames:
        out, stats = prune_frame_dsh(f.clone(), state_a, cfg)
        results_a.append((out.clone(), dict(stats)))

    state_b = DSHState()
    results_b = []
    for f in frames:
        out, stats = prune_frame_dsh(f.clone(), state_b, cfg)
        results_b.append((out.clone(), dict(stats)))

    for i in range(len(frames)):
        assert torch.equal(results_a[i][0], results_b[i][0]), (
            f"frame {i}: pruned tensors differ")
        assert results_a[i][1] == results_b[i][1], (
            f"frame {i}: stats differ")
    print("PASS  (5 frames, two runs identical)")


# ---------------------------------------------------------------------------
# Test 15 — variable-N reset in rank mode
# ---------------------------------------------------------------------------

def test_rank_variable_N():
    print("test_rank_variable_N ... ", end="")
    state = DSHState()
    cfg = _FakeCfg()
    cfg.dsh_rule = "rank"
    cfg.dsh_keep_frac = 0.30

    frame1 = _make_frame(20, seed=1)
    _, stats1 = prune_frame_dsh(frame1, state, cfg)
    assert stats1["history_reset"] is True and stats1["n_out"] == 20

    frame2 = _make_frame(20, seed=2)
    _, stats2 = prune_frame_dsh(frame2, state, cfg)
    assert stats2["history_reset"] is False
    assert stats2["n_out"] == max(1, math.ceil(0.30 * 20))

    frame3 = _make_frame(35, seed=3)
    _, stats3 = prune_frame_dsh(frame3, state, cfg)
    assert stats3["history_reset"] is True
    assert stats3["n_out"] == 35, "reset frame should keep all"

    frame4 = _make_frame(35, seed=4)
    _, stats4 = prune_frame_dsh(frame4, state, cfg)
    assert stats4["history_reset"] is False
    assert stats4["n_out"] == max(1, math.ceil(0.30 * 35))

    print("PASS  (N=20->20->35->35, resets correct, rank budget exact)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_first_frame_keeps_all,
        test_static_frame_prunes,
        test_novel_frame_keeps_all,
        test_retention_floor,
        test_variable_N,
        test_determinism,
        test_indices_sorted_in_range,
        test_mask_vs_prune_agree,
        test_state_reset,
        test_zero_norm_safety,
        test_rank_exact_keep_rate,
        test_rank_selects_perturbed,
        test_rank_scale_invariant,
        test_rank_determinism,
        test_rank_variable_N,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  ({e})")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"{failed}/{passed + failed} TESTS FAILED")
    print("=" * 60)
