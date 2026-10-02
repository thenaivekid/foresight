"""
test_class_pruner.py — CPU-only unit tests for the class pruner.

Tests with fake token tensors: no model, no GPU required.
Verifies the five properties the parent task requires:
  1. Retention floor: a frame is never emptied.
  2. Empty keep list is a no-op (fail safe, not fail closed).
  3. Ignore classes never remove more than the configured cap.
  4. Output token ordering / positions stay consistent.
  5. The pruner is deterministic.
"""
from __future__ import annotations

import math
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from class_pruner import (
    ClassPrunerState,
    compute_scores,
    embed_classes,
    prune_frame,
    select_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H = 32  # small hidden dim for tests

def _fake_embed_text(text):
    """Deterministic mock for backend.embed_text.
    Returns [1, 1, H] whose direction is a seeded function of the text."""
    torch.manual_seed(abs(hash(text)) % (2**31))
    v = torch.randn(1, 1, H)
    return v


def _make_vision_tokens(n, seed=42):
    """[1, N, H] with diverse directions, seeded."""
    torch.manual_seed(seed)
    return torch.randn(1, n, H)


# ---------------------------------------------------------------------------
# Test 1 — retention floor is respected (frame never emptied)
# ---------------------------------------------------------------------------

def test_retention_floor():
    print("test_retention_floor ... ", end="")
    for n_tokens in [1, 2, 3, 5, 10, 50, 200]:
        for floor in [0.01, 0.10, 0.30, 0.50, 0.90, 1.0]:
            embeds = _make_vision_tokens(n_tokens)
            # Create scores that would want to drop everything
            scores = -torch.ones(n_tokens)
            pruned, kept = select_tokens(embeds, scores, retention_floor=floor)
            min_keep = max(1, math.ceil(floor * n_tokens))
            assert pruned.shape[1] >= min_keep, (
                f"floor={floor}, n={n_tokens}: kept {pruned.shape[1]} < min {min_keep}")
            assert pruned.shape[1] >= 1, "frame was emptied!"

    # Also test via prune_frame (the full pipeline)
    cache = {}
    embeds = _make_vision_tokens(20)
    result, stats = prune_frame(
        embeds, ["ball"], ["giraffe", "submarine", "cactus"],
        _fake_embed_text, cache, retention_floor=0.30)
    assert result.shape[1] >= max(1, math.ceil(0.30 * 20)), (
        f"prune_frame violated floor: kept {result.shape[1]}")
    assert result.shape[1] >= 1
    print(f"PASS  (kept {result.shape[1]}/{embeds.shape[1]} tokens)")


# ---------------------------------------------------------------------------
# Test 2 — empty keep list is a no-op
# ---------------------------------------------------------------------------

def test_empty_keep_noop():
    print("test_empty_keep_noop ... ", end="")
    embeds = _make_vision_tokens(50, seed=99)
    cache = {}

    # Empty keep, some ignore
    result, stats = prune_frame(
        embeds, [], ["giraffe", "submarine"],
        _fake_embed_text, cache, retention_floor=0.30)
    assert stats is None, "expected no-op stats (None)"
    assert torch.equal(result, embeds), "expected identical tensors on no-op"

    # Empty keep, empty ignore
    result2, stats2 = prune_frame(
        embeds, [], [],
        _fake_embed_text, cache, retention_floor=0.30)
    assert stats2 is None
    assert torch.equal(result2, embeds)
    print("PASS")


# ---------------------------------------------------------------------------
# Test 3 — ignore classes never remove more than the configured cap
# ---------------------------------------------------------------------------

def test_ignore_cap():
    print("test_ignore_cap ... ", end="")
    N = 100
    embeds = _make_vision_tokens(N, seed=77)
    cache = {}

    # With keep=["ball"] and many ignore classes, retention floor still holds
    for floor in [0.30, 0.50, 0.70]:
        result, stats = prune_frame(
            embeds,
            ["ball"],
            ["giraffe", "submarine", "cactus", "volcano", "harpsichord",
             "igloo", "octopus", "windmill", "chandelier", "kayak"],
            _fake_embed_text, cache, retention_floor=floor)
        assert stats is not None, "pruning should have fired"
        min_keep = max(1, math.ceil(floor * N))
        assert result.shape[1] >= min_keep, (
            f"floor={floor}: kept {result.shape[1]} < min {min_keep}")
        assert stats["retention"] >= floor - 1e-9, (
            f"retention {stats['retention']} < floor {floor}")
    print("PASS  (retention respected with 10 ignore classes)")


# ---------------------------------------------------------------------------
# Test 4 — output token ordering / positions stay consistent
# ---------------------------------------------------------------------------

def test_order_preservation():
    print("test_order_preservation ... ", end="")
    N = 50
    torch.manual_seed(42)
    # Make tokens with known values so we can verify ordering
    embeds = torch.arange(N * H, dtype=torch.float32).reshape(1, N, H)

    # Create scores that prefer every other token
    scores = torch.zeros(N)
    scores[::2] = 1.0  # even indices score high
    scores[1::2] = -1.0  # odd indices score low

    pruned, kept = select_tokens(embeds, scores, retention_floor=0.30)

    # kept should be sorted (spatial order preserved)
    assert kept == sorted(kept), f"kept indices not sorted: {kept}"

    # Verify the tokens in pruned match the original at those indices
    for i, idx in enumerate(kept):
        assert torch.equal(pruned[0, i], embeds[0, idx]), (
            f"token {i} in pruned != original token {idx}")

    # Even indices should be preferred (they scored higher)
    even_kept = [k for k in kept if k % 2 == 0]
    min_keep = max(1, math.ceil(0.30 * N))
    assert len(even_kept) > 0, "no even-indexed tokens kept"
    print(f"PASS  (kept {len(kept)} tokens, {len(even_kept)} even-indexed)")


# ---------------------------------------------------------------------------
# Test 5 — the pruner is deterministic
# ---------------------------------------------------------------------------

def test_determinism():
    print("test_determinism ... ", end="")
    N = 80
    embeds = _make_vision_tokens(N, seed=123)

    # Run prune_frame twice with identical inputs
    for trial in range(3):
        cache1, cache2 = {}, {}
        r1, s1 = prune_frame(
            embeds.clone(), ["ball", "player"], ["giraffe"],
            _fake_embed_text, cache1, retention_floor=0.30, ignore_lambda=0.5)
        r2, s2 = prune_frame(
            embeds.clone(), ["ball", "player"], ["giraffe"],
            _fake_embed_text, cache2, retention_floor=0.30, ignore_lambda=0.5)
        assert torch.equal(r1, r2), f"trial {trial}: pruned tensors differ"
        assert s1 == s2, f"trial {trial}: stats differ"

    # Also verify compute_scores is deterministic
    keep_embs = torch.stack([cache1["ball"], cache1["player"]])
    ign_embs = torch.stack([cache1["giraffe"]])
    sc1 = compute_scores(embeds[0], keep_embs, ign_embs, 0.5)
    sc2 = compute_scores(embeds[0], keep_embs, ign_embs, 0.5)
    assert torch.equal(sc1, sc2), "compute_scores not deterministic"
    print("PASS")


# ---------------------------------------------------------------------------
# Test 6 — ClassPrunerState is thread-safe basics
# ---------------------------------------------------------------------------

def test_pruner_state():
    print("test_pruner_state ... ", end="")
    state = ClassPrunerState()

    k, ig, v = state.get_classes()
    assert k == [] and ig == [] and v == 0, "initial state wrong"

    state.set_classes(["ball"], ["giraffe"])
    k, ig, v = state.get_classes()
    assert k == ["ball"] and ig == ["giraffe"] and v == 1

    state.set_classes(["car", "person"], [])
    k, ig, v = state.get_classes()
    assert k == ["car", "person"] and ig == [] and v == 2
    print("PASS")


# ---------------------------------------------------------------------------
# Test 7 — embed_classes caching works
# ---------------------------------------------------------------------------

def test_embed_cache():
    print("test_embed_cache ... ", end="")
    cache = {}
    call_count = [0]
    original_fn = _fake_embed_text
    def counting_fn(text):
        call_count[0] += 1
        return original_fn(text)

    # First call: computes embeddings
    r1 = embed_classes(counting_fn, ["ball", "car"], cache)
    first_calls = call_count[0]
    assert first_calls == 2, f"expected 2 calls, got {first_calls}"
    assert "ball" in cache and "car" in cache

    # Second call: should use cache
    r2 = embed_classes(counting_fn, ["ball", "car"], cache)
    assert call_count[0] == 2, f"cache not used: {call_count[0]} calls"

    # Third call with one new name: only one new computation
    r3 = embed_classes(counting_fn, ["ball", "dog"], cache)
    assert call_count[0] == 3, f"expected 3 calls, got {call_count[0]}"
    assert "dog" in cache
    print("PASS")


# ---------------------------------------------------------------------------
# Test 8 — score formula correctness
# ---------------------------------------------------------------------------

def test_score_formula():
    print("test_score_formula ... ", end="")
    # Hand-crafted: token 0 aligns with keep, token 1 with ignore
    keep_dir = torch.zeros(H); keep_dir[0] = 1.0
    ign_dir = torch.zeros(H); ign_dir[1] = 1.0

    tokens = torch.zeros(2, H)
    tokens[0, 0] = 1.0  # aligns with keep
    tokens[1, 1] = 1.0  # aligns with ignore

    scores = compute_scores(tokens, keep_dir.unsqueeze(0), ign_dir.unsqueeze(0), 0.5)
    # token 0: cos_keep=1.0, cos_ign=0.0 -> score = 1.0 - 0.5*0.0 = 1.0
    # token 1: cos_keep=0.0, cos_ign=1.0 -> score = 0.0 - 0.5*1.0 = -0.5
    assert abs(scores[0].item() - 1.0) < 1e-5, f"token 0 score: {scores[0]:.6f}"
    assert abs(scores[1].item() - (-0.5)) < 1e-5, f"token 1 score: {scores[1]:.6f}"

    # With ignore_lambda=0, ignore term vanishes
    scores_no_ign = compute_scores(tokens, keep_dir.unsqueeze(0), ign_dir.unsqueeze(0), 0.0)
    assert abs(scores_no_ign[1].item() - 0.0) < 1e-5
    print("PASS")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_retention_floor()
    test_empty_keep_noop()
    test_ignore_cap()
    test_order_preservation()
    test_determinism()
    test_pruner_state()
    test_embed_cache()
    test_score_formula()
    print()
    print("=" * 60)
    print("ALL 8 TESTS PASSED")
    print("=" * 60)
