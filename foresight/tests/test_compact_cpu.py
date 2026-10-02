"""
test_compact_cpu.py — CPU-only tests for KVCacheManager.compact() and
position re-basing (_rebase_span_locked).

Proves on tiny fake caches (no model, no GPU):
  (a) After compaction the surviving tokens' positions are contiguous and correct.
  (b) The sink region is untouched.
  (c) Calling compact/ingest while a borrow is active raises loudly.

Run:
    python test_compact_cpu.py
"""
import os
import queue
import sys
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)
import threading
import torch
from transformers import DynamicCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(k, pos, inv_freq):
    """Apply half-split RoPE rotation at scalar position `pos`."""
    ang = float(pos) * inv_freq
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    # + convention matches Qwen3-VL forward RoPE; _rebase uses - for R(-delta)
    return k * cos + _rotate_half(k) * sin


def _get_k(cache, layer):
    """Read keys from either transformers 4.x or 5.x cache."""
    if hasattr(cache, 'layers'):
        return cache.layers[layer].keys
    return cache.key_cache[layer]


def _get_v(cache, layer):
    if hasattr(cache, 'layers'):
        return cache.layers[layer].values
    return cache.value_cache[layer]


def build_cache(n_tokens, n_layers, n_heads, head_dim, inv_freq):
    """Build a DynamicCache with `n_tokens` keys rotated at positions 0..n-1."""
    k_orig = torch.randn(1, n_heads, n_tokens, head_dim, dtype=torch.float64)
    cache = DynamicCache()
    for layer in range(n_layers):
        k_rotated = torch.zeros_like(k_orig)
        for t in range(n_tokens):
            k_rotated[:, :, t:t+1, :] = apply_rope(
                k_orig[:, :, t:t+1, :], float(t), inv_freq.double())
        v = torch.randn_like(k_orig)
        cache.update(k_rotated.clone(), v.clone(), layer)
    return cache, k_orig


def make_manager(cache, k_orig, sink, inv_freq, kv_budget):
    """Build a KVCacheManager around an existing cache without a model."""
    from foresight.manager import KVCacheManager
    mgr = KVCacheManager.__new__(KVCacheManager)
    mgr.cache = cache
    mgr.sink = sink
    mgr.next_pos = cache.get_seq_length()
    mgr.kv_budget = kv_budget
    mgr.evicted = 0
    mgr.compacted = 0
    mgr.rebased = 0
    mgr._lock = threading.Lock()
    mgr._sync = False
    mgr._borrowed = None
    mgr._compact_request = None
    mgr.prof = None
    mgr.b = None
    mgr._inv_freq_override = inv_freq.double()
    return mgr


def n_layers(cache):
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    return len(cache.key_cache)


# ---------------------------------------------------------------------------
# Test 1: positions are contiguous and correct after compaction
# ---------------------------------------------------------------------------

def test_positions_contiguous():
    print("TEST 1: positions contiguous after compaction ... ", end="", flush=True)
    D = 8
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    n_tokens = 12
    sink = 2
    n_lay = 2
    n_heads = 1

    cache, k_orig = build_cache(n_tokens, n_lay, n_heads, D, inv_freq)
    mgr = make_manager(cache, k_orig, sink, inv_freq, kv_budget=8)

    # target_len=7, keep_recent=3
    # middle: positions [2..8] (7 tokens, indices 0..6 within middle)
    # recent: positions [9,10,11]
    # keep_mid = 7 - 2 - 3 = 2
    # linspace(0, 6, 2).long() -> [0, 6] -> original positions 2, 8
    # new layout: [sink0, sink1, mid@2, mid@8, recent@9, recent@10, recent@11]
    # new positions: [0, 1, 2, 3, 4, 5, 6]
    removed = mgr.compact(target_len=7, keep_recent=3)

    assert removed == 5, f"expected 5 removed, got {removed}"
    assert mgr._len() == 7, f"expected len 7, got {mgr._len()}"
    assert mgr.next_pos == 7, f"expected next_pos 7, got {mgr.next_pos}"

    orig_indices = [0, 1, 2, 8, 9, 10, 11]
    new_positions = list(range(7))

    max_err = 0.0
    for layer in range(n_lay):
        k_layer = _get_k(mgr.cache, layer)
        for new_pos, orig_idx in zip(new_positions, orig_indices):
            expected = apply_rope(
                k_orig[:, :, orig_idx:orig_idx+1, :],
                float(new_pos), inv_freq.double())
            actual = k_layer[:, :, new_pos:new_pos+1, :]
            err = torch.abs(actual.double() - expected).max().item()
            max_err = max(max_err, err)
            assert err < 1e-6, (  # float32 inv_freq in _rebase_span_locked
                f"layer {layer} new_pos {new_pos} (orig {orig_idx}): "
                f"max err {err:.2e}")

    print(f"PASS  (max |err| = {max_err:.2e})")


# ---------------------------------------------------------------------------
# Test 2: sink region is untouched
# ---------------------------------------------------------------------------

def test_sink_untouched():
    print("TEST 2: sink region untouched ... ", end="", flush=True)
    D = 8
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    n_tokens = 10
    sink = 3
    n_lay = 2
    n_heads = 2

    cache, k_orig = build_cache(n_tokens, n_lay, n_heads, D, inv_freq)
    nl = n_layers(cache)

    sink_k_before = [_get_k(cache, l)[:, :, :sink, :].clone() for l in range(nl)]
    sink_v_before = [_get_v(cache, l)[:, :, :sink, :].clone() for l in range(nl)]

    mgr = make_manager(cache, k_orig, sink, inv_freq, kv_budget=7)
    removed = mgr.compact(target_len=7, keep_recent=2)
    assert removed > 0, "expected some removal"

    for l in range(nl):
        k_after = _get_k(mgr.cache, l)[:, :, :sink, :]
        v_after = _get_v(mgr.cache, l)[:, :, :sink, :]
        assert torch.equal(k_after, sink_k_before[l]), f"sink keys changed layer {l}!"
        assert torch.equal(v_after, sink_v_before[l]), f"sink values changed layer {l}!"

    print("PASS")


# ---------------------------------------------------------------------------
# Test 3: borrow guard blocks compact
# ---------------------------------------------------------------------------

def test_borrow_guard():
    print("TEST 3: borrow guard blocks compact ... ", end="", flush=True)
    D = 4
    inv_freq = torch.ones(D // 2)
    n_tokens = 6
    sink = 1

    cache, k_orig = build_cache(n_tokens, 1, 1, D, inv_freq)
    mgr = make_manager(cache, k_orig, sink, inv_freq, kv_budget=4)

    mgr._borrowed = ("test_reader", mgr._len(), mgr.next_pos)

    raised = False
    try:
        mgr.compact(target_len=4, keep_recent=2)
    except RuntimeError as e:
        assert "borrowed" in str(e).lower(), f"wrong error: {e}"
        raised = True
    assert raised, "compact() did not raise while borrowed!"
    print("PASS")


# ---------------------------------------------------------------------------
# Test 4: no-op when cache is small
# ---------------------------------------------------------------------------

def test_noop_when_small():
    print("TEST 4: no-op when cache fits target ... ", end="", flush=True)
    D = 4
    inv_freq = torch.ones(D // 2)
    cache, k_orig = build_cache(5, 1, 1, D, inv_freq)
    mgr = make_manager(cache, k_orig, 2, inv_freq, kv_budget=10)

    removed = mgr.compact(target_len=10, keep_recent=2)
    assert removed == 0
    assert mgr._len() == 5
    print("PASS")


# ---------------------------------------------------------------------------
# Test 5: compacted counter accumulates
# ---------------------------------------------------------------------------

def test_counter():
    print("TEST 5: compacted counter accumulates ... ", end="", flush=True)
    D = 4
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    cache, k_orig = build_cache(10, 1, 1, D, inv_freq)
    mgr = make_manager(cache, k_orig, 1, inv_freq, kv_budget=6)

    r1 = mgr.compact(target_len=6, keep_recent=2)
    assert mgr.compacted == r1
    print(f"PASS  (removed {r1})")


def test_request_handoff():
    print("TEST 6: controller request is consumed once ... ", end="", flush=True)
    D = 4
    inv_freq = torch.ones(D // 2)
    cache, k_orig = build_cache(5, 1, 1, D, inv_freq)
    mgr = make_manager(cache, k_orig, 1, inv_freq, kv_budget=8)
    mgr.request_compaction(12.0, 0.9, 0.2, 0.8)
    request = mgr.pop_compaction_request()
    assert request == {"vt": 12.0, "occupancy": 0.9, "novelty": 0.2, "confidence": 0.8}
    assert mgr.pop_compaction_request() is None
    print("PASS")


def test_ingester_executes_request_before_ingest():
    print("TEST 7: ingester executes request before ingest ... ", end="", flush=True)
    from foresight.input_ingester import input_ingester_thread

    class FakeBackend:
        def embed_text(self, text):
            return text

    class FakeManager:
        def __init__(self):
            self.b = FakeBackend()
            self.sink = 10
            self.kv_budget = 100
            self.events = []
            self.request = {"vt": 2.0, "occupancy": 0.9,
                            "novelty": 0.2, "confidence": 0.8}

        def pop_compaction_request(self):
            request, self.request = self.request, None
            return request

        def compact(self, target_len, keep_recent):
            self.events.append(("compact", target_len, keep_recent))
            return 25

        def ingest(self, embeds):
            self.events.append(("ingest", embeds))

        def evict(self):
            self.events.append(("evict",))
            return 0

        def seed(self, system_prompt):
            self.events.append(("seed", system_prompt))
            return self.sink

    class FakeConfig:
        system_prompt = "system"
        instruction = "task"
        icl_in_sink = False
        controller_prompt = ""
        kv_budget = 100
        compact_target_occupancy = 0.5
        compact_keep_recent_fraction = 0.5
        timestamp_tokens = False
        timestamp_fmt = ""
        gate_mode = "controller"
        goal_question = ""
        event = ""
        probe_default_s = 1.0
        deterministic = False
        video_id = "test"

    manager = FakeManager()
    frames = queue.Queue()
    frames.put((3.0, "frame"))
    stop = threading.Event()
    stop.set()
    input_ingester_thread(FakeConfig(), manager, frames, None, stop)
    names = [event[0] for event in manager.events]
    assert names.index("compact") < names.index("ingest"), names
    assert manager.events[names.index("compact")] == ("compact", 50, 20)
    print("PASS")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_positions_contiguous()
    test_sink_untouched()
    test_borrow_guard()
    test_noop_when_small()
    test_counter()
    test_request_handoff()
    test_ingester_executes_request_before_ingest()
    print("\nAll tests passed.")
