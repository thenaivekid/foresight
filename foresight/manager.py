"""
manager.py — the shared KV cache as a CONCURRENT, DB-like resource.

Design goals (v3, the "nothing blocks anyone" rewrite):

  * There is no longer a single GPU worker thread that serializes everything.
    Each component thread (encoder / orchestrator / writer) issues its own GPU
    work. The GPU still time-shares kernels (one device), but there is no
    *logical* head-of-line blocking anymore.

  * Concurrency control, MVCC-style:
      - The PRIMARY cache has exactly ONE writer: the orchestrator thread. So
        there is never a write-write conflict on it (the cache is a linear
        sequence; two appenders would be incoherent anyway).
      - READERS that must not be disturbed mid-flight (the writer thread) take a
        SNAPSHOT: `snapshot_clone()` returns an independent deep copy of the
        cache + the logical position. The reader then generates on its private
        copy, holding NO lock — so the orchestrator keeps mutating the primary
        concurrently and neither blocks the other. This is the read-snapshot
        isolation you get from an MVCC database.
      - The internal `_lock` is held only for the *brief* primary mutations
        (append / probe / evict) and for the clone copy. Read-read is free;
        a snapshot waits at most one in-flight op.

  * Two clocks survive eviction: `next_pos` (logical RoPE position, monotonic)
    vs physical cache length (write index). StreamingLLM eviction keeps a sink
    prefix + recent window; `next_pos` keeps running. That divergence is a
    CORRECTNESS problem, not bookkeeping (the survivors keep their original
    rotations), so eviction now either re-bases positions or fails loudly —
    see `on_evict` / `_rebase_locked`.

  * proactivity probe: splice a yes/no question, read the answer, ERASE it
    (truncate back) so it leaves no trace in the primary cache.

Honest limit: on ONE GPU the forwards still serialize on the SMs. Put the
writer (or encoder) on a second GPU with a model replica and this same code
becomes truly parallel — the snapshot already removes the cache coupling.
"""
import copy
import os
import threading
import time
import torch
from transformers import DynamicCache


class KVCacheManager:
    def __init__(self, backend, kv_budget, prof=None, sync=True, on_evict=None):
        self.b = backend
        self.kv_budget = kv_budget
        self.prof = prof
        # ---- what to do the instant the two clocks diverge -------
        # "raise"  (DEFAULT) — stop the run. Eviction has NEVER fired in any eval
        #          (max_seconds=300 -> ~55k tokens vs a 262144 budget), so this
        #          changes no existing result; it only converts "results past this
        #          point are silently untrustworthy" into a named failure. The
        #          moment kv_budget drops to 32768 for the compaction work
        #          (a prerequisite for compaction), eviction becomes routine and this fires.
        # "rebase" — re-rotate the surviving keys back onto a contiguous window
        #          (see _rebase_locked). The real fix, UNVERIFIED ON GPU.
        # "warn"   — the old behaviour: warn once and keep going. Kept only so an
        #          explicit "I know these numbers are wrong" run is possible.
        self.on_evict = on_evict or "raise"
        # fraction the window may overfill before one bulk eviction; see
        # _evict_locked. 0.0 restores evict-every-frame.
        self.evict_slack = float(os.environ.get("OMNIPRO_EVICT_SLACK", 0.25))
        # sync: torch.cuda.synchronize() around each op so per-op timing is true
        # GPU compute, not just kernel-launch wall time.
        self._sync = sync and torch.cuda.is_available()
        self.cache = DynamicCache()
        self.next_pos = 0                 # logical RoPE clock
        self.sink = 0                     # protected prefix length (system prompt)
        self.evicted = 0                  # total tokens dropped by eviction
        self.rebased = 0                  # total positions reclaimed by re-basing
        self.compacted = 0                # total tokens removed by compaction
        self._inv_freq_override = None    # testing hook: bypass _rope_inv_freq()
        self._lock = threading.Lock()     # guards PRIMARY mutations + snapshot
        # (label, phys0, pos0) while a reader is generating IN PLACE on the
        # primary; None otherwise. See borrow_begin().
        self._borrowed = None
        self._compact_request = None

    # ---- profiling helper ----
    def _rec(self, label, dur, wait):
        if self.prof is not None:
            self.prof.record_op(label, dur, wait)

    # ---- primary-cache primitives (call only while holding self._lock) ----
    def _len(self):
        return self.cache.get_seq_length()

    def _forward_primary(self, embeds):
        # ingest/seed only build the KV cache; they never read logits -> skip the
        # lm_head (want_logits=False) so every frame prefill is a bit cheaper.
        _, self.cache = self.b.forward(
            embeds, self.cache, pos_start=self.next_pos, phys_start=self._len(),
            want_logits=False)
        self.next_pos += embeds.shape[1]
        return None

    def _truncate(self, phys_len):
        if hasattr(self.cache, "crop"):
            self.cache.crop(phys_len)
        else:
            for i in range(len(self.cache.key_cache)):
                self.cache.key_cache[i] = self.cache.key_cache[i][:, :, :phys_len, :]
                self.cache.value_cache[i] = self.cache.value_cache[i][:, :, :phys_len, :]
        self._sync_len_locked(phys_len, "truncate")

    def _sync_len_locked(self, expected, op, before=None):
        """Make the cache's OWN reported length agree with the tensors we just
        rewrote, and VERIFY it — do not assume.

        This exists because of a real bug: `_evict_locked` used to fix up
        `_seen_tokens` in the transformers-4.x branch only, so on 5.x
        (`cache.layers`) the cache's bookkeeping kept the pre-eviction value while
        the tensors were shorter. Everything downstream is derived from
        `_len()`: `_forward_primary` passes it as `phys_start` (a wrong value
        writes the next frame at the wrong cache index), `borrow_begin` saves it
        as the erase point, and `occupancy()` feeds it to the compaction
        admission gate — a stale length there would make every refusal count in
        the compaction logs a fiction. Eviction has never fired, so the bug never bit; it
        becomes reachable the moment kv_budget drops to 32768.

        Rather than guess which attribute each transformers version keeps the
        length in, we set the ones we know, then assert the cache agrees."""
        if hasattr(self.cache, "_seen_tokens"):
            self.cache._seen_tokens = expected
        for layer in (getattr(self.cache, "layers", None) or []):
            if getattr(layer, "keys", None) is None:
                continue
            # Sliding-window layers carry their own monotonic counter. If it was
            # tracking the physical length we just changed, move it with us;
            # otherwise leave it alone and let the assertion below judge.
            cum = getattr(layer, "cumulative_length", None)
            if isinstance(cum, int) and before is not None and cum == before:
                layer.cumulative_length = expected
        got = self._len()
        if got != expected:
            raise RuntimeError(
                f"[manager] {op}: cache length bookkeeping disagrees with the "
                f"tensors — get_seq_length()={got}, tensors say {expected}. The "
                f"cache class ({type(self.cache).__name__}) keeps its length "
                f"somewhere _sync_len_locked does not know about; fix it there "
                f"before running, because phys_start / borrow_end / occupancy() "
                f"are all derived from this number.")

    def _rope_inv_freq(self):
        """Locate the text rotary's `inv_freq` (theta per head-dim pair).

        Resolved defensively for the same reason backend._resolve_modules is: the
        attribute has moved between transformers versions."""
        cands = []
        lm = getattr(self.b, "language_model", None)
        for owner in (lm, getattr(lm, "model", None), getattr(self.b, "model", None)):
            if owner is None:
                continue
            rot = getattr(owner, "rotary_emb", None)
            if rot is not None:
                cands.append(rot)
        for rot in cands:
            inv = getattr(rot, "inv_freq", None)
            if inv is not None:
                return inv
        raise RuntimeError(
            "position re-basing needs the text rotary's inv_freq and none of "
            "language_model[.model].rotary_emb / model.rotary_emb had it; "
            "re-run with on_evict='raise' and fix the resolution here first")

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _rebase_locked(self, delta):
        """POSITION RE-BASING. Rotate every surviving post-sink key
        back by `delta` positions, so the physical window is contiguous in RoPE
        space again and `next_pos` can be reset to the physical length.

        Why a single rotation is sufficient HERE: `backend.forward` feeds LINEAR
        mRoPE positions (all three axes = the same sequential index — see the
        Qwen3VLBackend docstring). RoPE is a rotation by p*theta_d per head-dim
        pair, and rotations compose, so shifting every axis of every token by the
        same -delta is exactly R(-delta) applied to the cached key:

            k' = k * cos(delta*theta) - rotate_half(k) * sin(delta*theta)

        Values are untouched (RoPE acts on q/k only). The sink keeps positions
        [0, sink) — StreamingLLM's attention-sink prefix — and the recent window
        lands immediately after it, which is precisely StreamingLLM's requirement
        that positions be assigned WITHIN the cache window rather than by absolute
        token index.

        HONEST STATUS: the ROTATION ITSELF is verified. Executed off-GPU against a
        numpy stand-in for the cache: re-basing the surviving window is identical
        (max |err| 3.3e-15, both the 4.x and 5.x cache layouts) to having encoded
        those keys at their new positions in the first place — R(-delta) . R(p) =
        R(p-delta), as the composition argument says. What remains UNVERIFIED is
        the one thing that check cannot see: that the cos/sin layout built here
        (half-split + rotate_half, matching backend.forward's linear mRoPE feed)
        is the same layout Qwen3-VL's rotary actually applies. If the real model
        interleaves its mRoPE sections differently, the composition still holds
        but against the wrong basis, and the emissions will collapse rather than
        degrade. So this stays opt-in (`on_evict="rebase"`), the default stays a
        hard failure, and the GPU check before trusting any number through this
        path is: one clip at a budget that never evicts vs. the same clip at a
        budget that does — graceful degradation, not collapse. If it collapses,
        the fix belongs in backend.py (a true `realign_rotary_suffix`), which
        this file must not reach into.
        """
        if delta <= 0:
            return
        inv_freq = self._rope_inv_freq()
        layers = getattr(self.cache, "layers", None)
        pairs = ([(l, "keys") for l in layers if getattr(l, "keys", None) is not None]
                 if layers is not None else
                 [(self.cache, i) for i in range(len(self.cache.key_cache))])
        for holder, key in pairs:
            k = getattr(holder, key) if isinstance(key, str) else holder.key_cache[key]
            # fp32 rotation, bf16 storage. Re-basing is applied ONCE PER
            # EVICTION and its error COMPOUNDS: measured on Qwen2.5-Omni's
            # theta=1e6 / d=128 rotary, rotating in bf16 drifts to rel_err 0.083
            # after 62 shifts and 0.244 after 248, against 0.033 / 0.105 when the
            # rotation is done in fp32. The remaining drift is bf16 *storage*
            # quantisation, which is why _evict_locked also evicts in large
            # chunks (hysteresis) rather than a few hundred tokens per frame --
            # fewer shifts is the other half of this fix.
            ang = (float(delta) * inv_freq.to(device=k.device, dtype=torch.float32))
            cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
            sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
            head = k[:, :, self.sink:, :].float()
            head = (head * cos - self._rotate_half(head) * sin).to(k.dtype)
            k = torch.cat([k[:, :, :self.sink, :], head], dim=2)
            if isinstance(key, str):
                setattr(holder, key, k)
            else:
                holder.key_cache[key] = k
        self.rebased += delta

    def _rebase_span_locked(self, phys_start, phys_end, deltas):
        """Per-token position re-basing for a physical span of the cache.

        Generalises _rebase_locked (which applies a uniform delta to every
        post-sink token) to the NON-CONTIGUOUS case: each token in the span
        [phys_start, phys_end) gets its own rotation delta.

        Args:
            phys_start, phys_end: physical cache indices bounding the span.
            deltas: 1-D float tensor of length (phys_end - phys_start).
                    deltas[i] is how many positions to shift the token at
                    physical index phys_start + i BACKWARD (positive means
                    the key moves to an earlier position).

        Mathematics: identical to _rebase_locked except the rotation angle is
        a per-token vector rather than a scalar. R(-d_i) . R(p_i) = R(p_i - d_i),
        so the key looks as if it was originally encoded at its new position.
        """
        span_len = phys_end - phys_start
        if span_len == 0:
            return
        if not isinstance(deltas, torch.Tensor):
            deltas = torch.tensor(deltas, dtype=torch.float32)
        assert len(deltas) == span_len, (
            f"_rebase_span_locked: deltas length {len(deltas)} != span {span_len}")
        if deltas.abs().max().item() == 0:
            return

        inv_freq = (self._inv_freq_override
                    if self._inv_freq_override is not None
                    else self._rope_inv_freq())
        inv_f = inv_freq.to(dtype=torch.float32)
        deltas = deltas.to(device=inv_f.device, dtype=torch.float32)
        if inv_f.dim() == 0:
            inv_f = inv_f.unsqueeze(0)
        # angles: [span_len, head_dim//2]
        angles = deltas.unsqueeze(-1) * inv_f.unsqueeze(0)
        cos_a = torch.cat([angles.cos(), angles.cos()], dim=-1)
        sin_a = torch.cat([angles.sin(), angles.sin()], dim=-1)

        layers = getattr(self.cache, "layers", None)
        pairs = ([(l, "keys") for l in layers
                  if getattr(l, "keys", None) is not None]
                 if layers is not None else
                 [(self.cache, i) for i in range(len(self.cache.key_cache))])

        for holder, key in pairs:
            k = (getattr(holder, key) if isinstance(key, str)
                 else holder.key_cache[key])
            span = k[:, :, phys_start:phys_end, :].clone()
            dev, dt = span.device, span.dtype
            c = cos_a.to(device=dev, dtype=dt).unsqueeze(0).unsqueeze(0)
            s = sin_a.to(device=dev, dtype=dt).unsqueeze(0).unsqueeze(0)
            span = span * c - self._rotate_half(span) * s
            k_new = torch.cat(
                [k[:, :, :phys_start, :], span, k[:, :, phys_end:, :]], dim=2)
            if isinstance(key, str):
                setattr(holder, key, k_new)
            else:
                holder.key_cache[key] = k_new

    def _evict_locked(self):
        n = self._len()
        # HYSTERESIS. Evicting the instant the window is full means one re-basing
        # rotation PER FRAME (hundreds per video), and re-basing error compounds
        # with every shift. Letting the cache overfill by evict_slack and then
        # cutting back to kv_budget in one go cuts the number of rotations by
        # ~1/evict_slack -- for a 900 s video at ~300 tok/s that is ~660 shifts
        # down to ~10. The window is therefore "at least kv_budget", oscillating
        # between kv_budget and kv_budget*(1+evict_slack).
        hi = int(self.kv_budget * (1.0 + self.evict_slack))
        if n <= hi:
            return 0
        keep_recent = self.kv_budget - self.sink
        # positions that vanish from the window: everything between the pinned
        # sink and the start of the recent window.
        delta = (n - keep_recent) - self.sink
        # ---- THE TWO CLOCKS DIVERGE HERE ------------------------------------
        # Until the first eviction, next_pos and the physical length advance
        # together on every path, so they are EQUAL and RoPE positions are exact.
        # From here on `next_pos` would keep climbing past the trained range
        # (Qwen3-VL-8B: max_position_embeddings=262144) while the cache holds only
        # `kv_budget` tokens, and the surviving keys would still carry their
        # ORIGINAL rotations — so every relative distance a query computes against
        # them is wrong by `delta`. This used to be a warn-once guard that "has
        # never fired"; lowering kv_budget to 32768 for the compaction work makes
        # eviction routine, at which point a guard is not enough.
        if self.on_evict == "raise":
            raise RuntimeError(
                f"[manager] EVICTION at len={n} (budget={self.kv_budget}) and "
                f"position re-basing is not enabled. {delta} positions would be "
                f"dropped from the window while the surviving keys keep their "
                f"original RoPE rotations -> every result past this point is "
                f"invalid. Choose "
                f"explicitly: KVCacheManager(..., on_evict='rebase') to re-rotate "
                f"the survivors (implemented, UNVERIFIED on GPU), or "
                f"on_evict='warn' to reproduce the old silent-drift behaviour, or "
                f"raise kv_budget so eviction cannot occur.")
        if self.on_evict == "rebase":
            self._rebase_locked(delta)
        elif not getattr(self, "_evict_warned", False):
            self._evict_warned = True
            print(f"[manager] WARNING: FIRST EVICTION at len={n} (budget="
                  f"{self.kv_budget}). next_pos ({self.next_pos}) now diverges from "
                  f"the physical window; RoPE positions will exceed the trained "
                  f"range. Results beyond this point are NOT trustworthy "
                  f"until position re-basing is implemented.",
                  flush=True)
        if hasattr(self.cache, "layers"):                 # transformers 5.x
            for layer in self.cache.layers:
                if getattr(layer, "keys", None) is None:
                    continue
                k, v = layer.keys, layer.values
                layer.keys = torch.cat([k[:, :, :self.sink], k[:, :, n - keep_recent:]], dim=2)
                layer.values = torch.cat([v[:, :, :self.sink], v[:, :, n - keep_recent:]], dim=2)
        else:                                             # transformers 4.x
            for i in range(len(self.cache.key_cache)):
                k, v = self.cache.key_cache[i], self.cache.value_cache[i]
                self.cache.key_cache[i] = torch.cat([k[:, :, :self.sink], k[:, :, n - keep_recent:]], dim=2)
                self.cache.value_cache[i] = torch.cat([v[:, :, :self.sink], v[:, :, n - keep_recent:]], dim=2)
        # Both branches, not just 4.x: the physical window is now sink+keep_recent
        # and every consumer of _len() must see that (see _sync_len_locked).
        self._sync_len_locked(self.sink + keep_recent, "evict", before=n)
        self.evicted += n - self.kv_budget
        if self.on_evict == "rebase":
            # the window is contiguous again: the logical clock is the physical
            # length, exactly as it was before the first eviction.
            self.next_pos = self._len()
        return n - self.kv_budget

    def occupancy(self):
        """Physical cache length / budget, in [0, 1] — the pressure signal
        `compaction.admit()` gates on. Cheap: no copy, one brief lock.

        Excludes an in-flight BORROW. The controller generates its tick in place
        on the primary (borrow_begin) and `borrow_end` erases every token it
        appended, so those tokens are not memory pressure — they are a transient
        the reader itself created and is about to remove. Counting them would let
        the gate's answer depend on HOW FAR INTO ITS OWN DECODE the caller was
        when it asked, which is neither a property of the cache nor stable across
        the schema walk. `_borrowed[1]` is the pre-borrow physical length."""
        if not self.kv_budget:
            return 0.0
        with self._lock:
            phys = self._borrowed[1] if self._borrowed is not None else self._len()
            return min(1.0, phys / float(self.kv_budget))

    def request_compaction(self, vt, occupancy, novelty, confidence):
        """Publish one admitted controller request for the sole cache writer."""
        with self._lock:
            self._compact_request = {
                "vt": float(vt), "occupancy": float(occupancy),
                "novelty": float(novelty), "confidence": float(confidence),
            }

    def pop_compaction_request(self):
        """Consume the newest admitted request; only the ingester calls this."""
        with self._lock:
            request = self._compact_request
            self._compact_request = None
            return request

    # ===== public API (thread-safe) =====
    def seed(self, system_text):
        """Seed the cache once; the resulting length becomes the eviction sink."""
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            self._forward_primary(self.b.embed_text(system_text))
            self.sink = self._len()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("seed", t2 - t1, t1 - t0)
        return self.sink

    def ingest(self, embeds):
        """Append projected visual (or text) tokens to the primary cache.
        ONLY the orchestrator thread calls this -> single-writer, no conflict."""
        t0 = time.time()
        self._assert_not_borrowed("ingest")
        with self._lock:
            t1 = time.time()
            self._forward_primary(embeds)
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("ingest.frame", t2 - t1, t1 - t0)

    def evict(self):
        t0 = time.time()
        # eviction re-lays the physical window, so a borrow's saved phys0 would no
        # longer mean what it meant. Never evict mid-borrow.
        self._assert_not_borrowed("evict")
        with self._lock:
            t1 = time.time()
            dropped = self._evict_locked()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("evict", t2 - t1, t1 - t0)
        return dropped

    def compact(self, target_len=None, keep_recent=None):
        """COMPACTION — selective thinning of the middle span of the KV cache.

        Unlike evict() which drops the ENTIRE middle span as a contiguous block,
        compact() uniformly sub-samples the middle to a target density, keeping
        tokens from across the full temporal span of the video history.

        SELECTION POLICY: UNIFORM STRIDE SUB-SAMPLING.
        --------------------------------------------------
        From the compactible middle span [sink, n - keep_recent), we keep
        `keep_mid` tokens chosen by torch.linspace(0, mid_len-1, keep_mid).long()
        — i.e. evenly spaced indices that always include the first and last token
        of the middle span. This is chosen over value-norm scoring or attention-
        based policies because:
          1. It makes zero assumptions about what is "informative" — safer on a
             frozen model where we cannot validate informativeness signals.
          2. It preserves temporal diversity: every part of the video history
             retains representation.
          3. Zero extra compute — no forward pass, no norm computation.
          4. Fully deterministic — no RNG, no floating-point ordering issues.

        POSITION RE-BASING: because the surviving tokens come from non-contiguous
        original positions, _rebase_span_locked applies a PER-TOKEN rotation
        R(-delta_i) to each surviving key, making it look as if it was encoded
        at its new contiguous position. This is the general case of the uniform
        delta that _rebase_locked uses for eviction.

        INVARIANT 3 (single writer): only the ingester (writer) calls this.
        The controller REQUESTS compaction through admit(); the ingester
        executes it here.

        Args:
            target_len: desired total cache length after compaction.
                        Defaults to kv_budget.
            keep_recent: number of recent tokens to protect.
                         Defaults to (target_len - sink) // 2.

        Returns: number of tokens removed (0 if compaction was unnecessary).
        """
        t0 = time.time()
        self._assert_not_borrowed("compact")
        with self._lock:
            t1 = time.time()
            n = self._len()
            if target_len is None:
                target_len = self.kv_budget
            if keep_recent is None:
                keep_recent = max(1, (target_len - self.sink) // 2)
            # Clamp keep_recent to available non-sink tokens
            keep_recent = min(keep_recent, n - self.sink)
            if keep_recent < 0:
                keep_recent = 0

            mid_start = self.sink
            mid_end = n - keep_recent
            mid_len = mid_end - mid_start

            if mid_len <= 0 or n <= target_len:
                self._rec("compact", 0.0, time.time() - t0)
                return 0  # nothing to compact

            keep_mid = target_len - self.sink - keep_recent
            if keep_mid < 0:
                keep_mid = 0
            if keep_mid >= mid_len:
                self._rec("compact", 0.0, time.time() - t0)
                return 0  # middle already fits

            # Uniform stride: evenly spaced indices including first and last
            if keep_mid > 0:
                indices = torch.linspace(0, mid_len - 1, keep_mid).long()
            else:
                indices = torch.tensor([], dtype=torch.long)

            # --- Build the new cache tensors ---
            layers = getattr(self.cache, "layers", None)
            if layers is not None:  # transformers 5.x
                for layer in layers:
                    if getattr(layer, "keys", None) is None:
                        continue
                    k, v = layer.keys, layer.values
                    k_parts = [k[:, :, :self.sink, :]]
                    v_parts = [v[:, :, :self.sink, :]]
                    if keep_mid > 0:
                        k_parts.append(k[:, :, mid_start:mid_end, :][:, :, indices, :])
                        v_parts.append(v[:, :, mid_start:mid_end, :][:, :, indices, :])
                    k_parts.append(k[:, :, mid_end:, :])
                    v_parts.append(v[:, :, mid_end:, :])
                    layer.keys = torch.cat(k_parts, dim=2)
                    layer.values = torch.cat(v_parts, dim=2)
            else:  # transformers 4.x
                for i in range(len(self.cache.key_cache)):
                    k = self.cache.key_cache[i]
                    v = self.cache.value_cache[i]
                    k_parts = [k[:, :, :self.sink, :]]
                    v_parts = [v[:, :, :self.sink, :]]
                    if keep_mid > 0:
                        k_parts.append(k[:, :, mid_start:mid_end, :][:, :, indices, :])
                        v_parts.append(v[:, :, mid_start:mid_end, :][:, :, indices, :])
                    k_parts.append(k[:, :, mid_end:, :])
                    v_parts.append(v[:, :, mid_end:, :])
                    self.cache.key_cache[i] = torch.cat(k_parts, dim=2)
                    self.cache.value_cache[i] = torch.cat(v_parts, dim=2)

            new_len = self.sink + keep_mid + keep_recent
            self._sync_len_locked(new_len, "compact", before=n)

            # --- Position re-basing (per-token) ---
            # Middle tokens: original pos was sink + indices[j], new pos is sink + j
            mid_deltas = (indices.float() - torch.arange(keep_mid).float()
                          if keep_mid > 0 else torch.tensor([]))
            # Recent tokens: all shift by the same amount
            recent_delta = float(mid_len - keep_mid)
            rec_deltas = torch.full((keep_recent,), recent_delta)
            all_deltas = (torch.cat([mid_deltas, rec_deltas])
                          if keep_mid > 0 else rec_deltas)

            self._rebase_span_locked(self.sink, new_len, all_deltas)

            removed = n - new_len
            self.compacted += removed
            self.next_pos = new_len  # contiguous after re-basing

            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("compact", t2 - t1, t1 - t0)
        return removed

    def probe(self, question, label):
        """PROBE-GATE (gate_mode='probe'): splice a yes/no question onto the PRIMARY
        cache, read one forward pass of logits, compute the yes-share, then ERASE
        the probe (truncate + restore the logical clock) so it leaves no trace."""
        from proactivity import yes_share
        t0 = time.time()
        self._assert_not_borrowed("probe")
        with self._lock:
            t1 = time.time()
            phys0, pos0 = self._len(), self.next_pos
            logits, self.cache = self.b.forward(
                self.b.embed_text(question), self.cache,
                pos_start=pos0, phys_start=phys0, want_logits=True)
            share = yes_share(logits, self.b.yes_ids, self.b.no_ids)
            self._truncate(phys0)
            self.next_pos = pos0
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec(label, t2 - t1, t1 - t0)
        return share

    # ---- IN-PLACE READ (the snapshot-free path) -------------------------------
    # `probe()` above already proves the mechanism: splice onto the PRIMARY, read,
    # truncate back. borrow_begin/borrow_end is the same trick opened up so a
    # caller can run a whole multi-token generation between the two, instead of a
    # single forward. It exists because snapshot_clone() copies the ENTIRE cache
    # every tick -- 144 KB/token, so ~8 GB on a 300 s clip -- purely to protect the
    # reader from a concurrent writer.
    #
    # WHEN THAT PROTECTION IS WORTH NOTHING: in lockstep (cfg.deterministic=True,
    # which is every benchmark number) the ingester is parked in
    # `while clock.get_next_check() <= vt: sleep(0.002)` for the whole duration of
    # a controller tick. There is no concurrent writer to protect against. The copy
    # is pure cost.
    #
    # We do NOT hold the lock across the generation. Holding it for the 2-4 s of a
    # tick would make a free-running ingester block on the cache, which breaks
    # INVARIANT 2. Instead the borrow is DECLARED: `_borrowed` is set, and every
    # primary mutation refuses loudly while it is. A silent corruption (an ingested
    # frame landing inside the borrow, then being truncated away by borrow_end)
    # becomes an immediate, named exception.
    def borrow_begin(self, label="borrow"):
        """Lend the PRIMARY cache to a reader. Returns (pos, phys) to generate at.

        The reader appends to `self.cache` exactly as it would to a clone, using
        the same pos_start/phys_start -- identical prefix, identical positions,
        therefore identical logits. `borrow_end()` erases every appended token."""
        t0 = time.time()
        with self._lock:
            if self._borrowed is not None:
                raise RuntimeError(
                    f"borrow_begin({label}) while cache is already borrowed by "
                    f"{self._borrowed[0]!r} -- two readers cannot share the primary")
            self._borrowed = (label, self._len(), self.next_pos)
        self._rec(f"{label}.begin", 0.0, time.time() - t0)
        return self._borrowed[2], self._borrowed[1]      # (pos, phys)

    def borrow_end(self):
        """Erase the borrow: truncate back to the pre-borrow length and restore the
        logical clock. Idempotent -- safe to call when nothing is borrowed."""
        if self._borrowed is None:
            return
        label, phys0, pos0 = self._borrowed
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            self._truncate(phys0)
            self.next_pos = pos0
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
            self._borrowed = None
        self._rec(f"{label}.end", t2 - t1, t1 - t0)

    def _assert_not_borrowed(self, op):
        if self._borrowed is not None:
            raise RuntimeError(
                f"{op}() on the primary cache while it is borrowed by "
                f"{self._borrowed[0]!r}. In lockstep this cannot happen (the "
                f"ingester waits on the clock); if you see it, the caller is "
                f"running free and must use snapshot_clone() instead.")

    def snapshot_clone(self):
        """MVCC read snapshot: return an INDEPENDENT clone of the primary cache
        plus its logical position + physical length. The caller (writer) then
        generates on the clone holding no lock, fully concurrent with the
        orchestrator's ongoing mutations of the primary.

        Costs a full deep copy of the cache on EVERY call. See borrow_begin() for
        the snapshot-free path used when there is provably no concurrent writer."""
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            clone = self._clone_cache()
            pos, phys = self.next_pos, self._len()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("snapshot_clone", t2 - t1, t1 - t0)
        return clone, pos, phys

    def _clone_cache(self):
        try:
            return copy.deepcopy(self.cache)
        except Exception:
            # manual fallback: clone the K/V tensors layer by layer
            dst = DynamicCache()
            src = self.cache
            if hasattr(src, "layers"):
                for layer in src.layers:
                    k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
                    if k is None:
                        continue
                    dst.update(k.clone(), v.clone(), len(dst.layers))
            else:
                for i in range(len(src.key_cache)):
                    dst.update(src.key_cache[i].clone(), src.value_cache[i].clone(), i)
            return dst
