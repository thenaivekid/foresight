"""
backend.py — the model adapter (the ONLY model-specific file).

Everything else in this project is model-agnostic. To support a different
VLM you write a new ModelBackend subclass; nothing else changes.

A backend must provide:
  * tokenizer handles: `eos_id`, `newline_ids`, plus `decode(ids)`.
  * `embed_text(text)  -> [1, L, H]`   text -> input embeddings
  * `embed_frame(pil)  -> [1, N, H]`   one image -> N vision-token embeddings
  * `forward(embeds, cache, pos_start, phys_start) -> (logits[V], cache)`
        run the text decoder over `embeds`, appending to `cache`, and return
        the LAST position's logits (CPU). `pos_start` is the logical RoPE
        position of the first new token; `phys_start` is the physical cache
        write index (these differ after eviction — see manager.py).

------------------------------------------------------------------------------
Qwen3VLBackend NOTE (UNTESTED on this machine — validate on the GPU server):
  * mRoPE positions are 3D (temporal, height, width). We feed *linear* positions
    (all three axes = sequential index), which is shape-correct for the mRoPE
    rotary but ignores the 2D image geometry. This is the one deliberate
    simplification; it mirrors the linear-cache approach used in an earlier
    prototype. If understanding quality is poor, the upgrade is to compute true
    positions via the model's `get_rope_index` per appended chunk.
  * Attribute layout (visual / language_model / lm_head) is resolved
    defensively because it has moved between transformers versions. If
    resolution fails you get a clear error listing what was tried.
------------------------------------------------------------------------------
"""
from abc import ABC, abstractmethod
import torch

# cuDNN's fused MHA graph fails on long KV sequences on this stack
# (torch 2.12 + GH200): scaled_dot_product_attention raised
# "Expected mha_graph.execute(...).is_good() to be true, but got false" at
# ~75k tokens and killed the controller thread mid-video. Disable that one
# backend; flash / mem-efficient / math remain available.
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass


class ModelBackend(ABC):
    eos_id: int
    newline_ids: set
    hidden_size: int

    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor: ...
    @abstractmethod
    def embed_frame(self, pil_image) -> torch.Tensor: ...
    @abstractmethod
    def forward(self, embeds, cache, pos_start, phys_start): ...
    @abstractmethod
    def decode(self, ids) -> str: ...


def _word_ids(tok, words):
    """Token ids for single-token surface forms (e.g. the newline stop token)."""
    out = []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    return out


class Qwen3VLBackend(ModelBackend):
    def __init__(self, cfg, role="full"):
        # role decides which half of the model to KEEP resident (for multi-GPU):
        #   "full"     -> vision + language (single-GPU: does everything)
        #   "vision"   -> only the ViT/merger (an encoder-only replica)
        #   "language" -> only the decoder + lm_head (ingester/controller replica)
        # The unused half is dropped after load; results are identical because each
        # role never calls the half it drops. device = which GPU this replica lives on.
        from transformers import AutoProcessor
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[cfg.dtype]
        self.device, self.dtype, self.cfg, self.role = cfg.device, dtype, cfg, role
        print(f"[backend] loading {cfg.model_id} ({cfg.dtype}, role={role}) on {cfg.device} ...", flush=True)

        # Qwen3-VL needs a recent transformers; class name may be
        # Qwen3VLForConditionalGeneration. Fall back to the generic loader.
        try:
            from transformers import Qwen3VLForConditionalGeneration as VLM
        except Exception:
            from transformers import AutoModelForImageTextToText as VLM
        # full -> load straight to the GPU; vision/language -> load to CPU then move
        # only the kept half to the GPU (peak = kept half).
        self.model = VLM.from_pretrained(
            cfg.model_id, torch_dtype=dtype,
            device_map=(cfg.device if role == "full" else None),
            low_cpu_mem_usage=True).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.tok = self.processor.tokenizer
        self._cap_image_resolution(cfg.max_pixels)

        self._resolve_modules()        # find visual / language_model / lm_head
        self.hidden_size = self.model.config.get_text_config().hidden_size \
            if hasattr(self.model.config, "get_text_config") else self.model.config.hidden_size

        self.eos_id = self.tok.eos_token_id
        self.newline_ids = set(_word_ids(self.tok, ["\n"]))
        # yes/no single-token ids for the PROBE-GATE (gate_mode="probe") yes-share
        self.yes_ids = _word_ids(self.tok, getattr(cfg, "yes_words", ["yes", "Yes", " yes", " Yes"]))
        self.no_ids = _word_ids(self.tok, getattr(cfg, "no_words", ["no", "No", " no", " No"]))

        self._apply_role(role)         # drop the unused half + free its GPU memory

    # -- keep only the half this role needs; free the rest; move kept half to GPU --
    def _apply_role(self, role):
        import gc
        if role == "full":
            return                      # already loaded straight to the GPU
        m = self.model
        inner = m.model if hasattr(m, "model") else m   # Qwen3VLModel (.visual/.language_model)
        if role == "vision":            # encoder-only: drop decoder + lm_head
            self.language_model = self.lm_head = self.embed_tokens = None
            if hasattr(m, "lm_head"):
                m.lm_head = None
            if hasattr(inner, "language_model"):
                inner.language_model = None
            gc.collect()
            inner.visual.to(self.device)                # move ONLY the ViT to GPU
        elif role == "language":        # decoder-only: drop the vision tower
            self.visual = self.get_image_features = None
            if hasattr(inner, "visual"):
                inner.visual = None
            gc.collect()
            inner.language_model.to(self.device)        # move ONLY the decoder to GPU
            if getattr(m, "lm_head", None) is not None:
                m.lm_head.to(self.device)
        gc.collect()
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    # -- cap vision tokens/frame by limiting image pixels (perf + KV memory) --
    def _cap_image_resolution(self, max_pixels):
        """For Qwen2VLImageProcessor the pixel-AREA bounds live in
        `size.longest_edge` (max pixels) / `size.shortest_edge` (min pixels) --
        these are total-pixel counts, not edge lengths. Vision tokens =
        pixels / (patch_size*merge_size)^2, so capping longest_edge caps the
        token count. Confirmed by the profiler's vis_tokens_per_frame sample."""
        if not max_pixels:
            return
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            return
        per_tok = (getattr(ip, "patch_size", 16) * getattr(ip, "merge_size", 2)) ** 2
        size = getattr(ip, "size", None)

        def _set(obj, key, val):            # SizeDict supports both attr + item
            try:
                setattr(obj, key, val)
            except Exception:
                pass
            try:
                obj[key] = val
            except Exception:
                pass

        if size is not None:
            cur_min = getattr(size, "shortest_edge", None) or (size.get("shortest_edge")
                      if hasattr(size, "get") else None) or per_tok
            _set(size, "longest_edge", max_pixels)
            _set(size, "shortest_edge", min(cur_min, max_pixels))
        # legacy flat attrs, if a future processor uses them
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        # also push onto the processor wrapper so apply-chat-template paths agree
        for attr in ("max_pixels",):
            if hasattr(self.processor, attr):
                setattr(self.processor, attr, max_pixels)
        print(f"[backend] capped image to <= {max_pixels} px "
              f"(~{max_pixels // per_tok} vision tokens/frame)", flush=True)

    # -- module resolution: tolerate transformers' shifting attribute layout --
    def _resolve_modules(self):
        m = self.model
        cand_lm = [lambda: m.model.language_model, lambda: m.language_model,
                   lambda: m.model]
        cand_vis = [lambda: m.model.visual, lambda: m.visual,
                    lambda: m.model.vision_tower]
        self.language_model = _first(cand_lm, "language model (text decoder)")
        self.visual = _first(cand_vis, "vision tower")
        self.lm_head = m.lm_head if hasattr(m, "lm_head") else m.get_output_embeddings()
        self.embed_tokens = self.model.get_input_embeddings()
        # preferred high-level image->features helper if present
        self.get_image_features = getattr(m, "get_image_features", None)

    # ---------------------------------------------------------------- text
    def embed_text(self, text):
        ids = self.tok(text, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.device)
        return self.embed_tokens(ids)               # [1, L, H]

    def embed_token(self, tok_id):
        ids = torch.tensor([[tok_id]], device=self.device)
        return self.embed_tokens(ids)               # [1, 1, H]

    # --------------------------------------------------------------- vision
    @torch.no_grad()
    def embed_frame(self, pil_image):
        feat = self.processor.image_processor(images=[pil_image], return_tensors="pt")
        pixel_values = feat["pixel_values"].to(self.device, self.dtype)
        grid_thw = feat["image_grid_thw"].to(self.device)
        if self.get_image_features is not None:
            out = self.get_image_features(pixel_values=pixel_values, image_grid_thw=grid_thw)
        else:
            out = self.visual(pixel_values, grid_thw=grid_thw)
        # transformers 5.x returns an output object whose `pooler_output` holds
        # the image embeds (a per-image tuple); older versions returned a plain
        # tensor or a tuple. Unwrap both forms to a single [N, H] tensor.
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        embeds = out[0] if isinstance(out, (list, tuple)) else out
        embeds = embeds.reshape(1, -1, self.hidden_size)               # [1, N, H]
        # Record the POST-MERGER token grid for spatial consumers (the CLIP-QDP
        # pruner resamples a 16x16 CLIP score map onto exactly this grid).
        # grid_thw is the PRE-merger patch grid; the merger folds it by
        # spatial_merge_size in each axis, so recover the factor from N itself
        # rather than trusting a config field name across transformers versions.
        try:
            _t, _h, _w = [int(x) for x in grid_thw[0].tolist()]
            _n = int(embeds.shape[1])
            _m = int(round(((_t * _h * _w) / max(_n, 1)) ** 0.5))
            _m = max(_m, 1)
            self.last_grid_hw = (_h // _m, _w // _m)
        except Exception:
            self.last_grid_hw = None
        return embeds.to(self.dtype)

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, embeds, cache, pos_start, phys_start, want_logits=True):
        """Run the decoder over `embeds`, append to `cache`, return (logits, cache).

        want_logits=False skips the lm_head entirely (the 151k-vocab projection):
        the INGESTER only prefills frames into the cache and never reads logits, so
        this saves a large matmul on every ingested chunk.

        When want_logits=True the returned logits stay ON THE GPU (float, last
        position only). The old code shipped the full 151k-vocab vector to CPU
        every token (~35-45 ms/tok of pure D2H sync); the caller now samples on GPU
        and only the chosen token id crosses the bus."""
        L = embeds.shape[1]
        cache_position = torch.arange(phys_start, phys_start + L, device=self.device)
        # linear positions, shaped [3, 1, L] for the 3D mRoPE rotary (all axes
        # equal -> behaves like ordinary 1D RoPE). See module docstring.
        lin = torch.arange(pos_start, pos_start + L, device=self.device)
        position_ids = lin.view(1, 1, L).expand(3, 1, L).contiguous()
        out = self.language_model(
            inputs_embeds=embeds.to(self.device, self.dtype),
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        if not want_logits:
            return None, out.past_key_values
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        return logits[0].float(), out.past_key_values                  # GPU (was .cpu())

    def decode(self, ids):
        return self.tok.decode(ids).strip()


def _first(candidates, what):
    for get in candidates:
        try:
            mod = get()
            if mod is not None:
                return mod
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate the {what} on this model; check the transformers "
        f"version / attribute layout in backend._resolve_modules")
