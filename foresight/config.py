"""
config.py — the single config for the asynchronous pipeline.

ONE proactivity mode only: the pure-generative CONTROLLER (probe_scheduler was
"model"). The encoder streams frames -> the ingester prefills them into a shared
KV cache -> the controller reads that cache each tick and emits ONE control JSON
(fps / have_enough_info / answer / question / next_check_s / compact_now), which
is both the output gate and the writer. There are no fixed yes/no gates here.
(There is no `new_event` field: the schema walk never produced it, so it would
be a permanently-false third state.)

SETTINGS ONLY. All prompt TEXT lives in `prompts.py` — one home for every string
we say to the model (per-task ICL for all 9 tasks, the generic controller DSL,
the probe-gate/system templates). This file just names and wires them. The eval
harness carries no prompt text either.

(The fixed-gate ablations, multi-GPU replicas, and VisionZip pruning live on the
`main` branch.)
"""
from dataclasses import dataclass, field

# ALL prompt text lives in prompts.py (settings here, content there).
from prompts import (CONTROLLER_PROMPT_GENERIC, GOAL_QUESTION, SYSTEM_PROMPT,
                     TASK_CONTROLLER_PROMPTS, TASK_WRITER_PROMPTS, WRITER_CUE,
                     WRITER_PROMPT)
@dataclass
class AsyncOmniConfig:
    # ---- model ----
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"               # primary GPU: ingester + shared KV cache
    # Multi-GPU replicas measured NO decode speedup (the per-tick cost is inherent
    # decode, not GPU contention) -> default single-GPU. Set to "cuda:1"/"cuda:2"
    # (with >=3 visible GPUs) to re-enable the split.
    writer_device: str = ""            # controller replica GPU ("" = share primary)
    encoder_device: str = ""           # vision-encoder replica GPU ("" = share primary)
    dtype: str = "bfloat16"            # float16 / bfloat16 / float32
    # Cap vision tokens/frame by limiting image pixel area. One vision token covers
    # a (patch_size*merge_size)^2 = 32x32 = 1024 px region for Qwen3-VL, so
    # 200704 px / 1024 = ~196 tokens/frame (actual ~180 after aspect resize).
    # 0 = processor default (~2040 tokens/frame, near-memoryless).
    max_pixels: int = 200704
    profile: bool = True              # collect + print the profiling summary

    # ---- reproducibility ----
    seed: int = 0                     # seeds python/numpy/torch (+cuda) RNGs
    deterministic: bool = False       # LIVE DEFAULT: independent async threads,
                                      # no encoder/ingester wait for controller ticks.
                                      # Explicit True opts into legacy lockstep eval;
                                      # never mix its results with wall-clock runs.

    # ---- absolute time signal ----
    # Prepend a short text timestamp before each frame's tokens so the model can
    # reason about real time (in-distribution for Qwen3-VL). {t} = video seconds.
    timestamp_tokens: bool = True
    timestamp_fmt: str = "\ntime {t:.1f}s\n"

    # ---- video / pacing ----
    video_path: str = ""
    fps: float = 1.0                  # encoder base frame rate (frames/s of video)
    max_seconds: float = 600.0
    realtime: bool = True             # pace the encoder to a wall clock (the
                                      # controller self-paces in VIDEO time, so it
                                      # needs realtime; batch fast-forwards the clip)
    speed: float = 1.0                # real-time multiplier (1.0 = live)
    frame_q_size: int = 8             # vis_q depth (encoder -> ingester buffer)

    # ---- encoder fps bounds (the controller steers fps within these) ----
    encoder_focus_fps: float = 3.0    # fps ceiling when the controller says "focus"
    encoder_idle_fps: float = 1.0     # fps floor (1-3 fps range; was 0.5)

    # ---- memory (shared linear KV cache) ----
    kv_budget: int = 262144           # 256K-token context (StreamingLLM eviction
                                      # past this; the system-prompt sink is pinned).

    # ---- the CONTROLLER ----
    # Self-paced cadence: the controller picks next_check_s each tick; clamp it so
    # it can't spin (min) or stall (max); fall back to default if unparsed.
    probe_default_s: float = 1.0
    probe_min_s: float = 0.2      # finer check grid (was 1.0) so vt lands closer to onsets
    probe_max_s: float = 1.5      # finer check grid (was 3.0)
    controller_max_tokens: int = 300  # cap for the control-JSON generation (free mode)

    # ---- SCHEMA-WALKED DECODE (the fix for "the diff never diffs") ------------
    # "schema": code force-feeds every key/punctuation as a batched PREFILL and the
    #           model only samples value slots; booleans are READ from the logits at
    #           the forced position (zero decode steps, continuous confidence).
    #           Measured motivation: 25 of the 35 tokens in a quiet tick were JSON
    #           structure the code already knew, and the prose rule "omit fps unless
    #           it changed" was obeyed 0/15 ticks. A diff is a DECODER CONSTRAINT,
    #           not a prompt instruction.
    # "free":   the legacy open-brace generate loop (kept for A/B).
    decode_mode: str = "schema"
    # Ablation: does the hit read need `seen` first?
    #   "before" -- describe the scene, then read the level (~1.3s/tick, current)
    #   "off"    -- read the level immediately, zero decodes (~0.15s/tick)
    #   "after"  -- read the level first, then describe (separates the effect of
    #               the perception step from the effect of its ORDER)
    # "off" would make the always-on trigger affordable; `seen` is also the single
    # biggest accuracy lever we have (F1 0.0 -> 0.255), so this must be measured.
    seen_mode: str = "before"
    # ---- FROZEN-PERCEPTION ablations -------------------------------------------
    # 20/58 videos emitted ONE byte-identical `seen` for the entire video; 31/58
    # emitted <=2 distinct descriptions ever. 81.6% of GT triggers were perception
    # failures, and a timestamp-only null model beat p_hit on 3 of 4 tasks.
    # Suspected cause: feeding the `seen` trace back into the prompt puts the
    # model's own last description a few tokens before the slot it must refill, so
    # greedy decode copies it. Two independent candidate fixes, BOTH DEFAULT OFF so
    # each is tested as one variable:
    seen_trace_in_prompt: bool = False  # the suspected cause
    now_anchor: bool = False            # "it is now Ns; describe the LATEST frame"

    # ---- PRIVILEGE THE PRESENT ------------------------------------------------
    # The task ICL is CONSTANT but was spliced fresh AFTER the video tokens every
    # tick, putting ~1400 tokens of instruction prose between the newest frame and
    # the point of generation — so the last thing the model saw before answering
    # was the manual, not the video. Seeding it into the pinned eviction sink puts
    # the newest frame adjacent to the decision AND prefills the ICL once per run
    # instead of once per tick.
    icl_in_sink: bool = True

    # ---- OUTPUT GATE ----------------------------------------------------------
    # "edge"       rising edge of the boolean level (original behaviour)
    # "hysteresis" Schmitt gate on the CONTINUOUS p_hit — only possible now that the
    #              logit read returns a real number. Targets PRECISION, measured at
    #              0.112 (89% of emits were false positives). Uses gate_high_thr /
    #              gate_low_thr / gate_rearm_s / debounce_s below.
    gate_strategy: str = "hysteresis"
    trigger_ema_tau_s: float = 0.5  # used only by experimental gate_strategy="ema_edge"
    distinct_sim_thr: float = 0.5   # word-overlap below this = a different occurrence
    # HOW THE CONTROLLER READS THE SHARED CACHE.
    #   "snapshot" -- MVCC deep copy every tick (mgr.snapshot_clone()). Safe under
    #                 any concurrency, and costs a full copy of the cache per tick:
    #                 144 KB/token, so ~8 GB and hundreds of ms on a 300 s clip.
    #   "inplace"  -- generate directly on the primary, then truncate the appended
    #                 tokens away (mgr.borrow_begin/borrow_end). Same prefix, same
    #                 pos_start/phys_start -> bit-identical logits, zero copy.
    # "inplace" is only legal when there is provably no concurrent writer, i.e.
    # deterministic=True (lockstep: the ingester waits on the clock for the whole
    # tick) and the controller shares the manager's GPU. controller.py refuses it
    # loudly and falls back to "snapshot" otherwise -- it never silently downgrades.
    controller_cache_mode: str = "snapshot"
    schema_max_seen_tokens: int = 12    # cap on the `seen` value slot
    schema_max_answer_tokens: int = 32  # cap on the `answer` value slot (hot ticks only)
    schema_max_int_tokens: int = 4      # cap on `event_time_s`
    schema_max_tail_tokens: int = 60    # cap on the `more` escape-hatch tail
    hit_threshold: float = 0.5          # P(true) above which the level is TRUE
    # PER-TASK firing thresholds. The single global 0.5 was wrong for every task:
    # some tasks scored time_f1 0.000 because p_hit never crossed 0.5 (gate could
    # never fire), others over-fired 6-8x. A p_hit threshold is only meaningful
    # per-task because the model's confidence SCALE shifts with how the task is
    # phrased (see evaluation/gates.py).
    #
    # Re-fitted offline by evaluation/resweep.py over a full development run (
    # 932 samples, 160,915 ticks; replay edge/level gate + refractory over saved
    # p_hit, greedy ±3s match, objective = time_f1). Pooled time_f1:
    #                                   fit-on-all   held-out(50%)
    #     old global p_hit>0.5            0.190          -        (10,904 emits)
    #     best single global              0.254          -
    #     per-task, ORIGINAL 156-cfg grid 0.316        0.308
    #     per-task, WIDE 1292-cfg grid    0.334        0.327      <- values below
    # Held-out moves in step with fit-on-all (+0.019 both), so the wide-grid gain is
    # REAL, not overfitting; and held-out ≈ fit-on-all (-0.008) means this fit barely
    # overfits at all — report the HELD-OUT number, it costs almost nothing.
    # Grid tuning is SATURATED: pushing refractory past 180s gained +0.0006 (noise).
    #
    # The F1 surface is FLAT near the optimum (fit-on-all vs held-out picked different
    # configs yet landed within 0.007 F1). Do not over-trust the exact constants; the
    # ROBUST finding is the regime:
    #   one-shot tasks (ETG/IEA/snapshot) -> high thr + very long refractory ("fire once")
    #   dense tasks (seq_step/narration)  -> LOW thr + short refractory
    #   counting (dedup/cumulative)       -> very high thr + short refractory
    # ⚠️ Offline SCREEN (assumes p_hit independent of the gate; firing feeds
    # `reported` back into the prompt, so it is weakly not) — confirm on GPU.
    task_hit_thresholds: dict = field(default_factory=lambda: {
        "cumulative_counting": 0.925,
        "dedup_counting": 0.992,
        "event_narration": 0.10,
        "explicit_target_grounding": 0.50,
        "instant_event_alert": 0.45,
        "realtime_state_monitor": 0.80,
        "semantic_condition_alert": 0.98,
        "sequential_step_instruction": 0.01,
        "snapshot_counting": 0.985,
    })
    # Coupled per-task gate mode + refractory (seconds), from the same resweep fit.
    # mode: "edge" = fire on the rising edge of p_hit>=thr; "level" = fire on every
    # tick above thr (both honour the refractory debounce). Selected by the adapter
    # per sample.task alongside task_hit_thresholds. All three knobs are ONE fitted
    # config — using the threshold without its mode/refractory does not reproduce it.
    task_gate_modes: dict = field(default_factory=lambda: {
        "cumulative_counting": "edge",
        "dedup_counting": "edge",
        "event_narration": "level",
        "explicit_target_grounding": "edge",
        "instant_event_alert": "edge",
        "realtime_state_monitor": "level",
        "semantic_condition_alert": "level",
        "sequential_step_instruction": "level",
        "snapshot_counting": "edge",
    })
    task_refractory_s: dict = field(default_factory=lambda: {
        "cumulative_counting": 7.0,
        "dedup_counting": 5.0,
        "event_narration": 7.0,
        "explicit_target_grounding": 300.0,
        "instant_event_alert": 600.0,
        "realtime_state_monitor": 7.0,
        "semantic_condition_alert": 10.0,
        "sequential_step_instruction": 7.0,
        "snapshot_counting": 600.0,
    })
    # ---- THE FULL PLAN JSON) ---------------------------
    # The method figure draws a 9-field plan. Measured on a full development run
    # (199,909 ticks), without forcing, only three of those fields — have_enough_info / answer /
    # event_time_s — are reliably emitted: `fps` fills 2.08%, `next_check_s` 3.13%
    # and `question_for_next` 4.85%, because all three sit behind the `more`
    # escape hatch. The flags below promote these fields into the forced spine,
    # ONE FLAG PER FIELD so every one of them stays independently ablatable.
    #
    # The cost asymmetry that makes this nearly free: prefilling k forced tokens
    # is ONE forward pass, decoding k tokens is k forward passes. `fps`,
    # `next_check_s`, `compact_now` and `replan_vision` are read as a softmax
    # restricted to their (small, single-token) value set at the forced position —
    # ZERO extra forwards, and a calibrated continuous confidence on top, exactly
    # the upgrade the logit read already gave `have_enough_info`.
    plan_fps: bool = True               # force-read fps every tick (was 2.08% via `more`).
                                        # fps is _clamp-ed to
                                        # encoder_idle_fps..encoder_focus_fps, so a free
                                        # decode buys nothing a restricted read cannot;
                                        # this drives the input (fps) gate.
    plan_cadence: bool = True           # force-read next_check_s every tick (was 3.13%).
                                        # The value is clamped to probe_min_s..probe_max_s;
                                        # consumer: next_check_vt.
    plan_question: bool = True          # force-sample question_for_next every tick (was
                                        # 4.85%). The deferred-question mechanism. It is
                                        # the ONE addition with a real token cost (~8
                                        # sampled tokens/tick), which is why it gets its
                                        # own ablation arm (`L1_noQ`).
    plan_count: bool = True              # force-sample count every tick on counting tasks.
                                        # A small capped decode (~3 tokens); the
                                        # consumer feeds it back into the next prompt.
    plan_phase: bool = True              # force-sample phase every tick on realtime_state_monitor.
                                        # ~6-token decode; the consumer feeds it
                                        # back as the remembered current state.
    plan_compact: bool = False          # OFF by default. The field itself is ~free (a
                                        # logit read); it is only useful together with
                                        # compaction.py's admission gate and a reduced
                                        # kv_budget. compact_now is a REQUEST; code
                                        # decides, and refusals are a reportable result.
    compact_target_occupancy: float = 0.50  # reduce an admitted cache to this budget fraction
    compact_keep_recent_fraction: float = 0.50  # fraction of retained non-sink tokens kept recent
    compact_occupancy_thr: float = 0.85  # pressure needed before honoring a model request
    compact_p_thr: float = 0.70          # minimum P(compact_now=true) for admission
    compact_novelty_max: float = 0.60    # protect recent context while it is changing
    prune_classes: bool = False         # OFF by default (experimental).
                                        # Consumer: class_pruner.prune_frame in the encoder.
    prune_retention_floor: float = 0.30 # never drop below 30% of a frame's tokens
    prune_ignore_lambda: float = 0.5    # weight on the ignore-class penalty

    # ---- TOKEN PRUNING MODE (upstream of the KV cache, in vision_stream) -----
    # "off"  : no pruning (default, and what every published number used)
    # "dsh"  : per-position EMA history + cosine novelty gate (vision-vision).
    #          Does NOT use the text embedding table, so the negative result in
    #          probe_class_score.json (Qwen visual tokens do not align with
    #          Qwen's own text embeddings) does not apply.
    # "clip" : query-conditioned scoring in OpenCLIP-aligned space, the space
    #          QueryStream actually used.
    # "random": CONTROL ARM. Drops the same fraction as `clip` but chooses at
    #          random, preserving order. Separates "removing tokens hurts" from
    #          "CLIP selects the wrong tokens" -- without it a pruning
    #          regression cannot be attributed to the scorer.
    # Default is "off": the 18-video sweep never showed CLIP beating baseline on
    # joint-F1, so pruning must be opted into, not inherited.
    prune_mode: str = "off"
    dsh_tau_temp: float = 0.90       # keep token if cos(now, history) < tau
    dsh_alpha: float = 0.10          # EMA rate for the per-position history
    dsh_retention_floor: float = 0.30  # threshold-rule floor only
    # Real Qwen temporal cosines are video-dependent (mean 0.82 on a dynamic
    # clip vs 0.999 on a static one, probe_dsh_v2_*.json), so no absolute tau
    # transfers. "rank" keeps a fixed most-novel FRACTION and is scale-free.
    dsh_rule: str = "rank"           # "rank" | "threshold"
    dsh_keep_frac: float = 0.30      # rank rule: fraction of tokens kept

    # CLIP-QDP (prune_mode="clip"). Scores each Qwen token by the CLIP-space
    # similarity between its image region and the question, then keeps the
    # above-mean tokens subject to a floor. Uses OpenCLIP-aligned space, which
    # probe_clip_qdp_steps.json confirms is genuinely query-sensitive.
    clip_model_id: str = "openai/clip-vit-large-patch14"
    clip_retention_floor: float = 0.50
    random_keep_frac: float = 0.50   # keep fraction for prune_mode="random"

    # DIAGNOSTIC (P1): shuffle each frame's tokens before ingest. backend.forward
    # assigns linear positions, so row-major order at a fixed stride is the ONLY
    # carrier of spatial layout; if shuffling does not move the score, geometry
    # is not encoded and pruning cannot be damaging it.
    permute_frame_tokens: bool = False
    plan_classes: bool = False          # OFF by default: requires a class-scoring
                                        # validation probe to pass. The cosine alignment
                                        # between post-merger visual tokens and text
                                        # embeddings is incidental, not trained; if a
                                        # correct class does not beat a decoy on 20 frames,
                                        # keep/ignore come OUT of the schema rather than
                                        # ship two fields whose consumer measures noise.
                                        # The slots exist behind this flag.
    schema_max_question_tokens: int = 8  # cap on the `question_for_next` value slot. 8 is
                                        # the figure-faithful budget: enough for one concrete
                                        # thing to check next tick, small enough that
                                        # forcing it on EVERY tick is affordable.
    schema_max_count_tokens: int = 4    # cap on the `count` integer slot. 4 digits
                                        # covers 0-9999; typical OmniPro counts are 1-20.
    schema_max_phase_tokens: int = 6    # cap on the `phase` state-label slot. 1-4 words;
                                        # 6 tokens is generous for a short label.
    schema_max_class_tokens: int = 3    # cap per keep/ignore class-list slot on replan ticks.
    # Value sets for the restricted reads. They live here, not in controller.py, so
    # they stay visibly consistent with the _clamp bounds that follow them:
    # fps_choices spans encoder_idle_fps..encoder_focus_fps (1..3) and
    # cadence_choices sits inside probe_min_s..probe_max_s (0.2..1.5). A choice
    # outside those bounds would be silently clamped away, i.e. a dead option.
    fps_choices: tuple = (1, 2, 3)
    cadence_choices: tuple = (0.5, 1.0, 1.5)
    # ---- time-only tasks: no answer decode ------------------------------------
    # instant_event_alert / semantic_condition_alert are scored time_only: the
    # scorer never reads the emitted text, so the ~32-token `answer` decode on
    # 39-49% of their ticks is pure waste. Membership is read from
    # evaluation/metrics.py:TIME_ONLY — never re-listed here, or the two copies
    # drift. Dropping the field is NOT enough on its own: `answer` also gates
    # firing, latches the level and carries dedup, so controller.py changes all
    # three together.
    time_only_bypass: bool = True
    # Dedup by event_time_s identity instead of word overlap (the `ev0`
    # rule): if the model reports an onset it has already spoken about, that is the
    # same occurrence, so stay quiet. Measured +0.053 macro time-F1 with NO free
    # parameter (the tuned ev10 variant is better still but selects its window on
    # the data it is scored on), and it fires BEFORE the answer decode, which is
    # what makes an answer-free path possible at all. Offline replay
    # (retime.py --dedup) — a real run must confirm it.
    ev0_dedup: bool = True
    event_dedup_window_s: float = 0.0  # exact identity by default; >0 is an ablation
    event_identity_in_prompt: bool = False  # opt-in code-owned onset ledger
    # The task name for the current sample, used ONLY to select the per-task field
    # whitelist. Empty = unknown, and controller.py then
    # falls back to recovering it from the per-task ICL prompt identity and, if
    # that fails too, to the FULL field set — an unknown task must never silently
    # lose its `answer`.
    task: str = ""
    # Log, every tick, what an UNRESTRICTED argmax would have produced at the
    # boolean slot. If it is not a boolean at all, the logit read is imposing
    # structure the model did not intend — we must know before trusting this path.
    verify_logit_read: bool = True
    # There is no free-form `note` field: it duplicated `seen` + `question_for_next`
    # and was essentially never filled (0/199,909 ticks).
    # MEMORY — the writer's own trace, fed back into the prompt:
    # WHAT I SAW (`seen`, consecutive duplicates collapsed) + WHAT I SAID
    # (`reported`, code-owned so the model cannot fake having answered).
    # Bounded so per-tick prompt cost is O(1), not O(stream length) — a memory that
    # grows without bound would make the system slower the longer it watches.
    seen_trace_ring: int = 10           # how many past observations to show

    # In-context "control language" (VISPROG-style): a compact JSON DSL the frozen
    # model emits each tick to drive its own probing. Taught via worked
    # (Situation -> Control) pairs that demonstrate the DECISIONS (stay quiet /
    # defer-with-question / fire-once / suppress-repeat), not just the syntax.
    # This GENERIC block is only the fallback: `task_controller_prompts` overrides
    # it for every one of the 9 OmniPro tasks. Text lives in prompts.py.
    controller_prompt: str = CONTROLLER_PROMPT_GENERIC

    # These drive the control-JSON generation.
    # SEEDED SAMPLING by default (writer_greedy=False), using Qwen's own
    # recommended settings (generation_config.json: do_sample=true, T=0.7,
    # top_p=0.8, top_k=20). Greedy (writer_greedy=True) overrides every sampling
    # field below.
    # Greedy is a prime suspect for the FROZEN-PERCEPTION bug: between ticks the
    # context changes by one frame (~185 of ~100k tokens), which barely moves the
    # logits, so argmax returns a byte-identical string by construction. Measured:
    # 578 consecutive ticks, one string, 100%.
    # Determinism does not require greedy: the generator is seeded (writer_seed)
    # and the walk is lockstep, so seeded sampling is equally bit-reproducible.
    writer_greedy: bool = False
    writer_seed: int = 3407
    writer_temperature: float = 0.7
    writer_top_p: float = 0.8
    writer_top_k: int = 20
    writer_repetition_penalty: float = 1.0
    writer_presence_penalty: float = 1.5

    
    # In eval the adapter sets `instruction` = the sample's task; for standalone runs the default below is used.
    instruction: str = "report the target event the instant it happens, and stay quiet otherwise"
    event: str = ""                   # the monitored condition (adapter: sample.event)
    video_id: str = ""                # set per-sample in eval; shown in controller logs

    # ---- PROBE-GATE system (gate_mode="probe"): the main-branch baseline --------
    # Fixed-cadence yes/no LOGIT gate + separate writer, restored for the
    # probe-vs-controller head-to-head). One forward pass per
    # probe reads yes_share = P(yes)/(P(yes)+P(no)); a Schmitt/hysteresis gate
    # (tuned "hyst2b") fires the writer, which snapshots the cache and answers.
    gate_mode: str = "controller"     # "controller" (icl DSL, default) | "probe"
    # Prompt TEXT for these lives in prompts.py; {event} / {instruction} are filled
    # at runtime by the ingester/writer.
    goal_question: str = GOAL_QUESTION
    writer_prompt: str = WRITER_PROMPT
    writer_cue: str = WRITER_CUE
    system_prompt: str = SYSTEM_PROMPT
    writer_max_tokens: int = 60
    writer_repeat_window: int = 8     # anti-loop guard for the free-running writer
    goal_threshold: float = 0.5       # non-hysteresis fallback threshold
    gate_hysteresis: bool = True      # hyst2b (best tuned gate on the 27-video eval)
    gate_high_thr: float = 0.5        # fire on a rising crossing (while armed)
    gate_low_thr: float = 0.40        # re-arm when the share falls below this...
    gate_rearm_s: float = 5.0         # ...OR this many seconds after a fire
    goal_gate_every: int = 1          # probe every frame (@1fps ~= controller's grid)
    debounce_s: float = 2.0           # min video-seconds between fires
    yes_words: list = field(default_factory=lambda: ["yes", "Yes", " yes", " Yes"])
    no_words: list = field(default_factory=lambda: ["no", "No", " no", " No"])

    # Per-task ICL prompts (override `controller_prompt` when sample.task matches;
    # the adapter selects by task). ALL 9 OmniPro tasks are covered. Text lives in
    # prompts.py. Counting / state-monitoring tasks additionally
    # carry `count` / `phase` (see plan_count / plan_phase).
    # Copied per instance so a caller mutating one config cannot corrupt the shared
    # module-level dict.
    task_controller_prompts: dict = field(
        default_factory=lambda: dict(TASK_CONTROLLER_PROMPTS))
    # Per-task writer prompts for the PROBE-GATE arm (same idea: the answer format
    # is task knowledge both systems get; the architecture is what differs).
    task_writer_prompts: dict = field(
        default_factory=lambda: dict(TASK_WRITER_PROMPTS))

