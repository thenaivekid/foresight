"""Small predefined live arms; no per-task threshold fitting or GT access."""
import dataclasses

ARMS = {
    "raw": ({}, "original"),
    "ev0": ({"ev0_dedup": True}, "original"),
    "cooldown5": ({"debounce_s": 5.0}, "original"),
    "strict_edge": ({"gate_strategy": "strict_edge"}, "original"),
    "short_ema": ({"gate_strategy": "ema_edge", "trigger_ema_tau_s": 0.5}, "original"),
    "ev0_cooldown5": ({"ev0_dedup": True, "debounce_s": 5.0}, "original"),
    "ev1": ({"ev0_dedup": True, "event_dedup_window_s": 1.0}, "original"),
    "stable_id_prompt": ({"ev0_dedup": True}, "stable_identity"),
    "id_memory": ({"ev0_dedup": True, "event_identity_in_prompt": True}, "original"),
    "unreported_prompt": ({"ev0_dedup": True}, "unreported"),
    "unreported_raw": ({}, "unreported"),
    "present_anchor": ({"ev0_dedup": True, "now_anchor": True}, "original"),
    # ---- KV-latch follow-up -------------------------------------------------
    # Observation: `p_hit` is read immediately after a 12-token `seen`
    # string sampled at T=0.7, and it tracks that string rather than the video.
    # At the first threshold crossing the `seen` word-overlap with the previous
    # tick is 0.13-0.16 against a 0.50-0.54 baseline, and 0/34 crossings had an
    # unchanged `seen`. These arms cut that channel (the seen_mode
    # experiment run live).
    "seen_after": ({"seen_mode": "after"}, "original"),
    "seen_off": ({"seen_mode": "off"}, "original"),
    # Combined candidate fix: read the level BEFORE describing (kills the
    # priming channel), define the level as an unreported occurrence (kills the
    # prompt-induced persistence), and fire only on the rising edge (the only
    # arm that separated from the rate-matched null).
    "fix": ({"seen_mode": "after", "gate_strategy": "strict_edge"}, "unreported"),
}

# Core arms, two independent async repetitions in separate debug jobs.
CORE_ARMS = ("raw", "ev0", "cooldown5", "strict_edge", "short_ema", "ev0_cooldown5", "ev1",
             "stable_id_prompt", "id_memory", "unreported_prompt")


def configure_arm(base, arm):
    from trigger_prompts import trigger_prompt
    changes, variant = ARMS[arm]
    prompts = {task: trigger_prompt(text, variant)
               for task, text in base.task_controller_prompts.items()}
    return dataclasses.replace(base, task_controller_prompts=prompts, **changes)