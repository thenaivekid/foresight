"""Opt-in, task-preserving prompt ablations. Never mutate the shipped prompts."""
import json
import re


_UNREPORTED_LEVEL = (
    "true only when a NEW occurrence required by the user's task is visible NOW, "
    "its answer is supported, and that occurrence has NOT already been reported. "
    "False for an ongoing or previously reported occurrence, even if it remains visible."
)

_GATE_CONTRACT = (
    "REPORT CONTRACT: a true decision is a request to report ONE new occurrence now. "
    "Read WHAT YOU HAVE ALREADY TOLD THE USER before deciding. After an occurrence "
    "has been reported, remain false for that same occurrence; rewording an answer "
    "or changing its timestamp does not make a new occurrence. A genuinely new "
    "occurrence may be reported immediately without an artificial quiet interval.\n"
)

_IDENTITY_GUIDANCE = (
    "\nEVENT ID CONSISTENCY: event_time_s is the original onset of the occurrence, "
    "not the current check time and not the time the answer was delivered. If you "
    "are still describing the SAME occurrence, reuse its SAME onset number. A later "
    "check or different wording is not a new event. Use a new onset only for visible "
    "evidence of a genuinely new occurrence. Never invent a future onset.\n"
)


def trigger_prompt(original, variant="original"):
    if variant == "original":
        return original
    if variant == "stable_identity":
        return original + _IDENTITY_GUIDANCE
    if variant != "unreported":
        raise ValueError(f"Unknown trigger prompt variant {variant!r}")

    # Replace both forms of the old level definition, preserving each task's
    # role, content format, counting/state rules, examples and planning fields.
    text = re.sub(r"(?m)^  have_enough_info\s*:[^\n]*\n",
                  "  have_enough_info  : " + _UNREPORTED_LEVEL + "\n", original)
    text = re.sub(r"What have_enough_info means here: .*?(?=HOW TO WRITE THE ANSWER)",
                  "What have_enough_info means here: " + _UNREPORTED_LEVEL + "\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"YOU DO NOT DECIDE WHEN TO ALERT[^\n]*\n", _GATE_CONTRACT, text)
    if _UNREPORTED_LEVEL not in text:
        raise ValueError("Prompt has no recognized decision definition")

    # Repair the worked examples as well as the prose: after a positive example
    # is reported, repeating that same onset must demonstrate false, not true.
    lines, seen = text.splitlines(keepends=True), set()
    for i, line in enumerate(lines):
        if not line.startswith('{"seen"'):
            continue
        row = json.loads(line)
        ev = row.get("event_time_s")
        if row.get("have_enough_info") and ev is not None:
            if ev in seen:
                row["have_enough_info"] = False
                row.pop("answer", None)
                row.pop("event_time_s", None)
                lines[i] = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                if i and lines[i - 1].startswith("At "):
                    prefix = re.match(r"At [\d.]+s", lines[i - 1]).group(0)
                    lines[i - 1] = (prefix + " the same occurrence has already been reported; "
                                    "stay quiet and preserve the task state:\n")
            seen.add(ev)
    return "".join(lines) + _IDENTITY_GUIDANCE