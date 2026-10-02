"""Pure event-identity helpers; never use onset IDs as delivery timestamps."""
import math


def event_key(value):
    """Legacy ev0 precision, with invalid identities explicitly rejected."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(value, 3) if math.isfinite(value) and value >= 0 else None


def already_reported(value, reported, tolerance_s=0.0):
    """Exact ev0 by default; a bounded tolerance is an explicit experiment."""
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("Event identity tolerance must be finite and nonnegative")
    key = event_key(value)
    if key is None or reported is None:
        return False
    if tolerance_s == 0:
        return key in reported
    return any(abs(key - old) <= tolerance_s + 1e-9
               for item in reported if (old := event_key(item)) is not None)


def identity_prompt(reported, limit=16):
    """Bounded, code-owned onset ledger; entries exist only after actual fire."""
    keys = sorted({key for item in reported if (key := event_key(item)) is not None})
    values = ", ".join(f"{key:g}" for key in keys[-limit:]) or "(none yet)"
    return ("\nALREADY REPORTED ONSET IDS (your previous event_time_s values, not delivery times): "
            + values + ". These occurrences have already been reported. For the SAME occurrence, "
            "reuse its original onset value; changing its time does not make it a new event.\n")