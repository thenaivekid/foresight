"""
proactivity.py — the proactivity decision (AsyncReasoning, Sec. 3.2), isolated.

The whole "proactive" behaviour reduces to one operation: given the model's
next-token logits after a spliced yes/no question, how much does it lean "yes"?
We return the normalized share P(yes) / (P(yes) + P(no)).

Why a *share* and not raw P(yes): on these models both yes/no probabilities are
tiny in absolute terms (the natural continuation is usually some other token),
so an absolute threshold is meaningless. The relative share is the usable
signal. The orchestrator thresholds this share to decide whether to acquire
more input (vision gate) or to emit output (writer gate).
"""
import torch


def yes_share(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    p_yes = max(probs[i].item() for i in yes_ids)
    p_no = max(probs[i].item() for i in no_ids)
    return p_yes / (p_yes + p_no + 1e-9)
