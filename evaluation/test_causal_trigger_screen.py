"""CPU-only tests for suppression-only, prefix-causal policy comparisons."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_trigger_screen import POLICIES, Policy, replay


def ticks(probs, times=None):
    times = list(range(len(probs))) if times is None else times
    return [{"arrival_s": float(t), "p_hit": p,
             "emission_index": i if p >= 0.5 else None}
            for i, (t, p) in enumerate(zip(times, probs))]


class TestCausalFilters(unittest.TestCase):
    def test_raw_keeps_every_existing_emission(self):
        self.assertEqual(replay(ticks([0.2, 0.5, 0.9, 0.1]), POLICIES[0]), [1, 2])

    def test_edge_one_per_high_episode(self):
        self.assertEqual(replay(ticks([0.1, 0.9, 0.8, 0.8, 0.1, 0.9]),
                                Policy("edge", "edge")), [1, 5])

    def test_confirm_does_not_use_future_tick(self):
        self.assertEqual(replay(ticks([0.1, 0.9, 0.8, 0.8]),
                                Policy("confirm", "confirm")), [2])

    def test_prefix_invariance_and_subset(self):
        tr = ticks([0.1, 0.9, 0.8, 0.2, 0.9, 0.99, 0.0, 0.9],
                   [0, 0.7, 3.2, 4.1, 5.9, 8.8, 10, 11.4])
        eligible = {t["emission_index"] for t in tr if t["emission_index"] is not None}
        for policy in POLICIES:
            complete = replay(tr, policy)
            self.assertTrue(set(complete) <= eligible)
            for n in range(len(tr) + 1):
                self.assertEqual(replay(tr[:n], policy), [i for i in complete if i < n], policy.name)

    def test_no_credit_for_current_probability_during_previous_gap(self):
        # One new high observation after a long quiet gap cannot accumulate
        # evidence over that preceding gap or overwrite its historical mean.
        tr = ticks([0.0, 1.0], [0, 10])
        for kind in ("mean", "ema", "integral"):
            p = Policy(kind, kind, window_s=2, tau_s=2)
            self.assertEqual(replay(tr, p), [])

    def test_constant_high_not_refired_without_rearm(self):
        for policy in POLICIES:
            if policy.kind not in ("level", "identity", "text_identity", "structured_change"):
                self.assertLessEqual(len(replay(ticks([0.9] * 12), policy)), 1)

    def test_identity_allows_new_onsets_on_a_high_plateau(self):
        tr = ticks([0.9] * 5)
        for t, event in zip(tr, [0, 0, 0, 3, 3]):
            t["event_time_s"] = event
        self.assertEqual(replay(tr, Policy("ev0", "identity")), [0, 3])

    def test_discrete_ewma_can_react_at_current_observation(self):
        tr = ticks([0.0, 1.0], [0, 10])
        self.assertEqual(replay(tr, Policy("point", "ema_point", tau_s=0.5)), [1])

    def test_bad_order_rejected(self):
        with self.assertRaises(ValueError):
            replay(ticks([0.1, 0.9], [2, 1]), POLICIES[0])

    def test_tolerant_identity_is_nontransitive_and_commit_only(self):
        tr = ticks([0.9] * 4)
        for t, ev in zip(tr, [0, 0.8, 1.6, 0]):
            t["event_time_s"] = ev
        p = Policy("ev1", "identity", identity_window_s=1)
        self.assertEqual(replay(tr, p), [0, 2])

    def test_state_return_counts_as_another_transition(self):
        tr = ticks([0.9] * 4)
        for t, value in zip(tr, ["A", "A", "B", "A"]):
            t["structured_key"] = value
        self.assertEqual(replay(tr, Policy("state", "structured_change")), [0, 2, 3])

    def test_text_dedup_does_not_turn_fixed_alerts_into_one_shot(self):
        tr = ticks([0.9] * 3)
        for t in tr:
            t.update(task="instant_event_alert", answer_text="fixed alert")
        self.assertEqual(replay(tr, Policy("text", "text_identity")), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()