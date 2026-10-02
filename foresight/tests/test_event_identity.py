"""CPU tests of the ACTUAL schema walk and fire/commit blocks, no model load."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import time
from unittest.mock import patch

import torch

import controller
from config import AsyncOmniConfig
from event_identity import already_reported, event_key, identity_prompt
from prompts import TASK_CONTROLLER_PROMPTS
from trigger_prompts import trigger_prompt


class IdentityHelpers(unittest.TestCase):
    def test_equivalent_numeric_forms(self):
        for value in (42, 42.0, "42", "42.000"):
            with self.subTest(value=value):
                self.assertEqual(event_key(value), 42.0)
                self.assertTrue(already_reported(value, {42.0}))

    def test_invalid_values_are_not_identities(self):
        for value in (None, "", "n/a", -1, float("nan"), float("inf"), "-inf", True):
            with self.subTest(value=value):
                self.assertIsNone(event_key(value))
                self.assertFalse(already_reported(value, {0.0, 1.0}))

    def test_zero_is_a_valid_onset(self):
        self.assertEqual(event_key(0), 0)
        self.assertTrue(already_reported(0, {0}))

    def test_exact_rule_does_not_suppress_nearby_onset(self):
        self.assertFalse(already_reported(42.01, {42.0}))
        self.assertTrue(already_reported(42.0001, {42.0}))

    def test_explicit_tolerance_boundaries(self):
        self.assertTrue(already_reported(43, {42}, 1))
        self.assertFalse(already_reported(43.001, {42}, 1))
        for tolerance in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                already_reported(42, {42}, tolerance)

    def test_memory_is_sample_local_and_read_only(self):
        previous = {42.0}
        self.assertTrue(already_reported(42, previous))
        self.assertFalse(already_reported(42, set()))
        self.assertEqual(previous, {42.0})

    def test_identity_ledger_is_bounded(self):
        text = identity_prompt(set(range(40)), limit=2)
        self.assertIn("38, 39.", text)
        self.assertNotIn("37,", text)
        self.assertIn("not delivery times", text)


class SchemaDedup(unittest.TestCase):
    def walk(self, *, enabled=True, seen=None, event="42", hit=True, question=False):
        cfg = AsyncOmniConfig(seen_mode="off", ev0_dedup=enabled)
        logits = torch.tensor([5.0, 0.0] if hit else [0.0, 5.0])
        backend = SimpleNamespace(embed_text=lambda text: text,
                                  tok=SimpleNamespace(decode=lambda ids: "true" if ids[0] == 0 else "false"))
        forced, progress = [], []

        def step(text):
            forced.append(text)
            return logits

        def decode(*args):
            key = forced[-1]
            text = event if '"event_time_s"' in key else (
                "Check the next occurrence" if '"question_for_next"' in key else "Decoded answer")
            return text, [99], logits

        fields = {"have_enough_info", "event_time_s", "answer"}
        if question:
            fields.add("question_for_next")
        with patch.object(controller, "_decode_until", side_effect=decode) as mock:
            diff, meta = controller._schema_tick(
                backend, cfg, None, step, "prompt", (0, 1), fields=fields,
                ev_seen=seen, progress=progress.append)
        return diff, meta, forced, progress, mock.call_count

    def test_duplicate_keeps_level_but_skips_answer_decode(self):
        diff, meta, forced, progress, n = self.walk(seen={42.0})
        self.assertTrue(diff["have_enough_info"])
        self.assertTrue(meta["dup_ev"])
        self.assertNotIn("answer", diff)
        self.assertFalse(any('"answer"' in s for s in forced))
        self.assertEqual(n, 1)
        self.assertEqual(progress, ["bool_ready", "event_id_ready"])

    def test_duplicate_does_not_skip_planning(self):
        diff, meta, _, _, n = self.walk(seen={42.0}, question=True)
        self.assertTrue(meta["dup_ev"])
        self.assertEqual(diff["question_for_next"], "Check the next occurrence")
        self.assertEqual(n, 2)

    def test_disabled_ev0_really_decodes_duplicates(self):
        diff, meta, _, progress, n = self.walk(enabled=False, seen={42.0})
        self.assertFalse(meta["dup_ev"])
        self.assertEqual(diff["answer"], "Decoded answer")
        self.assertEqual(n, 2)
        self.assertEqual(progress, ["bool_ready", "event_id_ready", "answer_ready"])

    def test_new_onset_is_not_a_duplicate(self):
        diff, meta, _, _, _ = self.walk(seen={41.0})
        self.assertFalse(meta["dup_ev"])
        self.assertIn("answer", diff)

    def test_schema_never_commits_an_unemitted_onset(self):
        seen = set()
        self.walk(seen=seen)
        self.assertEqual(seen, set())

    def test_invalid_onset_does_not_suppress_or_poison_json(self):
        for event in ("bad", "nan", "inf", "-1"):
            with self.subTest(event=event):
                diff, meta, _, _, _ = self.walk(seen={42.0}, event=event)
                self.assertFalse(meta["dup_ev"])
                self.assertFalse(meta["event_time_valid"])
                self.assertNotIn("event_time_s", diff)
                self.assertIn("answer", diff)

    def test_negative_boolean_decodes_nothing(self):
        diff, _, _, progress, n = self.walk(hit=False)
        self.assertFalse(diff["have_enough_info"])
        self.assertEqual(n, 0)
        self.assertEqual(progress, ["bool_ready"])


class ActualFireBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(controller.__file__).read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "controller_thread")
        gate = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and ast.unparse(n.test).startswith("cfg.gate_strategy == 'hysteresis'"))
        commit = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "fire"
                      and "reported.append" in ast.unparse(n))
        cls.gate_code = compile(ast.Module(body=[gate], type_ignores=[]), "actual_gate", "exec")
        cls.commit_code = compile(ast.Module(body=[commit], type_ignores=[]), "actual_commit", "exec")

    def test_duplicate_veto_in_every_gate(self):
        for strategy in ("level", "edge", "strict_edge", "hysteresis", "ema_edge"):
            for dup in (False, True):
                env = dict(cfg=AsyncOmniConfig(gate_strategy=strategy, debounce_s=0),
                           p_hit=0.9, level=True, armed=True, rising=True,
                           distinct=False, has_text=True, dup_ev=dup, vt=10, last_fire_vt=-1e9,
                           ema_value=None, ema_time=None, ema_armed=True, time=time, math=math)
                exec(self.gate_code, env)
                self.assertEqual(bool(env["fire"]), not dup, (strategy, dup))

    def test_only_real_fires_commit_identity_and_no_backdating(self):
        for fire in (False, True):
            emitted = []
            env = dict(fire=fire, vt=50.0, state={"event_time_s": 42.0}, emit_text="answer",
                       reported=[], reported_ev=set(), event_key=event_key, prof=None,
                       gen_s=1.0, vid="test", log=lambda *a: None,
                       evaluator=SimpleNamespace(record_trigger=lambda t, p: emitted.append(t),
                                                 record_write=lambda *a: None))
            exec(self.commit_code, env)
            self.assertEqual(env["reported_ev"], {42.0} if fire else set())
            self.assertEqual(emitted, [50.0] if fire else [])


class PromptContracts(unittest.TestCase):
    def test_original_variant_is_identical_for_all_tasks(self):
        for task, text in TASK_CONTROLLER_PROMPTS.items():
            self.assertEqual(trigger_prompt(text, "original"), text, task)

    def test_unreported_preserves_tasks_and_corrects_repeat_examples(self):
        import json
        for task, text in TASK_CONTROLLER_PROMPTS.items():
            with self.subTest(task=task):
                new = trigger_prompt(text, "unreported")
                self.assertIn("NOT already been reported", new)
                self.assertNotIn("YOU DO NOT DECIDE WHEN TO ALERT", new)
                self.assertNotIn("keep it true", new)
                seen = set()
                examples = [json.loads(s) for s in new.splitlines() if s.startswith('{"seen"')]
                self.assertGreaterEqual(len(examples), 4)
                for row in examples:
                    if row.get("have_enough_info") and "event_time_s" in row:
                        self.assertNotIn(row["event_time_s"], seen)
                        seen.add(row["event_time_s"])
                self.assertTrue(seen)
                for content in ("grid", "running total", "imperative", "NEW STATE"):
                    if content in text:
                        self.assertIn(content, new)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            trigger_prompt("x", "made_up")


if __name__ == "__main__":
    unittest.main(verbosity=2)