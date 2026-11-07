"""Offline checks for the public evidence-only protocol and paired inference."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_evicr as runner
from examples.offline_demo import package, dump
from export_evicr_qd_sft import system_message, user_message
from generate_evicr_qd_labels import messages, validate_package
from paired_statistics import analyze, exact_mcnemar
from run_test111_ter_structure_ablation import bootstrap_difference


class EvidenceOnlyTests(unittest.TestCase):
    def test_identical_teacher_student_and_online_prompt(self):
        q, d = "symbol-question", "[E1] symbolic evidence"
        expected = messages(q, d)
        self.assertEqual(expected, runner.critic_messages(q, d))
        self.assertEqual(expected[0]["content"], system_message())
        self.assertEqual(expected[1]["content"], user_message(q, d))

    def test_no_ks_changes_only_supplements_and_does_not_mutate(self):
        p = package()
        p["knowledge_supplements"] = [{"claim": "SYNTHETIC-SUPPLEMENT", "basis": "model_knowledge", "confidence": "low"}]
        original = copy.deepcopy(p)
        ablated = runner.delivered_package(p, True)
        self.assertEqual(p, original)
        self.assertEqual(ablated, {**p, "knowledge_supplements": []})

    def test_schema_rejects_missing_evidence_and_extra_answer(self):
        with self.assertRaises(ValueError):
            validate_package(package(), 2)
        with self.assertRaises(ValueError):
            validate_package({**package(), "y0": "bad"}, 1)

    def test_full_and_no_ks_runs_are_paired_without_answers_in_critic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"e_id": "E-HISTORY", "application_id": "A-HISTORY", "relation": "explanatory_support",
                       "application_question": "History Q", "application_answer": "Historical evidence", "knowledge_text": "Symbol A -> B"}
            frozen = [{"application_id": f"TEST-{i}", "question": f"Symbol question {i}", "split": "test", "retrieved": [{"payload": payload}]} for i in range(2)]
            initial = [{"application_id": r["application_id"], "question": r["question"], "answer": f"CURRENT-Y0-{i}"} for i, r in enumerate(frozen)]
            dump(root / "test.jsonl", frozen)
            dump(root / "y0.jsonl", initial)
            p = package()
            p["knowledge_supplements"] = [{"claim": "SUPPLEMENT-MARKER", "basis": "model_knowledge", "confidence": "low"}]
            seen = []
            def fake_chat(base, model, prompt, *args):
                text = json.dumps(prompt)
                seen.append((model, text))
                if model == "evicr-critic":
                    self.assertNotIn("CURRENT-Y0-", text)
                    return json.dumps(p)
                self.assertIn("CURRENT-Y0-", text)
                return "Synthetic refined answer"
            common = ["run_evicr.py", "--frozen-test", str(root / "test.jsonl"), "--ter-y0", str(root / "y0.jsonl")]
            with patch.object(runner, "call_chat", side_effect=fake_chat), patch.object(sys, "argv", common + ["--output-dir", str(root / "full")]):
                runner.main()
            self.assertEqual(len(seen), 4)
            seen.clear()
            with patch.object(runner, "call_chat", side_effect=fake_chat), patch.object(sys, "argv", common + ["--output-dir", str(root / "no_ks"), "--no-knowledge-supplements", "--reuse-critic", str(root / "full/critic_qd.jsonl")]):
                runner.main()
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(model == "generator" for model, _ in seen))
            self.assertTrue(all("SUPPLEMENT-MARKER" not in text for _, text in seen))
            # Frozen evaluation evidence may not carry the current answer/key.
            dump(root / "bad.jsonl", [{**r, "correct_answer": "SECRET-KEY"} for r in frozen])
            with self.assertRaises(ValueError):
                runner.load_inputs(root / "bad.jsonl", root / "y0.jsonl", 850)
            dump(root / "duplicate.jsonl", [frozen[0], frozen[0]])
            with self.assertRaises(ValueError):
                runner.load_inputs(root / "duplicate.jsonl", root / "y0.jsonl", 850)

    def test_synthetic_sft_and_inference_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "examples/offline_demo.py"), "--output-dir", tmp], check=True, capture_output=True)
            for p in (Path(tmp) / "sft").glob("critic_sft_*.jsonl"):
                if "rejected" in p.name:
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    text = json.dumps(row["messages"][:2])
                    self.assertNotIn("y_weak", text)
                    self.assertNotIn("y0", text)


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_identical_vectors(self):
        vector = [0, 1, 1, 0, 1]
        self.assertEqual(bootstrap_difference(vector, vector, random.Random(2), 500), [0.0, 0.0])

    def test_exact_mcnemar_and_cluster_resampling(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(5, 0), .0625)
        rows = [{"id": str(i), "year": "2021", "cluster_id": "shared",
                 "before_correct": before, "after_correct": after}
                for i, (before, after) in enumerate([(0, 1), (1, 0)])]
        result = analyze(rows, 500)
        self.assertEqual(result["paired_cluster_bootstrap_ci95_percentage_points"], [0, 0])
        self.assertEqual(result["singleton_mcnemar_n"], 0)
        self.assertEqual(result["paired_2x2"]["wrong_to_correct"], 1)


if __name__ == "__main__":
    unittest.main()
