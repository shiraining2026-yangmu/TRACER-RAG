import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflight_wrong_only_memory_v3 import bounded_stress_package, full_stress_package, stress_adapter, STRESS_POLICY
import run_three_model_wrong_only as original
import run_three_model_wrong_only_memory_v2 as memory_v2
import run_three_model_wrong_only_memory_v3 as queue


class ReachablePreflightTests(unittest.TestCase):
    def test_stress_fits_budget_and_rejects_impossible_schema(self):
        count = lambda value: len(value)
        package, tokens, size = bounded_stress_package(count, 640)
        self.assertLessEqual(tokens, 640)
        self.assertGreater(size, 0)
        self.assertEqual(size, len(package["answer_guidance"]))
        with self.assertRaises(ValueError):
            bounded_stress_package(count, 1)

    def test_adapter_only_replaces_exact_synthetic_package(self):
        bounded = {"stress": "bounded"}
        calls = []
        render = stress_adapter(lambda q, e, a, p: p, bounded, calls)
        real = {"answer_guidance": "实际评语"}
        self.assertIs(real, render("q", "e", "y0", real))
        self.assertIs(bounded, render("q", "e", "y0", full_stress_package()))
        self.assertEqual([True], calls)

    def test_bindings_freeze_new_contract_and_restore_older_versions(self):
        saved = original.PROTOCOL, original.CODE, original.commit_manifest, memory_v2.PROTOCOL
        with queue.version_bindings():
            self.assertEqual(queue.PROTOCOL, original.PROTOCOL)
            self.assertIn("preflight_wrong_only_memory_v3.py", original.CODE)
            self.assertIs(queue.summarize, original.summarize)
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                original.commit_manifest(work, {"protocol": queue.PROTOCOL})
                manifest = json.loads((work / "manifest.json").read_text())
                self.assertEqual(STRESS_POLICY, manifest["preflight_stress_policy"])
                self.assertEqual("release_after_request", manifest["cuda_memory_policy"])
        self.assertEqual(saved, (original.PROTOCOL, original.CODE, original.commit_manifest, memory_v2.PROTOCOL))

    def test_preflight_and_merge_use_v3_but_formal_runner_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = queue.Supervisor(Path(directory), Path(directory) / "output")
            for old, expected in (("preflight_wrong_only.py", "preflight_wrong_only_memory_v3.py"),
                                  ("summarize_ter_wrong_only.py", "summarize_ter_wrong_only_memory_v3.py"),
                                  ("run_evidence_refinement_wrong_questions.py", "run_evidence_refinement_wrong_questions.py")):
                with patch.object(original.Supervisor, "run") as run:
                    supervisor.run(old, "--flag")
                self.assertEqual((expected, "--flag"), run.call_args.args)


if __name__ == "__main__":
    unittest.main()
