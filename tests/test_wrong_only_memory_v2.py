import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_integrity import atomic_jsonl, sha256
from reuse_wrong_only_baselines import reuse_completed_baselines
from serve_benchmark_memory_control import with_memory_policy
import run_three_model_wrong_only as original
import run_three_model_wrong_only_memory_v2 as queue


class MemoryTests(unittest.TestCase):
    def test_release_runs_after_success_and_failure_without_changing_outputs(self):
        events = []
        expected = ["literal result"]
        def render(requests):
            events.append(requests)
            return expected
        wrapped = with_memory_policy(render, "release_after_request", lambda: events.append("release"))
        self.assertIs(expected, wrapped(["prompt"]))
        self.assertEqual([["prompt"], "release"], events)
        def fail(requests):
            raise RuntimeError("OOM")
        with self.assertRaises(RuntimeError):
            with_memory_policy(fail, "release_after_request", lambda: events.append("release_on_error"))([])
        self.assertEqual("release_on_error", events[-1])

    def test_version_bindings_restore_frozen_original_and_keep_two_remaining_models(self):
        saved = (original.PROTOCOL, original.CODE, original.MODELS)
        with queue.version_bindings():
            self.assertEqual(queue.PROTOCOL, original.PROTOCOL)
            self.assertEqual(["gemma2_9b_it", "yi15_9b_chat"], [m[0] for m in original.MODELS])
            self.assertIn("serve_benchmark_memory_control.py", original.CODE)
            self.assertIs(original.reuse_completed_baselines, reuse_completed_baselines)
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                original.commit_manifest(work, {"protocol": queue.PROTOCOL})
                self.assertEqual("release_after_request", json.loads((work / "manifest.json").read_text())["cuda_memory_policy"])
        self.assertEqual(saved, (original.PROTOCOL, original.CODE, original.MODELS))

    def test_both_services_release_cache_and_keep_original_model_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = queue.Supervisor(Path(directory), Path(directory) / "output")
            for script, kind in (("serve_single_generator.py", "generator"), ("serve_qwen_critic_adapter.py", "critic")):
                with patch.object(original.Supervisor, "serve") as serve:
                    supervisor.serve(script, ["--model", "frozen"], "base", "id", "log")
                args = serve.call_args.args
                self.assertEqual("serve_benchmark_memory_control.py", args[0])
                self.assertEqual(["--service-kind", kind, "--memory-policy", "release_after_request", "--model", "frozen"], args[1])

    def test_migration_preserves_32_token_baselines_and_excludes_old_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            source = app / "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_v1/gemma2_9b_it"
            target = source.parent.parent / "ter_wrong_only_batch4_memory_v2/gemma2_9b_it"
            source.mkdir(parents=True)
            (app / "run_frozen_evidence_generation.py").write_text("frozen runner")
            manifest = {"protocol": "ter-wrong-only-batch4-v1", "generation": {"max_new_tokens": 32}, "answer_critic_enabled": False}
            (source / "manifest.json").write_text(json.dumps(manifest))
            (source / "model_source.json").write_text("{}")
            runtime = {"model_id": "candidate"}
            (source / "deployment_preflight.json").write_text(json.dumps({"runtime": runtime,
                "baseline_batch_size": 4, "max_new_tokens": 32, "exact_official_token_ids_verified": True}))
            atomic_jsonl(source / "evicr_v3/2022/evidence_refinement_results.jsonl", [{"old": "not reusable"}])
            for arm, folder in (("dense", "dense_rag"), ("ter", "ter_direct")):
                for year, count in (("2022", 596), ("2021", 566)):
                    rows = [{"application_id": f"{year}-{i}", "question": f"q{i}", "answer": "答案：A",
                             "retrieved": [], "evidence": [], "status": "ok", "method": arm} for i in range(count)]
                    frozen = (app / f"questionbank_eval/dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                              if arm == "dense" else app / f"questionbank_eval/mandatory_choice_v3_{year}/alignrag_results.jsonl")
                    atomic_jsonl(frozen, rows)
                    prediction = source / folder / year / f"{folder}_results.jsonl"
                    atomic_jsonl(prediction, rows)
                    prediction.with_name("generation_contract.json").write_text(json.dumps({
                        "max_new_tokens": 32, "batch_size": 4, "invalid_choice_policy": "default_a",
                        "runtime": runtime, "runner_sha256": sha256(app / "run_frozen_evidence_generation.py"),
                        "evidence_input_sha256": sha256(frozen)}))
            audit = reuse_completed_baselines(source, target, app)
            for relative, digest in audit["files"].items():
                self.assertEqual(digest, sha256(target / relative))
            self.assertFalse((target / "evicr_v3").exists())
            self.assertFalse((target / "answer_refinement").exists())
            manifest["generation"]["max_new_tokens"] = 128
            (source / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                reuse_completed_baselines(source, target, app)

    def test_final_combination_references_completed_Mistral_with_original_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            supervisor = queue.Supervisor(app, app / "output")
            supervisor.output.mkdir()
            old = supervisor.baseline_source / "mistral_7b_instruct_v03"
            old.mkdir(parents=True)
            (old / "summary.json").write_text(json.dumps({"arms": {"evicr": {"correct": 521}}}))
            (old / "manifest.json").write_text("{}")
            def remaining():
                for slug, _, _ in original.MODELS:
                    target = supervisor.output / slug
                    target.mkdir()
                    (target / "summary.json").write_text(json.dumps({"arms": {"ter": {"correct": 1}}}))
            with queue.version_bindings(), patch.object(queue.subprocess, "run"), \
                 patch.object(original.Supervisor, "run_queue", side_effect=remaining):
                supervisor.run_queue()
            combined = json.loads((supervisor.output / "all_models_summary.json").read_text())
            self.assertEqual(3, len(combined))
            self.assertEqual(521, combined["mistral_7b_instruct_v03"]["evicr"]["correct"])
            reference = json.loads((supervisor.output / "completed_model_references.json").read_text())
            self.assertFalse(reference["mistral_7b_instruct_v03"]["predictions_relabelled"])


if __name__ == "__main__":
    unittest.main()
