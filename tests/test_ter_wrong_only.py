import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_integrity import atomic_jsonl, read_rows, sha256
import run_three_model_wrong_only as queue
import run_evidence_refinement_wrong_questions as evidence
from summarize_ter_wrong_only import merge_year, PROTOCOL, summarize


class WrongOnlyTests(unittest.TestCase):
    def test_evidence_batch4_routes_only_TER_errors_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [{"application_id": str(i), "question": f"题{i} A.甲 B.乙 C.丙 D.丁 E.戊",
                        "answer": "答案：A", "initial_answer": "答案：A",
                        "retrieved": [{"payload": {"e_id": "edge", "application_id": "historical", "knowledge_text": "资料"}}]}
                       for i in range(7)]
            atomic_jsonl(root / "ter_direct/2022/ter_direct_results.jsonl", sources)
            grade = root / "grading/ter_2022/grading.csv"
            grade.parent.mkdir(parents=True)
            with grade.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["application_id", "reference_status", "reference_answer", "score"])
                writer.writeheader()
                writer.writerows({"application_id": str(i), "reference_status": "valid",
                                  "reference_answer": "PRIVATE_KEY_SENTINEL", "score": "1" if i == 0 else "0"}
                                 for i in range(7))
            (root / "runtime.json").write_text(json.dumps({"runtime": {"model_id": "candidate"}}))
            package = {"evidence_status": "partial", "evidence_actions": [], "knowledge_supplements": [],
                       "change_support": "strong", "answer_guidance": "采用相关证据"}
            calls = []

            def factory(base, model, budget, *args):
                def invoke(prompts):
                    calls.append((model, budget, len(prompts)))
                    content = json.dumps(prompts, ensure_ascii=False)
                    self.assertNotIn("PRIVATE_KEY_SENTINEL", content)
                    self.assertNotIn("题0 A.", content)
                    if model == "critic":
                        self.assertNotIn("原始初稿", content)
                    return [json.dumps(package, ensure_ascii=False) if model == "critic" else "答案：B" for _ in prompts]
                return invoke

            argv = ["runner", "--source-results", str(root / "ter_direct/2022/ter_direct_results.jsonl"),
                    "--grading-csv", str(grade), "--output-dir", str(root / "evicr_v3/2022"),
                    "--generator-model", "candidate", "--critic-model", "critic", "--batch-size", "4",
                    "--generator-max-tokens", "32", "--critic-max-tokens", "640",
                    "--runtime-contract", str(root / "runtime.json"), "--resume"]
            with patch.object(sys, "argv", argv), patch.object(evidence, "batch_client", factory):
                evidence.main()
                evidence.main()
            self.assertEqual([("critic", 640, 4), ("candidate", 32, 4),
                              ("critic", 640, 2), ("candidate", 32, 2)], calls)
            merge_year(root, "2022")
            merged = read_rows(root / "evicr_v3/2022/evicr_v3_merged_results.jsonl")
            self.assertEqual(sources[0], merged[0])
            self.assertTrue(all(row["answer"] == "答案：B" for row in merged[1:]))
            self.assertFalse((root / "answer_refinement").exists())
            argv[argv.index("--critic-max-tokens") + 1] = "384"
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                evidence.main()

    def test_supervisor_has_no_answer_Critic_stage_and_uses_TER_grading(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "app"
            app.mkdir()
            supervisor = queue.Supervisor(app, app / "output")
            slug = "mistral_7b_instruct_v03"
            work = supervisor.output / slug
            calls = []
            for name in queue.CODE:
                (app / name).write_text("frozen runner")
            adapter = supervisor.adapter / "adapter_model.safetensors"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"adapter")
            runtime = {"model_id": "candidate"}
            report = {"runtime": runtime, "baseline_batch_size": 4, "refinement_batch_size": 4}

            def run(script, *args):
                calls.append((script, list(map(str, args))))
                if script == "download_benchmark_model.py":
                    (work / "model_source.json").write_text("{}")
                if script == "preflight_generator_deployment.py":
                    path = Path(args[args.index("--output") + 1])
                    path.write_text(json.dumps(report))

            with patch.object(queue, "reuse_completed_baselines"), patch.object(queue, "commit_manifest"), \
                 patch.object(queue, "sha256", return_value="hash"), \
                 patch.object(supervisor, "run", side_effect=run), patch.object(supervisor, "serve"), \
                 patch.object(supervisor, "grade"), patch.object(supervisor, "stop_servers"), \
                 patch.object(queue, "summarize"), patch.object(supervisor, "cleanup_weights"):
                supervisor.model(slug, "repo", None)
            self.assertNotIn("run_qwen35_answer_refinement.py", [name for name, _ in calls])
            v3 = [args for name, args in calls if name == "run_evidence_refinement_wrong_questions.py"]
            self.assertEqual(2, len(v3))
            for args, year in zip(v3, ("2022", "2021")):
                self.assertIn(f"ter_direct/{year}/ter_direct_results.jsonl", args[args.index("--source-results") + 1].replace("\\", "/"))
                self.assertIn(f"ter_{year}/grading.csv", args[args.index("--grading-csv") + 1].replace("\\", "/"))
                self.assertEqual("4", args[args.index("--batch-size") + 1])
                self.assertEqual("640", args[args.index("--critic-max-tokens") + 1])

    def test_queue_retries_current_model_then_hands_off(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = queue.Supervisor(root, root / "output")
            events = []

            def model(slug, repo, mirror):
                events.append(slug)
                supervisor.work = supervisor.output / slug
                supervisor.work.mkdir(parents=True, exist_ok=True)
                if len(events) == 1:
                    raise RuntimeError("temporary failure")
                (supervisor.work / "summary.json").write_text(json.dumps({"arms": {"ter": {"correct": 1}}}))

            with patch.object(queue, "MODELS", (("first", "repo1", None), ("second", "repo2", None))), \
                 patch.object(supervisor, "model", model), patch.object(supervisor, "stop_servers"), \
                 patch.object(queue.time, "sleep"):
                supervisor.run_queue()
            self.assertEqual(["first", "first", "second"], events)
            state = json.loads((supervisor.output / "state.json").read_text())
            self.assertEqual(("complete", PROTOCOL), (state["status"], state["protocol"]))

    def test_import_manifest_cannot_adopt_changed_baseline_or_mixed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "ter_direct/2022/ter_direct_results.jsonl"
            atomic_jsonl(source, [])
            audit = {"answer_refinement_reused": False, "evicr_reused": False,
                     "files": {str(source.relative_to(work)): sha256(source)}}
            (work / "baseline_reuse.json").write_text(json.dumps(audit))
            manifest = {"protocol": PROTOCOL, "answer_critic_enabled": False}
            queue.commit_manifest(work, manifest)
            queue.commit_manifest(work, manifest)
            with self.assertRaises(ValueError):
                queue.commit_manifest(work, {**manifest, "answer_critic_enabled": True})
            (work / "manifest.json").unlink()
            source.write_text("changed bytes")
            with self.assertRaises(ValueError):
                queue.commit_manifest(work, manifest)

    def test_summary_rescores_1138_and_rejects_wrong_set_or_token_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            bank = app / "questionbank_eval"
            work = bank / "model_benchmarks/ter_wrong_only_batch4_v1/test_model"
            work.mkdir(parents=True)
            for name in ("summarize_multimodel_benchmark.py", "benchmark_integrity.py"):
                shutil.copy2(Path(__file__).resolve().parents[1] / name, app / name)
            for name in ("run_frozen_evidence_generation.py", "run_evidence_refinement_wrong_questions.py"):
                (app / name).write_text("frozen runner")
            adapter = app / "adapter/adapter_model.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"critic")
            (work / "model_source.json").write_text("{}")
            report = {"runtime": {"model_id": "candidate"}, "max_new_tokens": 32,
                      "exact_official_token_ids_verified": True, "baseline_batch_size": 4,
                      "refinement_batch_size": 4, "answer_critic_probe_executed": False,
                      "evidence_critic_batch4": {"executable": True},
                      "candidate_refinement_stress_batch4_executable": True,
                      "evidence_critic_max_tokens": 640}
            for name in ("deployment_preflight.json", "refinement_deployment_preflight.json"):
                (work / name).write_text(json.dumps(report))

            def write_csv(path, rows):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            for year, total, scored in (("2021", 566, 554), ("2022", 596, 584)):
                rows = [{"application_id": f"{year}-{i}", "question": f"q{i}", "answer": "答案：A",
                         "initial_answer": "答案：A", "retrieved": [], "evidence": []} for i in range(total)]
                atomic_jsonl(bank / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl", rows)
                atomic_jsonl(bank / f"dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl", rows)
                atomic_jsonl(bank / f"mandatory_choice_v3_{year}/alignrag_results.jsonl", rows)
                refs = [{"application_id": row["application_id"], "reference_status": "valid",
                         "reference_answer": "B" if i == 0 else "A"} for i, row in enumerate(rows[:scored])]
                write_csv(bank / f"model_benchmarks/_reference/reference_{year}.csv", refs)
                update = {**rows[0], "answer": "答案：B", "critic_status": "valid",
                          "config": {"batch_size": 4, "generator_max_tokens": 32,
                                     "prompt_version": "evidence-refinement-questionbank-v3"}}
                atomic_jsonl(work / f"evicr_v3/{year}/evidence_refinement_results.jsonl", [update])
                for arm, folder, filename, outputs in (
                    ("dense", "dense_rag", "dense_rag_results.jsonl", rows),
                    ("ter", "ter_direct", "ter_direct_results.jsonl", rows),
                    ("evicr", "evicr_v3", "evicr_v3_merged_results.jsonl", [update] + rows[1:])):
                    atomic_jsonl(work / folder / year / filename, outputs)
                    grade = [{**ref, "score": str(int(outputs[i]["answer"][-1] == ref["reference_answer"]))}
                             for i, ref in enumerate(refs)]
                    write_csv(work / f"grading/{arm}_{year}/grading.csv", grade)
                    (work / f"grading/{arm}_{year}/grading_summary.json").write_text(json.dumps({
                        "correct": sum(int(row["score"]) for row in grade), "scored_questions": scored}))
                    runner = "run_evidence_refinement_wrong_questions.py" if arm == "evicr" else "run_frozen_evidence_generation.py"
                    contract = {"runtime": report["runtime"], "batch_size": 4, "max_new_tokens": 32,
                                "invalid_choice_policy": "default_a", "runner_sha256": sha256(app / runner)}
                    if arm == "evicr":
                        contract.update({"critic_max_tokens": 640, "critic_repair_attempts": 1,
                            "evidence_field_limit": 850, "limit": None,
                            "source_sha256": sha256(work / f"ter_direct/{year}/ter_direct_results.jsonl"),
                            "grading_sha256": sha256(work / f"grading/ter_{year}/grading.csv")})
                    (work / folder / year / "generation_contract.json").write_text(json.dumps(contract))
            manifest = {"protocol": PROTOCOL, "answer_critic_enabled": False,
                "generation": {"max_new_tokens": 32}, "evidence_critic_max_tokens": 640,
                "baseline_batch_size": 4, "refinement_batch_size": 4, "invalid_choice_policy": "default_a",
                "code_sha256": {name: sha256(app / name) for name in
                                 ("run_frozen_evidence_generation.py", "run_evidence_refinement_wrong_questions.py")},
                "input_sha256": {}, "model_source_sha256": sha256(work / "model_source.json"),
                "critic_adapter": str(adapter.parent), "critic_adapter_sha256": sha256(adapter),
                "pipeline": "TER mistakes -> evidence Critic -> reanswer"}
            (work / "manifest.json").write_text(json.dumps(manifest))
            result = summarize(work, "test_model")
            self.assertEqual((1138, 1136), (result["arms"]["evicr"]["correct"], result["arms"]["ter"]["correct"]))
            atomic_jsonl(work / "evicr_v3/2022/evidence_refinement_results.jsonl", [])
            with self.assertRaises(ValueError):
                summarize(work, "test_model")
            contract_path = work / "evicr_v3/2022/generation_contract.json"
            contract = json.loads(contract_path.read_text())
            contract["critic_max_tokens"] = 384
            contract_path.write_text(json.dumps(contract))
            with self.assertRaises(ValueError):
                summarize(work, "test_model")


if __name__ == "__main__":
    unittest.main()
