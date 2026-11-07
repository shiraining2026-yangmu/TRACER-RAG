import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_integrity import atomic_jsonl, read_rows, sha256
import run_qwen35_answer_refinement as answer_stage
from run_tcm_alignrag import fallback_critique
from summarize_qwen35_replay import merge_y1_wrong, summarize, PROTOCOL
import run_three_model_qwen35_replay as queue
from reuse_qwen35_baselines import reuse_completed_baselines


def source(key="a"):
    return {"application_id": key, "question": "题目 A.甲 B.乙 C.丙 D.丁 E.戊",
            "answer": "答案：A", "raw_answer": "初始判断：答案：A",
            "initial_answer": "答案：A",
            "retrieved": [{"payload": {"e_id": "E-ID", "application_id": "historical",
                                       "knowledge_text": "参考资料"}}],
            "reference_answer": "PRIVATE_KEY_SENTINEL"}


class ReplayTests(unittest.TestCase):
    def test_batch4_answer_stage_resumes_and_rejects_changed_batch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_jsonl(root / "ter.jsonl", [source(str(i)) for i in range(5)])
            (root / "runtime.json").write_text(json.dumps({"runtime": {"model_id": "candidate"}}))
            batches = []

            def factory(base, model, budget, *args):
                def invoke(prompts):
                    batches.append((model, len(prompts), budget))
                    return [json.dumps(fallback_critique()) if model == "critic" else "答案：B"
                            for _ in prompts]
                return invoke

            argv = ["runner", "--source-results", str(root / "ter.jsonl"), "--output", str(root / "y1.jsonl"),
                    "--generator-model", "candidate", "--critic-model", "critic", "--batch-size", "4",
                    "--runtime-contract", str(root / "runtime.json"), "--resume"]
            with patch.object(sys, "argv", argv), patch.object(answer_stage, "batch_client", factory):
                answer_stage.main()
                answer_stage.main()
            self.assertEqual([("critic", 4, 384), ("candidate", 4, 32),
                              ("critic", 1, 384), ("candidate", 1, 32)], batches)
            self.assertTrue(all(row["alignrag_config"]["execution_batch_size"] == 4
                                for row in read_rows(root / "y1.jsonl")))
            argv[argv.index("--batch-size") + 1] = "2"
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                answer_stage.main()

    def test_baseline_migration_preserves_bytes_and_excludes_partial_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            source_dir = app / "questionbank_eval/model_benchmarks/qwen35_replay_v1/test_model"
            target = source_dir.parent.parent / "qwen35_replay_batch4_v2/test_model"
            source_dir.mkdir(parents=True)
            (app / "run_frozen_evidence_generation.py").write_text("frozen baseline runner")
            (source_dir / "manifest.json").write_text(json.dumps({
                "protocol": "qwen35-two-pass-v1", "generation": {"max_new_tokens": 32}}))
            (source_dir / "model_source.json").write_text("{}")
            runtime = {"model_id": "candidate"}
            (source_dir / "deployment_preflight.json").write_text(json.dumps({
                "runtime": runtime, "baseline_batch_size": 4, "max_new_tokens": 32}))
            atomic_jsonl(source_dir / "answer_refinement/2022/alignrag_results.jsonl", [source()])
            for arm, folder in (("dense", "dense_rag"), ("ter", "ter_direct")):
                for year, total in (("2022", 596), ("2021", 566)):
                    rows = [{**source(f"{year}-{i}"), "evidence": [{"doc_id": "d", "text": "资料"}]}
                            for i in range(total)]
                    frozen = (app / "questionbank_eval/dense_chunk_rag_v1" /
                              f"{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                              if arm == "dense" else app / "questionbank_eval" /
                              f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
                    atomic_jsonl(frozen, rows)
                    prediction = source_dir / folder / year / f"{folder}_results.jsonl"
                    atomic_jsonl(prediction, [{**row, "status": "ok", "method": arm} for row in rows])
                    (prediction.parent / "generation_contract.json").write_text(json.dumps({
                        "max_new_tokens": 32, "batch_size": 4, "invalid_choice_policy": "default_a",
                        "runtime": runtime, "runner_sha256": sha256(app / "run_frozen_evidence_generation.py"),
                        "evidence_input_sha256": sha256(frozen)}))
            audit = reuse_completed_baselines(source_dir, target, app)
            self.assertFalse(audit["answer_refinement_reused"])
            self.assertFalse((target / "answer_refinement").exists())
            for relative, digest in audit["files"].items():
                self.assertEqual(digest, sha256(target / relative))
            # A new top-level version can adopt these committed baseline
            # contracts, while its answer-Critic output must still be empty.
            queue.commit_manifest(target, {"protocol": PROTOCOL, "refinement_batch_size": 4})
            self.assertEqual(PROTOCOL, json.loads((target / "manifest.json").read_text())["protocol"])
            with self.assertRaises(ValueError):
                queue.commit_manifest(target, {"protocol": PROTOCOL, "refinement_batch_size": 2})
            # Repeating a safe migration is idempotent; incompatible files fail.
            reuse_completed_baselines(source_dir, target, app)
            (target / "ter_direct/2021/ter_direct_results.jsonl").write_text("different")
            with self.assertRaises(ValueError):
                reuse_completed_baselines(source_dir, target, app)

    def test_supervisor_retries_current_model_before_automatic_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = queue.Supervisor(root, root / "output")
            events = []

            def model(slug, repo, mirror):
                events.append(slug)
                supervisor.work = supervisor.output / slug
                supervisor.work.mkdir(parents=True, exist_ok=True)
                if len(events) == 1:
                    raise RuntimeError("transient connection failure")
                (supervisor.work / "summary.json").write_text(json.dumps({"arms": {"dense": {"correct": 1}}}))

            with patch.object(queue, "MODELS", (("first", "repo1", None), ("second", "repo2", None))), \
                 patch.object(supervisor, "model", model), patch.object(supervisor, "stop_servers"), \
                 patch.object(queue.time, "sleep"):
                supervisor.run_queue()
            self.assertEqual(["first", "first", "second"], events)
            self.assertEqual("complete", json.loads((supervisor.output / "state.json").read_text())["status"])
            self.assertEqual({"first", "second"}, set(json.loads((supervisor.output / "all_models_summary.json").read_text())))

    def test_answer_stage_preserves_raw_y0_excludes_keys_and_resumes_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_jsonl(root / "ter.jsonl", [source()])
            (root / "runtime.json").write_text(json.dumps({"runtime": {"model_id": "candidate"}}))
            calls = []

            def factory(base, model, max_tokens, *args):
                self.assertEqual(384 if model == "critic" else 32, max_tokens)

                def call(prompts):
                    calls.extend(prompts)
                    text = json.dumps(prompts, ensure_ascii=False)
                    self.assertNotIn("PRIVATE_KEY_SENTINEL", text)
                    self.assertIn("初始判断：答案：A", text)
                    return [json.dumps(fallback_critique(), ensure_ascii=False)
                            if model == "critic" else "答案：B" for _ in prompts]

                return call

            argv = ["runner", "--source-results", str(root / "ter.jsonl"), "--output", str(root / "y1.jsonl"),
                    "--generator-model", "candidate", "--critic-model", "critic", "--batch-size", "1",
                    "--runtime-contract", str(root / "runtime.json"), "--resume"]
            with patch.object(sys, "argv", argv), patch.object(answer_stage, "batch_client", factory):
                answer_stage.main()
                self.assertEqual(2, len(calls))
                answer_stage.main()
                self.assertEqual(2, len(calls), "committed rows must not trigger any new model request")
            row = read_rows(root / "y1.jsonl")[0]
            self.assertEqual(("初始判断：答案：A", "答案：A", "答案：B"),
                             (row["initial_answer"], row["ter_answer"], row["answer"]))
            row["initial_answer"] = row["answer"]
            with self.assertRaises(ValueError):
                answer_stage.check_existing([row], {"a": source()})

    def test_v3_routes_on_y1_but_preserves_raw_y0_and_y1_correct_rows(self):
        original = [{**source(str(i)), "initial_answer": "初始判断：答案：A", "answer": "答案：B"}
                    for i in range(2)]
        grades = [{"application_id": "0", "reference_status": "valid", "score": "1"},
                  {"application_id": "1", "reference_status": "valid", "score": "0"}]
        update = {**original[1], "answer": "答案：A", "critic_status": "valid"}
        merged = merge_y1_wrong(original, [update], grades)
        self.assertEqual(original[0], merged[0])
        self.assertEqual("答案：A", merged[1]["answer"])
        for replacements in ([], [update, update], [{**update, "application_id": "0"}],
                             [{**update, "initial_answer": "答案：B"}]):
            with self.assertRaises(ValueError):
                merge_y1_wrong(original, replacements, grades)

    def test_completed_summary_recalculates_both_passes_and_rejects_missing_first_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            bank = app / "questionbank_eval"
            work = bank / "model_benchmarks/qwen35_replay_v1/test_model"
            work.mkdir(parents=True)
            (work / "manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "generation": {"max_new_tokens": 32},
                                                          "refinement_batch_size": 4}))
            report = {"runtime": {"model_id": "candidate"}, "max_new_tokens": 32,
                      "exact_official_token_ids_verified": True, "baseline_batch_size": 4, "refinement_batch_size": 4}
            for name in ("deployment_preflight.json", "refinement_deployment_preflight.json"):
                (work / name).write_text(json.dumps(report))
            contract = {"runtime": report["runtime"], "generator_max_tokens": 32,
                        "invalid_choice_policy": "default_a", "batch_size": 4}
            for year, total, scored in (("2021", 566, 554), ("2022", 596, 584)):
                ter = [{**source(f"{year}-{i}"), "evidence": []} for i in range(total)]
                for row in ter:
                    row.pop("reference_answer")
                atomic_jsonl(bank / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl", ter)
                atomic_jsonl(bank / f"dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl", ter)
                atomic_jsonl(bank / f"mandatory_choice_v3_{year}/alignrag_results.jsonl", ter)
                refs = [{"application_id": row["application_id"], "reference_status": "valid",
                         "reference_answer": "B" if i == 0 else "A"} for i, row in enumerate(ter[:scored])]
                self.write_csv(bank / f"model_benchmarks/_reference/reference_{year}.csv", refs)
                y1 = [{**row, "initial_answer": row["raw_answer"], "ter_answer": row["answer"],
                       "answer": "答案：B" if i in (0, 1) else "答案：A",
                       "critic_status": "fallback", "critic": fallback_critique(),
                       "alignrag_config": {"execution_batch_size": 4}}
                      for i, row in enumerate(ter)]
                update = {**y1[1], "answer": "答案：A", "critic_status": "valid", "config": {"batch_size": 4}}
                merged = [y1[0], update] + y1[2:]
                for arm, folder, filename, rows in (
                    ("dense", "dense_rag", "dense_rag_results.jsonl", ter),
                    ("ter", "ter_direct", "ter_direct_results.jsonl", ter),
                    ("answer_refinement", "answer_refinement", "alignrag_results.jsonl", y1),
                    ("evicr", "evicr_v3", "evicr_v3_merged_results.jsonl", merged)):
                    atomic_jsonl(work / folder / year / filename, rows)
                    stage_contract = {**contract}
                    if arm in {"answer_refinement", "evicr"}:
                        stage_contract["critic_max_tokens"] = 384 if arm == "answer_refinement" else 640
                    (work / folder / year / "generation_contract.json").write_text(json.dumps(stage_contract))
                    grades = [{**ref, "score": str(int(rows[i]["answer"] == "答案：" + ref["reference_answer"]))}
                              for i, ref in enumerate(refs)]
                    path = work / "grading" / f"{arm}_{year}"
                    self.write_csv(path / "grading.csv", grades)
                    (path / "grading_summary.json").write_text(json.dumps({
                        "correct": sum(int(row["score"]) for row in grades), "scored_questions": scored}))
                atomic_jsonl(work / "evicr_v3" / year / "evidence_refinement_results.jsonl", [update])
            result = summarize(work, "test_model")
            self.assertEqual({"dense", "ter", "evicr"}, set(result["arms"]))
            self.assertEqual(1138, result["arms"]["evicr"]["correct"])
            self.assertEqual(1136, result["arms"]["ter"]["correct"])
            self.assertEqual(566, result["validation"]["2021"]["answer_critic_rows"])
            atomic_jsonl(work / "answer_refinement/2021/alignrag_results.jsonl", [])
            with self.assertRaises(ValueError):
                summarize(work, "test_model")

    @staticmethod
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
