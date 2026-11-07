import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_integrity import atomic_jsonl, keyed, merge_wrong_only, read_rows, validate_predictions, freeze_protocol
from run_questionbank_rag_vs_alignrag import enforce_choices
import run_evidence_refinement_wrong_questions as v3
from download_benchmark_model import modelscope_file, exclude_duplicate_weights
from types import SimpleNamespace


class BenchmarkTests(unittest.TestCase):
    def test_resume_rejects_mixed_generation_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "contract.json", root / "result.jsonl"
            contract = {"max_tokens": 128, "adapter": "v2"}
            freeze_protocol(path, contract, output)
            freeze_protocol(path, contract, output)
            with self.assertRaises(ValueError):
                freeze_protocol(path, {**contract, "max_tokens": 32}, output)
            atomic_jsonl(output, [{"application_id": "a"}])
            with self.assertRaises(ValueError):
                freeze_protocol(root / "missing.json", contract, output)

    def test_invalid_outputs_cannot_earn_default_a_credit(self):
        contexts = [{"question": "q"}]
        result = enforce_choices(lambda prompts: ["无法判断"], contexts, ["无法判断"], invalid_policy="mark_invalid")
        self.assertEqual([("INVALID", "invalid_output")], result)
        rows = [{"application_id": "a", "question": "q", "answer": "INVALID", "choice_origin": "invalid_output"}]
        self.assertEqual({"a"}, set(validate_predictions(rows, {"a": rows[0]})))
        with self.assertRaises(ValueError):
            validate_predictions([{**rows[0], "choice_origin": "model"}], {"a": rows[0]})
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            atomic_jsonl(tmp / "results.jsonl", rows)
            with (tmp / "refs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["application_id", "reference_status", "reference_answer"])
                writer.writeheader()
                writer.writerow({"application_id": "a", "reference_status": "valid", "reference_answer": "A"})
            subprocess.run([sys.executable, str(ROOT / "grade_single_results.py"), "--results", str(tmp / "results.jsonl"),
                            "--reference-grading", str(tmp / "refs.csv"), "--output-dir", str(tmp / "grade"), "--label", "invalid"],
                           check=True, capture_output=True)
            summary = json.loads((tmp / "grade/grading_summary.json").read_text())
            self.assertEqual((0, 1, 1), (summary["correct"], summary["scored_questions"], summary["invalid_outputs"]))

    def test_duplicate_weight_filter_handles_both_repository_apis(self):
        names = ["model.safetensors.index.json", "model-00001-of-00002.safetensors",
                 "model-00002-of-00002.safetensors", "consolidated.safetensors", "config.json"]
        for items, key in (([{"Path": n} for n in names], "Path"),
                           ([SimpleNamespace(rfilename=n) for n in names], "rfilename")):
            filtered = exclude_duplicate_weights(items, key)
            actual = [item[key] if isinstance(item, dict) else getattr(item, key) for item in filtered]
            self.assertEqual(names[:3] + names[4:], actual)
        self.assertEqual([{"Path": "consolidated.safetensors"}],
                         exclude_duplicate_weights([{"Path": "consolidated.safetensors"}], "Path"))

    def test_completed_partial_download_commits_without_network(self):
        data = b"already downloaded and verified"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.safetensors.partial").write_bytes(data)
            item = {"Path": "weights.safetensors", "Size": len(data),
                    "Sha256": hashlib.sha256(data).hexdigest(), "Revision": "fixed"}
            with patch("download_benchmark_model.requests.get", side_effect=AssertionError("must not issue HTTP 416 range")):
                result = modelscope_file(root, "test/repo", item)
            self.assertEqual(len(data), result["size"])
            self.assertEqual(data, (root / "weights.safetensors").read_bytes())

    def test_wrong_only_merge_rejects_missing_extra_and_changed_evidence(self):
        original = [{"application_id": str(i), "question": f"q{i}", "answer": "答案：A", "retrieved": []} for i in range(2)]
        grade = [{"application_id": "0", "reference_status": "valid", "score": "1"},
                 {"application_id": "1", "reference_status": "valid", "score": "0"}]
        update = {**original[1], "initial_answer": "答案：A", "answer": "答案：B", "critic_status": "valid"}
        result = merge_wrong_only(original, [update], grade)
        self.assertEqual(original[0], result[0])
        self.assertEqual("答案：B", result[1]["answer"])
        for rows in ([], [update, update], [{**update, "application_id": "0"}],
                     [{**update, "retrieved": [{"changed": True}]}]):
            with self.assertRaises(ValueError):
                merge_wrong_only(original, rows, grade)

    def test_scoring_schema_selects_wrong_and_v3_resumes_without_key_in_prompt(self):
        evidence = [{"payload": {"e_id": "edge", "application_id": "historical", "knowledge_text": "依据"}}]
        source = [{"application_id": str(i), "question": f"题{i}\nA.甲 B.乙 C.丙 D.丁 E.戊",
                   "answer": "答案：A", "initial_answer": "答案：A", "retrieved": evidence} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            atomic_jsonl(tmp / "source.jsonl", source)
            with (tmp / "ref.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["application_id", "reference_answer", "reference_status"])
                writer.writeheader()
                writer.writerows([{"application_id": "0", "reference_answer": "A", "reference_status": "valid"},
                                  {"application_id": "1", "reference_answer": "B", "reference_status": "valid"}])
            subprocess.run([sys.executable, str(ROOT / "grade_single_results.py"), "--results", str(tmp / "source.jsonl"),
                            "--reference-grading", str(tmp / "ref.csv"), "--output-dir", str(tmp / "grade"), "--label", "test"],
                           check=True, capture_output=True)
            grading = tmp / "grade" / "grading.csv"
            self.assertEqual({"1"}, v3.load_grading_rows(grading)[0])
            package = {"evidence_status": "partial", "evidence_actions": [], "knowledge_supplements": [],
                       "change_support": "strong", "answer_guidance": "保留相关知识"}
            calls = []

            def factory(base, model, *args):
                def call(prompts):
                    calls.extend(prompts)
                    text = json.dumps(prompts, ensure_ascii=False)
                    self.assertNotIn("reference_answer", text)
                    self.assertNotIn("reference_status", text)
                    return [json.dumps(package, ensure_ascii=False) if model == "critic" else "答案：B" for _ in prompts]
                return call

            argv = ["runner", "--source-results", str(tmp / "source.jsonl"), "--grading-csv", str(grading),
                    "--output-dir", str(tmp / "v3"), "--generator-model", "generator", "--critic-model", "critic", "--resume"]
            with patch.object(sys, "argv", argv), patch.object(v3, "batch_client", factory):
                v3.main()
                self.assertEqual(2, len(calls))
                v3.main()
                self.assertEqual(2, len(calls), "resume must not regenerate completed samples")
            updates = read_rows(tmp / "v3" / "evidence_refinement_results.jsonl")
            self.assertEqual({"1"}, set(keyed(updates)))
            self.assertEqual("答案：B", updates[0]["answer"])

    def test_atomic_replacement_ignores_uncommitted_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            atomic_jsonl(path, [{"application_id": "a"}])
            path.with_suffix(".jsonl.tmp").write_text('{"truncated":')
            self.assertEqual([{"application_id": "a"}], read_rows(path))

    def test_eight_result_completion_gate_recalculates_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            bank = app / "questionbank_eval"
            work = bank / "model_benchmarks" / "test_model"
            refs = work.parent / "_reference"
            refs.mkdir(parents=True)
            for year, raw_count, scored_count in (("2021", 566, 554), ("2022", 596, 584)):
                rows = [{"application_id": f"{year}-{i}", "question": f"q{i}", "answer": "答案：A",
                         "initial_answer": "答案：A", "retrieved": [], "evidence": []} for i in range(raw_count)]
                atomic_jsonl(bank / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl", rows)
                atomic_jsonl(bank / "dense_chunk_rag_v1" / f"{year}_mandatory_choice_v3_answer_blind" / "retrieval.jsonl", rows)
                atomic_jsonl(bank / f"mandatory_choice_v3_{year}" / "alignrag_results.jsonl", rows)
                grades = [{"application_id": row["application_id"], "reference_status": "valid", "reference_answer": "A", "score": "1"}
                          for row in rows[:scored_count]]
                with (refs / f"reference_{year}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(grades[0])); writer.writeheader(); writer.writerows(grades)
                paths = {"no_rag": work / "no_rag" / f"{year}_mandatory_choice_v3_answer_blind" / "no_rag_results.jsonl",
                         "dense": work / "dense_rag" / year / "dense_rag_results.jsonl",
                         "ter": work / "ter_direct" / year / "ter_direct_results.jsonl",
                         "evicr": work / "evicr_v3" / year / "evicr_v3_merged_results.jsonl"}
                for arm, path in paths.items():
                    atomic_jsonl(path, rows)
                    grade_dir = work / "grading" / f"{arm}_{year}"
                    grade_dir.mkdir(parents=True)
                    (grade_dir / "grading_summary.json").write_text(json.dumps({"correct": scored_count, "scored_questions": scored_count}))
                    if arm == "ter":
                        with (grade_dir / "grading.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(grades[0])); writer.writeheader(); writer.writerows(grades)
                atomic_jsonl(work / "evicr_v3" / year / "evidence_refinement_results.jsonl", [])
            command = [sys.executable, str(ROOT / "summarize_multimodel_benchmark.py"), "--work-dir", str(work), "--slug", "test_model"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(1138, summary["arms"]["evicr"]["correct"])
            # Three-arm mode must be explicit; a missing no-RAG file must not
            # silently relax the deletion gate of a four-arm experiment.
            for year in ("2021", "2022"):
                (work / "no_rag" / f"{year}_mandatory_choice_v3_answer_blind" / "no_rag_results.jsonl").unlink()
            self.assertNotEqual(0, subprocess.run(command, capture_output=True).returncode)
            three_arm_command = command + ["--arms", "dense", "ter", "evicr"]
            result = subprocess.run(three_arm_command, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual({"dense", "ter", "evicr"}, set(summary["arms"]))
            # A truncated result must fail the deletion gate despite a summary
            # file left over from an earlier successful validation.
            atomic_jsonl(work / "ter_direct" / "2021" / "ter_direct_results.jsonl", [])
            self.assertNotEqual(0, subprocess.run(three_arm_command, capture_output=True).returncode)


if __name__ == "__main__":
    unittest.main()
