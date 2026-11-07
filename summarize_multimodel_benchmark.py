#!/usr/bin/env python3
"""Summarize per-year and combined accuracy after one model benchmark completes."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from benchmark_integrity import ANSWER, csv_rows, keyed, merge_wrong_only, read_rows, sha256, validate_predictions


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--arms", nargs="+", choices=("no_rag", "dense", "ter", "evicr"))
    args = parser.parse_args()
    if args.arms is None:
        manifest_path = args.work_dir / "manifest.json"
        protocol = load(manifest_path).get("protocol", "full") if manifest_path.exists() else "full"
        if protocol not in ("full", "dense_ter_evicr"):
            parser.error("Unknown experiment protocol in manifest")
        args.arms = (["dense", "ter", "evicr"] if protocol == "dense_ter_evicr"
                     else ["no_rag", "dense", "ter", "evicr"])
    if not {"dense", "ter", "evicr"} <= set(args.arms) or len(args.arms) != len(set(args.arms)):
        parser.error("Select dense, ter, evicr and optionally no_rag, without duplicates")
    arms = {
        "no_rag": "普通模型，无 RAG",
        "dense": "模型 + 普通向量 RAG",
        "ter": "模型 + TER（R4 直接生成）",
        "evicr": "模型 + TER + EviCR（v3）",
    }
    app = next(parent.parent for parent in args.work_dir.resolve().parents if parent.name == "questionbank_eval")
    reference_dir = app / "questionbank_eval/model_benchmarks/_reference"
    manifest_path = args.work_dir / "manifest.json"
    manifest = load(manifest_path) if manifest_path.exists() else {}
    result = {"model_slug": args.slug, "arms": {}, "validation": {},
              "evicr_protocol": "TER wrong-only v3 repair; TER correct answers preserved using offline labels",
              "interpretation": "selection-conditional diagnostic result; not a full-test blind Critic gain",
              "answer_key_used_for_model_input": False}
    if manifest.get("deployment_protocol") == "adapted_v2":
        reports = {name: load(args.work_dir / name) for name in (
            "deployment_preflight.json", "refinement_deployment_preflight.json")}
        for report in reports.values():
            if not report.get("exact_official_token_ids_verified") or report.get("max_new_tokens") != 128:
                raise ValueError("Missing corrected deployment checks")
        result["deployment_protocol"] = "adapted_v2"
        result["deployment_checks"] = {name: sha256(args.work_dir / name) for name in reports}
        result["generation"] = manifest["generation"]
        result["invalid_output_scoring"] = "wrong; never default A"
        for year in ("2021", "2022"):
            for directory in ("dense_rag", "ter_direct", "evicr_v3"):
                contract = load(args.work_dir / directory / year / "generation_contract.json")
                runtime = reports["refinement_deployment_preflight.json" if directory == "evicr_v3" else "deployment_preflight.json"]["runtime"]
                budget = contract.get("generator_max_tokens", contract.get("max_new_tokens"))
                if contract.get("runtime") != runtime or budget != 128 or contract.get("invalid_choice_policy") != "mark_invalid":
                    raise ValueError("Corrected run contract mismatch")
    for year, raw_count, scored_count in (("2021", 566, 554), ("2022", 596, 584)):
        questions_path = app / "questionbank_eval" / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl"
        frozen = keyed(read_rows(questions_path))
        if len(frozen) != raw_count:
            raise ValueError("Unexpected frozen question count")
        candidate_paths = {
            "no_rag": args.work_dir / "no_rag" / f"{year}_mandatory_choice_v3_answer_blind" / "no_rag_results.jsonl",
            "dense": args.work_dir / "dense_rag" / year / "dense_rag_results.jsonl",
            "ter": args.work_dir / "ter_direct" / year / "ter_direct_results.jsonl",
            "evicr": args.work_dir / "evicr_v3" / year / "evicr_v3_merged_results.jsonl",
        }
        paths = {key: candidate_paths[key] for key in args.arms}
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("Missing output for a requested experiment arm")
        all_rows = {key: read_rows(path) for key, path in paths.items()}
        refs = {str(row["application_id"]): row["reference_answer"]
                for row in csv_rows(reference_dir / f"reference_{year}.csv")
                if row.get("reference_status") == "valid"}
        if len(refs) != scored_count or not set(refs) <= set(frozen):
            raise ValueError("Reference IDs/count do not match frozen exams")
        for key, rows in all_rows.items():
            checked = validate_predictions(rows, frozen)
            scored = load(args.work_dir / "grading" / f"{key}_{year}" / "grading_summary.json")
            correct = sum(bool(ANSWER.fullmatch(checked[app_id]["answer"])) and checked[app_id]["answer"][-1] == answer
                          for app_id, answer in refs.items())
            if scored["scored_questions"] != scored_count or scored["correct"] != correct:
                raise ValueError("Stored grading differs from independently recalculated score")
        dense_path = app / "questionbank_eval" / "dense_chunk_rag_v1" / f"{year}_mandatory_choice_v3_answer_blind" / "retrieval.jsonl"
        ter_path = app / "questionbank_eval" / f"mandatory_choice_v3_{year}" / "alignrag_results.jsonl"
        for key, input_path, evidence_key in (("dense", dense_path, "evidence"), ("ter", ter_path, "retrieved")):
            if key not in all_rows:
                continue
            expected = keyed(read_rows(input_path))
            for row in all_rows[key]:
                if row[evidence_key] != expected[str(row["application_id"])][evidence_key]:
                    raise ValueError("Baseline did not preserve frozen retrieval")
        updates = read_rows(args.work_dir / "evicr_v3" / year / "evidence_refinement_results.jsonl")
        expected_merged = merge_wrong_only(all_rows["ter"], updates, csv_rows(args.work_dir / "grading" / f"ter_{year}" / "grading.csv"))
        if all_rows["evicr"] != expected_merged:
            raise ValueError("Merged EviCR result differs from exact wrong-only replacement")
        result["validation"][year] = {
            "raw_questions": raw_count, "scored_questions": scored_count, "refined_wrong_questions": len(updates),
            "question_sha256": sha256(questions_path), "dense_input_sha256": sha256(dense_path),
            "ter_input_sha256": sha256(ter_path), "result_sha256": {key: sha256(path) for key, path in paths.items()},
            "choice_origins": {key: dict(Counter(row.get("choice_origin", row.get("choice_status", "unknown")) for row in rows))
                               for key, rows in all_rows.items()},
            "critic_statuses": dict(Counter(row.get("critic_status", "unknown") for row in updates)),
        }
    for key, label in arms.items():
        if key not in args.arms:
            continue
        by_year = {year: load(args.work_dir / "grading" / f"{key}_{year}" / "grading_summary.json") for year in ("2021", "2022")}
        correct = sum(item["correct"] for item in by_year.values())
        total = sum(item["scored_questions"] for item in by_year.values())
        result["arms"][key] = {
            "label": label, "by_year": by_year, "correct": correct, "total": total,
            "accuracy_percent": round(correct / total * 100, 4),
        }
    temporary = args.work_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.work_dir / "summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
