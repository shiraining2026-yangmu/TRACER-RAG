#!/usr/bin/env python3
"""Verify the two-pass Qwen3.5 replay and report only dense / TER / EviCR."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmark_integrity import (atomic_jsonl, csv_rows, keyed,
                                 read_rows, sha256, validate_predictions)
from grade_single_results import choice
from run_qwen35_answer_refinement import check_existing

PROTOCOL = "qwen35-two-pass-batch4-v2"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge_y1_wrong(original, replacements, grading):
    source = keyed(original)
    grades = keyed(grading)
    if not set(grades) <= set(source):
        raise ValueError("Grading IDs absent from answer-refinement source")
    wrong = {key for key, row in grades.items()
             if row.get("reference_status") == "valid" and str(row.get("score")) == "0"}
    updates = validate_predictions(replacements, {key: source[key] for key in wrong})
    for key, row in updates.items():
        # Historical v3 routes on y1 mistakes but its generator/gate consume y0.
        if (row.get("initial_answer") != source[key]["initial_answer"].strip()
                or row.get("retrieved") != source[key]["retrieved"]
                or row.get("critic_status") not in {"valid", "repaired", "fallback"}):
            raise ValueError("V3 changed the candidate y0 or frozen evidence")
    return [updates.get(str(row["application_id"]), row) for row in original]


def merge_year(work, year):
    source = read_rows(work / "answer_refinement" / year / "alignrag_results.jsonl")
    updates = read_rows(work / "evicr_v3" / year / "evidence_refinement_results.jsonl")
    grade = csv_rows(work / "grading" / f"answer_refinement_{year}" / "grading.csv")
    result = merge_y1_wrong(source, updates, grade)
    atomic_jsonl(work / "evicr_v3" / year / "evicr_v3_merged_results.jsonl", result)


def summarize(work, slug):
    app = next(parent.parent for parent in work.resolve().parents if parent.name == "questionbank_eval")
    bank = app / "questionbank_eval"
    manifest = load(work / "manifest.json")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Missing two-pass protocol manifest")
    result = {"model_slug": slug, "protocol": PROTOCOL,
              "pipeline": "TER raw y0 -> answer Critic -> y1 -> select y1 mistakes -> evidence Critic v3 -> candidate answer; preserve y1 correct rows",
              "v3_initial_answer": "raw TER y0 (historical behavior)",
              "interpretation": "offline-label-conditioned repair; not a full-test blind deployment score",
              "answer_key_used_for_model_input": False,
              "generation": manifest["generation"], "arms": {}, "validation": {},
              "manifest_sha256": sha256(work / "manifest.json")}
    reports = {name: load(work / name) for name in
               ("deployment_preflight.json", "refinement_deployment_preflight.json")}
    for report in reports.values():
        if not report.get("exact_official_token_ids_verified") or report.get("max_new_tokens") != 32:
            raise ValueError("Missing official-template / 32-token deployment checks")
    if (manifest.get("refinement_batch_size") != 4
            or reports["deployment_preflight.json"].get("baseline_batch_size") != 4
            or reports["refinement_deployment_preflight.json"].get("refinement_batch_size") != 4):
        raise ValueError("Missing fixed batch-4 execution contract")
    for year, raw_count, scored_count in (("2021", 566, 554), ("2022", 596, 584)):
        questions = keyed(read_rows(bank / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl"))
        references = {str(row["application_id"]): row["reference_answer"]
                      for row in csv_rows(bank / "model_benchmarks/_reference" / f"reference_{year}.csv")
                      if row.get("reference_status") == "valid"}
        if len(questions) != raw_count or len(references) != scored_count or not set(references) <= set(questions):
            raise ValueError("Question/reference ID or count mismatch")
        paths = {
            "dense": work / "dense_rag" / year / "dense_rag_results.jsonl",
            "ter": work / "ter_direct" / year / "ter_direct_results.jsonl",
            "answer_refinement": work / "answer_refinement" / year / "alignrag_results.jsonl",
            "evicr": work / "evicr_v3" / year / "evicr_v3_merged_results.jsonl",
        }
        rows = {arm: read_rows(path) for arm, path in paths.items()}
        validated = {arm: validate_predictions(data, questions) for arm, data in rows.items()}
        for arm, directory in (("dense", "dense_rag"), ("ter", "ter_direct"),
                               ("answer_refinement", "answer_refinement"), ("evicr", "evicr_v3")):
            contract = load(work / directory / year / "generation_contract.json")
            expected_runtime = reports["deployment_preflight.json" if arm in {"dense", "ter"}
                                       else "refinement_deployment_preflight.json"]["runtime"]
            if (contract.get("runtime") != expected_runtime
                    or contract.get("max_new_tokens", contract.get("generator_max_tokens")) != 32
                    or contract.get("batch_size") != 4
                    or contract.get("invalid_choice_policy") != "default_a"):
                raise ValueError(f"Frozen generation contract mismatch: {arm}/{year}")
            if arm in {"answer_refinement", "evicr"} and contract.get("critic_max_tokens") != (384 if arm == "answer_refinement" else 640):
                raise ValueError(f"Changed Critic token budget: {arm}/{year}")
            correct = sum(choice(validated[arm][key]["answer"]) == ref for key, ref in references.items())
            grade_dir = work / "grading" / f"{arm}_{year}"
            stored = load(grade_dir / "grading_summary.json")
            grades = keyed(csv_rows(grade_dir / "grading.csv"))
            if (stored["correct"] != correct or stored["scored_questions"] != scored_count
                    or set(grades) != set(references)):
                raise ValueError(f"Independent score mismatch: {arm}/{year}")
            for key, ref in references.items():
                if (grades[key]["reference_answer"] != ref
                        or str(grades[key]["score"]) != str(int(choice(validated[arm][key]["answer"]) == ref))):
                    raise ValueError("Per-question grading mismatch")
        for arm, path, field in (
            ("dense", bank / "dense_chunk_rag_v1" / f"{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl", "evidence"),
            ("ter", bank / f"mandatory_choice_v3_{year}/alignrag_results.jsonl", "retrieved")):
            expected = keyed(read_rows(path))
            if any(row[field] != expected[key][field] for key, row in validated[arm].items()):
                raise ValueError("Frozen retrieval changed")
        if len(check_existing(rows["answer_refinement"], validated["ter"])) != raw_count:
            raise ValueError("Incomplete all-question answer-Critic pass")
        if any(row.get("alignrag_config", {}).get("execution_batch_size") != 4
               for row in rows["answer_refinement"]):
            raise ValueError("Mixed answer-refinement execution batches")
        updates = read_rows(work / "evicr_v3" / year / "evidence_refinement_results.jsonl")
        if any(row.get("config", {}).get("batch_size") != 4 for row in updates):
            raise ValueError("Mixed evidence-refinement execution batches")
        expected = merge_y1_wrong(rows["answer_refinement"], updates,
                                 csv_rows(work / "grading" / f"answer_refinement_{year}/grading.csv"))
        if rows["evicr"] != expected:
            raise ValueError("V3 merge must preserve y1 correct rows exactly")
        result["validation"][year] = {
            "raw_questions": raw_count, "scored_questions": scored_count,
            "answer_critic_rows": len(rows["answer_refinement"]), "v3_selected_y1_mistakes": len(updates),
            "answer_critic_accuracy": load(work / "grading" / f"answer_refinement_{year}/grading_summary.json"),
            "result_sha256": {arm: sha256(path) for arm, path in paths.items()},
            "choice_origins": {arm: dict(Counter(row.get("choice_origin", row.get("choice_status", "unknown"))
                                                  for row in data)) for arm, data in rows.items()},
        }
    for arm in ("dense", "ter", "evicr"):
        by_year = {year: load(work / "grading" / f"{arm}_{year}/grading_summary.json")
                   for year in ("2021", "2022")}
        correct = sum(data["correct"] for data in by_year.values())
        total = sum(data["scored_questions"] for data in by_year.values())
        result["arms"][arm] = {"correct": correct, "total": total,
                               "accuracy_percent": round(correct / total * 100, 4), "by_year": by_year}
    path = work / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--merge-year", choices=("2021", "2022"))
    args = parser.parse_args()
    if args.merge_year:
        merge_year(args.work_dir, args.merge_year)
    else:
        summarize(args.work_dir, args.slug)


if __name__ == "__main__":
    main()
