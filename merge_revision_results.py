#!/usr/bin/env python3
"""Merge affected-question reruns into the frozen eight-model results."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from benchmark_integrity import atomic_jsonl, keyed, read_rows, sha256


APP = Path(__file__).resolve().parent
REV = APP / "questionbank_eval/experiment1_6_revision_v1"
REFERENCE = APP / "questionbank_eval/model_benchmarks/_reference"
CHOICE = re.compile(r"答案\s*[：:]\s*([A-E])\b", re.I)

MODELS = (
    ("qwen2_5_7b_instruct", "Qwen2.5-7B-Instruct", "questionbank_eval/model_benchmarks/qwen2_5_7b_instruct", False),
    ("qwen3_8b", "Qwen3-8B", "questionbank_eval/model_benchmarks/qwen3_8b", False),
    ("internlm3_8b_instruct", "InternLM3-8B-Instruct", "questionbank_eval/model_benchmarks/internlm3_8b_instruct", False),
    ("meta_llama_3_1_8b_instruct", "Llama-3.1-8B-Instruct", "questionbank_eval/model_benchmarks/meta_llama_3_1_8b_instruct", False),
    ("mistral_7b_instruct_v03", "Mistral-7B-Instruct-v0.3", "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_v1/mistral_7b_instruct_v03", False),
    ("gemma2_9b_it", "Gemma-2-9B-it", "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v3/gemma2_9b_it", False),
    ("yi15_9b_chat", "Yi-1.5-9B-Chat", "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v5/yi15_9b_chat", False),
    ("qwen35_9b", "Qwen3.5-9B", "questionbank_eval/model_benchmarks", True),
)


def old_path(slug: str, source: str, arm: str, year: str) -> Path:
    if slug == "qwen35_9b":
        if arm == "dense":
            return APP / "questionbank_eval/dense_chunk_rag_v1" / f"{year}_mandatory_choice_v3_answer_blind/dense_rag_results.jsonl"
        if arm == "ter":
            return APP / "questionbank_eval" / f"mandatory_choice_v3_{year}/alignrag_results.jsonl"
        raise ValueError("Qwen3.5 EviCR is materialized from its y1 and replacement files")
    directory = APP / source
    filename = {"dense": "dense_rag_results.jsonl", "ter": "ter_direct_results.jsonl", "evicr": "evicr_v3_merged_results.jsonl"}[arm]
    subdirectory = {"dense": "dense_rag", "ter": "ter_direct", "evicr": "evicr_v3"}[arm]
    return directory / subdirectory / year / filename


def load_reference(year: str) -> dict[str, str]:
    result = {}
    with (REFERENCE / f"reference_{year}.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("reference_status") == "valid" and row.get("reference_answer") in set("ABCDE"):
                result[str(row["application_id"])] = row["reference_answer"]
    return result


def choice(row: dict[str, Any]) -> str:
    values = CHOICE.findall(str(row.get("answer", "")))
    return values[-1].upper() if values else ""


def qwen35_old_evicr(year: str) -> list[dict[str, Any]]:
    source = read_rows(APP / "questionbank_eval" / f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
    replacement_path = (APP / "questionbank_eval/evidence_refinement_wrong_2021_v3_adapter/evidence_refinement_results.jsonl"
                        if year == "2021" else APP / "questionbank_eval/evidence_refinement_wrong_2022_v3/evidence_refinement_results.jsonl")
    replacements = keyed(read_rows(replacement_path))
    return [replacements.get(str(row["application_id"]), row) for row in source]


def qwen35_old_ter(year: str) -> list[dict[str, Any]]:
    """The historical Qwen file stores raw TER y0 in initial_answer, then y1 in answer."""
    rows = read_rows(APP / "questionbank_eval" / f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
    result = []
    for row in rows:
        if not row.get("initial_answer"):
            raise ValueError(f"Missing historical Qwen TER y0: {row.get('application_id')}")
        result.append({**row, "answer": row["initial_answer"], "choice_status": "historical_ter_y0"})
    return result


def main() -> None:
    affected_rows = read_rows(REV / "rerun_questions_answer_blind.jsonl")
    affected = {str(row["application_id"]) for row in affected_rows}
    affected_by_year = {
        year: {str(row["application_id"]) for row in affected_rows if str(row.get("year")) == year}
        for year in ("2021", "2022")
    }
    out_root = REV / "final_results"
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"version": "experiment1_6_revision_final_v1", "affected_question_count": len(affected), "models": {}}
    changed_rows: list[dict[str, Any]] = []

    for slug, label, source, qwen35 in MODELS:
        model_summary: dict[str, Any] = {"label": label, "arms": {}, "historical_source": source}
        for arm in ("dense", "ter", "evicr"):
            arm_summary: dict[str, Any] = {"by_year": {}}
            for year in ("2021", "2022"):
                if qwen35 and arm == "evicr":
                    old = qwen35_old_evicr(year)
                elif qwen35 and arm == "ter":
                    old = qwen35_old_ter(year)
                else:
                    old = read_rows(old_path(slug, source, arm, year))
                new_path = REV / "model_runs" / slug / ({"dense": "dense_rag", "ter": "ter_direct", "evicr": "evicr_v3"}[arm]) / year
                if arm == "evicr":
                    new_path = new_path / "evicr_v3_merged_results.jsonl"
                elif arm == "dense":
                    new_path = new_path / "dense_rag_results.jsonl"
                else:
                    new_path = new_path / "ter_direct_results.jsonl"
                new = keyed(read_rows(new_path))
                old_by_id = keyed(old)
                year_affected = affected_by_year[year]
                if not year_affected <= set(old_by_id) or not year_affected <= set(new):
                    raise ValueError(f"{slug}/{arm}/{year} missing affected IDs")
                merged = [new.get(str(row["application_id"]), row) for row in old]
                if len(keyed(merged)) != len(old_by_id):
                    raise ValueError(f"duplicate IDs after merge: {slug}/{arm}/{year}")
                target = out_root / slug / arm / f"{year}.jsonl"
                atomic_jsonl(target, merged)
                reference = load_reference(year)
                correct = sum(choice(row) == reference.get(str(row["application_id"])) for row in merged if str(row["application_id"]) in reference)
                total = sum(str(row["application_id"]) in reference for row in merged)
                expected_scored = 554 if year == "2021" else 584
                expected_rows = 566 if year == "2021" else 596
                if total != expected_scored or len(merged) != expected_rows:
                    raise ValueError(f"Unexpected full-year coverage: {slug}/{arm}/{year} rows={len(merged)} scored={total}")
                arm_summary["by_year"][year] = {"correct": correct, "scored_questions": total,
                                                  "accuracy_percent": round(correct / total * 100, 4),
                                                  "result_sha256": sha256(target)}
                for app_id in sorted(year_affected & set(old_by_id) & set(new)):
                    old_choice = choice(old_by_id[app_id]); new_choice = choice(new[app_id])
                    changed_rows.append({"model": slug, "label": label, "arm": arm, "year": year,
                                         "application_id": app_id, "old_choice": old_choice, "new_choice": new_choice,
                                         "changed": old_choice != new_choice})
            total_correct = sum(item["correct"] for item in arm_summary["by_year"].values())
            total_scored = sum(item["scored_questions"] for item in arm_summary["by_year"].values())
            arm_summary.update({"correct": total_correct, "scored_questions": total_scored,
                                "accuracy_percent": round(total_correct / total_scored * 100, 4)})
            model_summary["arms"][arm] = arm_summary
        summary["models"][slug] = model_summary

    changed_path = REV / "affected_changes.jsonl"
    atomic_jsonl(changed_path, changed_rows)
    (REV / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (REV / "comparison_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "ordinary_vector_RAG", "TER", "TER+EviCR", "dense_correct", "ter_correct", "evicr_correct", "total"])
        for slug, model in summary["models"].items():
            writer.writerow([model["label"],
                             f"{model['arms']['dense']['accuracy_percent']:.2f}%",
                             f"{model['arms']['ter']['accuracy_percent']:.2f}%",
                             f"{model['arms']['evicr']['accuracy_percent']:.2f}%",
                             model["arms"]["dense"]["correct"], model["arms"]["ter"]["correct"],
                             model["arms"]["evicr"]["correct"], 1138])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
