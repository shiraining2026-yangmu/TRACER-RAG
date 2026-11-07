#!/usr/bin/env python3
"""Local-only grading of completed no-RAG results against original OCR keys."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from grade_questionbank_results import grade, parse_answer_key, write_csv
from run_dense_chunk_rag import atomic_json, file_hash, read_jsonl
from run_questionbank_no_rag import VERSION, resume_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answer-key-md", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = list(read_jsonl(args.questions))
    results, ids = resume_rows(args.results, frozen)
    if len(results) != len(frozen) or ids != {r["application_id"] for r in frozen}:
        raise ValueError("No-RAG output is incomplete")
    dense = list(read_jsonl(args.dense_results))
    if len(dense) != len(frozen) or {r["application_id"] for r in dense} != ids:
        raise ValueError("Dense RAG must have exactly the same question IDs")
    frozen_by_id = {r["application_id"]: r for r in frozen}
    if any(r["question"] != frozen_by_id[r["application_id"]]["question"] for r in dense):
        raise ValueError("Dense RAG questions differ from the frozen no-RAG questions")
    keys, issues, audit = parse_answer_key(args.answer_key_md)
    rows, summary = grade(args.results, keys, issues)
    dense_rows, dense_summary = grade(args.dense_results, keys, issues)
    dense_by_id = {r["application_id"]: r for r in dense_rows}
    pairs = Counter()
    compared = []
    for row in rows:
        other = dense_by_id[row["application_id"]]
        if row["reference_status"] != other["reference_status"]:
            raise ValueError("Paired key eligibility mismatch")
        if row["reference_status"] == "valid":
            label = ("no_rag_correct" if row["score"] else "no_rag_wrong") + "__" + ("dense_correct" if other["score"] else "dense_wrong")
            pairs[label] += 1
        else:
            label = "unscorable"
        compared.append({"application_id": row["application_id"], "unit": row["unit"], "source_number": row["source_number"],
                         "reference_answer": row["reference_answer"], "reference_status": row["reference_status"],
                         "no_rag_answer": row["model_answer"], "dense_answer": other["model_answer"],
                         "no_rag_score": row["score"], "dense_score": other["score"], "pair": label})
    summary.update(method=VERSION, results_sha256=file_hash(args.results), questions_sha256=file_hash(args.questions),
                   answer_key_sha256=file_hash(args.answer_key_md), dense_results_sha256=file_hash(args.dense_results),
                   answer_key_audit=audit, choice_origins=dict(Counter(r["choice_origin"] for r in results)),
                   dense_rag_comparison={"dense_correct": dense_summary["correct"],
                       "dense_accuracy_percent": dense_summary["score_rate_valid_key_matched"],
                       "dense_minus_no_rag_correct": dense_summary["correct"]-summary["correct"],
                       "dense_minus_no_rag_percentage_points": round((dense_summary["correct"]-summary["correct"])/summary["valid_key_matched_questions"]*100, 4),
                       "paired_counts_valid_keys": dict(pairs)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "no_rag_grading.csv", rows)
    with (args.output_dir / "no_rag_answers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["application_id", "year", "unit", "source_number", "question", "model_answer", "choice_origin"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        by_id = {r["application_id"]: r for r in rows}
        for result in results:
            writer.writerow({**{k: result.get(k, "") for k in fields if k != "model_answer"},
                             "model_answer": by_id[result["application_id"]]["model_answer"]})
    with (args.output_dir / "no_rag_vs_dense_paired.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compared[0]))
        writer.writeheader()
        writer.writerows(compared)
    atomic_json(args.output_dir / "grading_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
