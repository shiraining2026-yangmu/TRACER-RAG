#!/usr/bin/env python3
"""OFFLINE ONLY: grade frozen dense RAG results; never imported by inference."""
import argparse
import csv
import json
from pathlib import Path

from grade_questionbank_results import grade, parse_answer_key, write_csv
from run_dense_chunk_rag import atomic_json, file_hash, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answer-key-md", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results, questions = list(read_jsonl(args.results)), list(read_jsonl(args.questions))
    if len(results) != len(questions) or {r["application_id"] for r in results} != {r["application_id"] for r in questions}:
        raise ValueError("Require complete result file with exactly the frozen question IDs")
    keys, issues, audit = parse_answer_key(args.answer_key_md)
    rows, summary = grade(args.results, keys, issues)
    summary.update(results_sha256=file_hash(args.results), questions_sha256=file_hash(args.questions),
                   answer_key_sha256=file_hash(args.answer_key_md), answer_key_audit=audit,
                   choice_origins={k: sum(r.get("choice_origin")==k for r in results) for k in {r.get("choice_origin") for r in results}})
    summary["choice_status_counts"] = summary["choice_origins"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row, result in zip(rows, results):
        row["choice_status"] = result.get("choice_origin", "missing")
    write_csv(args.output_dir / "dense_rag_grading.csv", rows)
    with (args.output_dir / "dense_rag_answers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["application_id", "year", "unit", "source_number", "question", "model_answer", "choice_origin"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result, row in zip(results, rows):
            writer.writerow({**{k: result.get(k, "") for k in fields if k != "model_answer"}, "model_answer": row["model_answer"]})
    atomic_json(args.output_dir / "grading_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
