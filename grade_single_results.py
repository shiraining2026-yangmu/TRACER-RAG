#!/usr/bin/env python3
"""Offline-only grade one completed result JSONL against a frozen grading CSV.

The CSV is read only after inference; no answer key or target label is passed to
any model prompt.  This also avoids copying OCR key text into every benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from benchmark_integrity import invalid_prediction

CHOICE = re.compile(r"答案\s*[：:]\s*([A-E])\b", re.I)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def choice(value: object) -> str:
    values = CHOICE.findall(str(value))
    return values[-1].upper() if values else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference-grading", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    reference = {}
    with args.reference_grading.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("reference_status") == "valid" and row.get("reference_answer") in set("ABCDE"):
                reference[str(row["application_id"])] = row["reference_answer"]
    results = list(read_jsonl(args.results))
    if len({str(row.get("application_id")) for row in results}) != len(results):
        raise ValueError("Duplicate result application_id")
    rows, correct = [], 0
    for result in results:
        app_id = str(result.get("application_id", ""))
        if app_id not in reference:
            continue
        predicted = choice(result.get("answer", ""))
        if not predicted and not invalid_prediction(result):
            raise ValueError(f"Invalid answer format: {app_id}")
        score = int(predicted == reference[app_id])
        correct += score
        rows.append({
            "application_id": app_id, "year": result.get("year"), "unit": result.get("unit"),
            "source_number": result.get("source_number"), "reference_answer": reference[app_id],
            "reference_status": "valid", "model_answer": predicted, "score": score,
            "choice_status": result.get("choice_origin", result.get("choice_status", "")),
        })
    if len(rows) != len(reference):
        raise ValueError(f"Incomplete or mismatched result file: scored={len(rows)} expected={len(reference)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "grading.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "label": args.label, "scored_questions": len(rows), "correct": correct,
        "accuracy_percent": round(correct / len(rows) * 100, 4),
        "invalid_outputs": sum(row["choice_status"] == "invalid_output" for row in rows),
        "reference_grading": str(args.reference_grading),
        "answer_key_used_for_model_input": False,
    }
    (args.output_dir / "grading_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
