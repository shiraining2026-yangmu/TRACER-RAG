#!/usr/bin/env python3
"""Grade answer-blind question-bank outputs against OCR Markdown answer keys.

The keys are matched by (unit, source_number), because every exam unit restarts
its question numbering. OCR anomalies are never silently corrected: missing,
invalid, and contradictory keys are retained in the audit report as unscorable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


ANSWER_SECTION = re.compile(r"(?:^|\n)\s*#{1,6}\s*答案及解析\s*$", re.MULTILINE)
UNIT_HEADING = re.compile(r"^\s*#{0,6}\s*(第[一二三四五六七八九十]+)单元\s*$")
ANSWER_ENTRY = re.compile(r"(?<!\d)(\d{1,3})\s*\.?\s*【\s*答案\s*】\s*([A-Za-z])")
MODEL_CHOICE = re.compile(r"答案\s*[：:]\s*([A-E])\b", re.IGNORECASE)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def parse_answer_key(path: Path) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str], dict[str, int]]:
    """Return valid keys, key issues, and audit counts from the final answer section."""
    markdown = path.read_text(encoding="utf-8")
    matches = list(ANSWER_SECTION.finditer(markdown))
    if not matches:
        raise ValueError(f"No Markdown answer section heading in {path}")
    section = markdown[matches[-1].end():]
    seen: dict[tuple[str, int], list[str]] = defaultdict(list)
    active_unit: str | None = None
    for line in section.splitlines():
        heading = UNIT_HEADING.match(line)
        if heading:
            active_unit = heading.group(1)
            continue
        if active_unit is None:
            continue
        for number, choice in ANSWER_ENTRY.findall(line):
            seen[(active_unit, int(number))].append(choice.upper())

    valid: dict[tuple[str, int], str] = {}
    issues: dict[tuple[str, int], str] = {}
    for key, values in seen.items():
        invalid = sorted({value for value in values if value not in "ABCDE"})
        choices = sorted({value for value in values if value in "ABCDE"})
        if invalid:
            issues[key] = "invalid_key:" + "/".join(invalid)
        elif len(choices) == 1:
            valid[key] = choices[0]
        elif not choices:
            issues[key] = "missing_key_choice"
        else:
            issues[key] = "conflicting_keys:" + "/".join(choices)
    audit = {
        "parsed_entries": sum(len(values) for values in seen.values()),
        "distinct_key_positions": len(seen),
        "valid_key_positions": len(valid),
        "invalid_or_conflicting_key_positions": len(issues),
    }
    return valid, issues, audit


def extract_model_choice(value: object) -> str | None:
    matches = MODEL_CHOICE.findall(str(value))
    return matches[-1].upper() if matches else None


def normalize_unit(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith("单元") else text


def grade(result_path: Path, keys: dict[tuple[str, int], str], issues: dict[tuple[str, int], str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in read_jsonl(result_path):
        source_number = result.get("source_number")
        try:
            key_position = (normalize_unit(result.get("unit", "")), int(source_number))
        except (TypeError, ValueError):
            key_position = (normalize_unit(result.get("unit", "")), -1)
        reference = keys.get(key_position)
        key_status = "valid" if reference else issues.get(key_position, "missing_key")
        choice = extract_model_choice(result.get("answer", ""))
        score = int(reference is not None and choice == reference)
        rows.append(
            {
                "application_id": result.get("application_id", ""),
                "question_id": result.get("question_id", ""),
                "unit": key_position[0],
                "source_number": key_position[1],
                "data_status": result.get("data_status", ""),
                "reference_answer": reference or "",
                "reference_status": key_status,
                "model_answer": choice or "",
                "choice_status": result.get("choice_status", "missing"),
                "score": score,
            }
        )
    total = len(rows)
    valid_rows = [row for row in rows if row["reference_status"] == "valid"]
    correct = sum(row["score"] for row in rows)
    summary = {
        "result_file": str(result_path),
        "generated_questions": total,
        "valid_key_matched_questions": len(valid_rows),
        "unscorable_key_questions": total - len(valid_rows),
        "model_a_to_e_choices": sum(bool(row["model_answer"]) for row in rows),
        "no_model_choice": sum(not row["model_answer"] for row in rows),
        "correct": correct,
        "score_rate_all_generated": round(correct / total * 100, 2) if total else 0.0,
        "score_rate_valid_key_matched": round(correct / len(valid_rows) * 100, 2) if valid_rows else 0.0,
        "choice_status_counts": dict(sorted(Counter(str(row["choice_status"]) for row in rows).items())),
        "unscorable_reason_counts": dict(
            sorted(Counter(str(row["reference_status"]) for row in rows if row["reference_status"] != "valid").items())
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "application_id", "question_id", "unit", "source_number", "data_status",
        "reference_answer", "reference_status", "model_answer", "choice_status", "score",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key-md", type=Path, required=True)
    parser.add_argument("--plain-results", type=Path, required=True)
    parser.add_argument("--alignrag-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.answer_key_md, args.plain_results, args.alignrag_results):
        if not path.is_file():
            raise FileNotFoundError(path)
    keys, issues, key_audit = parse_answer_key(args.answer_key_md)
    plain_rows, plain_summary = grade(args.plain_results, keys, issues)
    align_rows, align_summary = grade(args.alignrag_results, keys, issues)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "plain_rag_grading.csv", plain_rows)
    write_csv(args.output_dir / "alignrag_grading.csv", align_rows)
    report = {"answer_key": {"path": str(args.answer_key_md), **key_audit}, "plain_rag": plain_summary, "alignrag": align_summary}
    (args.output_dir / "grading_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
