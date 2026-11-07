#!/usr/bin/env python3
"""Export a multiple-choice question bank to answer-blind JSONL inputs.

The source workbook is treated as a question-only input. This exporter never
reads or emits answer-key fields, even if a future workbook revision contains
one. The resulting JSONL can therefore be sent to retrieval/generation jobs
without exposing reference answers to either system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import openpyxl


REQUIRED_COLUMNS = (
    "题库ID",
    "年度",
    "考试",
    "单元",
    "原题号",
    "题型",
    "题干",
    "选项A",
    "选项B",
    "选项C",
    "选项D",
    "选项E",
    "共用选项组",
    "数据状态",
)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_question(row: dict[str, Any]) -> str:
    choices = []
    for label in "ABCDE":
        option = clean(row[f"选项{label}"])
        if option:
            choices.append(f"{label}. {option}")
    if len(choices) < 2:
        raise ValueError(f"{row['题库ID']}: fewer than two answer options")
    return "\n".join(
        (
            "以下为单选题。请结合题干、选项和提供的参考资料作答，并在回答结尾明确写出“答案：X”。",
            f"题干：{clean(row['题干'])}",
            "选项：",
            *choices,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbook = openpyxl.load_workbook(args.input, read_only=True, data_only=True)
    if "题库" not in workbook.sheetnames:
        raise ValueError("Workbook is missing sheet: 题库")
    sheet = workbook["题库"]
    headers = [clean(cell.value) for cell in sheet[5]]
    header_positions = {name: index for index, name in enumerate(headers, start=1) if name}
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in header_positions]
    if missing_columns:
        raise ValueError(f"Workbook is missing required question columns: {missing_columns}")

    # Some annual workbooks place answer-key columns between the choices and
    # the source metadata. Group just the approved columns into contiguous
    # ranges so those key columns are never requested from the workbook.
    selected_positions = sorted(header_positions[name] for name in REQUIRED_COLUMNS)
    read_ranges: list[tuple[int, int]] = []
    range_start = range_end = selected_positions[0]
    for position in selected_positions[1:]:
        if position == range_end + 1:
            range_end = position
        else:
            read_ranges.append((range_start, range_end))
            range_start = range_end = position
    read_ranges.append((range_start, range_end))
    row_iterators = [
        sheet.iter_rows(min_row=6, min_col=start, max_col=end, values_only=True)
        for start, end in read_ranges
    ]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    status_counts: dict[str, int] = {}
    for source_row, range_values in enumerate(zip(*row_iterators), start=6):
        raw: dict[str, Any] = {}
        for (start, end), values in zip(read_ranges, range_values):
            for position, value in zip(range(start, end + 1), values):
                header = headers[position - 1]
                if header in REQUIRED_COLUMNS:
                    raw[header] = value
        question_id = clean(raw["题库ID"])
        if not question_id:
            continue
        if question_id in seen:
            raise ValueError(f"Duplicate 题库ID: {question_id}")
        seen.add(question_id)
        stem = clean(raw["题干"])
        if not stem:
            raise ValueError(f"{question_id}: missing question stem")
        status = clean(raw["数据状态"]) or "未标注"
        status_counts[status] = status_counts.get(status, 0) + 1
        try:
            question = build_question(raw)
        except ValueError as error:
            skipped.append(
                {
                    "question_id": question_id,
                    "source_row": source_row,
                    "data_status": status,
                    "reason": str(error),
                }
            )
            continue
        rows.append(
            {
                "application_id": f"QB-{question_id}",
                "question_id": question_id,
                "question": question,
                # Retrieval intentionally receives the stem only. Options are supplied
                # to the generator, but never influence which evidence is recalled.
                "retrieval_query": stem,
                "split": "external_questionbank",
                "year": raw["年度"],
                "exam": clean(raw["考试"]),
                "unit": clean(raw["单元"]),
                "source_number": raw["原题号"],
                "question_type": clean(raw["题型"]),
                "shared_option_group": clean(raw["共用选项组"]),
                "data_status": status,
                "source_row": source_row,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "source_workbook": str(args.input),
                "source_sha256": sha256_file(args.input),
                "source_question_count": len(rows) + len(skipped),
                "eligible_question_count": len(rows),
                "skipped_question_count": len(skipped),
                "status_counts": status_counts,
                "skipped_questions": skipped,
                "contract": "Question-only export: no answer-key fields are read or emitted.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"eligible_question_count": len(rows), "skipped_question_count": len(skipped), "status_counts": status_counts},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
