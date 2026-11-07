"""Durable benchmark output and exact-ID validation (no model dependencies)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

ANSWER = re.compile(r"答案：[A-E]")


def invalid_prediction(row: dict) -> bool:
    return row.get("answer") == "INVALID" and row.get("choice_origin", row.get("choice_status")) == "invalid_output"


def completed_prediction(row: dict) -> bool:
    return bool(ANSWER.fullmatch(str(row.get("answer", "")))) or invalid_prediction(row)


def freeze_protocol(path: Path, contract: dict, output: Path) -> None:
    """Commit settings before generation, and reject mixed-protocol resumes."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Generation protocol changed; use a new output directory")
        return
    if output.exists() and output.stat().st_size:
        raise ValueError("Existing predictions lack a committed protocol; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = str(row.get("application_id", ""))
        if not key or key in result:
            raise ValueError("Missing or duplicate application_id")
        result[key] = row
    return result


def validate_predictions(rows: list[dict], expected: dict[str, dict]) -> dict[str, dict]:
    result = keyed(rows)
    if set(result) != set(expected):
        raise ValueError(f"Incomplete/mismatched IDs: actual={len(result)} expected={len(expected)}")
    for key, row in result.items():
        if row.get("question") != expected[key]["question"] or not completed_prediction(row):
            raise ValueError(f"Question mismatch or invalid answer: {key}")
    return result


def merge_wrong_only(original: list[dict], replacement: list[dict], grading: list[dict]) -> list[dict]:
    source = keyed(original)
    wrong = {str(row["application_id"]) for row in grading
             if row.get("reference_status") == "valid" and str(row.get("score")) == "0"}
    if not wrong <= set(source):
        raise ValueError("Grading IDs absent from source")
    target = {key: source[key] for key in wrong}
    updates = validate_predictions(replacement, target)
    for key, row in updates.items():
        if row.get("retrieved") != source[key].get("retrieved") or row.get("initial_answer") != source[key]["answer"]:
            raise ValueError("Refinement changed frozen evidence or initial answer")
        if row.get("critic_status") not in {"valid", "repaired", "fallback"}:
            raise ValueError("Incomplete Critic output")
    return [updates.get(str(row["application_id"]), row) for row in original]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
