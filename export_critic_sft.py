#!/usr/bin/env python3
"""Validate teacher critiques and export leakage-free Qwen Critic SFT data.

The exported student input contains only question, retrieved R4 evidence, and
weak answer. y_strong is deliberately excluded after it has helped construct
the teacher critique. All accepted records must be Build-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


EXPORT_VERSION = "tcm-alignrag-critic-sft-v2"
ALLOWED_VERDICTS = {"needs_revision", "acceptable", "insufficient_evidence"}
BALANCE_VERDICTS = ("needs_revision", "acceptable", "insufficient_evidence")
ALLOWED_ISSUE_TYPES = {
    "unsupported_claim",
    "historical_case_as_current_fact",
    "missing_constraint",
    "overstatement",
    "not_answering_question",
    "evidence_misuse",
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            yield row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_partition(application_id: str, validation_percent: int) -> str:
    digest = hashlib.sha256(f"{EXPORT_VERSION}:{application_id}".encode("utf-8")).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 100 < validation_percent else "train"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Teacher critique is not a JSON object")
    return value


def clean_string(value: Any, name: str, minimum: int = 1, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not (minimum <= len(value) <= maximum):
        raise ValueError(f"{name} length must be in [{minimum}, {maximum}]")
    return value


def compact_string(value: str, maximum: int) -> str:
    value = value.strip()
    return value if len(value) <= maximum else value[: maximum - 3].rstrip() + "..."


def validate_critique(value: dict[str, Any], evidence_count: int) -> dict[str, Any]:
    verdict = clean_string(value.get("verdict"), "verdict", maximum=64)
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict}")
    issues = value.get("issues")
    if not isinstance(issues, list) or len(issues) > 8:
        raise ValueError("issues must be a list containing at most 8 items")
    cleaned_issues = []
    for index, issue in enumerate(issues, 1):
        if not isinstance(issue, dict):
            raise ValueError(f"issues[{index}] must be an object")
        issue_type = clean_string(issue.get("type"), f"issues[{index}].type", maximum=64)
        if issue_type not in ALLOWED_ISSUE_TYPES:
            raise ValueError(f"Invalid issue type: {issue_type}")
        evidence_ids = issue.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
            raise ValueError(f"issues[{index}].evidence_ids must be a string list")
        if any(not re.fullmatch(r"E[1-9][0-9]*", item) for item in evidence_ids):
            raise ValueError(f"issues[{index}].evidence_ids contains an invalid ID")
        if any(int(item[1:]) > evidence_count for item in evidence_ids):
            raise ValueError(f"issues[{index}].evidence_ids exceeds retrieved evidence count")
        cleaned_issues.append(
            {
                "type": issue_type,
                "claim_or_location": clean_string(issue.get("claim_or_location"), f"issues[{index}].claim_or_location"),
                "why": clean_string(issue.get("why"), f"issues[{index}].why"),
                "evidence_ids": evidence_ids,
                "fix": clean_string(issue.get("fix"), f"issues[{index}].fix"),
            }
        )
    instructions = value.get("revision_instructions")
    if not isinstance(instructions, list) or not 1 <= len(instructions) <= 8:
        raise ValueError("revision_instructions must contain 1 to 8 items")
    cleaned_instructions = [clean_string(item, "revision_instruction") for item in instructions]
    if verdict == "needs_revision" and not cleaned_issues:
        raise ValueError("needs_revision requires at least one issue")
    return {
        "verdict": verdict,
        "issues": cleaned_issues,
        "revision_instructions": cleaned_instructions,
        "evidence_boundary": clean_string(value.get("evidence_boundary"), "evidence_boundary"),
    }


def compact_critique(value: dict[str, Any]) -> dict[str, Any]:
    """Keep supervision comfortably inside the fixed inference JSON budget."""
    issues = []
    for issue in value["issues"][:3]:
        issues.append(
            {
                "type": issue["type"],
                "claim_or_location": compact_string(issue["claim_or_location"], 160),
                "why": compact_string(issue["why"], 180),
                "evidence_ids": issue["evidence_ids"][:3],
                "fix": compact_string(issue["fix"], 180),
            }
        )
    return {
        "verdict": value["verdict"],
        "issues": issues,
        "revision_instructions": [compact_string(item, 180) for item in value["revision_instructions"][:3]],
        "evidence_boundary": compact_string(value["evidence_boundary"], 360),
    }


def critic_system_message() -> str:
    return (
        "你是中医领域 RAG 系统的证据一致性 Critic。仅根据给定的 R4 证据审查初始回答。"
        "K 知识可作为可复用依据；历史问答仅可作为相似案例参考，不能写成当前对象的确定事实。"
        "只有存在可定位的问题时才使用 needs_revision。若回答已忠实且回答了可回答部分，"
        "应使用 acceptable；不能仅因不能确诊而判为 insufficient_evidence。"
        "必须只输出一个紧凑合法的 JSON 对象，不要 Markdown 或解释文字。"
        "字段必须且只能为 verdict、issues、revision_instructions、evidence_boundary；"
        "issues 最多三项，revision_instructions 为一至三项。"
    )


def critic_user_message(question: str, evidence: str, weak_answer: str) -> str:
    schema = (
        '{"verdict":"needs_revision|acceptable|insufficient_evidence",'
        '"issues":[{"type":"...","claim_or_location":"...","why":"...",'
        '"evidence_ids":["E1"],"fix":"..."}],'
        '"revision_instructions":["..."],"evidence_boundary":"..."}'
    )
    return (
        f"问题：{question}\n\nR4 检索证据：\n{evidence}\n\n"
        f"初始回答 y_weak：\n{weak_answer}\n\n"
        f"按此 JSON 结构输出：{schema}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Critique-phase JSONL from generate_critic_supervision.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument(
        "--balance-train-per-verdict",
        type=int,
        default=0,
        help="When positive, deterministically resample each verdict to this training count; validation remains natural.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.validation_percent < 50:
        raise ValueError("--validation-percent must be between 1 and 49")
    if args.balance_train_per_verdict < 0:
        raise ValueError("--balance-train-per-verdict cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to write into non-empty {args.output_dir}; use --overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_fingerprints = set()
    reason_counts = Counter()
    for row in read_jsonl(args.input):
        sample_id = str(row.get("sample_id", ""))
        try:
            if row.get("generation_phase") != "critique" or row.get("dry_run"):
                raise ValueError("Input must contain completed critique-phase records, not dry runs")
            if row.get("split") != "build":
                raise ValueError("Only Build records may enter Critic SFT")
            question = clean_string(row.get("question"), "question", maximum=4000)
            evidence = clean_string(row.get("evidence"), "evidence", maximum=12000)
            weak_answer = clean_string(row.get("weak_answer"), "weak_answer", maximum=5000)
            if "answer" in row:
                raise ValueError("Source answer field is forbidden in critic supervision")
            evidence_count = evidence.count("[E")
            if not 1 <= evidence_count <= 3:
                raise ValueError("Expected 1 to 3 formatted R4 evidence blocks")
            critique = compact_critique(
                validate_critique(
                    extract_json(clean_string(row.get("teacher_critique"), "teacher_critique", maximum=10000)),
                    evidence_count,
                )
            )
            fingerprint = hashlib.sha256((question + "\n" + evidence + "\n" + weak_answer).encode("utf-8")).hexdigest()
            if fingerprint in seen_fingerprints:
                raise ValueError("Duplicate question/evidence/weak-answer supervision record")
            seen_fingerprints.add(fingerprint)
            partition = stable_partition(str(row["application_id"]), args.validation_percent)
            assistant = json.dumps(critique, ensure_ascii=False, separators=(",", ":"))
            messages = [
                {"role": "system", "content": critic_system_message()},
                {"role": "user", "content": critic_user_message(question, evidence, weak_answer)},
                {"role": "assistant", "content": assistant},
            ]
            accepted.append(
                {
                    "sample_id": sample_id,
                    "application_id": row["application_id"],
                    "source_split": "build",
                    "partition": partition,
                    "messages": messages,
                    "prompt_version": row.get("prompt_version"),
                    "export_version": EXPORT_VERSION,
                }
            )
        except Exception as exc:
            reason = str(exc)
            reason_counts[reason] += 1
            rejected.append({"sample_id": sample_id, "reason": reason})

    if len(accepted) < args.min_accepted:
        raise RuntimeError(f"Only {len(accepted)} records passed validation; expected at least {args.min_accepted}")
    accepted.sort(key=lambda row: row["sample_id"])
    train_rows = [row for row in accepted if row["partition"] == "train"]
    validation_rows = [row for row in accepted if row["partition"] == "validation"]
    if not train_rows or not validation_rows:
        raise RuntimeError("Need non-empty train and validation partitions")

    natural_train_verdicts = Counter(json.loads(row["messages"][-1]["content"])["verdict"] for row in train_rows)
    if args.balance_train_per_verdict:
        by_verdict = {
            verdict: [row for row in train_rows if json.loads(row["messages"][-1]["content"])["verdict"] == verdict]
            for verdict in BALANCE_VERDICTS
        }
        if any(not by_verdict[verdict] for verdict in BALANCE_VERDICTS):
            raise RuntimeError("Cannot balance because at least one verdict has no training samples")
        balanced_rows = []
        for verdict in BALANCE_VERDICTS:
            source_rows = by_verdict[verdict]
            for index in range(args.balance_train_per_verdict):
                source = source_rows[index % len(source_rows)]
                replica = dict(source)
                replica["sample_id"] = f"{source['sample_id']}::balance::{index // len(source_rows)}"
                replica["balance_source_sample_id"] = source["sample_id"]
                balanced_rows.append(replica)
        train_rows = sorted(
            balanced_rows,
            key=lambda row: hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(),
        )

    train_verdicts = Counter(json.loads(row["messages"][-1]["content"])["verdict"] for row in train_rows)

    write_jsonl(args.output_dir / "critic_sft_train.jsonl", train_rows)
    write_jsonl(args.output_dir / "critic_sft_validation.jsonl", validation_rows)
    write_jsonl(args.output_dir / "critic_sft_rejected.jsonl", rejected)
    manifest = {
        "export_version": EXPORT_VERSION,
        "input": str(args.input),
        "accepted": len(accepted),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "natural_train_verdicts": dict(natural_train_verdicts),
        "train_verdicts": dict(train_verdicts),
        "balance_train_per_verdict": args.balance_train_per_verdict or None,
        "rejected": len(rejected),
        "rejection_reasons": dict(reason_counts.most_common()),
        "student_input_contract": "question + R4 evidence + y_weak only; y_strong and source answer are excluded",
        "source_split_contract": "build only",
        "validation_percent": args.validation_percent,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
