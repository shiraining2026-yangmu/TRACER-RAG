#!/usr/bin/env python3
"""Validate and export q,D(q)-only EviCR supervision for QLoRA SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from generate_evicr_qd_labels import FIELDS, validate_package


VERSION = "evicr-qd-sft-v1"
FORBIDDEN = ("y0", "y_weak", "y_strong", "weak_answer", "strong_answer", "初始回答", "弱回答", "强回答")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition(application_id: str, validation_percent: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"{VERSION}:{application_id}".encode()).digest()[:8], "big")
    return "validation" if value % 100 < validation_percent else "train"


def system_message() -> str:
    return (
        "你是中医 RAG 的关系证据 Critic。只审查问题和 R4 检索证据；输入仅包含这两类材料。"
        "逐条判断 E1-E3 的相关性、适用条件和支持范围。K 知识可作为可复用依据，历史案例只能作为相似案例，"
        "不能写作当前对象事实。最多给出两条模型知识补充，且必须明确标为 model_knowledge，不能冒充 R4 证据。"
        "不直接作答。只输出一个合法 JSON 对象，不要 Markdown 或额外字段。"
    )


def user_message(question: str, evidence: str) -> str:
    schema = (
        '{"evidence_status":"supported|partial|insufficient|conflict",'
        '"evidence_actions":[{"evidence_id":"E1","action":"keep|downweight|discard","reason":"..."}],'
        '"knowledge_supplements":[{"claim":"...","basis":"model_knowledge","confidence":"high|medium|low"}],'
        '"change_support":"none|weak|strong","answer_guidance":"..."}'
    )
    return f"问题：\n{question}\n\nR4 检索证据：\n{evidence}\n\n输出：\n{schema}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.validation_percent < 50:
        raise ValueError("--validation-percent must be in [1, 49]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    for number, row in enumerate(read_jsonl(args.labels), 1):
        sample_id = str(row.get("sample_id", "")).strip()
        try:
            if row.get("schema_version") != "evicr-qd-labels-v1" or row.get("split") != "build":
                raise ValueError("not a Build evicr-qd label")
            if row.get("status") not in {"valid", "repaired"}:
                raise ValueError("invalid teacher label status")
            if not sample_id or sample_id in accepted:
                raise ValueError("missing or duplicate sample_id")
            question = str(row.get("question", "")).strip()
            evidence = str(row.get("evidence", "")).strip()
            if not question or not evidence:
                raise ValueError("blank question or evidence")
            prompt = user_message(question, evidence)
            combined = system_message() + "\n" + prompt
            lower = combined.lower()
            if any(token.lower() in lower for token in FORBIDDEN):
                raise ValueError("student input has an answer-like token")
            evidence_ids = row.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 3:
                raise ValueError("invalid evidence IDs")
            label = row.get("label")
            if not isinstance(label, dict) or set(label) != FIELDS:
                raise ValueError("invalid five-field label")
            label = validate_package(label, len(evidence_ids))
            fingerprint = hashlib.sha256((question + "\n" + evidence).encode("utf-8")).hexdigest()
            if any(item["qd_fingerprint"] == fingerprint for item in accepted.values()):
                raise ValueError("duplicate question/evidence fingerprint")
            accepted[sample_id] = {
                "sample_id": sample_id,
                "application_id": str(row.get("application_id", "")).strip(),
                "qd_fingerprint": fingerprint,
                "source_split": "build",
                "messages": [
                    {"role": "system", "content": system_message()},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(label, ensure_ascii=False, separators=(",", ":"))},
                ],
            }
        except (TypeError, ValueError) as exc:
            reason = str(exc)
            reasons[reason] += 1
            rejected.append({"line": str(number), "sample_id": sample_id, "reason": reason})
    if len(accepted) < args.min_accepted:
        raise RuntimeError(f"Only {len(accepted)} valid labels; need {args.min_accepted}")
    train, validation = [], []
    for row in sorted(accepted.values(), key=lambda value: value["sample_id"]):
        target = validation if partition(row["application_id"], args.validation_percent) == "validation" else train
        target.append(row)
    if not train or not validation:
        raise RuntimeError("deterministic split produced an empty partition")
    write_jsonl(args.output_dir / "critic_sft_train.jsonl", train)
    write_jsonl(args.output_dir / "critic_sft_validation.jsonl", validation)
    write_jsonl(args.output_dir / "critic_sft_rejected.jsonl", rejected)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "labels_sha256": sha256_file(args.labels),
        "n_accepted": len(accepted),
        "n_train": len(train),
        "n_validation": len(validation),
        "rejected": dict(sorted(reasons.items())),
        "student_input_contract": "system + question + R4 evidence only",
        "student_reads_y0": False,
        "student_reads_weak_or_strong_answer": False,
        "fingerprint": "sha256(question + newline + formatted R4 evidence)",
        "split": "deterministic application_id hash; Build only",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
