#!/usr/bin/env python3
"""Construct Build-only q,D(q) seeds for the EviCR Critic.

The legacy source may contain weak answers because it was produced by an older
pipeline.  This program reads only the Build question and retrieved R4 evidence
and deliberately drops every answer-like field before writing the new dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


VERSION = "evicr-qd-seed-v1"
FORBIDDEN_KEYS = {
    "weak_answer", "strong_answer", "teacher_critique", "answer", "y0", "y_weak", "y_strong"
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact(value: Any, maximum: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= maximum else text[: maximum - 3].rstrip() + "..."


def format_evidence(row: dict[str, Any], field_limit: int) -> tuple[str, list[str]]:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or not 1 <= len(retrieved) <= 3:
        raise ValueError("retrieved must contain one to three R4 records")
    application_id = str(row.get("application_id", "")).strip()
    blocks: list[str] = []
    edge_ids: list[str] = []
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"retrieved[{rank}] has no payload")
        if str(payload.get("application_id", "")).strip() == application_id:
            raise ValueError("self retrieval is forbidden")
        edge_id = compact(payload.get("e_id") or item.get("doc_id"), 120)
        relation = compact(payload.get("relation"), 120)
        historical_question = compact(payload.get("application_question"), field_limit)
        historical_answer = compact(payload.get("application_answer"), field_limit)
        knowledge = compact(payload.get("knowledge_text"), field_limit)
        if not edge_id or not relation or not knowledge:
            raise ValueError(f"retrieved[{rank}] lacks e_id, relation, or knowledge_text")
        edge_ids.append(edge_id)
        blocks.append(
            "\n".join(
                (
                    f"[E{rank} | {edge_id} | {relation}]",
                    f"历史问题：{historical_question}",
                    f"历史回答：{historical_answer}",
                    f"K 知识：{knowledge}",
                )
            )
        )
    return "\n\n".join(blocks), edge_ids


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Legacy Build retrieval JSONL; answer fields are discarded.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field-limit", type=int, default=850)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.field_limit < 100:
        raise ValueError("--field-limit must be at least 100")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for source_index, row in enumerate(read_jsonl(args.source), 1):
        if row.get("split") != "build":
            rejected["not_build"] += 1
            continue
        application_id = str(row.get("application_id", "")).strip()
        question = str(row.get("question", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        if not application_id or not question or not sample_id:
            rejected["missing_identifier_or_question"] += 1
            continue
        try:
            evidence, edge_ids = format_evidence(row, args.field_limit)
        except ValueError as exc:
            rejected[f"invalid_evidence:{str(exc)}"] += 1
            continue
        fingerprint = digest_text(question + "\n" + evidence)
        if fingerprint in seen:
            rejected["duplicate_question_evidence"] += 1
            continue
        seen.add(fingerprint)
        seed = {
            "schema_version": VERSION,
            "sample_id": sample_id,
            "application_id": application_id,
            "source_index": source_index,
            "split": "build",
            "question": question,
            "evidence": evidence,
            "evidence_ids": edge_ids,
            "qd_fingerprint": fingerprint,
        }
        if FORBIDDEN_KEYS & set(seed):
            raise AssertionError("Answer-like fields reached the q,D(q) seed")
        seeds.append(seed)
        if args.limit is not None and len(seeds) >= args.limit:
            break
    if not seeds:
        raise RuntimeError("No eligible Build q,D(q) seeds")

    write_jsonl(args.output_dir / "build_qd_seeds.jsonl", seeds)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "n_seeds": len(seeds),
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "source_split_contract": "Build only",
        "student_and_teacher_input_contract": "question + R4 evidence only",
        "dropped_source_fields": "all answer-like source fields, including legacy weak_answer",
        "evidence_field_limit": args.field_limit,
        "rejected": dict(sorted(rejected.items())),
    }
    (args.output_dir / "seed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
