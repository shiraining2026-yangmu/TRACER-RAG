#!/usr/bin/env python3
"""Select immutable R4 retrieval rows for Build-only Critic seeds.

This utility reuses a prior, hash-verified R4 retrieval run without copying its
target-answer fields. Each output row contains only the Critic seed metadata and
the fixed Top-k retrieved evidence required by the later supervision stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


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
                raise ValueError(f"Expected object at {path}:{number}")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(value: Any) -> str:
    return "".join(str(value or "").split())


def validate_retrieved(row: dict[str, Any], application_id: str, top_k: int) -> list[dict[str, Any]]:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or len(retrieved) != top_k:
        raise ValueError(f"{application_id}: expected exactly {top_k} retrieved evidence rows")
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"{application_id}: evidence {rank} has no payload")
        if str(payload.get("application_id", "")) == application_id:
            raise ValueError(f"{application_id}: self-retrieval in evidence {rank}")
        required = ("e_id", "application_id", "knowledge_id", "relation", "application_question", "application_answer", "knowledge_text")
        missing = [field for field in required if not str(payload.get(field, "")).strip()]
        if missing:
            raise ValueError(f"{application_id}: evidence {rank} missing {', '.join(missing)}")
    return retrieved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--r4-retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    seeds = list(read_jsonl(args.seeds))
    if not seeds:
        raise ValueError("Seed file is empty")
    by_application: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        application_id = str(seed.get("application_id", "")).strip()
        sample_id = str(seed.get("sample_id", "")).strip()
        question = normalize_question(seed.get("question"))
        if seed.get("split") != "build" or not application_id or not sample_id or not question:
            raise ValueError("Seeds must have Build split, application_id, sample_id, and non-empty question")
        if "answer" in seed:
            raise ValueError(f"Seed {sample_id} contains forbidden answer field")
        if application_id in by_application:
            raise ValueError(f"Duplicate seed application_id: {application_id}")
        by_application[application_id] = seed

    selected: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(args.r4_retrieval):
        application_id = str(row.get("application_id", "")).strip()
        seed = by_application.get(application_id)
        if seed is None:
            continue
        if application_id in selected:
            raise ValueError(f"Duplicate retrieval row for {application_id}")
        if normalize_question(row.get("question")) != normalize_question(seed.get("question")):
            raise ValueError(f"Question mismatch for {application_id}")
        selected[application_id] = validate_retrieved(row, application_id, args.top_k)

    missing = [application_id for application_id in by_application if application_id not in selected]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"R4 retrieval source misses {len(missing)} seed rows: {preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for seed in seeds:
            application_id = str(seed["application_id"])
            result = dict(seed)
            result["retrieved"] = selected[application_id]
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "contract": "Build-only seeds plus immutable R4 Top-k evidence; target answer excluded.",
        "count": len(seeds),
        "top_k": args.top_k,
        "seed_sha256": sha256_file(args.seeds),
        "r4_retrieval_sha256": sha256_file(args.r4_retrieval),
        "output": str(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
