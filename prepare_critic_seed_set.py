#!/usr/bin/env python3
"""Select a reproducible Build-only query set for AlignRAG critic supervision.

The selected JSONL intentionally does not carry the original application
answer. The answer is a source-side artifact and must not enter critic prompt
construction. R4 retrieval will later provide only non-self Build E evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


SELECTION_VERSION = "tcm-alignrag-critic-seed-v1"


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, application_id: str) -> str:
    return hashlib.sha256(f"{seed}:{application_id}".encode("utf-8")).hexdigest()


def normalized_question(question: str) -> str:
    return re.sub(r"\s+", "", question).strip()


def largest_remainder_quotas(bucket_sizes: dict[str, int], total: int) -> dict[str, int]:
    available = sum(bucket_sizes.values())
    if total > available:
        raise ValueError(f"Requested {total} seeds but only {available} eligible unique questions exist")
    quotas = {name: (size * total) // available for name, size in bucket_sizes.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        bucket_sizes,
        key=lambda name: (-((bucket_sizes[name] * total) % available), name),
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applications", type=Path, required=True)
    parser.add_argument("--build-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--min-question-chars", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --overwrite explicitly")

    relations_by_application: dict[str, set[str]] = defaultdict(set)
    for edge in read_jsonl(args.build_edges):
        if edge.get("split") != "build":
            raise ValueError(f"Non-Build edge in Build edge file: {edge.get('e_id')}")
        relations_by_application[str(edge["application_id"])].add(str(edge["relation"]))

    # Collapse atomic fragments that share an identical original question. One
    # representative application is selected deterministically for each query.
    candidates_by_question: dict[str, dict[str, Any]] = {}
    excluded = Counter()
    for app in read_jsonl(args.applications):
        if app.get("release", {}).get("split") != "build":
            continue
        application_id = str(app.get("application_id", ""))
        question = str(app.get("question", "")).strip()
        if not application_id or len(normalized_question(question)) < args.min_question_chars:
            excluded["short_or_missing_question"] += 1
            continue
        relation_types = sorted(relations_by_application.get(application_id, set()))
        if not relation_types:
            excluded["without_build_edge"] += 1
            continue
        candidate = {
            "application_id": application_id,
            "question": question,
            "source_dataset": str(app.get("source", {}).get("dataset", "unknown")),
            "source_record_hash": str(app.get("source", {}).get("record_hash", "")),
            "relation_types": relation_types,
        }
        key = normalized_question(question)
        previous = candidates_by_question.get(key)
        if previous is None or candidate["application_id"] < previous["application_id"]:
            candidates_by_question[key] = candidate

    candidates = list(candidates_by_question.values())
    if len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} eligible unique Build questions, need {args.count}")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        signature = "+".join(candidate["relation_types"])
        bucket = f"{candidate['source_dataset']}|{signature}"
        buckets[bucket].append(candidate)
    quotas = largest_remainder_quotas({name: len(rows) for name, rows in buckets.items()}, args.count)

    selected: list[dict[str, Any]] = []
    for bucket in sorted(buckets):
        rows = sorted(
            buckets[bucket],
            key=lambda row: (stable_rank(args.seed, row["application_id"]), row["application_id"]),
        )
        selected.extend(rows[:quotas[bucket]])
    selected.sort(key=lambda row: (stable_rank(args.seed, row["application_id"]), row["application_id"]))
    if len(selected) != args.count:
        raise AssertionError(f"Selected {len(selected)}, expected {args.count}")

    output_rows = []
    for index, row in enumerate(selected, 1):
        output_rows.append(
            {
                "sample_id": f"CRITIC-BUILD-{index:05d}",
                "application_id": row["application_id"],
                "question": row["question"],
                "split": "build",
                "source_dataset": row["source_dataset"],
                "source_record_hash": row["source_record_hash"],
                "relation_types": row["relation_types"],
                "selection_version": SELECTION_VERSION,
            }
        )
    if any("answer" in row for row in output_rows):
        raise AssertionError("Seed output must not contain source answers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    relation_counter = Counter(relation for row in output_rows for relation in row["relation_types"])
    dataset_counter = Counter(row["source_dataset"] for row in output_rows)
    manifest = {
        "selection_version": SELECTION_VERSION,
        "selection_contract": "Build applications only; one representative per normalized question; no source answer field.",
        "selected_count": len(output_rows),
        "eligible_unique_build_questions": len(candidates),
        "source_dataset_counts": dict(sorted(dataset_counter.items())),
        "relation_type_counts": dict(sorted(relation_counter.items())),
        "bucket_quotas": dict(sorted(quotas.items())),
        "excluded_application_counts": dict(sorted(excluded.items())),
        "seed": args.seed,
        "source_sha256": {
            "applications": sha256_file(args.applications),
            "build_edges": sha256_file(args.build_edges),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
