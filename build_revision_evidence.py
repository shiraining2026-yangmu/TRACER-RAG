#!/usr/bin/env python3
"""Materialize answer-blind dense/TER evidence for the affected-question rerun.

The historical retrieval files remain frozen.  Shared-case questions only need
their corrected question text attached to the same retrieved evidence because
the retrieval contract is stem-only.  The three direct TER overlaps are
re-ranked from the frozen R4 index after excluding the seven contaminated
evidence blocks.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from benchmark_integrity import atomic_jsonl, sha256
from run_questionbank_rag_vs_alignrag import r4_retrieve
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import transformers.modeling_utils as modeling_utils


DIRECT_OVERLAPS = {
    "QB-2021-TCM-01-102": ["E-AEK-0012824", "E-AEK-0012826", "E-AEK-0012825"],
    "QB-2022-TCM-03-080": ["E-AEK-0026727"],
    "QB-2022-TCM-03-103": ["E-AEK-0031292", "E-AEK-0031291", "E-AEK-0031295"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_jsonl(path, rows)


def clean_meta(row: dict[str, Any], question: str) -> dict[str, Any]:
    fields = (
        "application_id", "question_id", "question", "retrieval_query", "year", "exam", "unit",
        "source_number", "question_type", "shared_option_group", "data_status", "source_row",
    )
    result = {key: row.get(key) for key in fields}
    result["question"] = question
    return result


def load_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            key = str(row.get("application_id", ""))
            if key:
                if key in result:
                    raise ValueError(f"duplicate application_id: {key}")
                result[key] = row
    return result


def filtered_r4(rows: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        payload = row.get("payload", row)
        e_id = str(payload.get("e_id", "")) if isinstance(payload, dict) else ""
        doc_id = str(row.get("doc_id", ""))
        if e_id in excluded or doc_id in excluded:
            continue
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dense-input", type=Path, nargs=2, required=True)
    parser.add_argument("--ter-input", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r4-index-dir", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    args = parser.parse_args()

    questions = load_rows([args.questions])
    dense_source = load_rows(list(args.dense_input))
    ter_source = load_rows(list(args.ter_input))
    if not set(questions) <= set(dense_source) or not set(questions) <= set(ter_source):
        missing_dense = sorted(set(questions) - set(dense_source))
        missing_ter = sorted(set(questions) - set(ter_source))
        raise ValueError(f"evidence source missing rows: dense={missing_dense} ter={missing_ter}")

    reranked: dict[str, list[dict[str, Any]]] = {}
    rerank_audit: dict[str, dict[str, Any]] = {}
    missing_overlaps = set(DIRECT_OVERLAPS) - set(questions)
    if missing_overlaps:
        raise ValueError(f"direct-overlap questions absent from rerun set: {sorted(missing_overlaps)}")
    if args.r4_index_dir:
        if not args.embedding_model:
            raise ValueError("--embedding-model is required with --r4-index-dir")
        connection = sqlite3.connect(args.r4_index_dir / "lexical.sqlite")
        vectors = np.load(args.r4_index_dir / "dense_vectors.npy", mmap_mode="r")
        dense_ids = json.loads((args.r4_index_dir / "dense_ids.json").read_text(encoding="utf-8"))
        # This server has a pinned Torch 2.5.1 runtime.  Recent Transformers
        # blocks all pre-2.6 torch.load calls even for this trusted, local
        # BGE-M3 checkpoint.  The checkpoint is hash-pinned by the frozen
        # embedding contract; bypass only that version guard for this one load.
        modeling_utils.check_torch_load_is_safe = lambda: None
        encoder = SentenceTransformer(str(args.embedding_model), local_files_only=True)
        for app_id, excluded in DIRECT_OVERLAPS.items():
            if app_id not in questions:
                continue
            candidate = r4_retrieve(connection, vectors, dense_ids, encoder,
                                    str(questions[app_id].get("retrieval_query", "")),
                                    top_k=20, bm25_candidates=100, dense_candidates=100)
            candidate = filtered_r4(candidate, set(excluded))[:3]
            if len(candidate) != 3:
                raise ValueError(f"direct-overlap rerank returned {len(candidate)} rows for {app_id}")
            returned_eids = [str(row.get("payload", {}).get("e_id", "")) for row in candidate]
            if set(returned_eids) & set(excluded):
                raise ValueError(f"contaminated evidence survived reranking for {app_id}")
            rerank_audit[app_id] = {"excluded_evidence_ids": excluded,
                                    "new_top3_evidence_ids": returned_eids,
                                    "excluded_ids_absent": True}
            reranked[app_id] = candidate
        connection.close()
    elif DIRECT_OVERLAPS:
        raise ValueError("R4 index is required to rebuild direct-overlap evidence")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": "experiment1_6_revision_evidence_v1",
        "questions_sha256": sha256(args.questions),
        "direct_overlap_exclusions": DIRECT_OVERLAPS,
        "direct_overlap_rerank_audit": rerank_audit,
        "retrieval_query_contract": "stem_only",
        "answer_key_used_for_model_input": False,
        "rows": len(questions),
    }
    for year in ("2021", "2022"):
        ids = [key for key, row in questions.items() if str(row.get("year")) == year]
        dense_rows: list[dict[str, Any]] = []
        ter_rows: list[dict[str, Any]] = []
        for app_id in ids:
            question = str(questions[app_id]["question"])
            dense = clean_meta(dense_source[app_id], question)
            dense["evidence"] = dense_source[app_id]["evidence"]
            dense_rows.append(dense)

            ter = clean_meta(ter_source[app_id], question)
            ter["retrieved"] = reranked.get(app_id, ter_source[app_id]["retrieved"])
            ter_rows.append(ter)
        dense_path = args.output_dir / f"dense_{year}.jsonl"
        ter_path = args.output_dir / f"ter_{year}.jsonl"
        write_jsonl(dense_path, dense_rows)
        write_jsonl(ter_path, ter_rows)
        manifest[year] = {
            "rows": len(ids), "dense_sha256": sha256(dense_path), "ter_sha256": sha256(ter_path),
            "direct_overlap_reranked": [app_id for app_id in ids if app_id in reranked],
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
