#!/usr/bin/env python3
"""Run an isolated AEK R4 hybrid-retrieval experiment.

The run combines global BM25 and BGE-M3 dense retrieval.  Relation types are
soft ranking features only: they never exclude a candidate.  Build A answers
remain available as historical evidence, matching the original AEK contract.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from sentence_transformers import SentenceTransformer


ROUTES = {
    "diagnostic_support": ("\u8fa8\u8bc1", "\u8bc1\u578b", "\u8bc1\u5019", "\u8bca\u65ad", "\u5224\u65ad", "\u9274\u522b", "\u5c5e\u4e8e\u4ec0\u4e48\u8bc1", "\u75c5\u673a"),
    "treatment_or_operation_support": ("\u6cbb\u7597", "\u6cbb\u6cd5", "\u6cbb\u5219", "\u600e\u4e48\u529e", "\u5982\u4f55\u6cbb", "\u7528\u836f", "\u65b9\u5242", "\u63a8\u8350", "\u5efa\u8bae", "\u9488\u7078", "\u827e\u7078", "\u8c03\u7406", "\u64cd\u4f5c"),
    "explanatory_support": ("\u4e3a\u4ec0\u4e48", "\u539f\u56e0", "\u542b\u4e49", "\u662f\u4ec0\u4e48", "\u5982\u4f55\u7406\u89e3", "\u673a\u5236", "\u89e3\u91ca"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from error


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: str, limit: int) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
            if len(run) == 1:
                tokens.append(run)
        else:
            tokens.append(run)
    return tokens


def fts_query(text: str) -> str:
    values = list(dict.fromkeys(tokenize(text)))
    if not values:
        raise ValueError(f"No retrieval terms in {text!r}")
    return " OR ".join(f'"{value}"' for value in values[:96])


def char_bigrams(text: str) -> set[str]:
    compacted = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
    return {compacted[index:index + 2] for index in range(max(0, len(compacted) - 1))}


def overlap(left: str, right: str) -> float:
    a, b = char_bigrams(left), char_bigrams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def route_scores(question: str) -> dict[str, float]:
    counts = {relation: sum(marker in question for marker in markers) for relation, markers in ROUTES.items()}
    maximum = max(counts.values(), default=0)
    if not maximum:
        return {relation: 0.0 for relation in ROUTES}
    return {relation: count / maximum for relation, count in counts.items()}


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def dense_text(payload: dict[str, Any]) -> str:
    return "\n".join((
        "\u5386\u53f2\u95ee\u9898: " + compact(payload["application_question"], 600),
        "\u5386\u53f2\u56de\u7b54: " + compact(payload["application_answer"], 600),
        "K\u77e5\u8bc6: " + compact(payload["knowledge_text"], 900),
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--build-edges", type=Path, required=True)
    parser.add_argument("--applications", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/home/ubuntu/tcm_rag/models/Qwen3.5-9B")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bm25-candidates", type=int, default=100)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--encoding-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--evidence-char-limit", type=int, default=1000)
    parser.add_argument("--split", choices=("build", "dev", "test"), default="test")
    parser.add_argument("--phase", choices=("retrieve", "full", "generate"), default="full")
    parser.add_argument("--retrieval-results", type=Path, help="Required for --phase generate.")
    parser.add_argument("--rerank-weights", type=float, nargs=6,
                        default=[0.48, 0.35, 0.10, 0.07, 0.05, 0.05],
                        metavar=("BM25", "DENSE", "Q", "A", "K", "REL"),
                        help="Six nonnegative rerank weights; legacy defaults retained. Pass Dev-selected weights explicitly.")
    return parser.parse_args()


def materialize_documents(args: argparse.Namespace) -> list[tuple[str, dict[str, Any], str]]:
    applications = {
        row["application_id"]: row for row in read_jsonl(args.applications)
        if row.get("release", {}).get("split") == "build"
    }
    knowledge = {row["knowledge_id"]: row for row in read_jsonl(args.knowledge)}
    documents: list[tuple[str, dict[str, Any], str]] = []
    for edge in read_jsonl(args.build_edges):
        app = applications.get(edge["application_id"])
        item = knowledge.get(edge["knowledge_id"])
        if app is None or item is None:
            raise ValueError(f"Unresolvable Build E {edge['e_id']}")
        payload = {
            "e_id": edge["e_id"], "application_id": edge["application_id"],
            "knowledge_id": edge["knowledge_id"], "relation": edge["relation"],
            "application_question": app["question"], "application_answer": app["answer"],
            "knowledge_text": item["text"],
            "application_anchors": app.get("anchor_entities", []),
            "knowledge_anchors": item.get("anchor_entities", []),
        }
        lexical = "\n".join((payload["relation"], payload["application_question"], payload["application_answer"], payload["knowledge_text"], " ".join(payload["application_anchors"]), " ".join(payload["knowledge_anchors"])))
        documents.append((edge["e_id"], payload, lexical))
    if not documents:
        raise ValueError("No Build E documents")
    return documents


def load_encoder(model_path: Path) -> SentenceTransformer:
    return SentenceTransformer(str(model_path), device="cuda", trust_remote_code=True)


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    if args.index_dir.exists():
        raise FileExistsError(f"Refusing to overwrite index directory {args.index_dir}")
    documents = materialize_documents(args)
    args.index_dir.mkdir(parents=True)
    sqlite_path = args.index_dir / "lexical.sqlite"
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, application_id TEXT NOT NULL, relation TEXT NOT NULL, payload TEXT NOT NULL)")
        connection.execute("CREATE INDEX documents_application ON documents(application_id)")
        connection.execute("CREATE VIRTUAL TABLE fts USING fts5(doc_id UNINDEXED, terms, tokenize='unicode61')")
        for start in range(0, len(documents), 500):
            batch = documents[start:start + 500]
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", [(doc_id, payload["application_id"], payload["relation"], json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) for doc_id, payload, _ in batch])
            connection.executemany("INSERT INTO fts VALUES (?, ?)", [(doc_id, " ".join(tokenize(lexical))) for doc_id, _, lexical in batch])
        connection.commit()
    finally:
        connection.close()

    encoder = load_encoder(args.embedding_model)
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    try:
        for start in range(0, len(documents), args.encoding_batch_size):
            batch = documents[start:start + args.encoding_batch_size]
            encoded = encoder.encode([dense_text(payload) for _, payload, _ in batch], batch_size=args.encoding_batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            vectors.append(np.asarray(encoded, dtype=np.float32))
            ids.extend(doc_id for doc_id, _, _ in batch)
            if (start // args.encoding_batch_size + 1) % 100 == 0:
                print(f"[dense-index {min(start + len(batch), len(documents))}/{len(documents)}]", flush=True)
    finally:
        del encoder
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    matrix = np.concatenate(vectors, axis=0)
    np.save(args.index_dir / "dense_vectors.npy", matrix)
    (args.index_dir / "dense_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    metadata = {"document_count": len(documents), "embedding_dimension": int(matrix.shape[1]), "embedding_model": str(args.embedding_model), "retrieval_policy": "global BM25 top-100 + global BGE-M3 dense top-100 -> union -> soft relation-aware hybrid rerank", "source_hashes": {name: sha256(path) for name, path in {"build_edges": args.build_edges, "applications": args.applications, "knowledge": args.knowledge}.items()}}
    write_json(args.index_dir / "metadata.json", metadata)
    return metadata


def load_payloads(connection: sqlite3.Connection, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    return {doc_id: json.loads(payload) for doc_id, payload in connection.execute(f"SELECT doc_id, payload FROM documents WHERE doc_id IN ({placeholders})", ids)}


def retrieve(connection: sqlite3.Connection, dense_vectors: np.ndarray, dense_ids: list[str], encoder: SentenceTransformer, question: str, application_id: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    lexical_rows = connection.execute("SELECT fts.doc_id, -bm25(fts) FROM fts JOIN documents ON documents.doc_id=fts.doc_id WHERE fts MATCH ? AND documents.application_id != ? ORDER BY bm25(fts) LIMIT ?", (fts_query(question), application_id, args.bm25_candidates)).fetchall()
    query_vector = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    all_dense_scores = dense_vectors @ np.asarray(query_vector[0], dtype=np.float32)
    dense_take = min(len(dense_ids), args.dense_candidates + 8)
    dense_positions = np.argpartition(-all_dense_scores, dense_take - 1)[:dense_take]
    dense_positions = dense_positions[np.argsort(-all_dense_scores[dense_positions])]
    bm25 = {doc_id: float(score) for doc_id, score in lexical_rows}
    dense = {dense_ids[int(position)]: float(all_dense_scores[int(position)]) for position in dense_positions}
    candidate_ids = list(dict.fromkeys([*bm25, *dense]))
    payloads = load_payloads(connection, candidate_ids)
    candidate_ids = [doc_id for doc_id in candidate_ids if payloads[doc_id]["application_id"] != application_id]
    bm25_norm, dense_norm = normalize({doc_id: bm25.get(doc_id, 0.0) for doc_id in candidate_ids}), normalize({doc_id: dense.get(doc_id, 0.0) for doc_id in candidate_ids})
    route = route_scores(question)
    ranked = []
    for doc_id in candidate_ids:
        payload = payloads[doc_id]
        features = (bm25_norm[doc_id], dense_norm[doc_id],
                    overlap(question, payload["application_question"]),
                    overlap(question, payload["application_answer"]),
                    overlap(question, payload["knowledge_text"]),
                    route.get(payload["relation"], 0.0))
        score = sum(weight * feature for weight, feature in zip(args.rerank_weights, features))
        ranked.append({"doc_id": doc_id, "score": round(score, 6), "score_components": {"bm25": round(bm25_norm[doc_id], 6), "dense": round(dense_norm[doc_id], 6), "route_bonus": round(route.get(payload["relation"], 0.0), 6)}, "payload": payload})
    ranked.sort(key=lambda item: (-item["score"], item["doc_id"]))
    return ranked[:args.top_k]


def evidence(rows: list[dict[str, Any]], limit: int) -> str:
    blocks = []
    for rank, row in enumerate(rows, 1):
        payload = row["payload"]
        blocks.append("\n".join((f"[E{rank} | {payload['e_id']} | {payload['relation']}]", "\u5386\u53f2\u95ee\u9898: " + compact(payload["application_question"], limit // 2), "\u5386\u53f2\u56de\u7b54: " + compact(payload["application_answer"], limit // 2), "K\u77e5\u8bc6: " + compact(payload["knowledge_text"], limit))))
    return "\n\n".join(blocks)


def generate(args: argparse.Namespace, question: str, blocks: str) -> str:
    prompt = ("\u4f60\u6b63\u5728\u5b8c\u6210\u4e00\u9879\u4e2d\u533b\u77e5\u8bc6\u589e\u5f3a\u751f\u6210\u7814\u7a76\uff0c\u4e0d\u7528\u4e8e\u4e34\u5e8a\u8bca\u7597\u3002\n"
              "\u53ea\u53ef\u4f9d\u636e\u4e0b\u5217 Build E \u8bc1\u636e\u56de\u7b54\u3002\u5386\u53f2\u95ee\u7b54\u53ef\u4f5c\u4e3a\u76f8\u4f3c\u60c5\u5f62\u7684\u53c2\u8003\uff0cK \u77e5\u8bc6\u662f\u53ef\u590d\u7528\u4f9d\u636e\u3002\u4e0d\u8981\u628a\u76f8\u4f3c\u6848\u4f8b\u5199\u6210\u5f53\u524d\u5bf9\u8c61\u7684\u786e\u5b9a\u4e8b\u5b9e\uff0c\u4e5f\u4e0d\u8981\u5f15\u5165\u672a\u88ab\u8bc1\u636e\u652f\u6301\u7684\u5177\u4f53\u8bca\u65ad\u3001\u65b9\u5242\u6216\u5242\u91cf\u3002\u8bc1\u636e\u4e0d\u8db3\u65f6\u8bf7\u660e\u786e\u8bf4\u660e\u3002\n\n"
              f"\u95ee\u9898: {question}\n\n\u8bc1\u636e:\n{blocks}\n\n\u8bf7\u7ed9\u51fa\u7b80\u6d01\u4e14\u53ef\u8ffd\u6eaf\u7684\u56de\u7b54\u3002")
    body = json.dumps({"model": args.model, "messages": [{"role": "system", "content": "Follow the supplied evidence constraints exactly."}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": args.max_tokens, "stream": False}, ensure_ascii=False).encode("utf-8")
    request = Request(args.api_base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    except (HTTPError, URLError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(str(error)) from error


def main() -> None:
    args = parse_args()
    if any(not np.isfinite(w) or w < 0 for w in args.rerank_weights) or not any(args.rerank_weights):
        raise ValueError("Rerank weights must be finite, nonnegative and not all zero")
    for path in (args.queries, args.build_edges, args.applications, args.knowledge, args.embedding_model):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory {args.run_dir}")
    if args.phase == "generate":
        if args.retrieval_results is None or not args.retrieval_results.is_file():
            raise FileNotFoundError("--retrieval-results is required for --phase generate")
        output = list(read_jsonl(args.retrieval_results))
        if not output or any(row.get("split") != args.split or not row.get("retrieved") for row in output):
            raise ValueError("Retrieval rows must be non-empty and match the requested split")
        args.run_dir.mkdir(parents=True)
        started = time.monotonic()
        failures = 0
        for number, row in enumerate(output, 1):
            try:
                row["answer"] = generate(args, row["question"], evidence(row["retrieved"], args.evidence_char_limit))
            except RuntimeError as error:
                failures += 1
                row["error"] = str(error)
            print(f"[hybrid_r4 generate {number}/{len(output)}] {row['application_id']}", flush=True)
        write_jsonl(args.run_dir / "generation_results.jsonl", output)
        summary = {"status": "complete" if not failures else "generation_failed", "query_count": len(output), "evaluated_split": args.split, "generation_failures": failures, "retrieval_policy": "R4 retrieval loaded from a separately materialized, immutable retrieval run", "phase": args.phase, "retrieval_results_sha256": sha256(args.retrieval_results), "elapsed_seconds": round(time.monotonic() - started, 2)}
        write_json(args.run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
        if failures:
            raise RuntimeError(f"{failures} generation failures")
        return
    queries = [row for row in read_jsonl(args.queries) if row["split"] == args.split]
    if not queries:
        raise ValueError(f"No {args.split} queries")
    started = time.monotonic()
    metadata = build_index(args)
    args.run_dir.mkdir(parents=True)
    connection = sqlite3.connect(args.index_dir / "lexical.sqlite")
    dense_vectors = np.load(args.index_dir / "dense_vectors.npy", mmap_mode="r")
    dense_ids = json.loads((args.index_dir / "dense_ids.json").read_text(encoding="utf-8"))
    encoder = load_encoder(args.embedding_model)
    output, failures = [], 0
    try:
        for number, query in enumerate(queries, 1):
            retrieved = retrieve(connection, dense_vectors, dense_ids, encoder, query["question"], query["application_id"], args)
            row: dict[str, Any] = {**query, "retrieved": retrieved, "generated_at_utc": utc_now()}
            output.append(row)
            print(f"[hybrid_r4 retrieve {number}/{len(queries)}] {query['application_id']}", flush=True)
    finally:
        connection.close()
        del encoder
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    if args.phase == "full":
        for number, row in enumerate(output, 1):
            try:
                row["answer"] = generate(args, row["question"], evidence(row["retrieved"], args.evidence_char_limit))
            except RuntimeError as error:
                failures += 1
                row["error"] = str(error)
            print(f"[hybrid_r4 generate {number}/{len(output)}] {row['application_id']}", flush=True)
    write_jsonl(args.run_dir / "generation_results.jsonl", output)
    summary = {"status": "complete" if not failures else "generation_failed", "query_count": len(output), "evaluated_split": args.split, "generation_failures": failures, "index_document_count": metadata["document_count"], "retrieval_policy": metadata["retrieval_policy"], "rerank_weights": args.rerank_weights, "top_k": args.top_k, "bm25_candidates": args.bm25_candidates, "dense_candidates": args.dense_candidates, "embedding_model": str(args.embedding_model), "phase": args.phase, "source_hashes": {"queries": sha256(args.queries), **metadata["source_hashes"]}, "elapsed_seconds": round(time.monotonic() - started, 2)}
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"{failures} generation failures")


if __name__ == "__main__":
    main()
