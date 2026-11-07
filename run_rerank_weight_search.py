#!/usr/bin/env python3
"""Tune and validate the R4 hybrid reranking weights without touching Test data.

The script reuses an immutable R4 lexical/dense index. It creates candidate
features once for the Dev and Test query splits, selects a weight vector using
Dev MRR (Recall@3 breaks ties), and evaluates the selected vector once on
Test. The Test split is not read during weight selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


FEATURE_NAMES = ("bm25", "dense", "question_overlap", "answer_overlap", "knowledge_overlap", "relation_intent")
BASELINE_RAW = np.array((0.48, 0.35, 0.10, 0.07, 0.05, 0.05), dtype=np.float32)

ROUTES = {
    "diagnostic_support": ("\u8fa8\u8bc1", "\u8bc1\u578b", "\u8bc1\u5019", "\u8bca\u65ad", "\u5224\u65ad", "\u9274\u522b", "\u5c5e\u4e8e\u4ec0\u4e48\u8bc1", "\u75c5\u673a"),
    "treatment_or_operation_support": ("\u6cbb\u7597", "\u6cbb\u6cd5", "\u6cbb\u5219", "\u600e\u4e48\u529e", "\u5982\u4f55\u6cbb", "\u7528\u836f", "\u65b9\u5242", "\u63a8\u8350", "\u5efa\u8bae", "\u9488\u7078", "\u827e\u7078", "\u8c03\u7406", "\u64cd\u4f5c"),
    "explanatory_support": ("\u4e3a\u4ec0\u4e48", "\u539f\u56e0", "\u542b\u4e49", "\u662f\u4ec0\u4e48", "\u5982\u4f55\u7406\u89e3", "\u673a\u5236", "\u89e3\u91ca"),
}


@dataclass
class QueryFeatures:
    application_id: str
    question: str
    features: np.ndarray
    relevant: np.ndarray
    document_ids: list[str]
    knowledge_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bm25-candidates", type=int, default=100)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261103)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    tokens = list(dict.fromkeys(tokenize(text)))
    if not tokens:
        raise ValueError(f"No retrieval terms in {text!r}")
    return " OR ".join(f'"{token}"' for token in tokens[:96])


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


def load_payloads(connection: sqlite3.Connection, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    return {
        doc_id: json.loads(payload)
        for doc_id, payload in connection.execute(
            f"SELECT doc_id, payload FROM documents WHERE doc_id IN ({placeholders})", ids
        )
    }


def build_query_features(
    connection: sqlite3.Connection,
    dense_vectors: np.ndarray,
    dense_ids: list[str],
    encoder: SentenceTransformer,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> QueryFeatures:
    question = row["question"]
    application_id = row["application_id"]
    lexical_rows = connection.execute(
        "SELECT fts.doc_id, -bm25(fts) FROM fts JOIN documents ON documents.doc_id=fts.doc_id "
        "WHERE fts MATCH ? AND documents.application_id != ? ORDER BY bm25(fts) LIMIT ?",
        (fts_query(question), application_id, args.bm25_candidates),
    ).fetchall()
    query_vector = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    dense_scores = dense_vectors @ np.asarray(query_vector[0], dtype=np.float32)
    dense_take = min(len(dense_ids), args.dense_candidates + 8)
    positions = np.argpartition(-dense_scores, dense_take - 1)[:dense_take]
    positions = positions[np.argsort(-dense_scores[positions])]
    bm25 = {doc_id: float(score) for doc_id, score in lexical_rows}
    dense = {dense_ids[int(position)]: float(dense_scores[int(position)]) for position in positions}
    candidate_ids = list(dict.fromkeys([*bm25, *dense]))
    payloads = load_payloads(connection, candidate_ids)
    candidate_ids = [doc_id for doc_id in candidate_ids if payloads[doc_id]["application_id"] != application_id]
    bm25_norm = normalize({doc_id: bm25.get(doc_id, 0.0) for doc_id in candidate_ids})
    dense_norm = normalize({doc_id: dense.get(doc_id, 0.0) for doc_id in candidate_ids})
    route = route_scores(question)
    gold_knowledge = set(row["evaluation_knowledge_ids"])
    features, relevant, knowledge_ids = [], [], []
    for doc_id in candidate_ids:
        payload = payloads[doc_id]
        features.append((
            bm25_norm[doc_id],
            dense_norm[doc_id],
            overlap(question, payload["application_question"]),
            overlap(question, payload["application_answer"]),
            overlap(question, payload["knowledge_text"]),
            route.get(payload["relation"], 0.0),
        ))
        knowledge_ids.append(payload["knowledge_id"])
        relevant.append(payload["knowledge_id"] in gold_knowledge)
    return QueryFeatures(
        application_id=application_id,
        question=question,
        features=np.asarray(features, dtype=np.float32),
        relevant=np.asarray(relevant, dtype=bool),
        document_ids=candidate_ids,
        knowledge_ids=knowledge_ids,
    )


def simplex_weights(step: float) -> np.ndarray:
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("grid step must divide 1 exactly")
    rows: list[tuple[int, ...]] = []

    def compose(remaining: int, slots: int, prefix: tuple[int, ...]) -> None:
        if slots == 1:
            rows.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            compose(remaining - value, slots - 1, (*prefix, value))

    compose(units, len(FEATURE_NAMES), ())
    weights = np.asarray(rows, dtype=np.float32) / units
    # Preserve a hybrid core and retain the relation feature as a soft bonus.
    keep = (weights[:, 0] >= 0.10) & (weights[:, 1] >= 0.10) & (weights[:, 5] <= 0.20)
    return weights[keep]


def rank_arrays(examples: list[QueryFeatures], weights: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    hits, reciprocal_ranks = [], []
    for example in examples:
        scores = example.features @ weights
        order = np.argsort(-scores, kind="stable")[:top_k]
        labels = example.relevant[order]
        hits.append(float(labels.any()))
        ranks = np.flatnonzero(labels)
        reciprocal_ranks.append(0.0 if not len(ranks) else 1.0 / float(ranks[0] + 1))
    return np.asarray(hits, dtype=np.float64), np.asarray(reciprocal_ranks, dtype=np.float64)


def grid_metrics(examples: list[QueryFeatures], weights: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_candidates = max(len(example.document_ids) for example in examples)
    feature_tensor = torch.zeros((len(examples), max_candidates, len(FEATURE_NAMES)), dtype=torch.float32, device=device)
    relevant_tensor = torch.zeros((len(examples), max_candidates), dtype=torch.bool, device=device)
    mask = torch.zeros((len(examples), max_candidates), dtype=torch.bool, device=device)
    for index, example in enumerate(examples):
        count = len(example.document_ids)
        feature_tensor[index, :count] = torch.from_numpy(example.features).to(device)
        relevant_tensor[index, :count] = torch.from_numpy(example.relevant).to(device)
        mask[index, :count] = True
    recalls, mrrs = [], []
    for start in range(0, len(weights), 256):
        batch = torch.from_numpy(weights[start:start + 256]).to(device)
        scores = torch.einsum("qcf,bf->bqc", feature_tensor, batch)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        indices = torch.topk(scores, k=top_k, dim=2).indices
        labels = relevant_tensor.unsqueeze(0).expand(len(batch), -1, -1).gather(2, indices)
        recalls.append(labels.any(dim=2).float().mean(dim=1).cpu().numpy())
        inverse_rank = torch.tensor((1.0, 0.5, 1.0 / 3.0), dtype=torch.float32, device=device)
        mrrs.append((labels.float() * inverse_rank).amax(dim=2).mean(dim=1).cpu().numpy())
    return np.concatenate(recalls), np.concatenate(mrrs)


def normalized(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return values / total


def ablation_weights(selected: np.ndarray) -> dict[str, np.ndarray]:
    output = {"selected": selected}
    for name, indices in {
        "no_bm25": (0,),
        "no_dense": (1,),
        "no_local_overlap": (2, 3, 4),
        "no_relation_intent": (5,),
    }.items():
        value = selected.copy()
        value[list(indices)] = 0.0
        output[name] = normalized(value)
    for index, name in enumerate(FEATURE_NAMES):
        for factor, suffix in ((0.80, "minus20pct"), (1.20, "plus20pct")):
            value = selected.copy()
            value[index] *= factor
            output[f"sensitivity_{name}_{suffix}"] = normalized(value)
    return output


def bootstrap_interval(left: np.ndarray, right: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(left), len(left))
        differences[index] = float((left[sample] - right[sample]).mean())
    return tuple(float(value) for value in np.quantile(differences, (0.025, 0.975)))


def metrics_record(name: str, weights: np.ndarray, examples: list[QueryFeatures], top_k: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    hits, reciprocal_ranks = rank_arrays(examples, weights, top_k)
    return {
        "setting": name,
        "recall_at_3": float(hits.mean()),
        "mrr_at_3": float(reciprocal_ranks.mean()),
        **{name: float(value) for name, value in zip(FEATURE_NAMES, weights)},
    }, hits, reciprocal_ranks


def add_selected_comparison(
    record: dict[str, Any],
    selected_metrics: dict[str, Any],
    selected_hit: np.ndarray,
    selected_rr: np.ndarray,
    setting_hit: np.ndarray,
    setting_rr: np.ndarray,
    replicates: int,
    seed: int,
) -> None:
    """Append paired selected-minus-setting estimates with directionally aligned CIs."""
    record["selected_minus_setting_recall_at_3"] = selected_metrics["recall_at_3"] - record["recall_at_3"]
    record["selected_minus_setting_mrr_at_3"] = selected_metrics["mrr_at_3"] - record["mrr_at_3"]
    record["selected_minus_setting_recall_at_3_ci95"] = bootstrap_interval(
        selected_hit, setting_hit, replicates, seed
    )
    record["selected_minus_setting_mrr_at_3_ci95"] = bootstrap_interval(
        selected_rr, setting_rr, replicates, seed + 100
    )


def main() -> None:
    args = parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.run_dir}")
    for path in (args.queries, args.index_dir / "lexical.sqlite", args.index_dir / "dense_vectors.npy", args.index_dir / "dense_ids.json", args.embedding_model):
        if not path.exists():
            raise FileNotFoundError(path)
    queries = list(read_jsonl(args.queries))
    dev_rows = [row for row in queries if row["split"] == "dev"]
    test_rows = [row for row in queries if row["split"] == "test"]
    if not dev_rows or not test_rows:
        raise ValueError("Both Dev and Test query rows are required")
    args.run_dir.mkdir(parents=True)
    connection = sqlite3.connect(args.index_dir / "lexical.sqlite")
    dense_vectors = np.load(args.index_dir / "dense_vectors.npy", mmap_mode="r")
    dense_ids = json.loads((args.index_dir / "dense_ids.json").read_text(encoding="utf-8"))
    encoder = SentenceTransformer(str(args.embedding_model), device="cuda", trust_remote_code=True)
    try:
        # Select weights using Dev only. Test candidates and relevance labels are
        # not constructed until this protocol has produced a fixed vector.
        dev_examples = [build_query_features(connection, dense_vectors, dense_ids, encoder, row, args) for row in dev_rows]
        baseline = normalized(BASELINE_RAW)
        grid = simplex_weights(args.grid_step)
        grid = np.unique(np.vstack((grid, baseline)), axis=0)
        dev_recall, dev_mrr = grid_metrics(dev_examples, grid, args.top_k)
        distance_to_baseline = np.abs(grid - baseline).sum(axis=1)
        best_mrr = dev_mrr.max()
        candidates = np.flatnonzero(np.isclose(dev_mrr, best_mrr))
        best_recall = dev_recall[candidates].max()
        candidates = candidates[np.isclose(dev_recall[candidates], best_recall)]
        selected_index = candidates[np.argmin(distance_to_baseline[candidates])]
        selected = grid[selected_index]
        test_examples = [build_query_features(connection, dense_vectors, dense_ids, encoder, row, args) for row in test_rows]
    finally:
        connection.close()
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grid_rows = []
    for index in np.lexsort((distance_to_baseline, -dev_recall, -dev_mrr))[:100]:
        grid_rows.append({
            "dev_mrr_at_3": float(dev_mrr[index]),
            "dev_recall_at_3": float(dev_recall[index]),
            "l1_distance_to_baseline": float(distance_to_baseline[index]),
            **{name: float(value) for name, value in zip(FEATURE_NAMES, grid[index])},
        })

    selected_metrics, selected_hit, selected_rr = metrics_record("selected", selected, test_examples, args.top_k)
    baseline_metrics, baseline_hit, baseline_rr = metrics_record("published_baseline", baseline, test_examples, args.top_k)
    if not (np.isclose(baseline_metrics["recall_at_3"], 38.0 / 111.0) and np.isclose(baseline_metrics["mrr_at_3"], 0.3213213213213213)):
        raise RuntimeError("Baseline validation failed: candidate extraction no longer reproduces the published R4 Test metrics")

    dev_baseline_metrics, _, _ = metrics_record("published_baseline", baseline, dev_examples, args.top_k)
    dev_selected_metrics, _, _ = metrics_record("selected", selected, dev_examples, args.top_k)
    add_selected_comparison(
        baseline_metrics, selected_metrics, selected_hit, selected_rr, baseline_hit, baseline_rr,
        args.bootstrap_replicates, args.seed,
    )
    test_rows_output = [baseline_metrics, selected_metrics]
    dev_rows_output = [dev_baseline_metrics, dev_selected_metrics]
    test_metrics: dict[str, Any] = {"published_baseline": baseline_metrics, "selected": selected_metrics}
    dev_metrics: dict[str, Any] = {"published_baseline": dev_baseline_metrics, "selected": dev_selected_metrics}
    selected_ablation = ablation_weights(selected)
    for offset, (name, weights) in enumerate(selected_ablation.items()):
        dev_record, _, _ = metrics_record(name, weights, dev_examples, args.top_k)
        record, hit, rr = metrics_record(name, weights, test_examples, args.top_k)
        if name != "selected":
            add_selected_comparison(
                record, selected_metrics, selected_hit, selected_rr, hit, rr,
                args.bootstrap_replicates, args.seed + offset,
            )
        test_metrics[name] = record
        dev_metrics[name] = dev_record
        if name != "selected":
            test_rows_output.append(record)
            dev_rows_output.append(dev_record)

    selected_retrieval = []
    for example in test_examples:
        scores = example.features @ selected
        order = np.argsort(-scores, kind="stable")[:args.top_k]
        selected_retrieval.append({
            "application_id": example.application_id,
            "question": example.question,
            "retrieved": [
                {
                    "e_id": example.document_ids[int(index)],
                    "knowledge_id": example.knowledge_ids[int(index)],
                    "score": float(scores[int(index)]),
                    "relevant_knowledge": bool(example.relevant[int(index)]),
                }
                for index in order
            ],
        })

    with (args.run_dir / "dev_grid_top100.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    with (args.run_dir / "test_ablation_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in test_rows_output for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_rows_output:
            writer.writerow({key: json.dumps(value) if isinstance(value, tuple) else value for key, value in row.items()})
    with (args.run_dir / "dev_ablation_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in dev_rows_output for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dev_rows_output)
    write_jsonl(args.run_dir / "test_selected_retrieval.jsonl", selected_retrieval)
    summary = {
        "status": "complete",
        "selection_protocol": {
            "selection_split": "dev",
            "selection_query_count": len(dev_examples),
            "primary_metric": "MRR@3",
            "secondary_metric": "Recall@3",
            "tie_breaker": "minimum L1 distance from the pre-existing normalized weight vector",
            "candidate_grid_step": args.grid_step,
            "candidate_grid_count": int(len(grid)),
            "constraints": {"bm25_min": 0.10, "dense_min": 0.10, "relation_intent_max": 0.20},
        },
        "test_protocol": {
            "frozen_split": "test",
            "query_count": len(test_examples),
            "top_k": args.top_k,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.seed,
        },
        "published_baseline_raw_weights": {name: float(value) for name, value in zip(FEATURE_NAMES, BASELINE_RAW)},
        "published_baseline_normalized_weights": {name: float(value) for name, value in zip(FEATURE_NAMES, baseline)},
        "selected_weights": {name: float(value) for name, value in zip(FEATURE_NAMES, selected)},
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "input_hashes": {
            "queries": sha256(args.queries),
            "index_metadata": sha256(args.index_dir / "metadata.json"),
        },
    }
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
