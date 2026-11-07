#!/usr/bin/env python3
"""Answer an external answer-blind question bank with plain RAG and AlignRAG.

Plain RAG is deliberately restricted to lexical Top-3 retrieval over K only.
AlignRAG uses the frozen R4 relation-document index and the trained Critic for
one generation-critique-refinement pass. Source answer keys are forbidden from
the input contract and are never read by this runner.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from run_tcm_alignrag import (
    critic_repair_messages,
    evidence_from_row,
    fallback_critique,
    parse_critic,
)


FORBIDDEN_INPUT_KEYS = {
    "answer", "correct_answer", "标准答案", "正确答案", "答案", "答案内容", "解析",
}
ANSWER_PATTERN = re.compile(r"答案\s*[：:]\s*([A-E])\b", re.IGNORECASE)
LOOSE_CHOICE_PATTERN = re.compile(r"(?:选择|选|答案)\s*(?:为|是|：|:)?\s*([A-E])\b", re.IGNORECASE)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def batched(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def batch_client(
    api_base: str,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    max_retry_delay: int,
):
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = base + "/chat/completions/batch"
    else:
        url = base + "/v1/chat/completions/batch"
    headers = {"Content-Type": "application/json"}

    def call(message_batches: list[list[dict[str, str]]]) -> list[str]:
        if not message_batches:
            return []
        body = json.dumps(
            {
                "requests": [
                    {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False}
                    for messages in message_batches
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                request = Request(url, data=body, headers=headers, method="POST")
                with urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                responses = payload["data"]
                if not isinstance(responses, list) or len(responses) != len(message_batches):
                    raise RuntimeError("Batch endpoint returned an unexpected number of completions")
                texts = [str(item["choices"][0]["message"]["content"]).strip() for item in responses]
                if any(not text for text in texts):
                    raise RuntimeError("Batch endpoint returned an empty completion")
                return texts
            except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(min(max_retry_delay, 2**attempt))
        raise RuntimeError(f"Batch completion failed after {retries + 1} attempts: {last_error}")

    return call


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
    terms = list(dict.fromkeys(tokenize(text)))
    if not terms:
        raise ValueError(f"No retrievable terms in question: {text!r}")
    return " OR ".join(f'"{term}"' for term in terms[:96])


def bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def overlap(left: str, right: str) -> float:
    a, b = bigrams(left), bigrams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def route_scores(question: str) -> dict[str, float]:
    routes = {
        "diagnostic_support": ("辨证", "证型", "证候", "诊断", "判断", "鉴别", "病机"),
        "treatment_or_operation_support": ("治疗", "治法", "怎么办", "如何治", "用药", "方剂", "针灸", "调理", "操作"),
        "explanatory_support": ("为什么", "原因", "含义", "是什么", "机制", "解释"),
    }
    counts = {relation: sum(marker in question for marker in markers) for relation, markers in routes.items()}
    maximum = max(counts.values(), default=0)
    return {relation: (count / maximum if maximum else 0.0) for relation, count in counts.items()}


def r4_dense_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        (
            "历史问题: " + str(payload["application_question"])[:600],
            "历史回答: " + str(payload["application_answer"])[:600],
            "K知识: " + str(payload["knowledge_text"])[:900],
        )
    )


def compact(text: str, limit: int) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit]


def k_retrieve(connection: sqlite3.Connection, question: str, top_k: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT fts.doc_id, -bm25(fts) AS score, documents.payload "
        "FROM fts JOIN documents ON documents.doc_id = fts.doc_id "
        "WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
        (fts_query(question), top_k),
    ).fetchall()
    result = [{"doc_id": doc_id, "score": float(score), "payload": json.loads(payload)} for doc_id, score, payload in rows]
    if len(result) != top_k:
        raise ValueError(f"Plain RAG retrieved {len(result)} rather than Top-{top_k} K items")
    return result


def r4_load_payloads(connection: sqlite3.Connection, ids: list[str]) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _ in ids)
    return {
        doc_id: json.loads(payload)
        for doc_id, payload in connection.execute(
            f"SELECT doc_id, payload FROM documents WHERE doc_id IN ({placeholders})", ids
        )
    }


def r4_retrieve(
    connection: sqlite3.Connection,
    dense_vectors: np.ndarray,
    dense_ids: list[str],
    encoder: SentenceTransformer,
    question: str,
    top_k: int,
    bm25_candidates: int,
    dense_candidates: int,
) -> list[dict[str, Any]]:
    lexical_rows = connection.execute(
        "SELECT fts.doc_id, -bm25(fts) "
        "FROM fts JOIN documents ON documents.doc_id = fts.doc_id "
        "WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
        (fts_query(question), bm25_candidates),
    ).fetchall()
    query_vector = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    all_dense_scores = dense_vectors @ np.asarray(query_vector[0], dtype=np.float32)
    dense_take = min(len(dense_ids), dense_candidates + 8)
    positions = np.argpartition(-all_dense_scores, dense_take - 1)[:dense_take]
    positions = positions[np.argsort(-all_dense_scores[positions])]
    bm25 = {doc_id: float(score) for doc_id, score in lexical_rows}
    dense = {dense_ids[int(position)]: float(all_dense_scores[int(position)]) for position in positions}
    candidate_ids = list(dict.fromkeys([*bm25, *dense]))
    payloads = r4_load_payloads(connection, candidate_ids)
    bm25_norm = normalize({doc_id: bm25.get(doc_id, 0.0) for doc_id in candidate_ids})
    dense_norm = normalize({doc_id: dense.get(doc_id, 0.0) for doc_id in candidate_ids})
    route = route_scores(question)
    ranked: list[dict[str, Any]] = []
    for doc_id in candidate_ids:
        payload = payloads[doc_id]
        score = (
            0.48 * bm25_norm[doc_id]
            + 0.35 * dense_norm[doc_id]
            + 0.10 * overlap(question, payload["application_question"])
            + 0.07 * overlap(question, payload["application_answer"])
            + 0.05 * overlap(question, payload["knowledge_text"])
            + 0.05 * route.get(payload["relation"], 0.0)
        )
        ranked.append(
            {
                "doc_id": doc_id,
                "score": round(score, 6),
                "score_components": {
                    "bm25": round(bm25_norm[doc_id], 6),
                    "dense": round(dense_norm[doc_id], 6),
                    "route_bonus": round(route.get(payload["relation"], 0.0), 6),
                },
                "payload": payload,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["doc_id"]))
    result = ranked[:top_k]
    if len(result) != top_k:
        raise ValueError(f"AlignRAG retrieved {len(result)} rather than Top-{top_k} R4 documents")
    return result


def k_evidence(rows: list[dict[str, Any]], limit: int) -> str:
    return "\n\n".join(
        f"[K{rank} | {item['payload']['knowledge_id']}]\n{compact(item['payload']['text'], limit)}"
        for rank, item in enumerate(rows, 1)
    )


def plain_rag_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你正在完成中医执业医师单选题作答。展示的 K 检索资料仅供参考：相关时可用于辅助判断，"
                "不相关或不足时也必须结合题干和选项作出最佳选择，绝不能以证据不足、无法判断或建议咨询代替作答。"
                "只输出一行，严格写成“答案：X”（X 必须为 A、B、C、D、E 之一）；不要输出依据、解释或任何其他文字。"
            ),
        },
        {"role": "user", "content": f"题目：\n{question}\n\nK 检索证据：\n{evidence}"},
    ]


def align_initial_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你正在完成中医执业医师单选题作答。R4 检索证据仅是辅助参考，优先使用相关 K 知识，"
                "历史问答仅可作相似案例参考。检索内容不相关或不充分时，仍须结合题干和选项选择最合理的一项；"
                "不得因证据不足拒绝作答、只给泛泛建议或跳过选择。只输出一行，严格写成“答案：X”"
                "（X 必须为 A、B、C、D、E 之一）；不要输出依据、解释或任何其他文字。"
            ),
        },
        {"role": "user", "content": f"题目：\n{question}\n\nR4 检索参考资料：\n{evidence}"},
    ]


def questionbank_critic_messages(question: str, evidence: str, initial_answer: str) -> list[dict[str, str]]:
    schema = (
        '{"verdict":"needs_revision|acceptable|insufficient_evidence",'
        '"issues":[{"type":"unsupported_claim|historical_case_as_current_fact|missing_constraint|overstatement|not_answering_question|evidence_misuse",'
        '"claim_or_location":"...","why":"...","evidence_ids":["E1"],"fix":"..."}],'
        '"revision_instructions":["..."],"evidence_boundary":"..."}'
    )
    return [
        {
            "role": "system",
            "content": (
                "你是中医单选题的证据一致性 Critic。展示资料仅是辅助参考；审查时不得因为检索资料不完整"
                "而要求拒答、删除选项或输出无法判断。初稿必须保留并给出 A-E 中的一个选项。"
                "只输出一个合法、紧凑的 JSON 对象，不得使用 Markdown、解释文字或思考过程。"
                "verdict 只能为 needs_revision、acceptable 或 insufficient_evidence；issues 最多两项；"
                "revision_instructions 为一至两项；每项必须指向 E1 至 E3。若初稿已经选择选项且能回应题目，"
                "资料不充分本身不构成 needs_revision。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题目：{question}\n\nR4 检索参考资料：\n{evidence}\n\n初始回答 y0：\n{initial_answer}\n\n"
                f"请仅输出该 JSON：\n{schema}"
            ),
        },
    ]


def questionbank_refinement_messages(
    question: str, evidence: str, initial_answer: str, critique: str
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "根据 Critic 的有效批评精修中医单选题答案。R4 检索资料仅供辅助，不是拒答门槛："
                "资料不足时也必须依据题干和选项选择最合理答案。不得输出证据不足、无法判断、无法作答或就医建议"
                "来替代选择。只修正 Critic 指出的具体问题，保留合理判断。只输出一行，严格写成“答案：X”，"
                "其中 X 必须是 A、B、C、D、E 之一；不要输出依据、解释或任何其他文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题目：\n{question}\n\nR4 检索参考资料：\n{evidence}\n\n初始回答：\n{initial_answer}\n\n"
                f"Critic 批评：\n{critique}\n\n请输出精修后的最终答案。"
            ),
        },
    ]


def extract_choice(answer: str) -> str | None:
    """Return the final explicit A-E choice, preferring the required answer marker."""
    strict = ANSWER_PATTERN.findall(answer)
    if strict:
        return strict[-1].upper()
    loose = LOOSE_CHOICE_PATTERN.findall(answer)
    return loose[-1].upper() if loose else None


def choice_repair_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "将下面单选题作答改为合规的最终答案。必须作答，不能拒答；只输出一行，严格写成“答案：X”，"
                "其中 X 必须为 A、B、C、D、E 之一。根据原作答已表达的最佳判断选择，不能输出任何其他文字。"
            ),
        },
        {"role": "user", "content": f"题目：\n{question}\n\n原作答：\n{answer}"},
    ]


def enforce_choices(
    generator: Any, contexts: list[dict[str, Any]], answers: list[str], invalid_policy: str = "default_a"
) -> list[tuple[str, str]]:
    """Guarantee an answer marker, retrying only malformed model responses.

    The deterministic A fallback is intentionally marked in the output metadata so
    it cannot silently be mistaken for a model-compliant answer.
    """
    resolved: list[tuple[str, str] | None] = [None] * len(answers)
    repair_indices: list[int] = []
    for index, (context, answer) in enumerate(zip(contexts, answers)):
        choice = extract_choice(answer)
        if choice:
            resolved[index] = (f"答案：{choice}", "model")
        else:
            repair_indices.append(index)
    if repair_indices:
        repaired = generator(
            [choice_repair_messages(contexts[index]["question"], answers[index]) for index in repair_indices]
        )
        for index, repaired_answer in zip(repair_indices, repaired):
            contexts[index]["repair_raw_answer"] = repaired_answer
            choice = extract_choice(repaired_answer)
            if choice:
                resolved[index] = (f"答案：{choice}", "format_repair")
            else:
                resolved[index] = (("INVALID", "invalid_output") if invalid_policy == "mark_invalid"
                                   else ("答案：A", "forced_default_A"))
    return [item for item in resolved if item is not None]


def validate_question(row: dict[str, Any]) -> None:
    forbidden = FORBIDDEN_INPUT_KEYS & set(row)
    if forbidden:
        raise ValueError(f"Answer-key field(s) forbidden in external input: {sorted(forbidden)}")
    if not row.get("application_id") or not row.get("question") or not row.get("retrieval_query"):
        raise ValueError("External question needs non-empty application_id, question, and retrieval_query")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["application_id"]) for row in read_jsonl(path)}


def public_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "application_id",
            "question_id",
            "question",
            "retrieval_query",
            "year",
            "exam",
            "unit",
            "source_number",
            "question_type",
            "shared_option_group",
            "data_status",
            "source_row",
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--k-index", type=Path, required=True)
    parser.add_argument("--r4-index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--k-evidence-char-limit", type=int, default=850)
    parser.add_argument("--r4-evidence-field-limit", type=int, default=850)
    parser.add_argument("--generator-max-tokens", type=int, default=32)
    parser.add_argument("--critic-max-tokens", type=int, default=384)
    parser.add_argument("--critic-repair-attempts", type=int, default=1)
    parser.add_argument("--bm25-candidates", type=int, default=100)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-retry-delay", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke-test limit only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k != 3:
        raise ValueError("This comparison is fixed to Top-3 retrieval")
    for path in (args.questions, args.k_index, args.r4_index_dir / "lexical.sqlite", args.r4_index_dir / "dense_vectors.npy", args.r4_index_dir / "dense_ids.json", args.embedding_model):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = list(read_jsonl(args.questions))
    for row in rows:
        validate_question(row)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No external questions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plain_path = args.output_dir / "plain_rag_results.jsonl"
    align_path = args.output_dir / "alignrag_results.jsonl"
    if not args.resume and (plain_path.exists() or align_path.exists()):
        raise FileExistsError("Output exists; use --resume to continue safely")
    plain_done = completed_ids(plain_path) if args.resume else set()
    align_done = completed_ids(align_path) if args.resume else set()

    generator = batch_client(
        args.api_base, args.generator_model, args.generator_max_tokens,
        args.timeout, args.retries, args.max_retry_delay,
    )
    critic = batch_client(
        args.api_base, args.critic_model, args.critic_max_tokens,
        args.timeout, args.retries, args.max_retry_delay,
    )
    k_connection = sqlite3.connect(args.k_index)
    r4_connection = sqlite3.connect(args.r4_index_dir / "lexical.sqlite")
    dense_vectors = np.load(args.r4_index_dir / "dense_vectors.npy", mmap_mode="r")
    dense_ids = json.loads((args.r4_index_dir / "dense_ids.json").read_text(encoding="utf-8"))
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(args.embedding_model), device="cuda", trust_remote_code=True)
    started = time.monotonic()
    plain_failures = 0
    align_failures = 0
    try:
        numbered_rows = list(enumerate(rows, 1))
        for question_batch in batched(numbered_rows, args.batch_size):
            plain_contexts: list[dict[str, Any]] = []
            for number, question_row in question_batch:
                application_id = str(question_row["application_id"])
                if application_id in plain_done:
                    continue
                metadata = public_metadata(question_row)
                try:
                    retrieved = k_retrieve(k_connection, str(question_row["retrieval_query"]), args.top_k)
                    plain_contexts.append(
                        {
                            "number": number,
                            "application_id": application_id,
                            "metadata": metadata,
                            "question": str(question_row["question"]),
                            "retrieved": retrieved,
                            "messages": plain_rag_messages(
                                str(question_row["question"]), k_evidence(retrieved, args.k_evidence_char_limit)
                            ),
                        }
                    )
                except Exception as error:
                    plain_failures += 1
                    append_jsonl(plain_path, {**metadata, "error": str(error)})
                    print(f"[plain RAG {number}/{len(rows)}] {application_id} error", flush=True)
            if plain_contexts:
                try:
                    answers = generator([context["messages"] for context in plain_contexts])
                    for context, (answer, choice_status) in zip(plain_contexts, enforce_choices(generator, plain_contexts, answers)):
                        append_jsonl(
                            plain_path,
                            {**context["metadata"], "retrieved": context["retrieved"], "answer": answer, "choice_status": choice_status},
                        )
                        print(f"[plain RAG {context['number']}/{len(rows)}] {context['application_id']}", flush=True)
                except Exception as error:
                    for context in plain_contexts:
                        plain_failures += 1
                        append_jsonl(plain_path, {**context["metadata"], "error": str(error)})
                        print(f"[plain RAG {context['number']}/{len(rows)}] {context['application_id']} error", flush=True)

            align_contexts: list[dict[str, Any]] = []
            for number, question_row in question_batch:
                application_id = str(question_row["application_id"])
                if application_id in align_done:
                    continue
                metadata = public_metadata(question_row)
                question = str(question_row["question"])
                retrieval_query = str(question_row["retrieval_query"])
                try:
                    retrieved = r4_retrieve(
                        r4_connection, dense_vectors, dense_ids, encoder, retrieval_query,
                        args.top_k, args.bm25_candidates, args.dense_candidates,
                    )
                    evidence = evidence_from_row(
                        {"application_id": application_id, "question": question, "retrieved": retrieved},
                        args.r4_evidence_field_limit,
                    )
                    align_contexts.append(
                        {
                            "number": number,
                            "application_id": application_id,
                            "metadata": metadata,
                            "question": question,
                            "retrieved": retrieved,
                            "evidence": evidence,
                        }
                    )
                except Exception as error:
                    align_failures += 1
                    append_jsonl(align_path, {**metadata, "error": str(error)})
                    print(f"[R4+AlignRAG {number}/{len(rows)}] {application_id} error", flush=True)
            if not align_contexts:
                continue
            try:
                initial_answers = generator(
                    [align_initial_messages(context["question"], context["evidence"]) for context in align_contexts]
                )
                for context, initial_answer in zip(align_contexts, initial_answers):
                    context["initial_answer"] = initial_answer

                raw_critiques = critic(
                    [
                        questionbank_critic_messages(context["question"], context["evidence"], context["initial_answer"])
                        for context in align_contexts
                    ]
                )
                repair_contexts: list[dict[str, Any]] = []
                for context, raw_critique in zip(align_contexts, raw_critiques):
                    context["critic_raw"] = raw_critique
                    try:
                        context["critique"] = parse_critic(raw_critique, args.top_k)
                        context["critic_status"] = "valid"
                    except ValueError:
                        context["critic_status"] = "invalid"
                        repair_contexts.append(context)

                for _ in range(args.critic_repair_attempts):
                    if not repair_contexts:
                        break
                    repairs = critic(
                        [
                            critic_repair_messages(
                                context["question"], context["evidence"], context["initial_answer"], context["critic_raw"]
                            )
                            for context in repair_contexts
                        ]
                    )
                    remaining: list[dict[str, Any]] = []
                    for context, repair in zip(repair_contexts, repairs):
                        try:
                            context["critique"] = parse_critic(repair, args.top_k)
                            context["critic_status"] = "repaired"
                        except ValueError:
                            remaining.append(context)
                    repair_contexts = remaining
                for context in repair_contexts:
                    context["critique"] = fallback_critique()
                    context["critic_status"] = "fallback"

                final_answers = generator(
                    [
                        questionbank_refinement_messages(
                            context["question"], context["evidence"], context["initial_answer"],
                            json.dumps(context["critique"], ensure_ascii=False),
                        )
                        for context in align_contexts
                    ]
                )
                for context, (answer, choice_status) in zip(align_contexts, enforce_choices(generator, align_contexts, final_answers)):
                    append_jsonl(
                        align_path,
                        {
                            **context["metadata"],
                            "retrieved": context["retrieved"],
                            "initial_answer": context["initial_answer"],
                            "critic": context["critique"],
                            "critic_raw": context["critic_raw"],
                            "critic_status": context["critic_status"],
                            "answer": answer,
                            "choice_status": choice_status,
                            "alignrag_config": {
                                "iterations": 1,
                                "prompt_version": "tcm-alignrag-questionbank-mandatory-choice-v3",
                                "execution_batch_size": args.batch_size,
                            },
                        },
                    )
                    print(f"[R4+AlignRAG {context['number']}/{len(rows)}] {context['application_id']}", flush=True)
            except Exception as error:
                for context in align_contexts:
                    align_failures += 1
                    append_jsonl(align_path, {**context["metadata"], "error": str(error)})
                    print(f"[R4+AlignRAG {context['number']}/{len(rows)}] {context['application_id']} error", flush=True)
    finally:
        k_connection.close()
        r4_connection.close()
        del encoder
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    summary = {
        "status": "complete" if not (plain_failures or align_failures) else "completed_with_failures",
        "questions": len(rows),
        "plain_rag_completed": len(completed_ids(plain_path)),
        "alignrag_completed": len(completed_ids(align_path)),
        "plain_rag_new_failures": plain_failures,
        "alignrag_new_failures": align_failures,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "contract": {
            "plain_rag": "K-only lexical BM25 Top-3 -> base Qwen direct answer; no A/E, R4 rerank, or Critic.",
            "alignrag": "R4 hybrid relation-level Top-3 -> base Qwen y0 -> LoRA Critic -> one refinement y1.",
            "answer_key": "Question-bank input was validated to contain no answer-key fields.",
            "retrieval_query": "K and R4 retrieval receive the question stem only; answer options are not used for retrieval.",
            "answer_policy": "Evidence is auxiliary. Every output is forced to carry an A-E choice, with malformed outputs retried once.",
            "execution": (
                f"Micro-batched inference with batch_size={args.batch_size}; exam-mode prompts emit one mandatory A-E choice "
                f"(generator_max_tokens={args.generator_max_tokens}, critic_max_tokens={args.critic_max_tokens})."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
