#!/usr/bin/env python3
"""Run a leakage-free Test-111 EviCR ablation.

The Critic contract is deliberately narrow: C(q, D(q)).  It never receives an
initial answer, a weak answer, a reference answer, or an answer key.  The fixed
TER answer y0 is used only by the answer generator during the downstream
refinement stage.  The two refinement arms share the exact same persisted
Critic package; ``evicr_no_ks`` changes only ``knowledge_supplements`` to an
empty list before calling the generator.

This runner is resume-safe and writes a manifest containing source hashes and
per-record input hashes.  It is intentionally separate from historical
wrong-only and answer-aware experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CRITIC_FIELDS = {
    "evidence_status",
    "evidence_actions",
    "knowledge_supplements",
    "change_support",
    "answer_guidance",
}
EVIDENCE_STATUS = {"supported", "partial", "insufficient", "conflict"}
EVIDENCE_ACTIONS = {"keep", "downweight", "discard"}
CONFIDENCE = {"high", "medium", "low"}
CHANGE_SUPPORT = {"none", "weak", "strong"}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def file_hash(path: Path) -> str:
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


def evidence_text(row: dict[str, Any], field_limit: int) -> str:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or not 1 <= len(retrieved) <= 3:
        raise ValueError("Each frozen Test-111 row must contain one to three retrieved records")
    own_id = str(row.get("application_id", ""))
    blocks: list[str] = []
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"Evidence {rank} has no payload")
        if str(payload.get("application_id", "")) == own_id:
            raise ValueError(f"Self-retrieval detected for {own_id}")
        evidence_id = compact(payload.get("e_id") or item.get("doc_id"), 100)
        relation = compact(payload.get("relation"), 100)
        historical_question = compact(payload.get("application_question"), field_limit)
        historical_answer = compact(payload.get("application_answer"), field_limit)
        knowledge = compact(payload.get("knowledge_text"), field_limit)
        if not evidence_id or not knowledge:
            raise ValueError(f"Evidence {rank} lacks an ID or K knowledge")
        blocks.append(
            "\n".join(
                (
                    f"[E{rank} | {evidence_id} | {relation or 'unspecified_relation'}]",
                    f"历史问题：{historical_question}",
                    f"历史回答：{historical_answer}",
                    f"K 知识：{knowledge}",
                )
            )
        )
    return "\n\n".join(blocks)


def critic_messages(question: str, evidence: str) -> list[dict[str, str]]:
    """The only Critic prompt in this file: it contains q and D(q), nothing else."""
    schema = (
        '{"evidence_status":"supported|partial|insufficient|conflict",'
        '"evidence_actions":[{"evidence_id":"E1","action":"keep|downweight|discard",'
        '"reason":"..."}],'
        '"knowledge_supplements":[{"claim":"...","basis":"model_knowledge",'
        '"confidence":"high|medium|low"}],'
        '"change_support":"none|weak|strong","answer_guidance":"..."}'
    )
    return [
        {
            "role": "system",
            "content": (
                "你是中医 RAG 的关系证据 Critic。只审查问题和 R4 检索证据；"
                "输入仅包含这两类材料，不包含其他上下文。"
                "逐条判断 E1-E3 的相关性、适用条件和支持范围。K 知识可作为可复用依据，"
                "历史案例只能作为相似案例，不能写作当前对象事实。"
                "最多给出两条模型知识补充，且必须明确标为 model_knowledge，不能冒充 R4 证据。"
                "不直接替用户作答。只输出一个合法 JSON 对象，不要 Markdown 或额外字段。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：\n{question}\n\nR4 检索证据：\n{evidence}\n\n输出：\n{schema}",
        },
    ]


def critic_repair_messages(question: str, evidence: str, candidate: str) -> list[dict[str, str]]:
    """Formatting repair preserves the q,D(q)-only Critic contract."""
    return [
        {
            "role": "system",
            "content": (
                "将候选证据评议修复为合法 JSON。只可使用给定问题、R4 证据和候选文本；"
                "不添加任何未提供的上下文。"
                "字段必须且只能为 evidence_status、evidence_actions、knowledge_supplements、"
                "change_support、answer_guidance。只输出 JSON。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：\n{question}\n\nR4 检索证据：\n{evidence}\n\n"
                f"候选输出：\n{compact(candidate, 5000)}"
            ),
        },
    ]


def answer_messages(question: str, evidence: str, y0: str, package: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你正在精修一份中医 RAG 回答，不用于临床诊疗。保留初始回答中由展示证据支持、"
                "且直接回应问题的内容；根据证据评议只作必要修改。K 知识可以复用，历史案例仅是"
                "相似案例而非当前事实。模型知识补充不是检索证据，必须保留其不确定性与来源边界。"
                "不要声称确定诊断、处方、剂量、疗效或个体化治疗，不要提及 Critic 或编辑过程。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：\n{question}\n\nR4 检索证据：\n{evidence}\n\n"
                f"初始回答 y0：\n{y0}\n\n证据评议：\n"
                f"{json.dumps(package, ensure_ascii=False, separators=(',', ':'))}\n\n请输出精修回答。"
            ),
        },
    ]


def endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/v1/chat/completions" if not base.endswith("/v1") else base + "/chat/completions"


def call_chat(api_base: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> str:
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(endpoint(api_base), data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = str(payload["choices"][0]["message"]["content"]).strip()
            if not text:
                raise ValueError("Empty completion")
            return text
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Chat completion failed after four attempts: {last_error}")


def call_chat_batch(api_base: str, model: str, message_batch: list[list[dict[str, str]]], max_tokens: int, timeout: int) -> list[str]:
    """Use the local OpenAI-compatible batch endpoint for Qwen inference."""
    if not message_batch:
        return []
    body = json.dumps(
        {"requests": [{"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False} for messages in message_batch]},
        ensure_ascii=False,
    ).encode("utf-8")
    batch_endpoint = endpoint(api_base).replace("/chat/completions", "/chat/completions/batch")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(batch_endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=timeout * max(1, len(message_batch))) as response:
                payload = json.loads(response.read().decode("utf-8"))
            responses = payload.get("data")
            if not isinstance(responses, list) or len(responses) != len(message_batch):
                raise ValueError("Batch endpoint returned the wrong number of responses")
            texts = [str(item["choices"][0]["message"]["content"]).strip() for item in responses]
            if any(not text for text in texts):
                raise ValueError("Batch endpoint returned an empty completion")
            return texts
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Batch chat completion failed after four attempts: {last_error}")


def extract_object(value: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found")


def clean_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"Invalid {field}")
    return value.strip()


def validate_package(value: dict[str, Any], evidence_count: int) -> dict[str, Any]:
    if set(value) != CRITIC_FIELDS:
        raise ValueError("Critic JSON does not have the five-field contract")
    status = clean_text(value.get("evidence_status"), "evidence_status", 32)
    change = clean_text(value.get("change_support"), "change_support", 32)
    if status not in EVIDENCE_STATUS or change not in CHANGE_SUPPORT:
        raise ValueError("Unsupported status value")
    actions = value.get("evidence_actions")
    supplements = value.get("knowledge_supplements")
    if not isinstance(actions, list) or not 0 <= len(actions) <= evidence_count:
        raise ValueError("Invalid evidence_actions")
    if not isinstance(supplements, list) or not 0 <= len(supplements) <= 2:
        raise ValueError("Invalid knowledge_supplements")
    cleaned_actions = []
    seen_ids: set[str] = set()
    for item in actions:
        if not isinstance(item, dict) or set(item) != {"evidence_id", "action", "reason"}:
            raise ValueError("Invalid evidence action object")
        evidence_id = clean_text(item.get("evidence_id"), "evidence_id", 16)
        action = clean_text(item.get("action"), "action", 32)
        if evidence_id not in {f"E{index}" for index in range(1, evidence_count + 1)} or evidence_id in seen_ids:
            raise ValueError("Invalid or duplicate evidence ID")
        if action not in EVIDENCE_ACTIONS:
            raise ValueError("Invalid evidence action")
        seen_ids.add(evidence_id)
        cleaned_actions.append({"evidence_id": evidence_id, "action": action, "reason": clean_text(item.get("reason"), "reason", 420)})
    cleaned_supplements = []
    for item in supplements:
        if not isinstance(item, dict) or set(item) != {"claim", "basis", "confidence"}:
            raise ValueError("Invalid knowledge supplement object")
        if item.get("basis") != "model_knowledge":
            raise ValueError("Knowledge supplement must state model_knowledge")
        confidence = clean_text(item.get("confidence"), "confidence", 32)
        if confidence not in CONFIDENCE:
            raise ValueError("Invalid supplement confidence")
        cleaned_supplements.append(
            {"claim": clean_text(item.get("claim"), "claim", 520), "basis": "model_knowledge", "confidence": confidence}
        )
    return {
        "evidence_status": status,
        "evidence_actions": cleaned_actions,
        "knowledge_supplements": cleaned_supplements,
        "change_support": change,
        "answer_guidance": clean_text(value.get("answer_guidance"), "answer_guidance", 900),
    }


def neutral_package() -> dict[str, Any]:
    return {
        "evidence_status": "partial",
        "evidence_actions": [],
        "knowledge_supplements": [],
        "change_support": "none",
        "answer_guidance": "仅使用展示证据直接支持的内容，不能把历史案例视为当前对象事实。",
    }


def source_rows(frozen_path: Path, y0_path: Path, field_limit: int) -> list[dict[str, Any]]:
    frozen = list(read_jsonl(frozen_path))
    y0_rows = list(read_jsonl(y0_path))
    y0_by_id = {str(row.get("application_id", "")): row for row in y0_rows}
    if len(y0_by_id) != len(y0_rows):
        raise ValueError("TER-y0 source contains duplicate application IDs")
    rows = []
    for row in frozen:
        app_id = str(row.get("application_id", ""))
        question = clean_text(row.get("question"), "question", 6000)
        if row.get("split") != "test":
            raise ValueError(f"Frozen input must be Test-only: {app_id}")
        y0_row = y0_by_id.get(app_id)
        if y0_row is None:
            raise ValueError(f"TER-y0 source is missing {app_id}")
        if y0_row.get("question") != question:
            raise ValueError(f"Question mismatch for {app_id}")
        y0 = clean_text(y0_row.get("answer"), "TER-y0 answer", 12000)
        evidence = evidence_text(row, field_limit)
        rows.append({"application_id": app_id, "question": question, "evidence": evidence, "y0": y0})
    if len(rows) != 111:
        raise ValueError(f"Expected Test-111, got {len(rows)} rows")
    return rows


def load_existing(path: Path, rows: list[dict[str, Any]], required: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    allowed = {row["application_id"] for row in rows}
    output = {}
    for item in read_jsonl(path):
        app_id = str(item.get("application_id", ""))
        if app_id not in allowed or app_id in output or not required <= set(item):
            raise ValueError(f"Invalid checkpoint row in {path}: {app_id}")
        output[app_id] = item
    return output


def public_row(row: dict[str, Any], arm: str, answer: str, package: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "application_id": row["application_id"],
        "question": row["question"],
        "answer": answer,
        "arm": arm,
        "evidence_hash": digest_text(row["evidence"]),
        "critic": package,
        "protocol": {
            "critic_input": "q + D(q) only",
            "critic_reads_y0": False,
            "answer_generator_reads_y0": arm != "ter_y0",
            "answer_key_used_for_model_input": False,
            "knowledge_supplements_enabled": arm == "evicr_full",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--ter-y0", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--critic-api", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--generator-api")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--critic-max-tokens", type=int, default=640)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.evidence_field_limit < 100:
        raise ValueError("evidence field limit is implausibly short")
    generator_api = args.generator_api or args.critic_api
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = source_rows(args.frozen_test, args.ter_y0, args.evidence_field_limit)

    critic_path = args.output_dir / "critic_qd.jsonl"
    critics = load_existing(critic_path, rows, {"application_id", "question", "package", "critic_input_hash", "status"})
    for row in rows:
        app_id = row["application_id"]
        if app_id in critics and (
            critics[app_id]["question"] != row["question"]
            or critics[app_id]["critic_input_hash"] != digest_text(row["question"] + "\n" + row["evidence"])
        ):
            raise ValueError(f"Critic checkpoint mismatch for {app_id}")
    pending_critics = [row for row in rows if row["application_id"] not in critics]
    for start in range(0, len(pending_critics), 4):
        batch = pending_critics[start:start + 4]
        raw_batch = call_chat_batch(
            args.critic_api, args.critic_model,
            [critic_messages(row["question"], row["evidence"]) for row in batch],
            args.critic_max_tokens, args.timeout,
        )
        for row, raw in zip(batch, raw_batch):
            app_id = row["application_id"]
            try:
                package = validate_package(extract_object(raw), row["evidence"].count("[E"))
                status = "valid"
                repaired_raw = ""
            except (ValueError, TypeError) as first_error:
                repaired_raw = call_chat(args.critic_api, args.critic_model, critic_repair_messages(row["question"], row["evidence"], raw), args.critic_max_tokens, args.timeout)
                try:
                    package = validate_package(extract_object(repaired_raw), row["evidence"].count("[E"))
                    status = "repaired"
                except (ValueError, TypeError):
                    package = neutral_package()
                    status = f"fallback:{type(first_error).__name__}"
            critics[app_id] = {
                "application_id": app_id, "question": row["question"], "package": package,
                "status": status, "critic_input_hash": digest_text(row["question"] + "\n" + row["evidence"]),
                "raw": raw, "repair_raw": repaired_raw,
                "contract": "q + D(q) only; no y0, weak answer, or answer key",
            }
        write_jsonl(critic_path, [critics[key] for key in sorted(critics)])

    arms = {"ter_y0": args.output_dir / "ter_y0.jsonl", "evicr_full": args.output_dir / "evicr_full.jsonl", "evicr_no_ks": args.output_dir / "evicr_no_ks.jsonl"}
    for arm, path in arms.items():
        existing = load_existing(path, rows, {"application_id", "question", "answer", "arm", "protocol"})
        for row in rows:
            app_id = row["application_id"]
            if app_id in existing and (
                existing[app_id]["arm"] != arm or existing[app_id]["question"] != row["question"]
            ):
                raise ValueError(f"Answer checkpoint mismatch for {app_id}/{arm}")
        if arm == "ter_y0":
            for row in rows:
                app_id = row["application_id"]
                if app_id not in existing:
                    existing[app_id] = public_row(row, arm, row["y0"])
            write_jsonl(path, [existing[key] for key in sorted(existing)])
            continue
        pending_answers = [row for row in rows if row["application_id"] not in existing]
        for start in range(0, len(pending_answers), 4):
            batch = pending_answers[start:start + 4]
            message_batch = []
            delivered_batch = []
            for row in batch:
                delivered = dict(critics[row["application_id"]]["package"])
                if arm == "evicr_no_ks":
                    delivered["knowledge_supplements"] = []
                delivered_batch.append(delivered)
                message_batch.append(answer_messages(row["question"], row["evidence"], row["y0"], delivered))
            answer_batch = call_chat_batch(generator_api, args.generator_model, message_batch, args.answer_max_tokens, args.timeout)
            for row, delivered, answer in zip(batch, delivered_batch, answer_batch):
                app_id = row["application_id"]
                result = public_row(row, arm, answer, delivered)
                result["critic_status"] = critics[app_id]["status"]
                result["critic_input_hash"] = critics[app_id]["critic_input_hash"]
                result["source_y0_hash"] = digest_text(row["y0"])
                existing[app_id] = result
            write_jsonl(path, [existing[key] for key in sorted(existing)])

    status_counts: dict[str, int] = {}
    for item in critics.values():
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    manifest = {
        "status": "complete",
        "version": "test111-evicr-qd-ablation-v1",
        "n_questions": len(rows),
        "arms": list(arms),
        "critic_input_contract": "C(q, D(q)); no y0, weak answer, reference answer, or answer key",
        "critic_model": args.critic_model,
        "generator_model": args.generator_model,
        "critic_status_counts": status_counts,
        "source_sha256": {"frozen_test": file_hash(args.frozen_test), "ter_y0": file_hash(args.ter_y0)},
        "evidence_field_limit": args.evidence_field_limit,
        "knowledge_supplements_ablation": "The same persisted Critic package is used; only evicr_no_ks replaces knowledge_supplements with [].",
        "answer_key_used_for_model_input": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
