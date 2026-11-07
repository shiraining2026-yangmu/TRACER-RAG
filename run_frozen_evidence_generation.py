#!/usr/bin/env python3
"""Generate an exam baseline from a frozen, answer-blind evidence JSONL.

This is used for the ordinary dense-vector RAG and TER direct-generation arms.
It never opens an answer key; it only consumes questions plus pre-materialized
evidence from the already frozen retrieval system.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_dense_chunk_rag import atomic_json, digest, file_hash, log, read_jsonl
from run_questionbank_rag_vs_alignrag import batch_client, enforce_choices
from run_tcm_alignrag import evidence_from_row
from benchmark_integrity import completed_prediction, freeze_protocol

CHOICE = re.compile(r"答案：[A-E]$")


def dense_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"[D{rank} | {item['doc_id']}]\n{item['text']}" for rank, item in enumerate(row["evidence"], 1)
    )
    return [
        {"role": "system", "content": (
            "你正在完成中医执业医师单选题作答。展示的检索资料仅供参考：相关时可用于辅助判断，"
            "不相关或不足时也必须结合题干和选项作出最佳选择，绝不能以证据不足、无法判断或建议咨询代替作答。"
            "只输出一行，严格写成“答案：X”（X 必须为 A、B、C、D、E 之一）；不要输出依据、解释或任何其他文字。"
        )},
        {"role": "user", "content": f"题目：\n{row['question']}\n\n检索参考资料：\n{evidence}"},
    ]


def ter_messages(row: dict[str, Any], evidence_limit: int) -> list[dict[str, str]]:
    evidence = evidence_from_row(row, evidence_limit)
    return [
        {"role": "system", "content": (
            "你正在完成中医执业医师单选题作答。R4 检索证据仅是辅助参考，优先使用相关 K 知识，"
            "历史问答仅可作相似案例参考。检索内容不相关或不充分时，仍须结合题干和选项选择最合理的一项；"
            "不得因证据不足拒绝作答、只给泛泛建议或跳过选择。只输出一行，严格写成“答案：X”"
            "（X 必须为 A、B、C、D、E 之一）；不要输出依据、解释或任何其他文字。"
        )},
        {"role": "user", "content": f"题目：\n{row['question']}\n\nR4 检索参考资料：\n{evidence}"},
    ]


def validate(row: dict[str, Any], arm: str) -> None:
    forbidden = {"answer", "correct_answer", "标准答案", "正确答案", "答案", "答案内容", "解析"} & set(row)
    # Frozen TER retrieval is sourced from a prior Qwen result object, which
    # necessarily carries historical answer/critic fields.  For this arm those
    # fields are explicitly ignored: only question metadata and `retrieved` are
    # read into the new base-model prompt.  Dense retrieval JSONL must remain
    # strictly answer-blind.
    if forbidden and arm != "ter":
        raise ValueError(f"Answer-key fields forbidden in frozen evidence: {sorted(forbidden)}")
    if not row.get("application_id") or not row.get("question"):
        raise ValueError("Missing frozen question fields")
    if arm == "dense" and not isinstance(row.get("evidence"), list):
        raise ValueError("Dense row requires evidence list")
    if arm == "ter" and not isinstance(row.get("retrieved"), list):
        raise ValueError("TER row requires retrieved list")


def completed(path: Path, rows: list[dict[str, Any]], arm: str) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    existing = list(read_jsonl(path))
    allowed = {str(row["application_id"]): row for row in rows}
    done = set()
    for row in existing:
        key = str(row.get("application_id", ""))
        if key in done or key not in allowed or row.get("status") != "ok" or row.get("method") != arm or not completed_prediction(row):
            raise ValueError("Corrupt, duplicate, mismatched, or incomplete frozen-evidence output")
        field = "evidence" if arm == "dense" else "retrieved"
        if row.get("question") != allowed[key]["question"] or row.get(field) != allowed[key][field]:
            raise ValueError("Frozen question/evidence changed on resume")
        done.add(key)
    return existing, done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("dense", "ter"), required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8002")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--invalid-choice-policy", choices=("default_a", "mark_invalid"), default="default_a")
    parser.add_argument("--runtime-contract", type=Path, help="Passed deployment preflight; freezes corrected-run resume settings")
    args = parser.parse_args()
    rows = list(read_jsonl(args.evidence_input))
    for row in rows:
        validate(row, args.arm)
    if not rows:
        raise ValueError("No frozen evidence rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume")
    contract = {
        "arm": args.arm, "generator_model": args.generator_model, "api_base": args.api_base,
        "evidence_input": str(args.evidence_input), "evidence_input_sha256": file_hash(args.evidence_input),
        "answer_key_used_for_model_input": False, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "invalid_choice_policy": args.invalid_choice_policy,
        "evidence_field_limit": args.evidence_field_limit,
        "prompt_sha256": digest(json.dumps(dense_messages(rows[0]) if args.arm == "dense" else ter_messages(rows[0], args.evidence_field_limit), ensure_ascii=False)),
    }
    if args.runtime_contract:
        contract["runtime"] = json.loads(args.runtime_contract.read_text(encoding="utf-8"))["runtime"]
        contract["runner_sha256"] = file_hash(Path(__file__))
        freeze_protocol(args.output.with_name("generation_contract.json"), contract, args.output)
    existing, done = completed(args.output, rows, args.arm) if args.resume else ([], set())
    client = batch_client(args.api_base, args.generator_model, args.max_new_tokens, 300, 2, 20)
    pending = [row for row in rows if str(row["application_id"]) not in done]
    started = time.monotonic()
    initial = len(existing)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        if args.arm == "dense":
            prompts = [dense_messages(row) for row in batch]
        else:
            prompts = [ter_messages(row, args.evidence_field_limit) for row in batch]
        raw = client(prompts)
        contexts = [{"question": row["question"]} for row in batch]
        choices = enforce_choices(client, contexts, raw, invalid_policy=args.invalid_choice_policy)
        for row, context, raw_answer, (answer, origin) in zip(batch, contexts, raw, choices):
            public = {key: row.get(key) for key in (
                "application_id", "question_id", "question", "retrieval_query", "year", "exam", "unit",
                "source_number", "question_type", "shared_option_group", "data_status", "source_row",
            )}
            preserved = row.get("evidence") if args.arm == "dense" else row.get("retrieved")
            existing.append({**public, "evidence" if args.arm == "dense" else "retrieved": preserved,
                             # EviCR v3 consumes the TER direct answer as y0.  Dense
                             # RAG has no downstream Critic stage, so it does not need
                             # this duplicate field.
                             **({"initial_answer": answer} if args.arm == "ter" else {}),
                             "answer": answer, "raw_answer": raw_answer, "choice_origin": origin,
                             "repair_raw_answer": context.get("repair_raw_answer"),
                             "status": "ok", "method": args.arm})
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in existing:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(args.output)
        elapsed = time.monotonic() - started
        new_count = max(1, len(existing) - initial)
        summary = {
            "status": "complete" if len(existing) == len(rows) else "running", "arm": args.arm,
            "completed": len(existing), "total": len(rows), "model": args.generator_model,
            "choice_origins": dict(Counter(item["choice_origin"] for item in existing)),
            "evidence_input_sha256": file_hash(args.evidence_input),
            "elapsed_this_run_seconds": round(elapsed),
            "eta_seconds": round((len(rows) - len(existing)) * elapsed / new_count),
        }
        atomic_json(args.output.with_name("summary.json"), summary)
        log("frozen_evidence_generation", **summary)
    atomic_json(args.output.with_name("generation_contract.json"), contract)


if __name__ == "__main__":
    main()
