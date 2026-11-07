#!/usr/bin/env python3
"""Run an evidence-refinement ablation on questions already answered incorrectly.

The input is an existing answer-blind AlignRAG result file plus the local
grading CSV used only to select rows whose answer was wrong.  The answer key
is never placed in a model prompt.  The new pipeline is:

    frozen R4 evidence -> evidence critic -> answer generator

The evidence critic does not see the previous answer.  It may add a clearly
labelled ``model_knowledge`` supplement when the retrieved evidence is
incomplete, but it must keep that supplement separate from quoted/retrieved
evidence.  The answer model sees both the original evidence and the critic's
structured evidence package.
"""

from __future__ import annotations

import argparse
from benchmark_integrity import completed_prediction, freeze_protocol, sha256
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_integrity import atomic_jsonl, keyed, ANSWER


FORBIDDEN_KEYS = {
    "answer", "correct_answer", "标准答案", "正确答案", "答案", "答案内容", "解析",
    "reference_answer", "model_answer",
}
CHOICE = re.compile(r"答案\s*[：:]\s*([A-E])\b", re.IGNORECASE)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield value


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def batched(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def compact(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit]


def endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def batch_client(
    api_base: str,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    max_retry_delay: int,
):
    """Call the local OpenAI-compatible batch endpoint."""
    url = endpoint(api_base).replace("/chat/completions", "/chat/completions/batch")

    def call(message_batches: list[list[dict[str, str]]]) -> list[str]:
        if not message_batches:
            return []
        body = json.dumps(
            {
                "requests": [
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                        "stream": False,
                    }
                    for messages in message_batches
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                request = Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                responses = payload["data"]
                if not isinstance(responses, list) or len(responses) != len(message_batches):
                    raise RuntimeError("Batch endpoint returned an unexpected number of completions")
                texts = [str(item["choices"][0]["message"]["content"]).strip() for item in responses]
                if any(not text for text in texts):
                    raise RuntimeError("Batch endpoint returned an empty completion")
                return texts
            except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError,
                    json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(min(max_retry_delay, 2 ** attempt))
        raise RuntimeError(f"Batch completion failed after {retries + 1} attempts: {last_error}")

    return call


def evidence_from_result(row: dict[str, Any], per_field_limit: int) -> str:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or not retrieved:
        raise ValueError("Each source result needs a non-empty retrieved list")
    application_id = str(row.get("application_id", ""))
    blocks: list[str] = []
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"Evidence {rank} has no payload")
        if str(payload.get("application_id", "")) == application_id:
            raise ValueError(f"Self-retrieval detected for {application_id}")
        blocks.append(
            "\n".join(
                (
                    f"[E{rank} | {compact(payload.get('e_id'), 80)} | {compact(payload.get('relation'), 100)}]",
                    f"历史问题：{compact(payload.get('application_question'), per_field_limit)}",
                    f"历史回答：{compact(payload.get('application_answer'), per_field_limit)}",
                    f"K 知识：{compact(payload.get('knowledge_text'), per_field_limit)}",
                )
            )
        )
    return "\n\n".join(blocks)


def load_grading_rows(grading_csv: Path) -> tuple[set[str], dict[str, str]]:
    """Load wrong-row IDs and references for post-generation-only scoring.

    The reference choice is kept in a local evaluation map.  It is consulted
    only after a model answer has been generated and is never interpolated into
    a Critic or generator prompt.
    """
    wrong: set[str] = set()
    references: dict[str, str] = {}
    with grading_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("reference_status") != "valid":
                continue
            application_id = str(row.get("application_id", ""))
            reference = str(row.get("reference_answer", "")).strip().upper()
            if application_id and reference in "ABCDE":
                references[application_id] = reference
            score = row.get("score", "")
            if score != "0":
                continue
            wrong.add(application_id)
    return {item for item in wrong if item}, references


def evidence_critic_messages(question: str, evidence: str) -> list[dict[str, str]]:
    schema = (
        '{"evidence_status":"supported|partial|insufficient|conflict",'
        '"evidence_actions":[{"evidence_id":"E1","action":"keep|downweight|discard",'
        '"reason":"..."}],'
        '"knowledge_supplements":[{"claim":"...","confidence":"high|medium|low"}],'
        '"change_support":"none|weak|strong",'
        '"answer_guidance":"..."}'
    )
    return [
        {
            "role": "system",
            "content": (
                "你是中医单选题的证据整理 Critic。只审查问题和 R4 检索结果，不看任何初稿、标准答案或答案键。"
                "逐条判断 E1-E3 的相关性，保留 K 知识，降低或删除无关历史案例。"
                "你可以依据自身稳定的中医知识提出最多 2 条补充，并且必须放在 knowledge_supplements；"
                "这些补充不能伪装成检索证据。你不负责直接选择 A-E，只判断证据是否足以支持改变回答。"
                "change_support 只有在检索证据或高置信度补充明确支持改动时才可为 strong；证据不足时为 none。"
                "只输出一个紧凑 JSON，不要 Markdown、解释、重复证据或额外字段；每个 reason、claim、answer_guidance 尽量一句话。"
                "evidence_actions 最多 3 项，knowledge_supplements 最多 2 项。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：\n{question}\n\nR4 检索结果：\n{evidence}\n\n请严格输出：\n{schema}",
        },
    ]


def evidence_critic_repair_messages(
    question: str, evidence: str, invalid_output: str
) -> list[dict[str, str]]:
    """Ask the same critic to repair formatting without changing its evidence task."""
    return [
        {
            "role": "system",
            "content": (
                "把候选输出修复为合法 JSON。不得添加答案键、标准答案或新的检索文档。"
                "knowledge_supplements 若保留，basis 必须为 model_knowledge；只输出 JSON，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：\n{question}\n\nR4 检索结果：\n{evidence}\n\n"
                f"候选输出：\n{compact(invalid_output, 3600)}\n\n"
                "请按原任务的 JSON schema 修复输出。"
            ),
        },
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Critic did not return a JSON object")
    value = json.loads(candidate[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Critic JSON must be an object")
    return value


def validate_refinement(value: dict[str, Any], evidence_count: int) -> dict[str, Any]:
    status = value.get("evidence_status")
    if status not in {"supported", "partial", "insufficient", "conflict"}:
        raise ValueError("Invalid evidence_status")
    edits = value.get("evidence_actions")
    supplements = value.get("knowledge_supplements")
    change_support = value.get("change_support")
    guidance = value.get("answer_guidance")
    if not isinstance(edits, list) or len(edits) > 3:
        raise ValueError("evidence_actions must contain 0 to 3 items")
    if not isinstance(supplements, list) or len(supplements) > 2:
        raise ValueError("knowledge_supplements must contain 0 to 2 items")
    if change_support not in {"none", "weak", "strong"}:
        raise ValueError("Invalid change_support")
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("answer_guidance must be non-empty")
    clean_edits = []
    for item in edits:
        if not isinstance(item, dict) or item.get("action") not in {"keep", "downweight", "discard"}:
            raise ValueError("Invalid evidence edit")
        evidence_id = str(item.get("evidence_id", ""))
        if not re.fullmatch(r"E[1-9][0-9]*", evidence_id) or int(evidence_id[1:]) > evidence_count:
            raise ValueError(f"Invalid evidence id: {evidence_id}")
        clean_edits.append({
            "evidence_id": evidence_id,
            "action": item["action"],
            "reason": compact(item.get("reason"), 240),
        })
    clean_supplements = []
    for item in supplements:
        if not isinstance(item, dict):
            raise ValueError("Invalid knowledge supplement")
        if item.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError("Invalid supplement confidence")
        claim = compact(item.get("claim"), 360)
        if not claim:
            raise ValueError("Knowledge supplements need claim")
        clean_supplements.append({
            "claim": claim,
            "basis": "model_knowledge",
            "confidence": item["confidence"],
        })
    return {
        "evidence_status": status,
        "evidence_actions": clean_edits,
        "knowledge_supplements": clean_supplements,
        "change_support": change_support,
        "answer_guidance": compact(guidance, 360),
    }


def fallback_refinement() -> dict[str, Any]:
    return {
        "evidence_status": "partial",
        "evidence_actions": [],
        "knowledge_supplements": [],
        "change_support": "none",
        "answer_guidance": "证据不足时仍须结合题干、选项和模型知识作出最佳选择。",
    }


def answer_messages(
    question: str,
    evidence: str,
    initial_answer: str,
    refinement: dict[str, Any],
) -> list[dict[str, str]]:
    package = json.dumps(refinement, ensure_ascii=False, separators=(",", ":"))
    return [
        {
            "role": "system",
            "content": (
                "你正在完成中医执业医师单选题。R4 检索结果是辅助资料；证据整理器的 evidence_actions 用于筛选证据，"
                "knowledge_supplements 是模型自身知识，不能冒充 R4 证据。原始初稿 y0 是一个候选答案，默认保留其判断。"
                "只有证据整理结果明确支持改动时才改变 y0；change_support=none 时优先保留 y0。"
                "资料不足时不能拒答，必须依据题干、选项和可靠知识选择最合理的一项。"
                "最终只输出一行“答案：X”，X 必须是 A、B、C、D、E 之一；不要输出解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题目：\n{question}\n\n原始初稿 y0：\n{initial_answer}\n\n"
                f"原始 R4 检索结果：\n{evidence}\n\n"
                f"证据整理结果：\n{package}\n\n请输出最终选项。"
            ),
        },
    ]


def extract_choice(text: str) -> str | None:
    matches = CHOICE.findall(str(text))
    return matches[-1].upper() if matches else None


def repair_messages(question: str, raw_answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "必须完成单选题，只输出一行“答案：X”，X 为 A、B、C、D、E 之一。",
        },
        {"role": "user", "content": f"题目：\n{question}\n\n模型原始输出：\n{raw_answer}"},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True,
                        help="Existing AlignRAG result JSONL containing frozen R4 retrieved evidence.")
    parser.add_argument("--grading-csv", type=Path, required=True,
                        help="Local grading CSV; used only to select valid-key wrong rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wrong-from", choices=("alignrag", "plain_rag", "either"), default="alignrag",
                        help="Which grading CSV selects wrong rows. Default: AlignRAG.")
    # Kept for backward compatibility with the original single-Qwen service.
    # Cross-model benchmarks use the explicit generator/critic endpoints below:
    # a candidate base model answers, while the frozen Qwen3.5+LoRA model acts
    # only as the evidence Critic.
    parser.add_argument("--api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--generator-api-base", help="OpenAI-compatible endpoint for the evaluated base model")
    parser.add_argument("--critic-api-base", help="OpenAI-compatible endpoint for the frozen Qwen Critic")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--critic-max-tokens", type=int, default=640)
    parser.add_argument("--generator-max-tokens", type=int, default=32)
    parser.add_argument("--critic-repair-attempts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-retry-delay", type=int, default=20)
    parser.add_argument("--limit", type=int, help="Smoke-test limit after wrong-row selection.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--invalid-choice-policy", choices=("default_a", "mark_invalid"), default="default_a")
    parser.add_argument("--runtime-contract", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_results.is_file() or not args.grading_csv.is_file():
        raise FileNotFoundError("source-results and grading-csv must exist")
    wrong_ids, reference_answers = load_grading_rows(args.grading_csv)
    source_rows = []
    for row in read_jsonl(args.source_results):
        forbidden = FORBIDDEN_KEYS & set(row)
        # Existing result rows necessarily contain answer fields, so only check
        # the fields that could be forwarded into a prompt below.
        if not row.get("application_id") or not row.get("question"):
            continue
        if str(row["application_id"]) in wrong_ids:
            source_rows.append(row)
    source_rows.sort(key=lambda row: str(row["application_id"]))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        source_rows = source_rows[:args.limit]
    selected_by_id = keyed(source_rows)
    if args.limit is None and set(selected_by_id) != wrong_ids:
        raise ValueError("Some selected wrong question IDs are absent from source")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "evidence_refinement_results.jsonl"
    summary_path = args.output_dir / "summary.json"
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}; use --resume")
    if args.runtime_contract:
        contract = {
            "runtime": json.loads(args.runtime_contract.read_text(encoding="utf-8"))["runtime"],
            "runner_sha256": sha256(Path(__file__)),
            "source_sha256": sha256(args.source_results), "grading_sha256": sha256(args.grading_csv),
            "generator_model": args.generator_model, "critic_model": args.critic_model,
            "generator_max_tokens": args.generator_max_tokens, "critic_max_tokens": args.critic_max_tokens,
            "invalid_choice_policy": args.invalid_choice_policy, "batch_size": args.batch_size,
            "evidence_field_limit": args.evidence_field_limit, "critic_repair_attempts": args.critic_repair_attempts,
            "limit": args.limit, "wrong_from": args.wrong_from,
            "answer_key_used_for_model_input": False,
        }
        freeze_protocol(args.output_dir / "generation_contract.json", contract, output)
    # Resume only fully usable output rows.  A failed network/GPU batch must
    # remain pending rather than silently being treated as complete.
    completed = list(read_jsonl(output)) if args.resume and output.exists() else []
    done = set(keyed(completed))
    for row in completed:
        key = str(row["application_id"])
        expected = selected_by_id.get(key)
        if (expected is None or not completed_prediction(row)
                or row.get("question") != expected["question"]
                or row.get("retrieved") != expected["retrieved"]
                or row.get("initial_answer") != expected.get("initial_answer", "")
                or row.get("critic_status") not in {"valid", "repaired", "fallback"}):
            raise ValueError("Invalid or mismatched existing refinement row")
    if not output.exists():
        atomic_jsonl(output, [])
    generator_api_base = args.generator_api_base or args.api_base
    critic_api_base = args.critic_api_base or args.api_base
    critic = batch_client(critic_api_base, args.critic_model, args.critic_max_tokens, args.timeout, args.retries, args.max_retry_delay)
    generator = batch_client(generator_api_base, args.generator_model, args.generator_max_tokens, args.timeout, args.retries, args.max_retry_delay)
    failures = 0
    selected = [row for row in source_rows if str(row["application_id"]) not in done]
    live_correct = 0
    live_wrong = 0
    live_unscorable = 0
    if done:
        for row in read_jsonl(output):
            application_id = str(row.get("application_id", ""))
            if application_id not in done:
                continue
            choice = extract_choice(str(row.get("answer", "")))
            reference = reference_answers.get(application_id)
            if not reference:
                live_unscorable += 1
            elif choice == reference:
                live_correct += 1
            else:
                live_wrong += 1
    total_selected = len(source_rows)
    print(
        f"[evidence refinement] selected={total_selected} wrong rows; "
        f"resume_done={len(done)}; pending={len(selected)}; "
        "reference answers are evaluation-only",
        flush=True,
    )
    for batch in batched(selected, args.batch_size):
        contexts = []
        for row in batch:
            evidence = evidence_from_result(row, args.evidence_field_limit)
            contexts.append({
                "row": row,
                "application_id": str(row["application_id"]),
                "question": str(row["question"]),
                "initial_answer": str(row.get("initial_answer", "")).strip(),
                "evidence": evidence,
                "retrieved": row["retrieved"],
            })
        try:
            raw_critics = critic([evidence_critic_messages(item["question"], item["evidence"]) for item in contexts])
            repair_contexts = []
            for item, raw in zip(contexts, raw_critics):
                item["critic_raw"] = raw
                try:
                    item["refinement"] = validate_refinement(parse_json_object(raw), len(item["retrieved"]))
                    item["critic_status"] = "valid"
                except (ValueError, json.JSONDecodeError):
                    item["critic_status"] = "invalid"
                    repair_contexts.append(item)
            for _ in range(args.critic_repair_attempts):
                if not repair_contexts:
                    break
                repair_prompts = [
                    evidence_critic_repair_messages(item["question"], item["evidence"], item["critic_raw"])
                    for item in repair_contexts
                ]
                repaired = critic(repair_prompts)
                remaining = []
                for item, raw in zip(repair_contexts, repaired):
                    item["critic_raw"] = raw
                    try:
                        item["refinement"] = validate_refinement(parse_json_object(raw), len(item["retrieved"]))
                        item["critic_status"] = "repaired"
                    except (ValueError, json.JSONDecodeError):
                        remaining.append(item)
                repair_contexts = remaining
            for item in repair_contexts:
                item["refinement"] = fallback_refinement()
                item["critic_status"] = "fallback"

            answers = generator([
                answer_messages(
                    item["question"], item["evidence"], item["initial_answer"], item["refinement"]
                )
                for item in contexts
            ])
            repair_answer_indices = []
            for index, (item, answer) in enumerate(zip(contexts, answers)):
                choice = extract_choice(answer)
                initial_choice = extract_choice(item["initial_answer"])
                # Strong gate: evidence refinement can steer a correction only
                # when it explicitly reports strong support.  Otherwise preserve
                # the original answer candidate to prevent unsupported drift.
                if initial_choice and item["refinement"].get("change_support") != "strong":
                    item["answer"] = f"答案：{initial_choice}"
                    item["choice_status"] = "preserved_y0"
                elif choice:
                    item["answer"] = f"答案：{choice}"
                    item["choice_status"] = "model"
                else:
                    repair_answer_indices.append(index)
            if repair_answer_indices:
                repaired_answers = generator([repair_messages(contexts[index]["question"], answers[index]) for index in repair_answer_indices])
                for index, answer in zip(repair_answer_indices, repaired_answers):
                    contexts[index]["repair_raw_answer"] = answer
                    choice = extract_choice(answer)
                    contexts[index]["answer"] = (f"答案：{choice}" if choice else
                                                ("INVALID" if args.invalid_choice_policy == "mark_invalid" else "答案：A"))
                    contexts[index]["choice_status"] = ("format_repair" if choice else
                                                       ("invalid_output" if args.invalid_choice_policy == "mark_invalid" else "forced_default_A"))
            for index, item in enumerate(contexts):
                row = item["row"]
                completed.append({
                    "application_id": item["application_id"],
                    "question_id": row.get("question_id", ""),
                    "question": item["question"],
                    "retrieval_query": row.get("retrieval_query", ""),
                    "year": row.get("year"),
                    "exam": row.get("exam", ""),
                    "unit": row.get("unit", ""),
                    "source_number": row.get("source_number"),
                    "retrieved": item["retrieved"],
                    "initial_answer": item["initial_answer"],
                    "evidence_refinement": item["refinement"],
                    "critic_raw": item["critic_raw"],
                    "critic_status": item["critic_status"],
                    "answer": item["answer"],
                    "choice_status": item["choice_status"],
                    "raw_answer": answers[index],
                    "repair_raw_answer": item.get("repair_raw_answer"),
                    "config": {
                        "pipeline": "frozen R4 evidence -> evidence critic -> answer generator",
                        "wrong_from": args.wrong_from,
                        "batch_size": args.batch_size,
                        "generator_api_base": generator_api_base,
                        "critic_api_base": critic_api_base,
                        "generator_model": args.generator_model,
                        "critic_model": args.critic_model,
                        "prompt_version": "evidence-refinement-questionbank-v3",
                        "generator_max_tokens": args.generator_max_tokens,
                        "invalid_choice_policy": args.invalid_choice_policy,
                    },
                })
                # Atomic replacement keeps every committed row parseable even
                # if the server or supervisor is interrupted during a write.
                atomic_jsonl(output, completed)
                reference = reference_answers.get(item["application_id"])
                choice = extract_choice(item["answer"])
                if reference and choice:
                    if choice == reference:
                        live_correct += 1
                    else:
                        live_wrong += 1
                elif reference:
                    live_wrong += 1
                else:
                    live_unscorable += 1
                scored = live_correct + live_wrong
                accuracy = (live_correct / scored * 100.0) if scored else 0.0
                print(
                    f"[evidence refinement] {item['application_id']} "
                    f"answer={item['answer']} | cumulative "
                    f"correct={live_correct} wrong={live_wrong} unscorable={live_unscorable} "
                    f"accuracy={accuracy:.2f}% ({scored}/{total_selected})",
                    flush=True,
                )
        except Exception as error:
            # Do not append terminal error rows.  Completed prior batches are
            # already durable; raising leaves this batch pending for --resume.
            print(f"[evidence refinement] batch error: {error}", flush=True)
            raise
    summary = {
        "status": "complete" if failures == 0 else "completed_with_failures",
        "selected_wrong_rows": len(source_rows),
        "completed_rows": len({str(row.get('application_id')) for row in read_jsonl(output)}),
        "new_failures": failures,
        "post_generation_scoring": {
            "correct": live_correct,
            "wrong": live_wrong,
            "unscorable": live_unscorable,
            "scored": live_correct + live_wrong,
            "accuracy_on_scored": round(live_correct / (live_correct + live_wrong) * 100, 2)
            if live_correct + live_wrong else 0.0,
            "reference_answers_entered_model_prompt": False,
            "invalid_outputs": sum(row.get("choice_status") == "invalid_output" for row in completed),
        },
        "wrong_from": args.wrong_from,
        "answer_key_used_for_model_input": False,
        "pipeline": "frozen R4 evidence -> evidence critic -> answer generator",
        "generator_api_base": generator_api_base,
        "critic_api_base": critic_api_base,
        "prompt_version": "evidence-refinement-questionbank-v3",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
