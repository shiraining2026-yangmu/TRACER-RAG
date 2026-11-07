#!/usr/bin/env python3
"""Run one-round RAG+AlignRAG inference on immutable R4 retrieval output.

The critic receives question + displayed R4 evidence + initial answer. It does
not retrieve documents, see a source answer, or see a teacher strong answer.
Use --expected-split test for the final frozen Test-111 comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_VERDICTS = {"needs_revision", "acceptable", "insufficient_evidence"}
ALLOWED_ISSUE_TYPES = {
    "unsupported_claim",
    "historical_case_as_current_fact",
    "missing_constraint",
    "overstatement",
    "not_answering_question",
    "evidence_misuse",
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{number}")
                yield value


def compact(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def evidence_from_row(row: dict[str, Any], per_field_limit: int) -> str:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or not retrieved:
        raise ValueError("Each R4 retrieval row needs non-empty retrieved evidence")
    application_id = str(row.get("application_id", ""))
    blocks = []
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"Evidence {rank} has no payload")
        if payload.get("application_id") == application_id:
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


def endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def client(
    api_base: str,
    model: str,
    api_key_env: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    max_retry_delay: int,
):
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"

    def call(messages: Sequence[dict[str, str]]) -> str:
        body = json.dumps(
            {"model": model, "messages": list(messages), "temperature": 0.0, "max_tokens": max_tokens, "stream": False},
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                request = Request(endpoint(api_base), data=body, headers=headers, method="POST")
                with urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = str(payload["choices"][0]["message"]["content"]).strip()
                if not text:
                    raise RuntimeError("Model returned an empty completion")
                return text
            except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                retry_after = 0
                if isinstance(exc, HTTPError):
                    try:
                        retry_after = int(exc.headers.get("Retry-After", "0"))
                    except ValueError:
                        retry_after = 0
                time.sleep(max(retry_after, min(max_retry_delay, 2**attempt)))
        raise RuntimeError(f"Chat completion failed after {retries + 1} attempts: {last_error}")

    return call


def generator_initial_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你正在完成中医领域的证据增强问答，不用于临床诊疗。只可依据展示的 R4 证据回答。"
                "K 知识是可复用依据；历史问答仅作相似案例参考，不能当作当前对象事实。"
                "不得引入展示证据不支持的具体诊断、方剂、剂量、疗效或事实；证据不足时须明确说明。"
            ),
        },
        {"role": "user", "content": f"问题：{question}\n\nR4 检索证据：\n{evidence}\n\n请给出简洁回答。"},
    ]


def clean_critic_string(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{field} must contain 1 to {maximum} characters")
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    """Accept one JSON object, optionally wrapped in a Markdown JSON fence."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        trailing = text[offset + end:].strip()
        if trailing and trailing != "```":
            continue
        if not isinstance(value, dict):
            break
        return value
    raise ValueError("Critic did not return one parseable JSON object")


def validate_critic_payload(value: dict[str, Any], evidence_count: int) -> dict[str, Any]:
    required = {"verdict", "issues", "revision_instructions", "evidence_boundary"}
    if set(value) != required:
        raise ValueError("Critic JSON must contain exactly the required four fields")
    verdict = clean_critic_string(value.get("verdict"), "verdict", 64)
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(f"Unsupported verdict: {verdict}")
    raw_issues = value.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError("issues must be a list")
    issues: list[dict[str, Any]] = []
    # Some otherwise complete generations enumerate redundant low-priority issues.
    # Keep the first two in the normalized contract and retain the raw completion
    # separately for audit, rather than discarding the entire Critic response.
    for index, raw_issue in enumerate(raw_issues[:2], 1):
        if not isinstance(raw_issue, dict):
            raise ValueError(f"issues[{index}] must be an object")
        issue_type = clean_critic_string(raw_issue.get("type"), f"issues[{index}].type", 64)
        if issue_type not in ALLOWED_ISSUE_TYPES:
            raise ValueError(f"Unsupported issue type: {issue_type}")
        evidence_ids = raw_issue.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not 1 <= len(evidence_ids) <= 3
            or any(not isinstance(item, str) or not re.fullmatch(r"E[1-9][0-9]*", item) for item in evidence_ids)
            or any(int(item[1:]) > evidence_count for item in evidence_ids)
        ):
            raise ValueError(f"issues[{index}].evidence_ids is invalid")
        issues.append(
            {
                "type": issue_type,
                "claim_or_location": clean_critic_string(raw_issue.get("claim_or_location"), f"issues[{index}].claim_or_location", 160),
                "why": clean_critic_string(raw_issue.get("why"), f"issues[{index}].why", 180),
                "evidence_ids": evidence_ids,
                "fix": clean_critic_string(raw_issue.get("fix"), f"issues[{index}].fix", 180),
            }
        )
    instructions = value.get("revision_instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("revision_instructions must contain at least one item")
    cleaned_instructions = [clean_critic_string(item, "revision_instruction", 180) for item in instructions[:2]]
    if verdict == "needs_revision" and not issues:
        raise ValueError("needs_revision requires at least one issue")
    if verdict in {"acceptable", "insufficient_evidence"} and issues:
        raise ValueError(f"{verdict} must not include issues")
    return {
        "verdict": verdict,
        "issues": issues,
        "revision_instructions": cleaned_instructions,
        "evidence_boundary": clean_critic_string(value.get("evidence_boundary"), "evidence_boundary", 360),
    }


def parse_critic(text: str, evidence_count: int) -> dict[str, Any]:
    return validate_critic_payload(extract_json_object(text), evidence_count)


def fallback_critique() -> dict[str, Any]:
    """A conservative but answer-preserving fallback when local JSON decoding fails."""
    return {
        "verdict": "acceptable",
        "issues": [],
        "revision_instructions": [
            "Critic 输出无效；保留初始回答中由展示证据支持、且直接回应问题的内容。",
            "仅删除明确无依据的诊断、方剂、剂量、疗效或把历史案例写成当前事实的表述。",
            "若 K 知识支持一般性说明，应使用条件性表述作答，而不是仅以证据不足替代回答。",
        ],
        "evidence_boundary": "只能依据展示的 R4 证据；K 知识可复用，历史问答不得当作当前对象事实。",
    }


def critic_messages(question: str, evidence: str, initial_answer: str) -> list[dict[str, str]]:
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
                "你是经过 LoRA 微调的中医 RAG 证据一致性 Critic。"
                "仅根据显示的 R4 证据审查初始回答。必须只输出一个合法、紧凑的 JSON 对象，"
                "不得使用 Markdown、解释文字或思考过程。"
                "verdict 只能为 needs_revision、acceptable 或 insufficient_evidence。"
                "issues 最多两项；revision_instructions 为一至两项；每项必须指向 E1 至 E3。"
                "每个字段用一句简短中文完成，不要复述题目、证据或初始回答。"
                "只有存在可定位的问题时才用 needs_revision。若回答已忠实且回答了可回答部分，"
                "使用 acceptable；不能仅因无法给出确诊就判为 insufficient_evidence。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\nR4 检索证据：\n{evidence}\n\n初始回答 y0：\n{initial_answer}\n\n"
                f"按此结构输出：{schema}"
            ),
        },
    ]


def critic_repair_messages(question: str, evidence: str, initial_answer: str, invalid_critique: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "将候选 Critic 输出修复为一个合法 JSON 对象。只输出 JSON，不要 Markdown 或解释。"
                "字段必须且只能为 verdict、issues、revision_instructions、evidence_boundary。"
                "issues 最多两项；每项必须有 type、claim_or_location、why、evidence_ids、fix；"
                "revision_instructions 为一至两项。每个文本字段只写一句简短中文，"
                "不要复述证据、问题或初始回答。若候选无法可靠修复，请返回 acceptable、空 issues，"
                "并要求保留初始回答中支持且直接回应问题的内容。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\nR4 检索证据：\n{evidence}\n\n初始回答：\n{initial_answer}\n\n"
                f"待修复候选：\n{compact(invalid_critique, 2400)}"
            ),
        },
    ]


def refinement_messages(question: str, evidence: str, initial_answer: str, critique: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "根据 Critic 的有效批评精修回答。先保留初始回答中展示证据支持且直接回应问题的内容，"
                "只修改 Critic 明确指出的问题；不要把有依据的内容整体删成泛泛的就医建议。"
                "当 K 知识支持一般性说明时，应以条件性、非确诊表述回答用户可回答的部分。"
                "只有展示证据确实不能支持可回答内容时，才说明证据边界。"
                "不得把历史案例写成当前事实，不得新增无依据诊断、方剂、剂量、疗效或处置。"
                "不要提及批评、模型或编辑过程。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\nR4 检索证据：\n{evidence}\n\n"
                f"初始回答 y0：\n{initial_answer}\n\nCritic 批评 c0：\n{critique}\n\n请输出精修回答 y1。"
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--expected-split", choices=("build", "dev", "test"), default="test")
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--generator-max-tokens", type=int, default=512)
    parser.add_argument("--critic-max-tokens", type=int, default=1024)
    parser.add_argument("--critic-repair-attempts", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-retry-delay", type=int, default=120)
    parser.add_argument("--inter-request-delay", type=float, default=0.0)
    parser.add_argument("--limit", type=int, help="Process only the first N rows; intended for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Append missing application IDs to a partial output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume to continue")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.critic_repair_attempts < 0:
        raise ValueError("--critic-repair-attempts cannot be negative")
    generator = client(
        args.api_base, args.generator_model, args.api_key_env,
        args.generator_max_tokens, args.timeout, args.retries, args.max_retry_delay,
    )
    critic = client(
        args.api_base, args.critic_model, args.api_key_env,
        args.critic_max_tokens, args.timeout, args.retries, args.max_retry_delay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.resume and args.output.exists():
        completed = {str(row.get("application_id", "")): row for row in read_jsonl(args.output)}
        if not completed or any(not application_id for application_id in completed):
            raise ValueError("Existing output contains a missing application_id")
    count = 0
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in read_jsonl(args.input):
            application_id = str(row.get("application_id", ""))
            if application_id in completed:
                continue
            if args.limit is not None and count >= args.limit:
                break
            if row.get("split") != args.expected_split:
                raise ValueError(f"Expected split {args.expected_split}, received {row.get('split')}")
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError("Missing question")
            evidence = evidence_from_row(row, args.evidence_field_limit)
            y0 = generator(generator_initial_messages(question, evidence))
            raw_critique = critic(critic_messages(question, evidence, y0))
            critic_status = "valid"
            try:
                critique = parse_critic(raw_critique, len(row["retrieved"]))
            except ValueError:
                critique = None
                critic_status = "invalid"
                for _ in range(args.critic_repair_attempts):
                    repaired = critic(critic_repair_messages(question, evidence, y0, raw_critique))
                    try:
                        critique = parse_critic(repaired, len(row["retrieved"]))
                        critic_status = "repaired"
                        break
                    except ValueError:
                        raw_critique = repaired
            if critique is None:
                critique = fallback_critique()
                critic_status = "fallback"
            c0 = json.dumps(critique, ensure_ascii=False, separators=(",", ":"))
            y1 = generator(refinement_messages(question, evidence, y0, c0))
            result = dict(row)
            result.update({
                "initial_answer": y0,
                "critic": c0,
                "critic_raw": raw_critique,
                "critic_status": critic_status,
                "answer": y1,
                "evidence": evidence,
                "alignrag_config": {"iterations": 1, "prompt_version": "tcm-alignrag-inference-v2"},
            })
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            count += 1
            print(f"[alignrag {len(completed) + count}] {application_id}", flush=True)
            if args.inter_request_delay:
                time.sleep(args.inter_request_delay)
    print(json.dumps({"output": str(args.output), "processed_this_run": count, "total_completed": len(completed) + count, "expected_split": args.expected_split}, ensure_ascii=False))


if __name__ == "__main__":
    main()
