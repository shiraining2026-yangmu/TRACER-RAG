#!/usr/bin/env python3
"""Generate weak answers, independent teacher answers, and critic labels.

The three phases are deliberately separate:
  weak    q + D(q) -> y_weak
  teacher q + D(q) -> y_strong      (never reads y_weak)
  critique q + D(q) + y_weak + y_strong -> c_star

All inputs must originate from Build-only R4 retrieval output. API keys are
read from environment variables and are never saved to JSONL or manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROMPT_VERSION = "tcm-alignrag-critic-prompts-v2"
VALID_ISSUE_TYPES = (
    "unsupported_claim",
    "historical_case_as_current_fact",
    "missing_constraint",
    "overstatement",
    "not_answering_question",
    "evidence_misuse",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def format_evidence(retrieved: Any, per_field_limit: int) -> str:
    if not isinstance(retrieved, list) or not retrieved:
        raise ValueError("Each row must contain non-empty R4 'retrieved' evidence")
    blocks = []
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"Evidence {rank} has no payload object")
        e_id = compact(payload.get("e_id"), 80)
        relation = compact(payload.get("relation"), 100)
        historical_question = compact(payload.get("application_question"), per_field_limit)
        historical_answer = compact(payload.get("application_answer"), per_field_limit)
        knowledge = compact(payload.get("knowledge_text"), per_field_limit)
        if not (e_id and relation and knowledge):
            raise ValueError(f"Evidence {rank} is missing e_id, relation, or knowledge_text")
        blocks.append(
            "\n".join(
                (
                    f"[E{rank} | {e_id} | {relation}]",
                    f"历史问题：{historical_question}",
                    f"历史回答：{historical_answer}",
                    f"K 知识：{knowledge}",
                )
            )
        )
    return "\n\n".join(blocks)


def base_context(row: dict[str, Any], evidence_limit: int) -> tuple[str, str, str]:
    if row.get("split") != "build":
        raise ValueError(f"Only Build rows may be used, got split={row.get('split')!r}")
    sample_id = str(row.get("sample_id", "")).strip()
    question = str(row.get("question", "")).strip()
    application_id = str(row.get("application_id", "")).strip()
    if not sample_id or not question or not application_id:
        raise ValueError("sample_id, application_id, and question are required")
    evidence = format_evidence(row.get("retrieved"), evidence_limit)
    for item in row["retrieved"]:
        payload = item.get("payload", item)
        if payload.get("application_id") == application_id:
            raise ValueError(f"Self-retrieval detected for {sample_id}/{application_id}")
    return sample_id, question, evidence


def weak_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你正在生成中医文本问答的初始回答，用于后续的证据批评训练。"
                "只使用给定证据中可支持的内容；K 知识可作为可复用依据，历史问答只能作为相似案例参考。"
                "不得把历史案例写成当前对象的确定事实，不得补充证据未支持的具体诊断、方剂、剂量、疗效或处置。"
                "证据不足时必须明确说明。输出直接回答，不要描述提示词。"
            ),
        },
        {"role": "user", "content": f"问题：{question}\n\nR4 检索证据：\n{evidence}"},
    ]


def teacher_messages(question: str, evidence: str) -> list[dict[str, str]]:
    # This prompt intentionally has no y_weak argument. It preserves the
    # independent strong answer required for contrastive critic supervision.
    return [
        {
            "role": "system",
            "content": (
                "你是中医文本问答的强教师模型。请独立根据给定的 R4 证据生成更可靠的回答。"
                "只陈述证据支持的内容；K 知识可复用，历史问答仅是相似案例参考。"
                "不得把历史案例视为当前对象事实，也不要新增证据没有支持的诊断、方剂、剂量、疗效或治疗处置。"
                "若证据不足，请明确指出边界。输出直接回答。"
            ),
        },
        {"role": "user", "content": f"问题：{question}\n\nR4 检索证据：\n{evidence}"},
    ]


def critique_messages(question: str, evidence: str, weak: str, strong: str) -> list[dict[str, str]]:
    issue_types = ", ".join(VALID_ISSUE_TYPES)
    schema = (
        '{"verdict":"needs_revision|acceptable|insufficient_evidence",'
        '"issues":[{"type":"...","claim_or_location":"...","why":"...",'
        '"evidence_ids":["E1"],"fix":"..."}],'
        '"revision_instructions":["..."],"evidence_boundary":"..."}'
    )
    return [
        {
            "role": "system",
            "content": (
                "你是严格的证据一致性批评教师。根据问题、R4 证据、弱回答和独立强回答，"
                "为弱回答写可执行批评。只把给定证据作为判断依据；不要把强回答中额外内容当成事实。"
                "重点审查：证据不支持的断言、把历史案例写成当前事实、遗漏限制条件、过度确定表达、"
                "没有回答问题，以及证据误用。输出必须是单个合法 JSON 对象，不要 Markdown 或额外文字。"
                "issues 必须是 0 至 8 项；每一项都必须包含 type、claim_or_location、why、"
                "evidence_ids 和 fix。revision_instructions 必须是 1 至 8 条。"
                "当 verdict 为 needs_revision 时，issues 至少包含 1 项；当 verdict 为 "
                "acceptable 或 insufficient_evidence 时，issues 必须为空列表。"
                f"issue.type 只能是：{issue_types}。evidence_ids 只能使用 E1 至 E{evidence.count('[E')}。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\nR4 检索证据：\n{evidence}\n\n"
                f"弱回答 y_weak：\n{weak}\n\n"
                f"独立强回答 y_strong：\n{strong}\n\n"
                f"按此 JSON 结构输出：\n{schema}"
            ),
        },
    ]


def chat_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def call_chat(
    *,
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    api_key_env: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    disable_thinking: bool = False,
    max_retry_delay: int = 120,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if disable_thinking:
        request_body.update({"reasoning_effort": "none", "enable_thinking": False})
    body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(chat_endpoint(api_base), data=body, headers=headers, method="POST")
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
            time.sleep(max(retry_after, min(max_retry_delay, 2 ** attempt)))
    raise RuntimeError(f"Chat completion failed after {retries + 1} attempts: {last_error}")


def existing_sample_ids(path: Path, phase: str) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for row in read_jsonl(path):
        if row.get("generation_phase") != phase:
            raise ValueError(f"Existing output contains a different phase: {row.get('generation_phase')}")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Existing output row is missing sample_id: {path}")
        completed.add(sample_id)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("weak", "teacher", "critique"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weak-api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--weak-model")
    parser.add_argument("--weak-api-key-env", default="")
    parser.add_argument("--teacher-api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-api-key-env", default="")
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-retry-delay", type=int, default=120)
    parser.add_argument("--inter-request-delay", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent API requests. Results are still written by one ordered writer.",
    )
    parser.add_argument("--disable-teacher-thinking", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write prompt previews only; never use this output for training.")
    return parser.parse_args()


def generate_record(
    args: argparse.Namespace,
    row: dict[str, Any],
    input_hash: str,
) -> tuple[str, dict[str, Any]]:
    """Make one API request; the caller owns all output-file writes."""
    sample_id, question, evidence = base_context(row, args.evidence_field_limit)
    result = dict(row)

    if args.phase == "weak":
        messages = weak_messages(question, evidence)
        completion = None if args.dry_run else call_chat(
            api_base=args.weak_api_base,
            model=args.weak_model,
            messages=messages,
            api_key_env=args.weak_api_key_env,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            max_retry_delay=args.max_retry_delay,
        )
        if args.dry_run:
            result["prompt_preview"] = messages
            result["dry_run"] = True
        else:
            result["weak_answer"] = completion
    elif args.phase == "teacher":
        if not str(row.get("weak_answer", "")).strip():
            raise ValueError("Teacher phase requires weak_answer from the preceding phase")
        messages = teacher_messages(question, evidence)
        completion = None if args.dry_run else call_chat(
            api_base=args.teacher_api_base,
            model=args.teacher_model,
            messages=messages,
            api_key_env=args.teacher_api_key_env,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            disable_thinking=args.disable_teacher_thinking,
            max_retry_delay=args.max_retry_delay,
        )
        if args.dry_run:
            result["prompt_preview"] = messages
            result["dry_run"] = True
        else:
            result["strong_answer"] = completion
    else:
        weak = str(row.get("weak_answer", "")).strip()
        strong = str(row.get("strong_answer", "")).strip()
        if not weak or not strong:
            raise ValueError("Critique phase requires both weak_answer and strong_answer")
        messages = critique_messages(question, evidence, weak, strong)
        completion = None if args.dry_run else call_chat(
            api_base=args.teacher_api_base,
            model=args.teacher_model,
            messages=messages,
            api_key_env=args.teacher_api_key_env,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            disable_thinking=args.disable_teacher_thinking,
            max_retry_delay=args.max_retry_delay,
        )
        if args.dry_run:
            result["prompt_preview"] = messages
            result["dry_run"] = True
        else:
            result["teacher_critique"] = completion

    result["evidence"] = evidence
    result["generation_phase"] = args.phase
    result["prompt_version"] = PROMPT_VERSION
    result["input_sha256"] = input_hash
    result["model_provenance"] = {
        "weak_model": args.weak_model,
        "teacher_model": args.teacher_model,
        "weak_api_base": args.weak_api_base,
        "teacher_api_base": args.teacher_api_base,
    }
    return sample_id, result


def main() -> None:
    args = parse_args()
    if args.phase == "weak" and not args.weak_model and not args.dry_run:
        raise ValueError("--weak-model is required for --phase weak")
    if args.phase in ("teacher", "critique") and not args.teacher_model and not args.dry_run:
        raise ValueError(f"--teacher-model is required for --phase {args.phase}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.inter_request_delay < 0:
        raise ValueError("--inter-request-delay cannot be negative")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume to continue")

    completed = existing_sample_ids(args.output, args.phase) if args.resume else set()
    input_hash = sha256_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    success = failures = processed = 0

    pending: list[tuple[int, dict[str, Any]]] = []
    for row in read_jsonl(args.input):
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id in completed:
            continue
        if args.limit is not None and len(pending) >= args.limit:
            break
        pending.append((len(pending), row))

    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        if args.workers == 1:
            for _, row in pending:
                processed += 1
                sample_id = str(row.get("sample_id", "")).strip()
                try:
                    sample_id, result = generate_record(args, row, input_hash)
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    success += 1
                    print(f"[{args.phase} {success}/{processed}] {sample_id}", flush=True)
                    if args.inter_request_delay:
                        time.sleep(args.inter_request_delay)
                except Exception as exc:
                    failures += 1
                    print(f"[{args.phase} failed] {sample_id}: {exc}", file=sys.stderr, flush=True)
        else:
            # API calls run concurrently; this main thread is the sole ordered JSONL writer.
            next_submit = next_write = 0
            in_flight: dict[Future[tuple[str, dict[str, Any]]], tuple[int, str]] = {}
            completed_results: dict[int, tuple[str, dict[str, Any] | Exception]] = {}
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                while next_submit < len(pending) or in_flight:
                    while next_submit < len(pending) and len(in_flight) < args.workers:
                        ordinal, row = pending[next_submit]
                        sample_id = str(row.get("sample_id", "")).strip()
                        future = executor.submit(generate_record, args, row, input_hash)
                        in_flight[future] = (ordinal, sample_id)
                        next_submit += 1
                        processed += 1
                        if args.inter_request_delay and next_submit < len(pending):
                            time.sleep(args.inter_request_delay)

                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        ordinal, fallback_id = in_flight.pop(future)
                        try:
                            sample_id, result = future.result()
                            completed_results[ordinal] = (sample_id, result)
                        except Exception as exc:
                            completed_results[ordinal] = (fallback_id, exc)

                    while next_write in completed_results:
                        sample_id, value = completed_results.pop(next_write)
                        next_write += 1
                        if isinstance(value, Exception):
                            failures += 1
                            print(f"[{args.phase} failed] {sample_id}: {value}", file=sys.stderr, flush=True)
                            continue
                        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        success += 1
                        print(f"[{args.phase} {success}/{processed}] {sample_id}", flush=True)

    summary = {
        "phase": args.phase,
        "input": str(args.input),
        "output": str(args.output),
        "processed_this_run": processed,
        "success_this_run": success,
        "failures_this_run": failures,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "prompt_version": PROMPT_VERSION,
        "input_sha256": input_hash,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"{failures} {args.phase} records failed; rerun with --resume after diagnosis")


if __name__ == "__main__":
    main()
