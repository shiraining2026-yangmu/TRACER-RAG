#!/usr/bin/env python3
"""Label Build q,D(q) evidence-review tasks with a strong model.

This is a separate protocol from the historical answer-aware supervision.  The
teacher receives exactly a question and fixed R4 evidence, then emits the five
EviCR fields.  Neither the input file nor this program accepts initial answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VERSION = "evicr-qd-labels-v1"
FIELDS = {"evidence_status", "evidence_actions", "knowledge_supplements", "change_support", "answer_guidance"}
STATUSES = {"supported", "partial", "insufficient", "conflict"}
ACTIONS = {"keep", "downweight", "discard"}
CONFIDENCE = {"high", "medium", "low"}
CHANGE_SUPPORT = {"none", "weak", "strong"}
FORBIDDEN_KEYS = {"weak_answer", "strong_answer", "teacher_critique", "answer", "y0", "y_weak", "y_strong"}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qd_hash(question: str, evidence: str) -> str:
    return hashlib.sha256((question + "\n" + evidence).encode("utf-8")).hexdigest()


def json_object(value: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character == "{":
            try:
                parsed, _ = decoder.raw_decode(value[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("no JSON object found")


def string(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"invalid {name}")
    return value.strip()


def validate_package(value: dict[str, Any], evidence_count: int) -> dict[str, Any]:
    if set(value) != FIELDS:
        raise ValueError("output must contain exactly five EviCR fields")
    evidence_status = string(value.get("evidence_status"), "evidence_status", 32)
    change_support = string(value.get("change_support"), "change_support", 32)
    if evidence_status not in STATUSES or change_support not in CHANGE_SUPPORT:
        raise ValueError("invalid categorical EviCR field")
    actions = value.get("evidence_actions")
    if not isinstance(actions, list) or len(actions) != evidence_count:
        raise ValueError("evidence_actions must evaluate every evidence block")
    clean_actions: list[dict[str, str]] = []
    expected = {f"E{index}" for index in range(1, evidence_count + 1)}
    observed: set[str] = set()
    for item in actions:
        if not isinstance(item, dict) or set(item) != {"evidence_id", "action", "reason"}:
            raise ValueError("invalid evidence action")
        evidence_id = string(item.get("evidence_id"), "evidence_id", 16)
        action = string(item.get("action"), "action", 32)
        if evidence_id not in expected or evidence_id in observed or action not in ACTIONS:
            raise ValueError("invalid evidence action identifier")
        observed.add(evidence_id)
        clean_actions.append({"evidence_id": evidence_id, "action": action, "reason": string(item.get("reason"), "reason", 420)})
    if observed != expected:
        raise ValueError("not all evidence blocks were evaluated")
    supplements = value.get("knowledge_supplements")
    if not isinstance(supplements, list) or len(supplements) > 2:
        raise ValueError("invalid knowledge_supplements")
    clean_supplements: list[dict[str, str]] = []
    for item in supplements:
        if not isinstance(item, dict) or set(item) != {"claim", "basis", "confidence"} or item.get("basis") != "model_knowledge":
            raise ValueError("invalid knowledge supplement")
        confidence = string(item.get("confidence"), "confidence", 32)
        if confidence not in CONFIDENCE:
            raise ValueError("invalid knowledge supplement confidence")
        clean_supplements.append({"claim": string(item.get("claim"), "claim", 520), "basis": "model_knowledge", "confidence": confidence})
    return {
        "evidence_status": evidence_status,
        "evidence_actions": sorted(clean_actions, key=lambda item: item["evidence_id"]),
        "knowledge_supplements": clean_supplements,
        "change_support": change_support,
        "answer_guidance": string(value.get("answer_guidance"), "answer_guidance", 900),
    }


def messages(question: str, evidence: str) -> list[dict[str, str]]:
    schema = (
        '{"evidence_status":"supported|partial|insufficient|conflict",'
        '"evidence_actions":[{"evidence_id":"E1","action":"keep|downweight|discard","reason":"..."}],'
        '"knowledge_supplements":[{"claim":"...","basis":"model_knowledge","confidence":"high|medium|low"}],'
        '"change_support":"none|weak|strong","answer_guidance":"..."}'
    )
    return [
        {"role": "system", "content": (
            "你是中医 RAG 的关系证据 Critic。只审查问题和 R4 检索证据；输入仅包含这两类材料。"
            "逐条判断 E1-E3 的相关性、适用条件和支持范围。K 知识可作为可复用依据，历史案例只能作为相似案例，"
            "不能写作当前对象事实。最多给出两条模型知识补充，且必须明确标为 model_knowledge，不能冒充 R4 证据。"
            "不直接作答。只输出一个合法 JSON 对象，不要 Markdown 或额外字段。"
        )},
        {"role": "user", "content": f"问题：\n{question}\n\nR4 检索证据：\n{evidence}\n\n输出：\n{schema}"},
    ]


def repair_messages(question: str, evidence: str, candidate: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "将候选证据评议修复为合法 JSON。只可使用给定问题、R4 证据和候选文本；不添加其他上下文。"
            "字段必须且只能为 evidence_status、evidence_actions、knowledge_supplements、change_support、answer_guidance。"
            "每个 E1-E3 都须有一项 evidence_actions。只输出 JSON。"
        )},
        {"role": "user", "content": f"问题：\n{question}\n\nR4 检索证据：\n{evidence}\n\n候选输出：\n{candidate[:6000]}"},
    ]


def endpoint(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/chat/completions") else (base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions")


def chat(base: str, model: str, api_key: str, prompt: list[dict[str, str]], max_tokens: int, timeout: int, retries: int) -> str:
    body = json.dumps({"model": model, "messages": prompt, "temperature": 0.0, "max_tokens": max_tokens, "stream": False}, ensure_ascii=False).encode("utf-8")
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(endpoint(base), data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = str(payload["choices"][0]["message"]["content"]).strip()
            if not content:
                raise ValueError("empty completion")
            return content
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"teacher request failed after {retries + 1} attempts: {type(error).__name__}")


def write_record(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def existing(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return output
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in output or row.get("status") not in {"valid", "repaired"}:
            raise ValueError(f"invalid checkpoint at {path}: {sample_id}")
        output[sample_id] = row
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_seed(seed: dict[str, Any]) -> tuple[str, str, str, list[str], str, str]:
    if FORBIDDEN_KEYS & set(seed):
        raise ValueError("Seed has an answer-like field")
    if seed.get("schema_version") != "evicr-qd-seed-v1" or seed.get("split") != "build":
        raise ValueError("Only Build evicr-qd seeds are accepted")
    sample_id = string(seed.get("sample_id"), "sample_id", 128)
    question = string(seed.get("question"), "question", 6000)
    evidence = string(seed.get("evidence"), "evidence", 20000)
    evidence_ids = seed.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 3:
        raise ValueError("invalid seed evidence IDs")
    application_id = string(seed.get("application_id"), "application_id", 256)
    return sample_id, application_id, question, evidence_ids, evidence, qd_hash(question, evidence)


def generate_label(
    item: tuple[str, str, str, list[str], str, str], args: argparse.Namespace, api_key: str
) -> dict[str, Any]:
    sample_id, application_id, question, evidence_ids, evidence, input_hash = item
    raw = chat(args.api_base, args.model, api_key, messages(question, evidence), args.max_tokens, args.timeout, args.retries)
    repair_used = False
    try:
        package = validate_package(json_object(raw), len(evidence_ids))
        status = "valid"
    except (ValueError, TypeError) as first_error:
        repair_used = True
        last_error: Exception = first_error
        candidate = raw
        package = None
        for _ in range(3):
            repaired = chat(args.api_base, args.model, api_key, repair_messages(question, evidence, candidate), max(args.max_tokens, 1200), args.timeout, args.retries)
            candidate = repaired
            try:
                package = validate_package(json_object(repaired), len(evidence_ids))
                break
            except (ValueError, TypeError) as exc:
                last_error = exc
        if package is None:
            raise RuntimeError(f"unrecoverable label {sample_id}: {type(first_error).__name__}/{type(last_error).__name__}")
        status = "repaired"
    return {
        "schema_version": VERSION,
        "sample_id": sample_id,
        "application_id": application_id,
        "split": "build",
        "question": question,
        "evidence": evidence,
        "evidence_ids": evidence_ids,
        "input_sha256": input_hash,
        "label": package,
        "status": status,
        "repair_used": repair_used,
        "teacher_model": args.model,
        "teacher_input_contract": "question + R4 evidence only",
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be in [1, 16]")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(f"API key environment variable is empty: {args.api_key_env}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to write into non-empty {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "teacher_qd_labels.jsonl"
    completed = existing(labels_path) if args.resume else {}
    seeds = list(read_jsonl(args.seeds))
    pending: list[tuple[str, str, str, list[str], str, str]] = []
    for seed in seeds:
        item = normalize_seed(seed)
        sample_id, _, _, _, _, input_hash = item
        prior = completed.get(sample_id)
        if prior is not None:
            if prior.get("input_sha256") != input_hash:
                raise ValueError(f"checkpoint input mismatch: {sample_id}")
            continue
        pending.append(item)
    if args.limit is not None:
        pending = pending[:args.limit]
    failures_path = args.output_dir / "teacher_qd_failures.jsonl"
    with labels_path.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_label, item, args, api_key): item for item in pending}
            output_count = 0
            for future in as_completed(futures):
                item = futures[future]
                sample_id = item[0]
                try:
                    record = future.result()
                except Exception as exc:
                    with failures_path.open("a", encoding="utf-8", newline="\n") as failure_handle:
                        write_record(failure_handle, {
                            "sample_id": sample_id,
                            "input_sha256": item[5],
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "teacher_model": args.model,
                        })
                    print(json.dumps({"failed_sample_id": sample_id, "error": type(exc).__name__}, ensure_ascii=False), flush=True)
                    continue
                write_record(handle, record)
                completed[record["sample_id"]] = record
                output_count += 1
                if output_count % 20 == 0:
                    print(json.dumps({"completed": len(completed), "newly_labeled": output_count}, ensure_ascii=False), flush=True)
    manifest = {
        "version": VERSION,
        "status": "complete" if len(completed) == len(seeds) else "partial",
        "n_completed": len(completed),
        "n_seed_rows": len(seeds),
        "seed_sha256": sha256_file(args.seeds),
        "teacher_model": args.model,
        "workers": args.workers,
        "teacher_input_contract": "question + R4 evidence only",
        "answer_like_fields_accepted": False,
        "api_key_environment_variable": args.api_key_env,
    }
    (args.output_dir / "label_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
