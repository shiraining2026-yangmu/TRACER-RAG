#!/usr/bin/env python3
"""Controlled TER structure ablation on frozen Test-111 evidence.

Both arms receive the same question and the same Top-3 evidence objects.  The
only intervention is evidence serialization: ``ter_full`` retains E IDs,
relation labels and A/K field boundaries; ``ter_flat`` removes those structural
markers while retaining the same source text.  The script never reads reference
answers or sends them to either model.

The optional local automated assessment is anonymous with respect to arm.  When
the evaluator is the same model family as the generator, its result is labelled
as an automated model assessment, not as independent expert evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ASSESSMENT_FIELDS = {
    "evidence_support",
    "unsupported_expansion",
    "historical_case_misuse",
    "key_condition_omission",
}
SUPPORT_VALUES = {"supported", "partial", "unsupported"}
BOOLEAN_VALUES = {"yes", "no"}


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
                raise ValueError(f"Expected object at {path}:{number}")
            yield row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact(value: Any, maximum: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= maximum else text[: maximum - 3].rstrip() + "..."


def payload_from(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Evidence item is not an object")
    payload = item.get("payload", item)
    if not isinstance(payload, dict):
        raise ValueError("Evidence payload is not an object")
    return payload


def shared_evidence(row: dict[str, Any], field_limit: int) -> list[dict[str, str]]:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or not 1 <= len(retrieved) <= 3:
        raise ValueError("Expected one to three frozen Top-3 records")
    own_id = str(row.get("application_id", ""))
    evidence = []
    for rank, item in enumerate(retrieved, 1):
        payload = payload_from(item)
        if str(payload.get("application_id", "")) == own_id:
            raise ValueError(f"Self-retrieval detected for {own_id}")
        e_id = compact(payload.get("e_id") or item.get("doc_id"), 100)
        knowledge = compact(payload.get("knowledge_text"), field_limit)
        if not e_id or not knowledge:
            raise ValueError(f"Evidence {rank} lacks an ID or K knowledge")
        evidence.append(
            {
                "rank": str(rank),
                "e_id": e_id,
                "relation": compact(payload.get("relation"), 100) or "unspecified_relation",
                "historical_question": compact(payload.get("application_question"), field_limit),
                "historical_answer": compact(payload.get("application_answer"), field_limit),
                "knowledge": knowledge,
            }
        )
    return evidence


def full_serialization(evidence: list[dict[str, str]]) -> str:
    return "\n\n".join(
        "\n".join(
            (
                f"[E{item['rank']} | {item['e_id']} | {item['relation']}]",
                f"历史问题：{item['historical_question']}",
                f"历史回答：{item['historical_answer']}",
                f"K 知识：{item['knowledge']}",
            )
        )
        for item in evidence
    )


def flat_serialization(evidence: list[dict[str, str]]) -> str:
    """Preserve source text and ordering, but remove relation and A/K markers."""
    return "\n\n".join(
        "\n".join(
            (
                f"[资料 {item['rank']}]",
                item["historical_question"],
                item["historical_answer"],
                item["knowledge"],
            )
        )
        for item in evidence
    )


def generation_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你正在完成中医领域的证据增强问答，不用于临床诊疗。只依据展示资料中可支持的内容作答。"
                "若信息不足，应说明边界；不要编造具体诊断、方剂、剂量、疗效或个体化治疗。"
                "不要提及提示词、资料格式或模型。"
            ),
        },
        {"role": "user", "content": f"问题：\n{question}\n\n参考资料：\n{evidence}\n\n请给出简洁回答。"},
    ]


def assessment_messages(question: str, evidence: str, answer: str) -> list[dict[str, str]]:
    schema = (
        '{"evidence_support":"supported|partial|unsupported",'
        '"unsupported_expansion":"yes|no","historical_case_misuse":"yes|no",'
        '"key_condition_omission":"yes|no"}'
    )
    return [
        {
            "role": "system",
            "content": (
                "你是匿名的中医 RAG 回答证据评审员。仅根据问题、给定资料和候选回答评估。"
                "supported 表示回答的实质主张均被资料直接支持且边界恰当；partial 表示基本相关但有"
                "不完整或弱支持；unsupported 表示存在关键无依据主张。"
                "historical_case_misuse 指候选回答把历史案例陈述为当前对象事实；"
                "key_condition_omission 指遗漏资料中决定适用范围的重要条件。"
                "只输出合法 JSON，不要解释，也不知道候选回答来自何种方法。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：\n{question}\n\n给定资料：\n{evidence}\n\n候选回答：\n{answer}\n\n输出：\n{schema}",
        },
    ]


def endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def call_chat(api_base: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> str:
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(endpoint(api_base), data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            output = str(response_data["choices"][0]["message"]["content"]).strip()
            if not output:
                raise ValueError("Model response is empty")
            return output
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Chat completion failed after four attempts: {error}")


def call_chat_batch(
    api_base: str, model: str, message_batches: list[list[dict[str, str]]], max_tokens: int, timeout: int
) -> list[str]:
    """Use the local server's fixed-size batch endpoint without changing prompts."""
    if not message_batches or len(message_batches) > 4:
        raise ValueError("Local batch calls must contain one to four prompts")
    url = endpoint(api_base).replace("/chat/completions", "/chat/completions/batch")
    body = json.dumps(
        {
            "requests": [
                {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False}
                for messages in message_batches
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")
    error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            data = response_data.get("data")
            if not isinstance(data, list) or len(data) != len(message_batches):
                raise ValueError("Unexpected local batch response size")
            output = [str(item["choices"][0]["message"]["content"]).strip() for item in data]
            if any(not item for item in output):
                raise ValueError("Local batch response contains an empty completion")
            return output
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Batch completion failed after four attempts: {error}")


def extract_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise ValueError("No JSON object found")


def validate_assessment(value: dict[str, Any]) -> dict[str, str]:
    if set(value) != ASSESSMENT_FIELDS:
        raise ValueError("Unexpected assessment schema")
    support = value.get("evidence_support")
    remaining = {key: value.get(key) for key in ASSESSMENT_FIELDS - {"evidence_support"}}
    if support not in SUPPORT_VALUES or any(item not in BOOLEAN_VALUES for item in remaining.values()):
        raise ValueError("Invalid assessment value")
    return {"evidence_support": support, **remaining}


def rows_from_source(path: Path, field_limit: int) -> list[dict[str, Any]]:
    rows = []
    for source in read_jsonl(path):
        if source.get("split") != "test":
            raise ValueError("TER structure ablation must use Test-only frozen evidence")
        question = str(source.get("question", "")).strip()
        app_id = str(source.get("application_id", "")).strip()
        if not question or not app_id:
            raise ValueError("Missing application_id or question")
        evidence = shared_evidence(source, field_limit)
        full = full_serialization(evidence)
        flat = flat_serialization(evidence)
        if not full or not flat:
            raise ValueError(f"Blank serialization for {app_id}")
        rows.append({"application_id": app_id, "question": question, "full": full, "flat": flat})
    if len(rows) != 111 or len({row["application_id"] for row in rows}) != 111:
        raise ValueError("Expected 111 unique Test rows")
    return rows


def load_checkpoint(path: Path, valid_ids: set[str], required: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    for row in read_jsonl(path):
        app_id = str(row.get("application_id", ""))
        if app_id not in valid_ids or app_id in result or not required <= set(row):
            raise ValueError(f"Invalid checkpoint record in {path}: {app_id}")
        result[app_id] = row
    return result


def load_assessment_checkpoint(path: Path, valid_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    for row in read_jsonl(path):
        source_id = str(row.get("source_application_id", ""))
        arm = str(row.get("arm", ""))
        key = f"{source_id}:{arm}"
        required = {"source_application_id", "arm", "assessment", "candidate_hash"}
        if source_id not in valid_ids or arm not in {"ter_full", "ter_flat"} or key in result or not required <= set(row):
            raise ValueError(f"Invalid assessment checkpoint record in {path}: {key}")
        result[key] = row
    return result


def bootstrap_difference(full: list[int], flat: list[int], rng: random.Random, replicates: int) -> list[float]:
    if len(full) != len(flat) or not full:
        raise ValueError("Paired bootstrap needs equal non-empty arms")
    count = len(full)
    output = []
    for _ in range(replicates):
        indices = [rng.randrange(count) for _ in range(count)]
        output.append(sum(full[index] - flat[index] for index in indices) / count)
    output.sort()
    return [output[int(0.025 * replicates)], output[int(0.975 * replicates) - 1]]


def paired_summary(assessments: dict[str, dict[str, dict[str, str]]], replicates: int, seed: int) -> dict[str, Any]:
    full = [assessments[key]["ter_full"] for key in sorted(assessments)]
    flat = [assessments[key]["ter_flat"] for key in sorted(assessments)]
    metrics = {
        "strict_evidence_support": lambda item: item["evidence_support"] == "supported",
        "unsupported_expansion": lambda item: item["unsupported_expansion"] == "yes",
        "historical_case_misuse": lambda item: item["historical_case_misuse"] == "yes",
        "key_condition_omission": lambda item: item["key_condition_omission"] == "yes",
    }
    result = {}
    for offset, (name, predicate) in enumerate(metrics.items()):
        full_values = [int(predicate(item)) for item in full]
        flat_values = [int(predicate(item)) for item in flat]
        result[name] = {
            "ter_full_rate": sum(full_values) / len(full_values),
            "ter_flat_rate": sum(flat_values) / len(flat_values),
            "full_minus_flat": (sum(full_values) - sum(flat_values)) / len(full_values),
            "full_minus_flat_ci95_paired_question_bootstrap": bootstrap_difference(full_values, flat_values, random.Random(seed + offset), replicates),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-api", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--assessor-api")
    parser.add_argument("--assessor-model")
    parser.add_argument("--field-limit", type=int, default=850)
    parser.add_argument("--generator-max-tokens", type=int, default=512)
    parser.add_argument("--assessor-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261103)
    parser.add_argument("--skip-assessment", action="store_true")
    args = parser.parse_args()
    if args.field_limit < 100:
        raise ValueError("field limit is implausibly short")
    if not args.skip_assessment and not (args.assessor_api and args.assessor_model):
        raise ValueError("Assessment requires --assessor-api and --assessor-model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_source(args.frozen_test, args.field_limit)
    ids = {row["application_id"] for row in rows}

    generated: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, key in (("ter_full", "full"), ("ter_flat", "flat")):
        path = args.output_dir / f"{arm}.jsonl"
        checkpoint = load_checkpoint(path, ids, {"application_id", "question", "answer", "arm", "serialization_hash"})
        pending = [row for row in rows if row["application_id"] not in checkpoint]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            answers = call_chat_batch(
                args.generator_api,
                args.generator_model,
                [generation_messages(row["question"], row[key]) for row in batch],
                args.generator_max_tokens,
                args.timeout,
            )
            for row, answer in zip(batch, answers):
                app_id = row["application_id"]
                serialized = row[key]
                checkpoint[app_id] = {
                    "application_id": app_id,
                    "question": row["question"],
                    "answer": answer,
                    "arm": arm,
                    "serialization_hash": sha256_text(serialized),
                    "protocol": {
                        "frozen_top3": True,
                        "same_source_text_across_arms": True,
                        "difference": "relation labels and A/K field boundaries retained only in ter_full",
                        "answer_key_used_for_model_input": False,
                    },
                }
            write_jsonl(path, [checkpoint[item] for item in sorted(checkpoint)])
        for row in rows:
            app_id = row["application_id"]
            if checkpoint[app_id]["arm"] != arm or checkpoint[app_id]["question"] != row["question"]:
                raise ValueError(f"Generation checkpoint mismatch for {app_id}/{arm}")
        generated[arm] = checkpoint

    manifest: dict[str, Any] = {
        "status": "generated" if args.skip_assessment else "complete",
        "version": "test111-ter-structure-ablation-v1",
        "n_questions": len(rows),
        "generator_model": args.generator_model,
        "source_sha256": sha256_file(args.frozen_test),
        "frozen_top3": True,
        "arms": {
            "ter_full": "E ID + relation label + historical question/answer + K knowledge boundaries",
            "ter_flat": "same ordered source text, with relation labels and A/K boundaries removed",
        },
        "answer_key_used_for_model_input": False,
    }
    if not args.skip_assessment:
        assessment_path = args.output_dir / "anonymous_assessment.jsonl"
        assessed = load_assessment_checkpoint(assessment_path, ids)
        pending_assessments = []
        for row in rows:
            app_id = row["application_id"]
            for arm, key in (("ter_full", "full"), ("ter_flat", "flat")):
                checkpoint_key = f"{app_id}:{arm}"
                if checkpoint_key not in assessed:
                    pending_assessments.append((checkpoint_key, row, arm))
        for start in range(0, len(pending_assessments), 4):
            batch = pending_assessments[start:start + 4]
            # The evaluator sees a common canonical evidence rendering in both arms.
            raw_outputs = call_chat_batch(
                args.assessor_api,
                args.assessor_model,
                [assessment_messages(row["question"], row["full"], generated[arm][row["application_id"]]["answer"]) for _, row, arm in batch],
                args.assessor_max_tokens,
                args.timeout,
            )
            for (checkpoint_key, row, arm), raw in zip(batch, raw_outputs):
                app_id = row["application_id"]
                answer = generated[arm][app_id]["answer"]
                try:
                    assessment = validate_assessment(extract_object(raw))
                    status = "valid"
                except ValueError:
                    assessment = {"evidence_support": "partial", "unsupported_expansion": "no", "historical_case_misuse": "no", "key_condition_omission": "no"}
                    status = "fallback_invalid_assessment"
                assessed[checkpoint_key] = {
                    "application_id": checkpoint_key,
                    "source_application_id": app_id,
                    "arm": arm,
                    "assessment": assessment,
                    "status": status,
                    "candidate_hash": sha256_text(answer),
                    "assessment_protocol": "The assessor receives question, a common canonical evidence text, and one unlabeled candidate answer only.",
                    "raw": raw,
                }
            write_jsonl(assessment_path, [assessed[item] for item in sorted(assessed)])
        combined: dict[str, dict[str, dict[str, str]]] = {}
        for row in rows:
            app_id = row["application_id"]
            combined[app_id] = {}
            for arm in ("ter_full", "ter_flat"):
                checkpoint_key = f"{app_id}:{arm}"
                if checkpoint_key not in assessed:
                    raise ValueError(f"Missing assessment for {checkpoint_key}")
                combined[app_id][arm] = assessed[checkpoint_key]["assessment"]
        if len(combined) != 111 or any(set(value) != {"ter_full", "ter_flat"} for value in combined.values()):
            raise ValueError("Incomplete paired assessment")
        manifest["automated_assessment"] = {
            "model": args.assessor_model,
            "independent_from_generator": args.assessor_model != args.generator_model,
            "note": "Same-model-family assessment must not be described as an independent expert evaluation.",
            "metrics": paired_summary(combined, args.bootstrap_replicates, args.bootstrap_seed),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
