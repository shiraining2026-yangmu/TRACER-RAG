#!/usr/bin/env python3
"""Blind automated evidence-fidelity assessment for Test-111 EviCR ablations.

The evaluator receives each question, its frozen canonical Top-3 evidence, and
three anonymized answers.  It is never shown method names, Critic JSON, answer
keys, or reference answers.  This is explicitly an automated anonymous rating,
not an expert annotation or clinical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VERSION = "evicr-qd-fidelity-v1"
ARMS = ("ter_y0", "evicr_full", "evicr_no_ks")
SUPPORT = {"supported", "partial", "unsupported"}
BINARY = {"yes", "no"}
FIELDS = ("evidence_support", "unsupported_expansion", "historical_case_misuse", "key_condition_omission")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any, maximum: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= maximum else text[: maximum - 3].rstrip() + "..."


def evidence_text(row: dict[str, Any], field_limit: int) -> str:
    retrieved = row.get("retrieved")
    if not isinstance(retrieved, list) or not 1 <= len(retrieved) <= 3:
        raise ValueError("frozen test must contain one to three retrieved blocks")
    own_id = str(row.get("application_id", ""))
    blocks = []
    for rank, item in enumerate(retrieved, 1):
        payload = item.get("payload", item) if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            raise ValueError("retrieved payload missing")
        if str(payload.get("application_id", "")) == own_id:
            raise ValueError("self retrieval in frozen test")
        evidence_id = compact(payload.get("e_id") or item.get("doc_id"), 100)
        relation = compact(payload.get("relation"), 100)
        knowledge = compact(payload.get("knowledge_text"), field_limit)
        historical_question = compact(payload.get("application_question"), field_limit)
        historical_answer = compact(payload.get("application_answer"), field_limit)
        if not evidence_id or not knowledge:
            raise ValueError("frozen evidence lacks ID or K knowledge")
        blocks.append("\n".join((f"[E{rank} | {evidence_id} | {relation or 'unspecified_relation'}]", f"历史问题：{historical_question}", f"历史回答：{historical_answer}", f"K 知识：{knowledge}")))
    return "\n\n".join(blocks)


def slot_order(application_id: str) -> list[str]:
    order = list(ARMS)
    random.Random(int.from_bytes(hashlib.sha256((VERSION + application_id).encode()).digest()[:8], "big")).shuffle(order)
    return order


def judge_messages(question: str, evidence: str, answers: dict[str, str]) -> list[dict[str, str]]:
    rubric = {
        "evidence_support": "supported|partial|unsupported",
        "unsupported_expansion": "yes|no",
        "historical_case_misuse": "yes|no",
        "key_condition_omission": "yes|no",
    }
    return [
        {"role": "system", "content": (
            "你是中医 RAG 回答的匿名证据忠实性评审器。仅依据题目与展示的固定 R4 证据评价每个匿名回答。"
            "历史案例仅可作相似案例，不是当前对象事实；模型不能把未被证据支持的诊断、方剂、剂量、疗效或处置写成事实。"
            "对每个匿名回答独立输出四项判断：evidence_support 指其核心结论被证据充分支持、部分支持或不支持；"
            "unsupported_expansion 指是否添加展示证据未支持的实质性断言；historical_case_misuse 指是否将历史案例当作当前事实；"
            "key_condition_omission 指是否遗漏证据明确的关键适用条件或限制。不要选择正确答案，不要比较回答，也不要输出解释、方法名称或 Markdown。"
        )},
        {"role": "user", "content": json.dumps({"question": question, "fixed_top3_evidence": evidence, "anonymous_answers": answers, "required_schema": {"ratings": {code: rubric for code in answers}}}, ensure_ascii=False)},
    ]


def repair_messages(question: str, evidence: str, answers: dict[str, str], candidate: str) -> list[dict[str, str]]:
    prompt = judge_messages(question, evidence, answers)
    prompt[0] = {"role": "system", "content": "将匿名证据忠实性评审修复为合法 JSON。仅依据给定题目、固定 R4 证据、匿名回答和候选输出；不得引入外部信息。只输出一个 JSON 对象，包含 ratings，且每个匿名回答均有四个规定字段。"}
    prompt[1] = {"role": "user", "content": prompt[1]["content"] + "\n\n候选输出：\n" + candidate[:6000]}
    return prompt


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
            text = str(payload["choices"][0]["message"]["content"]).strip()
            if not text:
                raise ValueError("empty completion")
            return text
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"judge request failed after {retries + 1} attempts: {type(error).__name__}")


def json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("no JSON object")


def validate_rating(value: Any, codes: set[str]) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"ratings"} or not isinstance(value["ratings"], dict) or set(value["ratings"]) != codes:
        raise ValueError("invalid rating object")
    clean: dict[str, dict[str, str]] = {}
    for code, rating in value["ratings"].items():
        if not isinstance(rating, dict) or set(rating) != set(FIELDS):
            raise ValueError("rating has missing or extra fields")
        clean_rating = {field: str(rating[field]).strip().lower() for field in FIELDS}
        if clean_rating["evidence_support"] not in SUPPORT or any(clean_rating[field] not in BINARY for field in FIELDS[1:]):
            raise ValueError("rating has invalid category")
        clean[code] = clean_rating
    return clean


def load_answers(path: Path, expected_arm: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        app_id = str(row.get("application_id", ""))
        if not app_id or app_id in output or row.get("arm") != expected_arm:
            raise ValueError(f"invalid {expected_arm} arm record: {app_id}")
        question, answer = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"blank {expected_arm} answer")
        output[app_id] = {"question": question, "answer": answer}
    return output


def load_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    frozen = list(read_jsonl(args.frozen_test))
    if len(frozen) != 111 or any(row.get("split") != "test" for row in frozen):
        raise ValueError("expected exactly frozen Test-111 rows")
    arms = {
        "ter_y0": load_answers(args.ter_y0, "ter_y0"),
        "evicr_full": load_answers(args.evicr_full, "evicr_full"),
        "evicr_no_ks": load_answers(args.evicr_no_ks, "evicr_no_ks"),
    }
    ids = {str(row.get("application_id", "")) for row in frozen}
    if any(set(answer_rows) != ids for answer_rows in arms.values()):
        raise ValueError("all arms must cover the same frozen Test-111 IDs")
    inputs = []
    for row in frozen:
        app_id = str(row["application_id"])
        question = str(row.get("question", "")).strip()
        if not question or any(arm[app_id]["question"] != question for arm in arms.values()):
            raise ValueError(f"question mismatch: {app_id}")
        evidence = evidence_text(row, args.evidence_field_limit)
        ordering = slot_order(app_id)
        codes = ("回答甲", "回答乙", "回答丙")
        slot_to_arm = dict(zip(codes, ordering))
        answers = {code: arms[arm][app_id]["answer"] for code, arm in slot_to_arm.items()}
        input_hash = hashlib.sha256(json.dumps({"question": question, "evidence": evidence, "answers": answers}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        inputs.append({"application_id": app_id, "question": question, "evidence": evidence, "answers": answers, "slot_to_arm": slot_to_arm, "input_sha256": input_hash})
    return sorted(inputs, key=lambda row: row["application_id"])


def existing(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        app_id = str(row.get("application_id", ""))
        if not app_id or app_id in rows or row.get("status") not in {"valid", "repaired"}:
            raise ValueError("invalid judgement checkpoint")
        rows[app_id] = row
    return rows


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float | None:
    total = discordant_a + discordant_b
    if not total:
        return None
    tail = sum(math.comb(total, index) for index in range(0, min(discordant_a, discordant_b) + 1)) / (2**total)
    return min(1.0, 2 * tail)


def is_positive(value: str, field: str) -> bool:
    return value == "supported" if field == "evidence_support" else value == "yes"


def paired_stats(rows: list[dict[str, Any]], baseline: str, comparison: str, field: str, seed: int, draws: int) -> dict[str, Any]:
    pairs = [(row["ratings"][baseline][field], row["ratings"][comparison][field]) for row in rows]
    before = [is_positive(left, field) for left, _ in pairs]
    after = [is_positive(right, field) for _, right in pairs]
    a = sum(left and right for left, right in zip(before, after))
    b = sum(left and not right for left, right in zip(before, after))
    c = sum(not left and right for left, right in zip(before, after))
    d = sum(not left and not right for left, right in zip(before, after))
    point = 100 * (sum(after) - sum(before)) / len(pairs)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(draws):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        deltas.append(100 * sum(int(after[index]) - int(before[index]) for index in indices) / len(pairs))
    deltas.sort()
    low, high = deltas[int(0.025 * (draws - 1))], deltas[int(0.975 * (draws - 1))]
    return {
        "field": field,
        "n_pairs": len(pairs),
        "baseline_positive_percent": round(100 * sum(before) / len(pairs), 3),
        "comparison_positive_percent": round(100 * sum(after) / len(pairs), 3),
        "comparison_minus_baseline_percentage_points": round(point, 3),
        "paired_bootstrap_95ci_percentage_points": [round(low, 3), round(high, 3)],
        "paired_2x2": {"both_positive": a, "baseline_only": b, "comparison_only": c, "both_negative": d},
        "exact_mcnemar_p_value": None if exact_mcnemar(b, c) is None else round(exact_mcnemar(b, c), 8),
    }


def summarize(rows: list[dict[str, Any]], output_dir: Path, draws: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "version": VERSION,
        "n_questions": len(rows),
        "automated_anonymous_evaluation": True,
        "arm_distributions": {},
        "comparisons": {},
    }
    for arm in ARMS:
        distribution: dict[str, dict[str, int]] = {}
        for field in FIELDS:
            counts = Counter(row["ratings"][arm][field] for row in rows)
            values = sorted(SUPPORT) if field == "evidence_support" else sorted(BINARY)
            distribution[field] = {value: counts.get(value, 0) for value in values}
        summary["arm_distributions"][arm] = distribution
    for baseline, comparison in (("ter_y0", "evicr_full"), ("evicr_full", "evicr_no_ks")):
        name = f"{comparison}_minus_{baseline}"
        summary["comparisons"][name] = [paired_stats(rows, baseline, comparison, field, 20261103 + index, draws) for index, field in enumerate(FIELDS)]
    (output_dir / "fidelity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["comparison,field,n_pairs,baseline_positive_percent,comparison_positive_percent,delta_percentage_points,bootstrap_ci_low,bootstrap_ci_high,both_positive,baseline_only,comparison_only,both_negative,mcnemar_p"]
    for comparison, results in summary["comparisons"].items():
        for item in results:
            table = item["paired_2x2"]
            ci = item["paired_bootstrap_95ci_percentage_points"]
            lines.append(",".join(map(str, (comparison, item["field"], item["n_pairs"], item["baseline_positive_percent"], item["comparison_positive_percent"], item["comparison_minus_baseline_percentage_points"], ci[0], ci[1], table["both_positive"], table["baseline_only"], table["comparison_only"], table["both_negative"], item["exact_mcnemar_p_value"]))))
    (output_dir / "fidelity_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--ter-y0", type=Path, required=True)
    parser.add_argument("--evicr-full", type=Path, required=True)
    parser.add_argument("--evicr-no-ks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs(args)
    if args.dry_run:
        preview = judge_messages(inputs[0]["question"], inputs[0]["evidence"], inputs[0]["answers"])
        print(json.dumps({"n_inputs": len(inputs), "prompt_preview": preview, "critic_or_answer_key_visible_to_judge": False}, ensure_ascii=False))
        return
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(f"API key environment variable is empty: {args.api_key_env}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to write into non-empty {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "anonymous_judgements.jsonl"
    completed = existing(path) if args.resume else {}
    with path.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as handle:
        for index, item in enumerate(inputs, 1):
            old = completed.get(item["application_id"])
            if old is not None:
                if old.get("input_sha256") != item["input_sha256"]:
                    raise ValueError("checkpoint input mismatch")
                continue
            raw = chat(args.api_base, args.model, api_key, judge_messages(item["question"], item["evidence"], item["answers"]), args.max_tokens, args.timeout, args.retries)
            repair_used = False
            try:
                rating_by_code = validate_rating(json_object(raw), set(item["answers"]))
                status = "valid"
            except ValueError:
                repair_used = True
                repaired = chat(args.api_base, args.model, api_key, repair_messages(item["question"], item["evidence"], item["answers"], raw), args.max_tokens, args.timeout, args.retries)
                rating_by_code = validate_rating(json_object(repaired), set(item["answers"]))
                status = "repaired"
            ratings = {item["slot_to_arm"][code]: rating for code, rating in rating_by_code.items()}
            record = {"application_id": item["application_id"], "status": status, "repair_used": repair_used, "input_sha256": item["input_sha256"], "slot_to_arm": item["slot_to_arm"], "ratings": ratings}
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed[item["application_id"]] = record
            if index % 10 == 0:
                print(json.dumps({"completed": len(completed), "total": len(inputs)}, ensure_ascii=False), flush=True)
    if len(completed) != len(inputs):
        raise RuntimeError("incomplete anonymous assessment")
    rows = [completed[item["application_id"]] for item in inputs]
    summary = summarize(rows, args.output_dir, args.bootstrap_draws)
    manifest = {"version": VERSION, "status": "complete", "n_questions": len(rows), "judge_model": args.model, "judge_input": "question + frozen canonical Top-3 evidence + anonymous candidate answers", "methods_hidden_from_judge": True, "critic_json_hidden_from_judge": True, "answer_key_hidden_from_judge": True, "automated_anonymous_evaluation": True, "not_expert_annotation": True, "bootstrap_unit": "question", "bootstrap_draws": args.bootstrap_draws, "source_sha256": {"frozen_test": sha256_file(args.frozen_test), "ter_y0": sha256_file(args.ter_y0), "evicr_full": sha256_file(args.evicr_full), "evicr_no_ks": sha256_file(args.evicr_no_ks)}, "rating_status_counts": dict(Counter(row["status"] for row in rows)), "summary_file": "fidelity_summary.json"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
