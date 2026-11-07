"""Answer-blind deployment checks before each corrected candidate benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import requests
from transformers import AutoTokenizer

from run_frozen_evidence_generation import dense_messages, ter_messages
from run_questionbank_rag_vs_alignrag import extract_choice
from run_evidence_refinement_wrong_questions import answer_messages, evidence_from_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--critic-present", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--require-batch4", action="store_true",
                        help="Require executable batch 4; record numerical output differences")
    parser.add_argument("--refinement-source-root", type=Path)
    args = parser.parse_args()
    base = args.api_base.rstrip("/")
    response = requests.get(base + "/v1/benchmark_runtime", timeout=30)
    response.raise_for_status()
    runtime = response.json()
    if runtime["model_id"] != args.model or runtime["input_adapter_version"] != "official-template-no-extra-special-tokens-v2":
        raise ValueError("Unexpected model/adapter runtime")
    if runtime["model_type"] == "gemma2" and runtime["attention_implementation"] != "eager":
        raise ValueError("Gemma-2 requires faithful attention softcapping in this runtime")
    tokenizer = AutoTokenizer.from_pretrained(runtime["model_path"], local_files_only=True)

    def official_ids(messages):
        try:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        except Exception as exc:
            if "system role not supported" not in str(exc).lower():
                raise
            merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
            ids = tokenizer.apply_chat_template(merged, tokenize=True, return_dict=False, add_generation_prompt=True)
        return ids["input_ids"] if isinstance(ids, dict) else list(ids)

    def call(messages, batch=False):
        entries = [{"model": args.model, "messages": value, "max_tokens": args.max_new_tokens, "temperature": 0}
                   for value in messages]
        response = requests.post(base + ("/v1/chat/completions/batch" if batch else "/v1/chat/completions"),
                                 json={"requests": entries} if batch else entries[0], timeout=240)
        response.raise_for_status()
        values = response.json()["data"] if batch else [response.json()]
        if len(values) != len(entries):
            raise ValueError("Batch response cardinality mismatch")
        for value, original in zip(values, messages):
            if value["input_adapter_version"] != runtime["input_adapter_version"]:
                raise ValueError("Generation server changed during preflight")
            expected = hashlib.sha256(json.dumps(official_ids(original)).encode()).hexdigest()
            if value["input_ids_sha256"] != expected:
                raise ValueError("Actual model input differs from the official tokenizer: special-token/template bug")
        return values

    # Synthetic interface tasks; no exam reference answer or answer key is read.
    prompts = [[{"role": "system", "content": "按用户指令只输出指定的一行文字，不添加解释。"},
                {"role": "user", "content": f"请原样输出：答案：{letter}"}] for letter in "ABCD"]
    singles = [call([prompt])[0] for prompt in prompts]
    try:
        batch = call(prompts, batch=True) if not args.critic_present or args.require_batch4 else None
    except requests.HTTPError as exc:
        if "out of memory" not in exc.response.text.lower():
            raise
        batch = None
    try:
        batch2 = call(prompts[:2], batch=True)
    except requests.HTTPError as exc:
        if "out of memory" not in exc.response.text.lower():
            raise
        batch2 = None
    text = lambda value: value["choices"][0]["message"]["content"]
    identical = batch is not None and all(text(a) == text(b) for a, b in zip(singles, batch))
    identical2 = batch2 is not None and all(text(a) == text(b) for a, b in zip(singles[:2], batch2))
    report = {"runtime": runtime, "synthetic_prompts_sha256": hashlib.sha256(
                  json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
              "single_vs_batch4_identical": identical, "single_vs_batch2_identical": identical2,
              "baseline_batch_size": 4 if identical else 1, "refinement_batch_size": 2 if identical2 else 1,
              "synthetic_valid_choices": sum(extract_choice(text(item)) is not None for item in singles),
              "smoke_finish_reasons": [value["choices"][0]["finish_reason"] for value in singles],
              "no_answer_keys_read": True, "max_new_tokens": args.max_new_tokens,
              "exact_official_token_ids_verified": True, "critic_co_loaded": args.critic_present}
    if any(value["choices"][0]["finish_reason"] == "length" for value in singles):
        raise ValueError("Synthetic one-line completions hit output limit; inspect deployment before scoring")
    app = Path(__file__).resolve().parent
    longest = []
    real_prompts, real_singles = [], []
    refinement_lengths = []
    stress_prompts = []
    stress_package = {
        "evidence_status": "partial", "change_support": "strong",
        "evidence_actions": [{"evidence_id": f"E{i}", "action": "keep", "reason": "证" * 240} for i in (1, 2, 3)],
        "knowledge_supplements": [{"claim": "证" * 360, "basis": "model_knowledge", "confidence": "high"} for _ in range(2)],
        "answer_guidance": "证" * 360,
    }
    for arm in ("dense", "ter"):
        for year in ("2022", "2021"):
            path = (app / "questionbank_eval/dense_chunk_rag_v1" / f"{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                    if arm == "dense" else app / "questionbank_eval" / f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
            messages, lengths = [], []
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                prompt = dense_messages(row) if arm == "dense" else ter_messages(row, 850)
                messages.append(prompt)
                lengths.append(len(official_ids(prompt)))
                if arm == "ter":
                    refined = answer_messages(row["question"], evidence_from_result(row, 850), "答案：A", stress_package)
                    refinement_lengths.append(len(official_ids(refined)))
                    stress_prompts.append(refined)
            if max(lengths) + args.max_new_tokens > runtime["context_limit"]:
                raise ValueError(f"Frozen {arm}/{year} input exceeds model context")
            # Select the exact token maximum, not a character-length proxy.
            prompt = messages[lengths.index(max(lengths))]
            value = call([prompt])[0]
            longest.append({"arm": arm, "year": year, "usage": value["usage"],
                            "all_prompt_count": len(lengths), "max_prompt_tokens": max(lengths),
                            "finish_reason": value["choices"][0]["finish_reason"]})
            real_prompts.append(prompt)
            real_singles.append(value)
    report["long_prompt_generation_checks"] = longest
    report["refinement_stress_max_prompt_tokens"] = max(refinement_lengths)
    if max(refinement_lengths) + args.max_new_tokens > runtime["context_limit"]:
        raise ValueError("Refinement input stress check exceeds model context; compact uniformly before running")
    # Compare real long prompts against single inference, not only tiny probes.
    if not args.critic_present or args.require_batch4:
        try:
            batch = call(real_prompts, batch=True)
            report["long_single_vs_batch4_identical"] = all(text(a) == text(b) for a, b in zip(real_singles, batch))
        except requests.HTTPError as exc:
            if args.require_batch4 or "out of memory" not in exc.response.text.lower():
                raise
            report["long_single_vs_batch4_identical"] = False
        if not report["long_single_vs_batch4_identical"]:
            report["baseline_batch_size"] = 1
    # Large length disparity catches attention-mask/left-padding mistakes.
    probes = [real_prompts[0], prompts[0]]
    try:
        pair = call(probes, batch=True)
        report["mixed_length_single_vs_batch2_identical"] = all(
            text(a) == text(b) for a, b in zip([real_singles[0], singles[0]], pair))
    except requests.HTTPError as exc:
        if "out of memory" not in exc.response.text.lower():
            raise
        report["mixed_length_single_vs_batch2_identical"] = False
    if not report["mixed_length_single_vs_batch2_identical"]:
        report["refinement_batch_size"] = 1
        if not args.critic_present:
            report["baseline_batch_size"] = 1
    if args.require_batch4:
        # Batch-dependent numerical differences are reported, never hidden as
        # exact equivalence. This version explicitly fixes execution batch 4.
        if batch is None:
            raise ValueError("Required candidate batch 4 did not execute")
        started = time.monotonic()
        call([real_prompts[0], prompts[0], real_prompts[1], prompts[1]], batch=True)
        report["mixed_length_batch4_executable"] = True
        report["mixed_length_batch4_seconds"] = round(time.monotonic() - started, 3)
        report["baseline_batch_size"] = 4
        report["batch_policy"] = "fixed4-v2; output text can differ across batch sizes"
        if args.critic_present:
            if not args.refinement_source_root:
                raise ValueError("Batch4 shared-VRAM checks need this candidate's TER source")
            from benchmark_integrity import read_rows
            from run_qwen35_answer_refinement import refine_batch
            from run_questionbank_rag_vs_alignrag import questionbank_critic_messages
            from run_tcm_alignrag import evidence_from_row
            from run_evidence_refinement_wrong_questions import evidence_critic_messages

            critic_tokenizer = AutoTokenizer.from_pretrained(
                app.parent / "models/Qwen3.5-9B", local_files_only=True)
            sources = []
            for year in ("2022", "2021"):
                sources.extend(read_rows(args.refinement_source_root / "ter_direct" / year /
                                         "ter_direct_results.jsonl"))

            def critic_length(row):
                messages = questionbank_critic_messages(
                    row["question"], evidence_from_row(row, 850), row["raw_answer"])
                return len(critic_tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False))

            selected = sorted(sources, key=critic_length)[-4:]
            calls = []

            def critic_client(budget):
                def invoke(messages):
                    print(json.dumps({"stage": "batch4_preflight_critic", "budget": budget,
                                      "batch_size": len(messages)}, ensure_ascii=False), flush=True)
                    started = time.monotonic()
                    response = requests.post("http://127.0.0.1:8001/v1/chat/completions/batch",
                        json={"requests": [{"model": "qwen-critic-repair-v2", "messages": prompt,
                              "max_tokens": budget, "temperature": 0} for prompt in messages]}, timeout=420)
                    response.raise_for_status()
                    rows = response.json()["data"]
                    if len(rows) != len(messages):
                        raise ValueError("Critic batch response cardinality mismatch")
                    values = [text(row).strip() for row in rows]
                    if any(not value for value in values):
                        raise ValueError("Empty batch4 Critic completion")
                    calls.append({"budget": budget, "batch_size": len(messages),
                                  "seconds": round(time.monotonic() - started, 3)})
                    return values
                return invoke

            def candidate(messages):
                return [text(row) for row in call(messages, batch=True)]

            started = time.monotonic()
            refined = refine_batch(selected, candidate, critic_client(384))
            report["answer_refinement_batch4"] = {
                "executable": True, "seconds": round(time.monotonic() - started, 3),
                "selected_ids": [row["application_id"] for row in selected],
                "critic_prompt_tokens": [critic_length(row) for row in selected],
                "critic_statuses": [row["critic_status"] for row in refined]}
            started = time.monotonic()
            critic_client(640)([evidence_critic_messages(row["question"], evidence_from_result(row, 850))
                                for row in selected])
            report["evidence_critic_batch4"] = {
                "executable": True, "seconds": round(time.monotonic() - started, 3)}
            longest_stress = sorted(stress_prompts, key=lambda messages: len(official_ids(messages)))[-4:]
            call(longest_stress, batch=True)
            report["candidate_refinement_stress_batch4_executable"] = True
            report["critic_batch4_calls"] = calls
            report["refinement_batch_size"] = 4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
