#!/usr/bin/env python3
"""Check shared-VRAM batch4 evidence refinement without an answer Critic pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from transformers import AutoTokenizer

from benchmark_integrity import read_rows, sha256
from run_evidence_refinement_wrong_questions import (
    answer_messages, batch_client, evidence_critic_messages, evidence_from_result,
    load_grading_rows, parse_json_object, validate_refinement, fallback_refinement,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = Path(__file__).resolve().parent
    # This existing check verifies official token IDs and real candidate batch4
    # with both models resident. Its answer-Critic probe is deliberately disabled.
    candidate_report = args.output.with_name("shared_candidate_preflight.json")
    subprocess.run([sys.executable, "-u", str(app / "preflight_generator_deployment.py"),
                    "--api-base", args.api_base, "--model", args.model, "--max-new-tokens", "32",
                    "--require-batch4", "--output", str(candidate_report)], check=True)
    report = json.loads(candidate_report.read_text())
    runtime = report["runtime"]
    baseline = json.loads((args.source_root / "deployment_preflight.json").read_text())
    if runtime != baseline["runtime"] or report["baseline_batch_size"] != 4:
        raise ValueError("Candidate runtime changed or batch4 failed with Critic resident")
    critic = batch_client("http://127.0.0.1:8001", "qwen-critic-repair-v2", 640, 420, 2, 20)
    candidate = batch_client(args.api_base, args.model, 32, 300, 2, 20)
    tokenizer = AutoTokenizer.from_pretrained(app.parent / "models/Qwen3.5-9B", local_files_only=True)
    sources = []
    counts = {}
    for year in ("2022", "2021"):
        wrong, _ = load_grading_rows(args.source_root / "grading" / f"ter_{year}/grading.csv")
        selected = [row for row in read_rows(args.source_root / "ter_direct" / year / "ter_direct_results.jsonl")
                    if row["application_id"] in wrong]
        if {row["application_id"] for row in selected} != wrong:
            raise ValueError("TER wrong-row selection is incomplete")
        sources.extend(selected)
        counts[year] = len(selected)

    def length(row):
        prompt = evidence_critic_messages(row["question"], evidence_from_result(row, 850))
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False, return_dict=False)
        return len(ids)

    selected = sorted(sources, key=length)[-4:]
    started = time.monotonic()
    raw = critic([evidence_critic_messages(row["question"], evidence_from_result(row, 850)) for row in selected])
    elapsed = time.monotonic() - started
    packages, statuses = [], []
    for row, value in zip(selected, raw):
        try:
            packages.append(validate_refinement(parse_json_object(value), len(row["retrieved"])))
            statuses.append("valid")
        except (ValueError, json.JSONDecodeError):
            packages.append(fallback_refinement())
            statuses.append("invalid_probe_only")
    candidate([answer_messages(row["question"], evidence_from_result(row, 850), row["answer"], package)
               for row, package in zip(selected, packages)])
    # The largest allowed evidence package catches shared-VRAM failures even
    # when the sampled Critic's actual output is shorter.
    stress_package = {"evidence_status": "partial", "change_support": "strong",
        "evidence_actions": [{"evidence_id": f"E{i}", "action": "keep", "reason": "证" * 240} for i in (1, 2, 3)],
        "knowledge_supplements": [{"claim": "证" * 360, "basis": "model_knowledge", "confidence": "high"} for _ in range(2)],
        "answer_guidance": "证" * 360}
    stress = [answer_messages(row["question"], evidence_from_result(row, 850), row["answer"], stress_package)
              for row in sources]
    candidate_tokenizer = AutoTokenizer.from_pretrained(runtime["model_path"], local_files_only=True)

    def candidate_length(prompt):
        try:
            ids = candidate_tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                                          enable_thinking=False, return_dict=False)
        except Exception as exc:
            if "system role not supported" not in str(exc).lower():
                raise
            merged = [{"role": "user", "content": prompt[0]["content"] + "\n\n" + prompt[1]["content"]}]
            ids = candidate_tokenizer.apply_chat_template(merged, tokenize=True, add_generation_prompt=True, return_dict=False)
        return len(ids["input_ids"] if isinstance(ids, dict) else ids)

    stress = sorted(stress, key=candidate_length)[-4:]
    if stress and max(map(candidate_length, stress)) + 32 > runtime["context_limit"]:
        raise ValueError("Evidence-refinement stress input exceeds candidate context")
    candidate(stress)
    report.update({"critic_co_loaded": True, "answer_critic_probe_executed": False,
        "refinement_batch_size": 4, "evidence_critic_max_tokens": 640,
        "candidate_refinement_stress_batch4_executable": True,
        "candidate_refinement_stress_prompt_tokens": list(map(candidate_length, stress)),
        "ter_wrong_counts": counts, "candidate_preflight_sha256": sha256(candidate_report),
        "evidence_critic_batch4": {"executable": True, "seconds": round(elapsed, 3),
            "selected_ids": [row["application_id"] for row in selected],
            "critic_prompt_tokens": list(map(length, selected)), "statuses": statuses},
        "probe_predictions_committed_to_results": False})
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
