#!/usr/bin/env python3
"""Replay the historical answer-Critic pass using this candidate's raw TER y0.

This stage processes every frozen question and never reads a grading CSV.
The subsequent evidence-v3 stage selects mistakes in its y1 output, but still
receives raw TER y0 as initial_answer, exactly as the historical Qwen pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmark_integrity import (atomic_jsonl, completed_prediction, freeze_protocol,
                                 keyed, read_rows, sha256)
from run_questionbank_rag_vs_alignrag import (
    batch_client, batched, enforce_choices, questionbank_critic_messages,
    questionbank_refinement_messages,
)
from run_tcm_alignrag import (critic_repair_messages, evidence_from_row,
                              fallback_critique, parse_critic)


def initial_answer(row: dict) -> str:
    # Do not substitute the normalized A-E answer or historical Qwen's draft.
    raw = row.get("raw_answer")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Candidate TER row lacks a raw initial completion")
    return raw.strip()


def check_existing(rows: list[dict], source: dict[str, dict]) -> set[str]:
    done = keyed(rows)
    for key, row in done.items():
        expected = source.get(key)
        if (expected is None or not completed_prediction(row)
                or row.get("question") != expected["question"]
                or row.get("retrieved") != expected["retrieved"]
                or row.get("initial_answer") != initial_answer(expected)
                or row.get("ter_answer") != expected["answer"]
                or row.get("critic_status") not in {"valid", "repaired", "fallback"}):
            raise ValueError(f"Mismatched/incomplete answer-Critic checkpoint: {key}")
        parse_critic(json.dumps(row["critic"], ensure_ascii=False), 3)
    return set(done)


def refine_batch(source: list[dict], generator, critic) -> list[dict]:
    contexts = [{"row": row, "question": row["question"],
                 "initial_answer": initial_answer(row),
                 "evidence": evidence_from_row(row, 850)} for row in source]
    raw = critic([questionbank_critic_messages(c["question"], c["evidence"], c["initial_answer"])
                  for c in contexts])
    if len(raw) != len(contexts):
        raise ValueError("Answer-Critic response count mismatch")
    pending = []
    for context, value in zip(contexts, raw):
        context["critic_raw"] = value
        try:
            context["critic"] = parse_critic(value, 3)
            context["critic_status"] = "valid"
        except ValueError:
            pending.append(context)
    if pending:
        repaired = critic([critic_repair_messages(c["question"], c["evidence"],
                           c["initial_answer"], c["critic_raw"]) for c in pending])
        if len(repaired) != len(pending):
            raise ValueError("Critic repair response count mismatch")
        for context, value in zip(pending, repaired):
            context["critic_repair_raw"] = value
            try:
                context["critic"] = parse_critic(value, 3)
                context["critic_status"] = "repaired"
            except ValueError:
                context["critic"] = fallback_critique()
                context["critic_status"] = "fallback"
    answers = generator([questionbank_refinement_messages(c["question"], c["evidence"],
                         c["initial_answer"], json.dumps(c["critic"], ensure_ascii=False))
                         for c in contexts])
    if len(answers) != len(contexts):
        raise ValueError("Answer refinement response count mismatch")
    resolved = enforce_choices(generator, contexts, answers, invalid_policy="default_a")
    if len(resolved) != len(contexts):
        raise ValueError("Answer format repair response count mismatch")
    result = []
    for c, raw_answer, (answer, status) in zip(contexts, answers, resolved):
        row = c["row"]
        public = {key: row.get(key) for key in (
            "application_id", "question_id", "question", "retrieval_query", "year", "exam",
            "unit", "source_number", "question_type", "shared_option_group", "data_status", "source_row")}
        result.append({**public, "retrieved": row["retrieved"],
                       "initial_answer": c["initial_answer"], "ter_answer": row["answer"],
                       "critic": c["critic"], "critic_raw": c["critic_raw"],
                       "critic_repair_raw": c.get("critic_repair_raw"),
                       "critic_status": c["critic_status"], "answer": answer,
                       "raw_answer": raw_answer, "choice_status": status,
                       "repair_raw_answer": c.get("repair_raw_answer"),
                       "alignrag_config": {"iterations": 1,
                           "prompt_version": "tcm-alignrag-questionbank-mandatory-choice-v3",
                           "generator_max_tokens": 32, "critic_max_tokens": 384,
                           "answer_key_used_for_model_input": False}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-api-base", default="http://127.0.0.1:8002")
    parser.add_argument("--critic-api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = read_rows(args.source_results)
    expected = keyed(source)
    if not source or any(not completed_prediction(row) for row in source):
        raise ValueError("TER input must contain completed candidate predictions")
    runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))["runtime"]
    app = Path(__file__).resolve().parent
    contract = {"protocol": "qwen35-two-pass-batch4-v2", "source_sha256": sha256(args.source_results),
                "runtime": runtime, "generator_model": args.generator_model,
                "critic_model": args.critic_model, "batch_size": args.batch_size,
                "generator_max_tokens": 32, "critic_max_tokens": 384,
                "critic_repair_attempts": 1, "evidence_field_limit": 850,
                "invalid_choice_policy": "default_a", "answer_key_used_for_model_input": False,
                "code_sha256": {name: sha256(app / name) for name in
                    (Path(__file__).name, "run_questionbank_rag_vs_alignrag.py", "run_tcm_alignrag.py")}}
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume")
    freeze_protocol(args.output.with_name("generation_contract.json"), contract, args.output)
    completed = read_rows(args.output) if args.output.exists() else []
    done = check_existing(completed, expected)
    pending = [row for row in source if str(row["application_id"]) not in done]
    print(json.dumps({"stage": "answer_critic", "resume_done": len(done),
                      "pending": len(pending), "total": len(source)}), flush=True)
    if not args.output.exists():
        atomic_jsonl(args.output, [])
    generator = batch_client(args.generator_api_base, args.generator_model, 32, 300, 2, 20)
    critic = batch_client(args.critic_api_base, args.critic_model, 384, 300, 2, 20)
    for batch in batched(pending, args.batch_size):
        updates = refine_batch(batch, generator, critic)
        for row in updates:
            row["alignrag_config"]["execution_batch_size"] = args.batch_size
        check_existing(updates, expected)
        for row in updates:
            completed.append(row)
            atomic_jsonl(args.output, completed)
            print(json.dumps({"stage": "answer_critic", "completed": len(completed),
                              "total": len(source), "application_id": row["application_id"],
                              "critic_status": row["critic_status"]}), flush=True)
    print(json.dumps({"stage": "answer_critic", "status": "complete", "total": len(completed),
                      "critic_statuses": dict(Counter(row["critic_status"] for row in completed))}), flush=True)


if __name__ == "__main__":
    main()
