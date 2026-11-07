#!/usr/bin/env python3
"""Run answer-blind, evidence-first EviCR on a frozen set of arbitrary size.

The Critic uses the exact system/user message builders used by q,D-only SFT.
Only the answer reviser receives y0. There is no answer-key-based routing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from export_evicr_qd_sft import system_message, user_message
from generate_evicr_qd_labels import validate_package
from run_test111_evicr_qd_ablation import (
    answer_messages, call_chat, critic_repair_messages, digest_text,
    evidence_text, extract_object, file_hash, public_row, read_jsonl, write_jsonl,
)

VERSION = "evicr-qd-public-v1"
FORBIDDEN_INPUT_KEYS = {"answer", "correct_answer", "standard_answer", "answer_key", "y0", "y_weak", "weak_answer", "strong_answer", "teacher_critique"}


def critic_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_message()},
            {"role": "user", "content": user_message(question, evidence)}]


def delivered_package(package: dict, no_supplements: bool) -> dict:
    delivered = copy.deepcopy(package)
    if no_supplements:
        delivered["knowledge_supplements"] = []
    return delivered


def load_inputs(frozen: Path, baseline: Path, field_limit: int) -> list[dict]:
    evidence_rows, answers = list(read_jsonl(frozen)), list(read_jsonl(baseline))
    if not evidence_rows:
        raise ValueError("Frozen evidence is empty")
    ids = [str(r.get("application_id", "")).strip() for r in evidence_rows]
    baseline_ids = [str(r.get("application_id", "")).strip() for r in answers]
    if any(not k for k in ids + baseline_ids) or len(set(ids)) != len(ids) or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("Inputs require unique, nonempty application IDs")
    if set(ids) != set(baseline_ids):
        raise ValueError("Frozen evidence and TER-y0 must have exactly the same IDs")
    by_id = dict(zip(baseline_ids, answers))
    rows = []
    for source, app_id in zip(evidence_rows, ids):
        if FORBIDDEN_INPUT_KEYS.intersection(source):
            raise ValueError("Frozen input contains a current answer or answer-key field")
        if source.get("split") != "test":
            raise ValueError("Evaluation evidence must be Test-only")
        question = str(source.get("question", "")).strip()
        initial = by_id[app_id]
        if not question or initial.get("question") != question:
            raise ValueError("Missing or mismatched question")
        y0 = str(initial.get("answer", "")).strip()
        if not y0:
            raise ValueError("Empty TER-y0 answer")
        rows.append({"application_id": app_id, "question": question,
                     "evidence": evidence_text(source, field_limit), "y0": y0,
                     "evidence_count": len(source["retrieved"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--ter-y0", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--critic-api", default="http://127.0.0.1:8001")
    parser.add_argument("--critic-model", default="evicr-critic")
    parser.add_argument("--generator-api", default="http://127.0.0.1:8002")
    parser.add_argument("--generator-model", default="generator")
    parser.add_argument("--critic-max-tokens", type=int, default=900)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--evidence-field-limit", type=int, default=850)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reuse-critic", type=Path, help="Use an immutable critic_qd.jsonl for a controlled no-supplements run")
    parser.add_argument("--no-knowledge-supplements", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write prompt/contract previews without any API calls")
    args = parser.parse_args()
    if args.evidence_field_limit < 100:
        raise ValueError("Evidence field limit must be >= 100")
    if min(args.critic_max_tokens, args.answer_max_tokens, args.timeout) <= 0:
        raise ValueError("Generation budgets and timeout must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory for each frozen run")
    if args.no_knowledge_supplements and not args.reuse_critic and not args.dry_run:
        raise ValueError("A no-supplements ablation must reuse --reuse-critic from the full run")
    rows = load_inputs(args.frozen_test, args.ter_y0, args.evidence_field_limit)
    source_hashes = {"frozen_test": file_hash(args.frozen_test), "ter_y0": file_hash(args.ter_y0)}
    if args.reuse_critic:
        source_hashes["reused_critic"] = file_hash(args.reuse_critic)
    arm = "evicr_no_ks" if args.no_knowledge_supplements else "evicr_full"
    manifest = {
        "version": VERSION, "status": "dry_run" if args.dry_run else "running",
        "n_questions": len(rows), "arm": arm, "source_sha256": source_hashes,
        "critic_input_contract": "system + q + D(q); no current y0 or answer key",
        "critic_prompt_version": "evicr-qd-sft-v1",
        "critic_model": args.critic_model, "generator_model": args.generator_model,
        "all_questions_processed": True, "answer_key_used_for_routing": False,
        "evidence_field_limit": args.evidence_field_limit,
        "critic_max_tokens": args.critic_max_tokens, "answer_max_tokens": args.answer_max_tokens,
        "fallback_policy": "one JSON repair attempt, then fail; never silently substitute a label",
    }
    reused = {}
    if args.reuse_critic:
        for record in read_jsonl(args.reuse_critic):
            app_id = record.get("application_id")
            if not app_id or app_id in reused:
                raise ValueError("Duplicate or missing ID in reused Critic output")
            reused[app_id] = record
        if set(reused) != {r["application_id"] for r in rows}:
            raise ValueError("Reused Critic IDs do not match frozen evaluation set")
        for row in rows:
            prior = reused[row["application_id"]]
            if prior.get("critic_prompt_hash") != digest_text(json.dumps(critic_messages(row["question"], row["evidence"]), ensure_ascii=False, sort_keys=True)):
                raise ValueError("Reused Critic prompt hash does not match")
            validate_package(prior["package"], row["evidence_count"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_manifest()
    if args.dry_run:
        previews = []
        for row in rows:
            prompt = critic_messages(row["question"], row["evidence"])
            previews.append({"application_id": row["application_id"], "critic_messages": prompt,
                             "critic_prompt_hash": digest_text(json.dumps(prompt, ensure_ascii=False, sort_keys=True))})
        write_jsonl(args.output_dir / "critic_prompt_preview.jsonl", previews)
        print(json.dumps(manifest, ensure_ascii=False))
        return
    critics, outputs = [], []
    try:
        for row in rows:
            prompt = critic_messages(row["question"], row["evidence"])
            prompt_hash = digest_text(json.dumps(prompt, ensure_ascii=False, sort_keys=True))
            if reused:
                record = reused[row["application_id"]]
                package = validate_package(record["package"], row["evidence_count"])
                record = {**record, "status": "reused"}
            else:
                raw = call_chat(args.critic_api, args.critic_model, prompt, args.critic_max_tokens, args.timeout)
                status = "valid"
                try:
                    package = validate_package(extract_object(raw), row["evidence_count"])
                except (ValueError, TypeError):
                    raw = call_chat(args.critic_api, args.critic_model, critic_repair_messages(row["question"], row["evidence"], raw), args.critic_max_tokens, args.timeout)
                    package = validate_package(extract_object(raw), row["evidence_count"])
                    status = "repaired"
                record = {"application_id": row["application_id"], "question": row["question"],
                          "package": package, "critic_prompt_hash": prompt_hash,
                          "critic_input_hash": digest_text(row["question"] + "\n" + row["evidence"]), "status": status}
            critics.append(record)
            delivered = delivered_package(package, args.no_knowledge_supplements)
            response = call_chat(args.generator_api, args.generator_model,
                                 answer_messages(row["question"], row["evidence"], row["y0"], delivered),
                                 args.answer_max_tokens, args.timeout)
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Generator returned an empty answer")
            result = public_row(row, arm, response, delivered)
            result["critic_prompt_hash"] = prompt_hash
            result["source_y0_hash"] = digest_text(row["y0"])
            outputs.append(result)
            write_jsonl(args.output_dir / "critic_qd.jsonl", critics)
            write_jsonl(args.output_dir / (arm + ".jsonl"), outputs)
        manifest["status"] = "complete"
        manifest["n_generated"] = len(outputs)
    except Exception:
        manifest["status"] = "failed"
        manifest["n_generated"] = len(outputs)
        save_manifest()
        raise
    save_manifest()
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
