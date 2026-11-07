#!/usr/bin/env python3
"""Closed-book same-generator exam baseline: questions/options only, no evidence.

Uses the frozen answer-blind exams and the same batch client, decoding settings,
and explicit answer-format fallback policy as dense RAG. No answer keys are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from run_dense_chunk_rag import atomic_json, contract, digest, file_hash, log, questions, read_jsonl

VERSION = "same-qwen-no-rag-v1"


def messages(row):
    # Only the frozen question/option string crosses the model-input boundary.
    return [
        {"role": "system", "content": (
            "你正在完成中医执业医师单选题作答。请结合题干和选项作出最佳选择，"
            "绝不能以证据不足、无法判断或建议咨询代替作答。"
            "只输出一行，严格写成“答案：X”（X 必须为 A、B、C、D、E 之一）；不要输出依据、解释或任何其他文字。")},
        {"role": "user", "content": f"题目：\n{row['question']}"},
    ]


def resume_rows(path, frozen):
    rows = list(read_jsonl(path)) if path.exists() else []
    by_id = {row["application_id"]: row for row in frozen}
    ids = set()
    for row in rows:
        key = row.get("application_id")
        if (key in ids or key not in by_id or row.get("status") != "ok"
                or row.get("method") != VERSION or not re.fullmatch(r"答案：[A-E]", row.get("answer", ""))
                or row.get("question") != by_id[key]["question"]):
            raise ValueError("Corrupt, duplicate, mismatched, or invalid existing no-RAG result")
        ids.add(key)
    return rows, ids


def generate(args):
    from run_questionbank_rag_vs_alignrag import batch_client, enforce_choices
    exams = questions(args.questions)
    with urllib.request.urlopen(args.api_base.rstrip("/") + "/v1/models", timeout=30) as response:
        served = {m["id"] for m in json.load(response)["data"]}
    if args.generator_model not in served:
        raise ValueError("The expected same-generator endpoint is not available")
    client = batch_client(args.api_base, args.generator_model, 32, 300, 2, 20)
    total = sum(len(rows) for _, rows in exams)
    finished_prior_years = 0
    for question_path, frozen in exams:
        out = args.output_dir / question_path.stem.replace("questions_", "")
        out.mkdir(parents=True, exist_ok=True)
        spec = {"method": VERSION, "questions_sha256": file_hash(question_path), "model": args.generator_model,
                "temperature": 0.0, "max_tokens": 32, "batch_size": args.batch_size,
                "prompt_template_sha256": digest(json.dumps(messages({"question": "{question}"}), ensure_ascii=False)),
                "prompt_template": messages({"question": "{question}"}),
                "format_policy": "existing one repair then explicitly marked forced_default_A",
                "retrieval": False, "critic": False, "script_sha256": file_hash(Path(__file__))}
        contract(out / "generation_contract.json", spec)
        target = out / "no_rag_results.jsonl"
        completed, ids = resume_rows(target, frozen)
        pending = [row for row in frozen if row["application_id"] not in ids]
        started = time.monotonic()
        initial = len(completed)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start+args.batch_size]
            raw = client([messages(row) for row in batch])
            resolved = enforce_choices(client, batch, raw)
            if len(resolved) != len(batch):
                raise ValueError("Answer-format resolver returned the wrong batch size")
            for row, original, (answer, origin) in zip(batch, raw, resolved):
                completed.append({**row, "answer": answer, "raw_answer": original, "choice_origin": origin,
                                  "choice_status": origin, "status": "ok", "method": VERSION})
            temporary = target.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as f:
                for row in completed:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(target)
            elapsed = time.monotonic() - started
            seconds_per_question = elapsed/max(len(completed)-initial, 1)
            summary = {"status": "complete" if len(completed)==len(frozen) else "running", "completed": len(completed),
                       "total": len(frozen), "choice_origins": dict(Counter(r["choice_origin"] for r in completed)),
                       "elapsed_this_run_seconds": round(elapsed),
                       "eta_seconds": round((len(frozen)-len(completed))*seconds_per_question)}
            atomic_json(out / "summary.json", summary)
            state = {"status": "running", "pid": os.getpid(), "exam": question_path.name,
                     "completed": finished_prior_years+len(completed), "total": total,
                     "eta_all_seconds": round((total-finished_prior_years-len(completed))*seconds_per_question),
                     "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            atomic_json(args.output_dir / "chain_state.json", state)
            log("no_rag_generation", exam=question_path.name, **summary)
        finished_prior_years += len(completed)
    atomic_json(args.output_dir / "chain_state.json", {"status": "complete", "pid": os.getpid(), "completed": total,
                "total": total, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    log("no_rag_complete", completed=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--generator-model", default="qwen-generator-repair-v2")
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=4)
    # Existing valid rows are always checked and retained.  This explicit flag
    # makes that intended crash-recovery behavior visible to benchmark drivers.
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output_dir / "writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            atomic_json(args.output_dir / "chain_state.json", {"status": "starting", "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            generate(args)
        except Exception as exc:
            atomic_json(args.output_dir / "chain_state.json", {"status": "failed", "pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
