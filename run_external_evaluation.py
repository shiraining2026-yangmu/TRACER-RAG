#!/usr/bin/env python3
"""Run answer-blind external transfer evaluations with one local model endpoint.

CMExam is used as an independent multiple-choice task. MedQuAD is used as an
open-ended medical QA task. The reference answer is written to a separate
file and is never included in model prompts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import time
from pathlib import Path
from urllib.request import Request, urlopen


CHOICE = re.compile(r"(?:答案|answer|choice)\s*[：:：]?\s*([A-E])\b", re.I)
LOOSE = re.compile(r"\b([A-E])\b", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sample(items: list[dict], size: int, key: str) -> list[dict]:
    if len(items) <= size:
        return items
    ranked = sorted(items, key=lambda item: hashlib.sha256(
        (key + "\0" + str(item["row_id"])).encode("utf-8")).hexdigest())
    return ranked[:size]


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.split(r"\n\s*选项\s*[：:]", value, maxsplit=1)[0]
    value = re.sub(r"^以下为单选题。请结合题干、选项和提供的参考资料作答，并在回答结尾明确写出“答案：X”。\s*", "", value)
    value = re.sub(r"^题干\s*[：:]\s*", "", value.strip())
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def excluded_questions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = normalize_question(row.get("retrieval_query") or row.get("question", ""))
                if value:
                    excluded.add(value)
    return excluded


def post_batch(api_base: str, model: str, prompts: list[list[dict[str, str]]], max_tokens: int) -> list[dict]:
    url = api_base.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions/batch"
    body = json.dumps({"requests": [{"model": model, "messages": messages,
                                      "temperature": 0.0, "max_tokens": max_tokens,
                                      "stream": False} for messages in prompts]},
                      ensure_ascii=False).encode("utf-8")
    for attempt in range(5):
        try:
            started = time.perf_counter()
            req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=900) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data", [])
            if len(data) != len(prompts):
                raise RuntimeError(f"unexpected completion count {len(data)} != {len(prompts)}")
            elapsed = (time.perf_counter() - started) / len(data)
            out = []
            for item in data:
                content = str(item["choices"][0]["message"].get("content", "")).strip()
                if not content:
                    raise RuntimeError("empty completion")
                out.append({"text": content, "usage": item.get("usage", {}),
                            "elapsed_seconds": elapsed})
            return out
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise AssertionError("unreachable")


def cmexam_rows(path: Path, size: int, excluded: set[str]) -> tuple[list[dict], dict[str, str], int]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            options = item.get("Options") or []
            question = str(item.get("Question", "")).strip()
            if not question or not options or item.get("Answer") not in "ABCDE":
                continue
            rows.append({"row_id": index, "question": question,
                         "options": options, "reference": item["Answer"]})
    before = len(rows)
    rows = [row for row in rows if normalize_question(row["question"]) not in excluded]
    excluded_count = before - len(rows)
    rows = stable_sample(rows, size, "cmexam-test-a4aaeb2d")
    return rows, {str(row["row_id"]): row["reference"] for row in rows}, excluded_count


def medquad_rows(path: Path, size: int) -> tuple[list[dict], dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for index, item in enumerate(csv.DictReader(f)):
            question, answer = str(item.get("Question", "")).strip(), str(item.get("Answer", "")).strip()
            if question and answer:
                rows.append({"row_id": index, "question": question, "reference": answer})
    rows = stable_sample(rows, size, "medquad-5b0961fb")
    return rows, {str(row["row_id"]): row["reference"] for row in rows}


def choice_prompt(row: dict) -> list[dict[str, str]]:
    options = "\n".join(f"{item.get('key', '')}. {item.get('value', '')}" for item in row["options"])
    return [{"role": "system", "content":
             "你正在完成独立的医学单选题。只输出一行‘答案：X’，X 必须为 A、B、C、D、E 之一；不要输出解释。"},
            {"role": "user", "content": f"题目：\n{row['question']}\n\n选项：\n{options}"}]


def open_prompt(row: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content":
             "你正在回答公开医学问答。请基于医学常识给出简洁、谨慎、可核查的回答；不要编造引用，不要输出思考过程。"},
            {"role": "user", "content": row["question"]}]


def choice_repair_prompt(question: str, raw: str) -> list[dict[str, str]]:
    return [{"role": "system", "content":
             "将单选题原作答修复为合规答案。根据原作答已表达的最佳判断选择；只输出一行‘答案：X’，X 为 A-E。"},
            {"role": "user", "content": f"题目：\n{question}\n\n原作答：\n{raw}"}]


def parsed_choice(text: str) -> str:
    values = CHOICE.findall(text or "") or LOOSE.findall(text or "")
    return values[-1].upper() if values else ""


def run_task(task: str, rows: list[dict], refs: dict[str, str], api_base: str,
             model: str, output: Path, batch_size: int, max_tokens: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[str(item["row_id"])] = item
    pending = [row for row in rows if str(row["row_id"]) not in existing]
    with output.open("a", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            prompts = [choice_prompt(row) if task == "cmexam" else open_prompt(row) for row in batch]
            results = post_batch(api_base, model, prompts, max_tokens)
            repair_indices = [i for i, result in enumerate(results)
                              if task == "cmexam" and not parsed_choice(result["text"])]
            repaired = post_batch(api_base, model,
                                  [choice_repair_prompt(batch[i]["question"], results[i]["text"])
                                   for i in repair_indices], 32) if repair_indices else []
            repair_by_index = dict(zip(repair_indices, repaired))
            for index, (row, result) in enumerate(zip(batch, results)):
                item = {"row_id": row["row_id"], "task": task, "question": row["question"],
                        "answer": result["text"], "usage": result.get("usage", {}),
                        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                        "protocol": {"temperature": 0.0, "batch_size": batch_size,
                                     "max_tokens": max_tokens, "reference_in_prompt": False,
                                     "source_row_sha256": hashlib.sha256(
                                         json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}}
                if task == "cmexam":
                    item["options"] = row["options"]
                    repair = repair_by_index.get(index)
                    candidate = repair["text"] if repair else result["text"]
                    parsed = parsed_choice(candidate)
                    item["raw_answer"] = result["text"]
                    item["repair_raw_answer"] = repair["text"] if repair else ""
                    item["answer"] = f"答案：{parsed or 'A'}"
                    item["choice_status"] = ("forced_default_A" if not parsed and repair else
                                              "format_repair" if repair else "model")
                    if repair:
                        item["usage"]["repair_prompt_tokens"] = int(repair.get("usage", {}).get("prompt_tokens", 0) or 0)
                        item["usage"]["repair_completion_tokens"] = int(repair.get("usage", {}).get("completion_tokens", 0) or 0)
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
            print(json.dumps({"task": task, "completed": len(existing) + min(start + batch_size, len(pending)),
                              "total": len(rows)}, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cmexam", type=Path, required=True)
    p.add_argument("--medquad", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--api-base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cmexam-size", type=int, default=500)
    p.add_argument("--medquad-size", type=int, default=300)
    p.add_argument("--exclude-questions", type=Path, nargs="*", default=[])
    p.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    args = p.parse_args()
    excluded = excluded_questions(args.exclude_questions)
    cm_rows, cm_ref, excluded_count = cmexam_rows(args.cmexam, args.cmexam_size, excluded)
    mq_rows, mq_ref = medquad_rows(args.medquad, args.medquad_size)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cmexam_reference.csv").write_text(
        "row_id,reference_answer\n" + "\n".join(f"{k},{v}" for k, v in cm_ref.items()) + "\n", encoding="utf-8")
    with (args.output_root / "medquad_reference.json").open("w", encoding="utf-8") as f:
        json.dump(mq_ref, f, ensure_ascii=False, indent=2)
    manifest = {"version": "external_transfer_v1", "model": args.model,
                "cmexam": {"source_sha256": sha256(args.cmexam), "commit": "a4aaeb2d9bc021bb358f45803776f5fe6a06ebe9",
                            "repository": "https://huggingface.co/datasets/fzkuji/CMExam",
                            "citation": "Liu et al., Benchmarking Large Language Models on CMExam, arXiv:2306.03030 (2023)",
                            "license_note": "Dataset card states Apache-2.0 for this unofficial preprocessing and research-only/non-commercial use; verify original-source terms before redistribution.",
                            "sample_size": len(cm_rows), "exact_overlap_excluded": excluded_count,
                            "exclusion_source_sha256": [sha256(path) for path in args.exclude_questions],
                            "selection": "exclude normalized exact overlap then sha256 stable ranking"},
                "medquad": {"source_sha256": sha256(args.medquad), "commit": "5b0961fbaa6d7f9c344c5d59c29943fb900c2eca",
                            "repository": "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset",
                            "citation": "Ben Abacha and Demner-Fushman, A Question-Entailment Approach to Question Answering, BMC Bioinformatics (2019)",
                            "license_note": "No explicit license was present in the dataset card/repository metadata; restrict use to research evaluation and do not redistribute source records pending rights confirmation.",
                            "sample_size": len(mq_rows), "selection": "sha256 stable ranking"},
                "reference_in_prompt": False, "batch_size": args.batch_size,
                "pretraining_contamination_not_ruled_out": True,
                "test_split_used": "CMExam test.json only; no train/validation rows used"}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_task("cmexam", cm_rows, cm_ref, args.api_base, args.model,
             args.output_root / "cmexam_predictions.jsonl", args.batch_size, 32)
    run_task("medquad", mq_rows, mq_ref, args.api_base, args.model,
             args.output_root / "medquad_predictions.jsonl", args.batch_size, 192)


if __name__ == "__main__":
    main()
