#!/usr/bin/env python3
"""Summarize answer-blind CMExam and MedQuAD external evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


CHOICE = re.compile(r"(?:答案|answer|choice)\s*[：:：]?\s*([A-E])\b", re.I)
LOOSE = re.compile(r"\b([A-E])\b", re.I)


def choice(text: str) -> str:
    values = CHOICE.findall(text or "") or LOOSE.findall(text or "")
    return values[-1].upper() if values else ""


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[一-鿿]", (text or "").lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    lcs = lcs_length(p, r)
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    args = p.parse_args()
    cm_ref = {}
    with (args.run_root / "cmexam_reference.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cm_ref[str(row["row_id"])] = row["reference_answer"]
    mq_ref = json.loads((args.run_root / "medquad_reference.json").read_text(encoding="utf-8"))
    cm = read_jsonl(args.run_root / "cmexam_predictions.jsonl")
    mq = read_jsonl(args.run_root / "medquad_predictions.jsonl")
    cm_correct = sum(choice(row["answer"]) == cm_ref.get(str(row["row_id"]), "") for row in cm)
    rouge = [rouge_l(row["answer"], mq_ref.get(str(row["row_id"]), "")) for row in mq]
    usage = [row.get("usage", {}) for row in cm + mq]
    elapsed = sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in cm + mq)
    summary = {"version": "external_transfer_summary_v1", "model": json.loads(
        (args.run_root / "manifest.json").read_text(encoding="utf-8"))["model"],
               "cmexam": {"rows": len(cm), "correct": cm_correct,
                           "accuracy_percent": round(100 * cm_correct / len(cm), 4) if cm else None,
                           "choice_parse_failures": sum(not choice(row["answer"]) for row in cm)},
               "medquad": {"rows": len(mq), "rouge_l_mean": round(sum(rouge) / len(rouge), 6) if rouge else None,
                           "rouge_l_median": round(sorted(rouge)[len(rouge) // 2], 6) if rouge else None,
                           "automatic_reference_overlap_only": True},
               "evidence_fidelity": {"status": "not_measured", "reason": "External tasks were direct transfer evaluations without a frozen evidence corpus."},
               "pretraining_contamination": {"status": "not_ruled_out",
                                            "interpretation": "Public benchmark split and public QA sources may have appeared in pretraining; results are transfer evidence, not contamination-proof generalization."},
               "expert_evaluation": {"status": "annotation_template_required", "not_claimed": True},
               "cost_evaluation": {"status": "token_and_elapsed_reported",
                                    "monetary_cost": None,
                                    "reason": "No user-specified local GPU tariff was supplied; monetary cost is not imputed."},
               "usage": {"prompt_tokens": sum(int(u.get("prompt_tokens", 0) or 0) for u in usage),
                         "completion_tokens": sum(int(u.get("completion_tokens", 0) or 0) for u in usage),
                         "repair_prompt_tokens": sum(int(u.get("repair_prompt_tokens", 0) or 0) for u in usage),
                         "repair_completion_tokens": sum(int(u.get("repair_completion_tokens", 0) or 0) for u in usage),
                         "elapsed_seconds_reported": round(elapsed, 3)},
               "choice_format_audit": {"forced_default_A": sum(row.get("choice_status") == "forced_default_A" for row in cm),
                                       "format_repaired": sum(row.get("choice_status") == "format_repair" for row in cm)}}
    # This is intentionally a blind form. Its mapping key stays with the
    # analysis files and is not part of the sheet given to raters.
    annotation_key = {"model": summary["model"], "items": {}}
    with (args.run_root / "expert_annotation_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["blinded_item_id", "task", "question", "candidate_response", "clinical_correctness_0_2",
                         "completeness_0_2", "safety_0_2", "unsupported_claim_0_1", "expert_id", "notes"])
        for row in cm:
            blind_id = hashlib.sha256(f"external-annotation-v1\0cmexam\0{row['row_id']}".encode()).hexdigest()[:12]
            annotation_key["items"][blind_id] = {"task": "cmexam", "row_id": row["row_id"]}
            writer.writerow([blind_id, "cmexam", row["question"], row["answer"], "", "", "", "", "", ""])
        for row in mq:
            blind_id = hashlib.sha256(f"external-annotation-v1\0medquad\0{row['row_id']}".encode()).hexdigest()[:12]
            annotation_key["items"][blind_id] = {"task": "medquad", "row_id": row["row_id"]}
            writer.writerow([blind_id, "medquad", row["question"], row["answer"], "", "", "", "", "", ""])
    (args.run_root / "expert_annotation_key_do_not_share.json").write_text(
        json.dumps(annotation_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
