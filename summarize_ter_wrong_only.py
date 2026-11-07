#!/usr/bin/env python3
"""Audit fixed32/batch4 TER-wrong-only results and independently rescore three arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark_integrity import (ANSWER, atomic_jsonl, csv_rows, keyed, merge_wrong_only,
                                 read_rows, sha256, validate_predictions)
from grade_single_results import choice

PROTOCOL = "ter-wrong-only-batch4-v1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge_year(work, year):
    source = read_rows(work / "ter_direct" / year / "ter_direct_results.jsonl")
    updates = read_rows(work / "evicr_v3" / year / "evidence_refinement_results.jsonl")
    grading = csv_rows(work / "grading" / f"ter_{year}/grading.csv")
    merged = merge_wrong_only(source, updates, grading)
    atomic_jsonl(work / "evicr_v3" / year / "evicr_v3_merged_results.jsonl", merged)


def summarize(work, slug):
    app = next(parent.parent for parent in work.resolve().parents if parent.name == "questionbank_eval")
    bank = app / "questionbank_eval"
    manifest = load(work / "manifest.json")
    if (manifest.get("protocol") != PROTOCOL or manifest.get("answer_critic_enabled") is not False
            or manifest.get("generation", {}).get("max_new_tokens") != 32
            or manifest.get("evidence_critic_max_tokens") != 640
            or manifest.get("baseline_batch_size") != 4 or manifest.get("refinement_batch_size") != 4
            or manifest.get("invalid_choice_policy") != "default_a"):
        raise ValueError("Not the fixed32/batch4 TER-wrong-only protocol")
    if (work / "answer_refinement").exists():
        raise ValueError("Answer-Critic results must remain in their separate historical run")
    for name, digest in manifest["code_sha256"].items():
        if sha256(app / name) != digest:
            raise ValueError(f"Frozen code changed: {name}")
    for name, digest in manifest["input_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    if (sha256(work / "model_source.json") != manifest["model_source_sha256"]
            or sha256(Path(manifest["critic_adapter"]) / "adapter_model.safetensors") != manifest["critic_adapter_sha256"]):
        raise ValueError("Candidate provenance or fixed Critic adapter changed")
    reports = {name: load(work / name) for name in
               ("deployment_preflight.json", "refinement_deployment_preflight.json")}
    for report in reports.values():
        if not report.get("exact_official_token_ids_verified") or report.get("max_new_tokens") != 32:
            raise ValueError("Missing official-template / 32-token deployment checks")
    refinement = reports["refinement_deployment_preflight.json"]
    if (reports["deployment_preflight.json"].get("baseline_batch_size") != 4
            or refinement.get("refinement_batch_size") != 4
            or refinement.get("answer_critic_probe_executed") is not False
            or not refinement.get("evidence_critic_batch4", {}).get("executable")
            or not refinement.get("candidate_refinement_stress_batch4_executable")
            or refinement.get("evidence_critic_max_tokens") != 640):
        raise ValueError("Missing evidence-only shared-VRAM batch4 checks")
    if manifest.get("baseline_reuse_sha256"):
        audit_path = work / "baseline_reuse.json"
        if sha256(audit_path) != manifest["baseline_reuse_sha256"]:
            raise ValueError("Baseline reuse audit changed")
        for relative, digest in load(audit_path)["files"].items():
            if sha256(work / relative) != digest:
                raise ValueError("Imported baseline no longer matches its original bytes")
    audits = {}
    for year, raw_count, scored_count in (("2021", 566, 554), ("2022", 596, 584)):
        questions = keyed(read_rows(bank / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl"))
        refs = {str(row["application_id"]): row["reference_answer"]
                for row in csv_rows(bank / f"model_benchmarks/_reference/reference_{year}.csv")
                if row.get("reference_status") == "valid"}
        if len(questions) != raw_count or len(refs) != scored_count or not set(refs) <= set(questions):
            raise ValueError("Question/reference IDs or counts changed")
        for arm, folder, filename in (("dense", "dense_rag", "dense_rag_results.jsonl"),
                                       ("ter", "ter_direct", "ter_direct_results.jsonl"),
                                       ("evicr", "evicr_v3", "evicr_v3_merged_results.jsonl")):
            rows = read_rows(work / folder / year / filename)
            checked = validate_predictions(rows, questions)
            if any(not ANSWER.fullmatch(str(row.get("answer", ""))) for row in rows):
                raise ValueError("Default-A historical policy requires explicit A-E answers")
            grades = keyed(csv_rows(work / "grading" / f"{arm}_{year}/grading.csv"))
            if set(grades) != set(refs):
                raise ValueError("Per-question grading coverage mismatch")
            for key, ref in refs.items():
                if (grades[key]["reference_answer"] != ref
                        or str(grades[key]["score"]) != str(int(choice(checked[key]["answer"]) == ref))):
                    raise ValueError("Per-question grading differs from independent scoring")
            contract = load(work / folder / year / "generation_contract.json")
            runtime = reports["refinement_deployment_preflight.json" if arm == "evicr"
                              else "deployment_preflight.json"]["runtime"]
            runner = "run_evidence_refinement_wrong_questions.py" if arm == "evicr" else "run_frozen_evidence_generation.py"
            if (contract.get("runtime") != runtime or contract.get("batch_size") != 4
                    or contract.get("generator_max_tokens", contract.get("max_new_tokens")) != 32
                    or contract.get("invalid_choice_policy") != "default_a"
                    or contract.get("runner_sha256") != manifest["code_sha256"][runner]):
                raise ValueError(f"Generation contract mismatch: {arm}/{year}")
            if arm == "evicr":
                if (contract.get("critic_max_tokens") != 640 or contract.get("critic_repair_attempts") != 1
                        or contract.get("evidence_field_limit") != 850 or contract.get("limit") is not None
                        or contract.get("source_sha256") != sha256(work / "ter_direct" / year / "ter_direct_results.jsonl")
                        or contract.get("grading_sha256") != sha256(work / "grading" / f"ter_{year}/grading.csv")):
                    raise ValueError("Evidence Critic must consume the full TER-wrong set and TER y0")
        ter = read_rows(work / "ter_direct" / year / "ter_direct_results.jsonl")
        if any(row.get("initial_answer") != row["answer"] for row in ter):
            raise ValueError("TER initial answer must be this candidate's direct y0")
        updates = read_rows(work / "evicr_v3" / year / "evidence_refinement_results.jsonl")
        if any(row.get("config", {}).get("batch_size") != 4
               or row.get("config", {}).get("generator_max_tokens") != 32
               or row.get("config", {}).get("prompt_version") != "evidence-refinement-questionbank-v3" for row in updates):
            raise ValueError("Mixed evidence-refinement row contracts")
        merged = merge_wrong_only(ter, updates, csv_rows(work / "grading" / f"ter_{year}/grading.csv"))
        if merged != read_rows(work / "evicr_v3" / year / "evicr_v3_merged_results.jsonl"):
            raise ValueError("TER correct and unscorable rows must be preserved exactly")
        audits[year] = {"v3_selected_TER_mistakes": len(updates), "answer_critic_rows": 0,
                        "TER_correct_rows_preserved_exactly": True, "TER_y0_verified": True}
    subprocess.run([sys.executable, str(app / "summarize_multimodel_benchmark.py"),
                    "--work-dir", str(work), "--slug", slug, "--arms", "dense", "ter", "evicr"],
                   check=True, stdout=subprocess.DEVNULL)
    result = load(work / "summary.json")
    result.update({"protocol": PROTOCOL, "pipeline": manifest["pipeline"],
                   "generation": manifest["generation"], "answer_critic_enabled": False,
                   "manifest_sha256": sha256(work / "manifest.json"), "wrong_only_audit": audits})
    temporary = work / "summary.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(work / "summary.json")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--merge-year", choices=("2021", "2022"))
    args = parser.parse_args()
    if args.merge_year:
        merge_year(args.work_dir, args.merge_year)
    else:
        summarize(args.work_dir, args.slug)


if __name__ == "__main__":
    main()
