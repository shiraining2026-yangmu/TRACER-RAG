"""Adopt complete original wrong-only baselines without importing refinement."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from benchmark_integrity import sha256, read_rows
from run_frozen_evidence_generation import completed


def reuse_completed_baselines(source: Path, target: Path, app: Path):
    if not source.exists():
        return None
    if source.parent.name != "ter_wrong_only_batch4_v1" or source.name != target.name:
        raise ValueError("Only the matching original wrong-only model may supply baselines")
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest.get("protocol") != "ter-wrong-only-batch4-v1"
            or manifest.get("generation", {}).get("max_new_tokens") != 32
            or manifest.get("answer_critic_enabled") is not False):
        raise ValueError("Not the original 32-token TER-wrong-only experiment")
    report = json.loads((source / "deployment_preflight.json").read_text())
    if (report.get("baseline_batch_size") != 4 or report.get("max_new_tokens") != 32
            or not report.get("exact_official_token_ids_verified")):
        raise ValueError("Original baselines lack batch4/32 official-template checks")
    files = [source / "model_source.json", source / "deployment_preflight.json"]
    for arm, folder in (("dense", "dense_rag"), ("ter", "ter_direct")):
        for year, count in (("2022", 596), ("2021", 566)):
            prediction = source / folder / year / f"{folder}_results.jsonl"
            contract_path = prediction.with_name("generation_contract.json")
            contract = json.loads(contract_path.read_text())
            frozen = (app / f"questionbank_eval/dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                      if arm == "dense" else app / f"questionbank_eval/mandatory_choice_v3_{year}/alignrag_results.jsonl")
            if (contract.get("max_new_tokens") != 32 or contract.get("batch_size") != 4
                    or contract.get("invalid_choice_policy") != "default_a"
                    or contract.get("runtime") != report["runtime"]
                    or contract.get("runner_sha256") != sha256(app / "run_frozen_evidence_generation.py")
                    or contract.get("evidence_input_sha256") != sha256(frozen)):
                raise ValueError("Baseline generation contract changed")
            rows, done = completed(prediction, read_rows(frozen), arm)
            if len(rows) != count or len(done) != count:
                raise ValueError("Only a complete baseline can be reused")
            files.extend([prediction, contract_path])
    audit = {"source_protocol": manifest["protocol"], "source_directory": str(source.resolve()),
        "source_manifest_sha256": sha256(source / "manifest.json"), "copied_stages": ["dense_rag", "ter_direct"],
        "answer_refinement_reused": False, "evicr_reused": False,
        "files": {str(path.relative_to(source)): sha256(path) for path in files}}
    for path in files:
        destination = target / path.relative_to(source)
        if destination.exists() and sha256(destination) != sha256(path):
            raise ValueError("Refusing to overwrite a different migration target")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    (target / "baseline_reuse.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2) + "\n")
    return audit
