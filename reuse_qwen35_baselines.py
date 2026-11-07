"""Copy only complete, unchanged 32-token batch-4 baselines from replay v1."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from benchmark_integrity import read_rows, sha256
from run_frozen_evidence_generation import completed


def reuse_completed_baselines(source: Path, target: Path, app: Path) -> dict | None:
    if not source.exists():
        return None
    if source.parent.name != "qwen35_replay_v1" or source.name != target.name:
        raise ValueError("Only matching strict replay-v1 baselines may be reused")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("protocol") != "qwen35-two-pass-v1"
            or manifest.get("generation", {}).get("max_new_tokens") != 32):
        raise ValueError("Source is not the strict 32-token two-pass replay")
    files = [source / "model_source.json", source / "deployment_preflight.json"]
    report = json.loads(files[1].read_text(encoding="utf-8"))
    if report.get("baseline_batch_size") != 4 or report.get("max_new_tokens") != 32:
        raise ValueError("Source baselines did not use batch 4 / 32 tokens")
    for arm, folder in (("dense", "dense_rag"), ("ter", "ter_direct")):
        for year, total in (("2022", 596), ("2021", 566)):
            directory = source / folder / year
            prediction = directory / f"{folder}_results.jsonl"
            contract_path = directory / "generation_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            frozen = (app / "questionbank_eval/dense_chunk_rag_v1" /
                      f"{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                      if arm == "dense" else app / "questionbank_eval" /
                      f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
            if (contract.get("max_new_tokens") != 32 or contract.get("batch_size") != 4
                    or contract.get("invalid_choice_policy") != "default_a"
                    or contract.get("runtime") != report["runtime"]
                    or contract.get("runner_sha256") != sha256(app / "run_frozen_evidence_generation.py")
                    or contract.get("evidence_input_sha256") != sha256(frozen)):
                raise ValueError(f"Incompatible baseline contract: {arm}/{year}")
            rows, done = completed(prediction, read_rows(frozen), arm)
            if len(rows) != total or len(done) != total:
                raise ValueError(f"Incomplete baseline: {arm}/{year}")
            files.extend([prediction, contract_path])
    audit = {"source_protocol": manifest["protocol"], "source_directory": str(source.resolve()),
             "source_manifest_sha256": sha256(source / "manifest.json"),
             "copied_stages": ["dense_rag", "ter_direct"],
             "answer_refinement_reused": False, "evicr_reused": False,
             "files": {str(path.relative_to(source)): sha256(path) for path in files}}
    for path in files:
        destination = target / path.relative_to(source)
        if destination.exists():
            if sha256(destination) != sha256(path):
                raise ValueError(f"Migration would overwrite a different file: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    target.mkdir(parents=True, exist_ok=True)
    (target / "baseline_reuse.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
