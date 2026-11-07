#!/usr/bin/env python3
"""Resume three candidates using TER mistakes -> evidence Critic -> reanswer."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

from benchmark_integrity import freeze_protocol, sha256
from reuse_qwen35_baselines import reuse_completed_baselines
from run_three_model_qwen35_replay import Supervisor as BaseSupervisor, StopRequested, atomic_json, MODELS
from summarize_ter_wrong_only import PROTOCOL, summarize

CODE = ("run_three_model_wrong_only.py", "summarize_ter_wrong_only.py", "preflight_wrong_only.py",
        "run_three_model_qwen35_replay.py", "summarize_qwen35_replay.py", "run_qwen35_answer_refinement.py",
        "run_frozen_evidence_generation.py", "run_evidence_refinement_wrong_questions.py",
        "run_questionbank_rag_vs_alignrag.py", "run_tcm_alignrag.py", "benchmark_integrity.py",
        "serve_single_generator.py", "serve_qwen_critic_adapter.py", "grade_single_results.py",
        "preflight_generator_deployment.py", "reuse_qwen35_baselines.py",
        "summarize_multimodel_benchmark.py", "download_benchmark_model.py")


def commit_manifest(work, manifest):
    if not (work / "manifest.json").exists():
        audit_path = work / "baseline_reuse.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("answer_refinement_reused") is not False or audit.get("evicr_reused") is not False:
                raise ValueError("Only unchanged baseline predictions may be imported")
            for relative, expected in audit["files"].items():
                path = work / relative
                if not path.resolve().is_relative_to(work.resolve()) or sha256(path) != expected:
                    raise ValueError("Imported baseline changed before protocol commit")
        elif any(work.glob("*/*/*results.jsonl")):
            raise ValueError("Unregistered predictions precede this protocol")
    freeze_protocol(work / "manifest.json", manifest,
                    work / "evicr_v3/2022/evidence_refinement_results.jsonl")


class Supervisor(BaseSupervisor):
    def state(self, phase, status="running", **extra):
        self.stage = phase
        atomic_json(self.work / "state.json", {"status": status, "phase": phase,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "pid": os.getpid(), "protocol": PROTOCOL, **extra})

    def model(self, slug, repo, mirror):
        self.work = self.output / slug
        self.work.mkdir(parents=True, exist_ok=True)
        if (self.work / "summary.json").exists():
            summarize(self.work, slug)
            self.cleanup_weights(slug)
            self.state("complete", "complete")
            return
        if not (self.work / "manifest.json").exists():
            reuse_completed_baselines(self.baseline_source / slug, self.work, self.app)
        path = self.model_root / slug
        model_id = "benchmark-" + slug
        self.state("download_model")
        args = ["--repo", repo, "--path", path, "--manifest", self.work / "model_source.json"]
        if mirror:
            args.extend(["--modelscope", mirror])
        self.run("download_benchmark_model.py", *args)
        bank = self.app / "questionbank_eval"
        inputs = [bank / f"questions_{year}_mandatory_choice_v3_answer_blind.jsonl" for year in ("2021", "2022")]
        inputs += [bank / f"dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                   for year in ("2021", "2022")]
        inputs += [bank / f"mandatory_choice_v3_{year}/alignrag_results.jsonl" for year in ("2021", "2022")]
        inputs += [self.reference / f"reference_{year}.csv" for year in ("2021", "2022")]
        manifest = {
            "protocol": PROTOCOL, "repo": repo, "reported_arms": ["dense", "ter", "evicr"],
            "pipeline": "candidate TER y0 -> select TER mistakes offline -> evidence Critic v3 -> candidate answer; preserve TER correct rows",
            "generation": {"max_new_tokens": 32, "do_sample": False, "quantization": "NF4 4-bit double quantization"},
            "answer_critic_enabled": False, "evidence_critic_max_tokens": 640,
            "baseline_batch_size": 4, "refinement_batch_size": 4,
            "critic_repair_attempts": 1, "evidence_field_limit": 850,
            "invalid_choice_policy": "default_a", "invalid_output_note": "historical forced A fallback; explicitly tagged and counted",
            "v3_selection": "this candidate's TER mistakes; preserve its TER correct rows",
            "v3_initial_answer": "this candidate's normalized TER answer y0",
            "critic_base": str(self.root / "models/Qwen3.5-9B"), "critic_adapter": str(self.adapter),
            "critic_adapter_sha256": sha256(self.adapter / "adapter_model.safetensors"),
            "model_source_sha256": sha256(self.work / "model_source.json"),
            "input_sha256": {str(p): sha256(p) for p in inputs},
            "code_sha256": {name: sha256(self.app / name) for name in CODE},
            "answer_key_used_for_model_input": False,
            "reuse_policy": "verified strict replay-v1 batch4/32 baselines only; no answer-Critic, v3, probe, or adapted128 predictions imported",
            "baseline_reuse_sha256": sha256(self.work / "baseline_reuse.json")
            if (self.work / "baseline_reuse.json").exists() else None,
        }
        commit_manifest(self.work, manifest)
        self.state("generator_startup")
        self.serve("serve_single_generator.py", ["--model-path", path, "--model-id", model_id,
                   "--host", "127.0.0.1", "--port", "8002"], self.generator_base, model_id,
                   self.work / "logs/generator_service.log")
        self.state("deployment_preflight")
        live = self.work / "deployment_preflight_live.json"
        self.run("preflight_generator_deployment.py", "--api-base", self.generator_base, "--model", model_id,
                 "--max-new-tokens", 32, "--require-batch4", "--output", live)
        report = json.loads(live.read_text())
        report_path = self.work / "deployment_preflight.json"
        if (self.work / "baseline_reuse.json").exists():
            if json.loads(report_path.read_text())["runtime"] != report["runtime"]:
                raise ValueError("Imported baselines and current candidate runtime differ")
        else:
            atomic_json(report_path, report)
        if report["baseline_batch_size"] != 4:
            raise ValueError("Required baseline batch4 did not execute")
        for arm, directory in (("dense", "dense_rag"), ("ter", "ter_direct")):
            for year in ("2022", "2021"):
                self.state(f"{arm}_{year}")
                source = (bank / f"dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                          if arm == "dense" else bank / f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
                out = self.work / directory / year / f"{directory}_results.jsonl"
                self.run("run_frozen_evidence_generation.py", "--arm", arm, "--evidence-input", source,
                         "--output", out, "--generator-model", model_id, "--api-base", self.generator_base,
                         "--batch-size", 4, "--max-new-tokens", 32, "--invalid-choice-policy", "default_a",
                         "--runtime-contract", report_path, "--resume")
                self.grade(out, year, arm)
        self.state("critic_startup")
        self.serve("serve_qwen_critic_adapter.py", ["--model", self.root / "models/Qwen3.5-9B",
                   "--adapter", self.adapter, "--generator-model-id", "qwen-generator-unused",
                   "--critic-model-id", self.critic_id, "--host", "127.0.0.1", "--port", "8001"],
                   self.critic_base, self.critic_id, self.work / "logs/critic_service.log")
        self.state("refinement_deployment_preflight")
        refinement_report = self.work / "refinement_deployment_preflight.json"
        self.run("preflight_wrong_only.py", "--api-base", self.generator_base, "--model", model_id,
                 "--source-root", self.work, "--output", refinement_report)
        for year in ("2022", "2021"):
            self.state(f"evidence_v3_{year}")
            ter = self.work / "ter_direct" / year / "ter_direct_results.jsonl"
            out = self.work / "evicr_v3" / year
            self.run("run_evidence_refinement_wrong_questions.py", "--source-results", ter,
                     "--grading-csv", self.work / "grading" / f"ter_{year}/grading.csv",
                     "--output-dir", out, "--generator-api-base", self.generator_base,
                     "--critic-api-base", self.critic_base, "--generator-model", model_id,
                     "--critic-model", self.critic_id, "--batch-size", 4,
                     "--generator-max-tokens", 32, "--critic-max-tokens", 640,
                     "--invalid-choice-policy", "default_a", "--runtime-contract", refinement_report, "--resume")
            self.run("summarize_ter_wrong_only.py", "--work-dir", self.work, "--slug", slug, "--merge-year", year)
            self.grade(out / "evicr_v3_merged_results.jsonl", year, "evicr")
        self.stop_servers()
        self.state("validate_summary")
        summarize(self.work, slug)
        self.cleanup_weights(slug)
        self.state("complete", "complete")

    def run_queue(self):
        for slug, repo, mirror in MODELS:
            for attempt in range(1, 4):
                print(json.dumps({"model": slug, "attempt": attempt, "protocol": PROTOCOL}), flush=True)
                try:
                    self.model(slug, repo, mirror)
                    break
                except Exception as error:
                    print(json.dumps({"model": slug, "error": str(error), "phase": self.stage,
                                      "action": "retry from committed checkpoints"}), flush=True)
                    self.state(self.stage, "retry_pending", error=str(error))
                    self.stop_servers()
                    if attempt == 3:
                        raise
                    time.sleep(30)
                finally:
                    self.stop_servers()
        combined = {slug: json.loads((self.output / slug / "summary.json").read_text())["arms"]
                    for slug, _, _ in MODELS}
        atomic_json(self.output / "all_models_summary.json", combined)
        self.work = self.output
        self.state("complete", "complete", completed_models=list(combined))
        print("All three TER-wrong-only benchmarks complete and independently verified", flush=True)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    app = Path(__file__).resolve().parent
    output = args.output_root or app / "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_v1"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(app, output)

        def stop(signum, frame):
            raise StopRequested()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            supervisor.run_queue()
        except StopRequested:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            supervisor.state(supervisor.stage, "interrupted")
            supervisor.stop_servers()
            raise SystemExit(143)
        finally:
            supervisor.stop_servers()


if __name__ == "__main__":
    main()
