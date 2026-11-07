#!/usr/bin/env python3
"""Persistent sequential three-model replay of historical Qwen3.5's full chain."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import requests

from benchmark_integrity import freeze_protocol, sha256
from summarize_qwen35_replay import PROTOCOL, summarize
from reuse_qwen35_baselines import reuse_completed_baselines

MODELS = (
    ("mistral_7b_instruct_v03", "mistralai/Mistral-7B-Instruct-v0.3", None),
    ("gemma2_9b_it", "google/gemma-2-9b-it", "google/gemma-2-9b-it"),
    ("yi15_9b_chat", "01-ai/Yi-1.5-9B-Chat", "01ai/Yi-1.5-9B-Chat"),
)
CODE = ("run_three_model_qwen35_replay.py", "run_qwen35_answer_refinement.py",
        "summarize_qwen35_replay.py", "run_frozen_evidence_generation.py",
        "run_evidence_refinement_wrong_questions.py", "run_questionbank_rag_vs_alignrag.py",
        "run_tcm_alignrag.py", "benchmark_integrity.py", "serve_single_generator.py",
        "serve_qwen_critic_adapter.py", "grade_single_results.py", "preflight_generator_deployment.py",
        "reuse_qwen35_baselines.py")


class StopRequested(BaseException):
    pass


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def commit_manifest(work, manifest):
    guard = work / "dense_rag/2022/dense_rag_results.jsonl"
    audit_path = work / "baseline_reuse.json"
    if not (work / "manifest.json").exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("answer_refinement_reused") is not False or audit.get("evicr_reused") is not False:
            raise ValueError("Imported refinement predictions are forbidden")
        for relative, expected in audit["files"].items():
            path = work / relative
            if not path.resolve().is_relative_to(work.resolve()) or sha256(path) != expected:
                raise ValueError("Imported baseline changed before manifest commit")
        # Baseline predictions already have individually verified contracts.
        # The new top-level contract must precede the fresh answer-Critic pass.
        guard = work / "answer_refinement/2022/alignrag_results.jsonl"
    freeze_protocol(work / "manifest.json", manifest, guard)


class Supervisor:
    def __init__(self, app, output):
        self.app, self.output, self.root = app, output, app.parent
        self.reference = app / "questionbank_eval/model_benchmarks/_reference"
        self.model_root = self.root / "models_benchmark"
        self.adapter = app / "runs/alignrag_critic_3k_v1/selective_repair_v2/qwen_critic_lora_balanced_v2_len2560"
        self.generator_base, self.critic_base = "http://127.0.0.1:8002", "http://127.0.0.1:8001"
        self.critic_id = "qwen-critic-repair-v2"
        self.servers = []
        self.stage = "starting"
        self.work = output
        self.baseline_source = app / "questionbank_eval/model_benchmarks/qwen35_replay_v1"

    def state(self, phase, status="running", **extra):
        self.stage = phase
        atomic_json(self.work / "state.json", {"status": status, "phase": phase,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "pid": os.getpid(), "protocol": PROTOCOL, **extra})

    def run(self, script, *args):
        subprocess.run([sys.executable, "-u", str(self.app / script), *map(str, args)], check=True)

    def stop_servers(self):
        for process in reversed(self.servers):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
        self.servers.clear()

    def serve(self, script, args, base, model_id, log):
        # A competing service must never silently supply this experiment's API.
        try:
            requests.get(base + "/v1/models", timeout=3)
        except requests.RequestException:
            pass
        else:
            raise RuntimeError(f"API port already occupied before launching {model_id}")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            process = subprocess.Popen([sys.executable, "-u", str(self.app / script), *map(str, args)],
                                       stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        self.servers.append(process)
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError(f"{model_id} exited during startup; see {log}")
            try:
                response = requests.get(base + "/v1/models", timeout=3)
                response.raise_for_status()
                if any(item.get("id") == model_id for item in response.json().get("data", [])):
                    return
            except (requests.RequestException, ValueError):
                pass
            time.sleep(5)
        raise RuntimeError(f"Timed out waiting for {model_id}; see {log}")

    def grade(self, results, year, label):
        self.run("grade_single_results.py", "--results", results, "--reference-grading",
                 self.reference / f"reference_{year}.csv", "--output-dir",
                 self.work / "grading" / f"{label}_{year}", "--label", f"{label}_{year}")

    def cleanup_weights(self, slug):
        # This is called only after an independent all-output completion audit.
        path = self.model_root / slug
        allowed = {model[0] for model in MODELS}
        if path.is_symlink() or path.resolve().parent != self.model_root.resolve() or path.name not in allowed:
            raise ValueError("Unsafe candidate weight cleanup path")
        if path.exists():
            shutil.rmtree(path)
        (self.work / "model_deleted_after_verified_summary").touch()

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
        manifest = {"protocol": PROTOCOL, "repo": repo, "reported_arms": ["dense", "ter", "evicr"],
                    "generation": {"max_new_tokens": 32, "do_sample": False, "quantization": "NF4 4-bit double quantization"},
                    "answer_critic_max_tokens": 384, "evidence_critic_max_tokens": 640,
                    "baseline_batch_size": 4, "refinement_batch_size": 4,
                    "critic_repair_attempts": 1, "evidence_field_limit": 850,
                    "invalid_choice_policy": "default_a", "invalid_output_note": "historical forced A fallback; counted and explicitly tagged",
                    "v3_selection": "this candidate's answer-refinement y1 mistakes; preserve its y1 correct rows",
                    "v3_initial_answer": "this candidate's raw TER y0, historical behavior",
                    "critic_base": str(self.root / "models/Qwen3.5-9B"), "critic_adapter": str(self.adapter),
                    "critic_adapter_sha256": sha256(self.adapter / "adapter_model.safetensors"),
                    "model_source_sha256": sha256(self.work / "model_source.json"),
                    "input_sha256": {str(p): sha256(p) for p in inputs},
                    "code_sha256": {name: sha256(self.app / name) for name in CODE},
                    "answer_key_used_for_model_input": False,
                    "deployment": "official candidate chat template; Gemma eager attention; validated padding and shared-VRAM checks",
                    "reuse_policy": "verified complete replay-v1 batch4/32-token dense and TER may be copied; answer refinement starts fresh; 128-token predictions excluded",
                    "baseline_reuse_sha256": sha256(self.work / "baseline_reuse.json")
                    if (self.work / "baseline_reuse.json").exists() else None}
        commit_manifest(self.work, manifest)
        self.state("generator_startup")
        self.serve("serve_single_generator.py", ["--model-path", path, "--model-id", model_id,
                   "--host", "127.0.0.1", "--port", "8002"], self.generator_base, model_id,
                   self.work / "logs/generator_service.log")
        self.state("deployment_preflight")
        self.run("preflight_generator_deployment.py", "--api-base", self.generator_base, "--model", model_id,
                 "--max-new-tokens", 32, "--require-batch4", "--output", self.work / "deployment_preflight.json")
        report = json.loads((self.work / "deployment_preflight.json").read_text())
        baseline_batch = report["baseline_batch_size"]
        for arm, directory in (("dense", "dense_rag"), ("ter", "ter_direct")):
            for year in ("2022", "2021"):
                self.state(f"{arm}_{year}")
                source = (bank / f"dense_chunk_rag_v1/{year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"
                          if arm == "dense" else bank / f"mandatory_choice_v3_{year}/alignrag_results.jsonl")
                out = self.work / directory / year / f"{directory}_results.jsonl"
                self.run("run_frozen_evidence_generation.py", "--arm", arm, "--evidence-input", source,
                         "--output", out, "--generator-model", model_id, "--api-base", self.generator_base,
                         "--batch-size", baseline_batch, "--max-new-tokens", 32, "--invalid-choice-policy", "default_a",
                         "--runtime-contract", self.work / "deployment_preflight.json", "--resume")
                self.grade(out, year, arm)
        self.state("critic_startup")
        self.serve("serve_qwen_critic_adapter.py", ["--model", self.root / "models/Qwen3.5-9B",
                   "--adapter", self.adapter, "--generator-model-id", "qwen-generator-unused",
                   "--critic-model-id", self.critic_id, "--host", "127.0.0.1", "--port", "8001"],
                   self.critic_base, self.critic_id, self.work / "logs/critic_service.log")
        self.state("refinement_deployment_preflight")
        report_path = self.work / "refinement_deployment_preflight.json"
        self.run("preflight_generator_deployment.py", "--api-base", self.generator_base, "--model", model_id,
                 "--max-new-tokens", 32, "--critic-present", "--require-batch4",
                 "--refinement-source-root", self.work, "--output", report_path)
        refinement_batch = json.loads(report_path.read_text())["refinement_batch_size"]
        if baseline_batch != 4 or refinement_batch != 4:
            raise ValueError("Batch4 replay may not silently fall back to another batch size")
        for year in ("2022", "2021"):
            self.state(f"answer_critic_{year}")
            ter = self.work / "ter_direct" / year / "ter_direct_results.jsonl"
            y1 = self.work / "answer_refinement" / year / "alignrag_results.jsonl"
            self.run("run_qwen35_answer_refinement.py", "--source-results", ter, "--output", y1,
                     "--generator-model", model_id, "--critic-model", self.critic_id,
                     "--batch-size", refinement_batch, "--runtime-contract", report_path, "--resume")
            self.grade(y1, year, "answer_refinement")
            self.state(f"evidence_v3_{year}")
            out = self.work / "evicr_v3" / year
            self.run("run_evidence_refinement_wrong_questions.py", "--source-results", y1,
                     "--grading-csv", self.work / "grading" / f"answer_refinement_{year}/grading.csv",
                     "--output-dir", out, "--generator-api-base", self.generator_base,
                     "--critic-api-base", self.critic_base, "--generator-model", model_id,
                     "--critic-model", self.critic_id, "--batch-size", refinement_batch,
                     "--generator-max-tokens", 32, "--critic-max-tokens", 640,
                     "--invalid-choice-policy", "default_a", "--runtime-contract", report_path, "--resume")
            self.run("summarize_qwen35_replay.py", "--work-dir", self.work, "--slug", slug, "--merge-year", year)
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
        combined = {slug: json.loads((self.output / slug / "summary.json").read_text(encoding="utf-8"))["arms"]
                    for slug, _, _ in MODELS}
        atomic_json(self.output / "all_models_summary.json", combined)
        self.work = self.output
        self.state("complete", "complete", completed_models=list(combined))
        print("All three Qwen3.5 two-pass replays complete and independently verified", flush=True)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    app = Path(__file__).resolve().parent
    output = args.output_root or app / "questionbank_eval/model_benchmarks/qwen35_replay_batch4_v2"
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
            # Prevent an interrupted stage from retaining a misleading running marker.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            supervisor.state(supervisor.stage, "interrupted")
            supervisor.stop_servers()
            raise SystemExit(143)
        finally:
            supervisor.stop_servers()


if __name__ == "__main__":
    main()
