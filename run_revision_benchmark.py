#!/usr/bin/env python3
"""Resumable 61-question rerun for the eight historical model protocols.

Only the affected questions are regenerated.  Full-paper percentages are
computed later by replacing these IDs in the frozen historical result files.
Candidate weights are downloaded into tmpfs and removed after each completed
model; the fixed Qwen3.5+LoRA Critic remains under the protected model path.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import requests

from benchmark_integrity import atomic_jsonl, csv_rows, keyed, merge_wrong_only, read_rows, sha256


APP = Path(__file__).resolve().parent
ROOT = APP.parent
REV = APP / "questionbank_eval/experiment1_6_revision_v1"
REFERENCE = APP / "questionbank_eval/model_benchmarks/_reference"
CRITIC_MODEL = ROOT / "models/Qwen3.5-9B"
CRITIC_ADAPTER = APP / "runs/alignrag_critic_3k_v1/selective_repair_v2/qwen_critic_lora_balanced_v2_len2560"
CRITIC_ID = "qwen-critic-repair-v2"

MODELS = (
    ("qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B-Instruct", None, "questionbank_eval/model_benchmarks/qwen2_5_7b_instruct", False, False),
    ("qwen3_8b", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B", "questionbank_eval/model_benchmarks/qwen3_8b", False, False),
    ("internlm3_8b_instruct", "internlm/internlm3-8b-instruct", "Shanghai_AI_Laboratory/internlm3-8b-instruct", "questionbank_eval/model_benchmarks/internlm3_8b_instruct", True, False),
    ("meta_llama_3_1_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct", "LLM-Research/Meta-Llama-3.1-8B-Instruct", "questionbank_eval/model_benchmarks/meta_llama_3_1_8b_instruct", False, False),
    ("mistral_7b_instruct_v03", "mistralai/Mistral-7B-Instruct-v0.3", None, "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_v1/mistral_7b_instruct_v03", False, False),
    ("gemma2_9b_it", "google/gemma-2-9b-it", "google/gemma-2-9b-it", "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v3/gemma2_9b_it", False, False),
    ("yi15_9b_chat", "01-ai/Yi-1.5-9B-Chat", "01ai/Yi-1.5-9B-Chat", "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v5/yi15_9b_chat", False, True),
    ("qwen35_9b", "", None, "questionbank_eval/model_benchmarks/qwen35_replay_v1", False, True),
)


def write_subset_references(question_ids: set[str]) -> None:
    manifest = {}
    for year in ("2021", "2022"):
        source = REFERENCE / f"reference_{year}.csv"
        rows = [row for row in csv_rows(source) if str(row.get("application_id")) in question_ids]
        if not rows:
            raise ValueError(f"No reference rows for {year}")
        target = REV / f"reference_{year}_affected.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        manifest[year] = {"rows": len(rows), "sha256": sha256(target)}
    (REV / "affected_reference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_ids() -> set[str]:
    return {str(row["application_id"]) for row in read_rows(REV / "rerun_questions_answer_blind.jsonl")}


def run_command(args: list[Any], log: Path | None = None) -> None:
    rendered = [str(item) for item in args]
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            subprocess.run([sys.executable, "-u", *rendered], cwd=APP, stdout=handle, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run([sys.executable, "-u", *rendered], cwd=APP, check=True)


class Runner:
    def __init__(self, output: Path):
        self.output = output
        self.processes: list[subprocess.Popen] = []
        self.stage = "starting"
        self.model_root = Path("/dev/shm/alignrag_revision_models")
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.model_root = self.model_root.resolve()

    def state(self, phase: str, status: str = "running", **extra: Any) -> None:
        self.stage = phase
        payload = {"status": status, "phase": phase, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
        (self.output / "state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def stop(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=15)
        self.processes.clear()

    def serve(self, script: str, args: list[Any], base: str, model_id: str, log: Path) -> None:
        try:
            requests.get(base + "/v1/models", timeout=2)
        except requests.RequestException:
            pass
        else:
            raise RuntimeError(f"API port already occupied: {base}")
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("ab")
        process = subprocess.Popen([sys.executable, "-u", str(APP / script), *[str(x) for x in args]],
                                   cwd=APP, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        self.processes.append(process)
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"{model_id} exited during startup; inspect {log}")
            try:
                response = requests.get(base + "/v1/models", timeout=3)
                response.raise_for_status()
                if any(item.get("id") == model_id for item in response.json().get("data", [])):
                    return
            except (requests.RequestException, ValueError):
                pass
            time.sleep(3)
        raise RuntimeError(f"Timed out waiting for {model_id}")

    def runtime_contract(self, work: Path, model_id: str) -> Path:
        runtime = requests.get("http://127.0.0.1:8002/v1/benchmark_runtime", timeout=30).json()
        path = work / "runtime_contract.json"
        path.write_text(json.dumps({"runtime": runtime, "model": model_id, "batch_size": 4}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def grade(self, results: Path, year: str, label: str, work: Path) -> Path:
        out = work / "grading" / f"{label}_{year}"
        run_command(["grade_single_results.py", "--results", results, "--reference-grading",
                     REV / f"reference_{year}_affected.csv", "--output-dir", out, "--label", f"{label}_{year}"],
                    work / "logs" / "grading.log")
        return out / "grading.csv"

    def merge_evicr(self, work: Path, year: str, qwen35: bool) -> None:
        source_path = (work / "answer_refinement" / year / "alignrag_results.jsonl"
                       if qwen35 else work / "ter_direct" / year / "ter_direct_results.jsonl")
        ter = read_rows(source_path)
        updates = read_rows(work / "evicr_v3" / year / "evidence_refinement_results.jsonl")
        grading = csv_rows(work / "grading" / f"{'answer_refinement' if qwen35 else 'ter'}_{year}" / "grading.csv")
        if qwen35:
            source = keyed(ter)
            wrong = {str(row["application_id"]) for row in grading if row.get("reference_status") == "valid" and str(row.get("score")) == "0"}
            replacement = keyed(updates)
            if set(replacement) != wrong:
                raise ValueError(f"Qwen3.5 EviCR IDs differ from wrong y1 IDs: {len(replacement)} vs {len(wrong)}")
            merged = []
            for row in ter:
                key = str(row["application_id"])
                if key in replacement:
                    update = replacement[key]
                    if update.get("retrieved") != row.get("retrieved") or update.get("initial_answer") != row.get("initial_answer"):
                        raise ValueError(f"Qwen3.5 EviCR changed frozen y0 evidence/input: {key}")
                    merged.append(update)
                else:
                    merged.append(row)
        else:
            merged = merge_wrong_only(ter, updates, grading)
        out = work / "evicr_v3" / year / "evicr_v3_merged_results.jsonl"
        atomic_jsonl(out, merged)
        self.grade(out, year, "evicr", work)

    def model(self, slug: str, repo: str, mirror: str | None, historical: str, trust_remote: bool, yi_wrapper: bool) -> None:
        work = self.output / slug
        work.mkdir(parents=True, exist_ok=True)
        done_marker = work / "complete.json"
        if done_marker.exists():
            return
        model_id = "benchmark-" + slug
        is_qwen35 = slug == "qwen35_9b"
        if is_qwen35:
            model_path = CRITIC_MODEL
        else:
            model_path = self.model_root / slug
            self.state(f"{slug}:download_model")
            args = ["download_benchmark_model.py", "--repo", repo, "--path", model_path, "--manifest", work / "model_source.json"]
            if mirror:
                args += ["--modelscope", mirror]
            run_command(args, work / "logs" / "download.log")

        try:
            self.state(f"{slug}:generator_startup")
            generator_args: list[Any] = []
            script = "serve_single_generator.py"
            if yi_wrapper:
                script = "serve_benchmark_memory_control_yi_eos_v1.py"
                generator_args += ["--service-kind", "generator", "--memory-policy", "release_after_request"]
            generator_args += ["--model-path", model_path, "--model-id", model_id, "--host", "127.0.0.1", "--port", 8002]
            if trust_remote:
                generator_args.append("--trust-remote-code")
            self.serve(script, generator_args, "http://127.0.0.1:8002", model_id, work / "logs" / "generator_service.log")
            contract = self.runtime_contract(work, model_id)

            for arm, directory in (("dense", "dense_rag"), ("ter", "ter_direct")):
                for year in ("2022", "2021"):
                    self.state(f"{slug}:{arm}_{year}")
                    source = REV / "evidence" / f"{arm}_{year}.jsonl"
                    output = work / directory / year / f"{directory}_results.jsonl"
                    run_command(["run_frozen_evidence_generation.py", "--arm", arm, "--evidence-input", source,
                                 "--output", output, "--generator-model", model_id, "--api-base", "http://127.0.0.1:8002",
                                 "--batch-size", 4, "--max-new-tokens", 32, "--invalid-choice-policy", "default_a",
                                 "--runtime-contract", contract, "--resume"], work / "logs" / "generation.log")
                    self.grade(output, year, arm, work)

            self.state(f"{slug}:critic_startup")
            self.serve("serve_qwen_critic_adapter.py", ["--model", CRITIC_MODEL, "--adapter", CRITIC_ADAPTER,
                         "--generator-model-id", "qwen-generator-unused", "--critic-model-id", CRITIC_ID,
                         "--host", "127.0.0.1", "--port", 8001], "http://127.0.0.1:8001", CRITIC_ID,
                        work / "logs" / "critic_service.log")

            for year in ("2022", "2021"):
                ter = work / "ter_direct" / year / "ter_direct_results.jsonl"
                if is_qwen35:
                    self.state(f"{slug}:answer_critic_{year}")
                    y1 = work / "answer_refinement" / year / "alignrag_results.jsonl"
                    run_command(["run_qwen35_answer_refinement.py", "--source-results", ter, "--output", y1,
                                 "--generator-api-base", "http://127.0.0.1:8002", "--critic-api-base", "http://127.0.0.1:8001",
                                 "--generator-model", model_id, "--critic-model", CRITIC_ID, "--batch-size", 4,
                                 "--runtime-contract", contract, "--resume"], work / "logs" / "answer_critic.log")
                    self.grade(y1, year, "answer_refinement", work)
                    source_for_evicr = y1
                    grading_label = "answer_refinement"
                else:
                    source_for_evicr = ter
                    grading_label = "ter"
                self.state(f"{slug}:evidence_v3_{year}")
                out = work / "evicr_v3" / year
                run_command(["run_evidence_refinement_wrong_questions.py", "--source-results", source_for_evicr,
                             "--grading-csv", work / "grading" / f"{grading_label}_{year}" / "grading.csv",
                             "--output-dir", out, "--generator-api-base", "http://127.0.0.1:8002",
                             "--critic-api-base", "http://127.0.0.1:8001", "--generator-model", model_id,
                             "--critic-model", CRITIC_ID, "--batch-size", 4, "--generator-max-tokens", 32,
                             "--critic-max-tokens", 640, "--invalid-choice-policy", "default_a",
                             "--runtime-contract", contract, "--resume"], work / "logs" / "evidence_refinement.log")
                self.merge_evicr(work, year, is_qwen35)

            manifest = {
                "protocol": "experiment1_6_revision_v1",
                "model": slug, "historical_source": historical, "reported_arms": ["dense", "ter", "evicr"],
                "rows_per_year": {"2021": 30, "2022": 31}, "batch_size": 4,
                "generator_max_tokens": 32, "evidence_critic_max_tokens": 640,
                "answer_critic_max_tokens": 384 if is_qwen35 else None,
                "answer_critic_enabled": is_qwen35,
                "answer_key_used_for_model_input": False,
                "revision_questions_sha256": sha256(REV / "rerun_questions_answer_blind.jsonl"),
            }
            (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            done_marker.write_text(json.dumps({"status": "complete", "model": slug}, ensure_ascii=False) + "\n", encoding="utf-8")
        finally:
            self.stop()
            if not is_qwen35 and model_path.exists():
                resolved = model_path.resolve()
                if resolved.parent != self.model_root or resolved.name != slug:
                    raise ValueError(f"refusing to remove unexpected candidate path: {resolved}")
                shutil.rmtree(resolved)

    def run_queue(self) -> None:
        for slug, repo, mirror, historical, trust_remote, yi_wrapper in MODELS:
            if (self.output / slug / "complete.json").exists():
                continue
            for attempt in range(1, 4):
                try:
                    print(json.dumps({"model": slug, "attempt": attempt, "status": "starting"}), flush=True)
                    self.model(slug, repo, mirror, historical, trust_remote, yi_wrapper)
                    break
                except Exception as exc:
                    self.stop()
                    self.state(f"{slug}:{self.stage}", "retry_pending", error=str(exc), attempt=attempt)
                    print(json.dumps({"model": slug, "attempt": attempt, "error": str(exc)}), flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(30)
        self.state("complete", "complete", completed_models=[slug for slug, *_ in MODELS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REV / "model_runs")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ids = load_ids()
    write_subset_references(ids)
    runner = Runner(args.output_root)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        runner.run_queue()
    except KeyboardInterrupt:
        runner.stop(); runner.state(runner.stage, "interrupted"); raise SystemExit(143)
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
