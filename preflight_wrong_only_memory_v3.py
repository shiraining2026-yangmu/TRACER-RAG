#!/usr/bin/env python3
"""Keep the frozen batch4 probes, with a stress JSON reachable in 640 tokens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import requests

STRESS_POLICY = "qwen-640-token-reachable-json-v1"
BUDGET = 640


def full_stress_package():
    return {"evidence_status": "partial", "change_support": "strong",
        "evidence_actions": [{"evidence_id": f"E{i}", "action": "keep", "reason": "证" * 240} for i in (1, 2, 3)],
        "knowledge_supplements": [{"claim": "证" * 360, "basis": "model_knowledge", "confidence": "high"} for _ in range(2)],
        "answer_guidance": "证" * 360}


def serialized(package):
    return json.dumps(package, ensure_ascii=False, separators=(",", ":"))


def bounded_stress_package(token_count, budget=BUDGET):
    # Enumerate rather than assume token counts are monotonic under BPE merges.
    chosen = None
    for size in range(241):
        package = full_stress_package()
        for action in package["evidence_actions"]:
            action["reason"] = "证" * size
        for claim in package["knowledge_supplements"]:
            claim["claim"] = "证" * size
        package["answer_guidance"] = "证" * size
        tokens = token_count(serialized(package))
        if tokens <= budget:
            chosen = package, tokens, size
    if chosen is None:
        raise ValueError("Even an empty stress schema exceeds the Critic output budget")
    return chosen


def stress_adapter(answer_messages, bounded, substitutions):
    unreachable = full_stress_package()
    def render(question, evidence, initial_answer, package):
        if package == unreachable:
            substitutions.append(True)
            package = bounded
        return answer_messages(question, evidence, initial_answer, package)
    return render


def main():
    from transformers import AutoTokenizer
    import preflight_wrong_only as frozen

    app = Path(__file__).resolve().parent
    args = sys.argv[1:]
    output = Path(args[args.index("--output") + 1])
    expected = hashlib.sha256((app / "serve_benchmark_memory_control.py").read_bytes()).hexdigest()
    def memory_snapshot():
        values = {}
        for port in (8001, 8002):
            response = requests.get(f"http://127.0.0.1:{port}/v1/benchmark_memory", timeout=15)
            response.raise_for_status()
            value = response.json()
            if (value.get("memory_policy") != "release_after_request" or value.get("wrapper_sha256") != expected
                    or value["reserved_bytes"] - value["allocated_bytes"] > 1024**3):
                raise ValueError("Wrong memory wrapper or unused CUDA reservations remain")
            values[str(port)] = value
        return values
    before = memory_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(app.parent / "models/Qwen3.5-9B", local_files_only=True)
    def token_count(value):
        return len(tokenizer.encode(value, add_special_tokens=False))
    bounded, tokens, size = bounded_stress_package(token_count)
    substitutions = []
    saved = frozen.answer_messages
    frozen.answer_messages = stress_adapter(saved, bounded, substitutions)
    try:
        frozen.main()
    finally:
        frozen.answer_messages = saved
    report = json.loads(output.read_text())
    if len(substitutions) != sum(report["ter_wrong_counts"].values()):
        raise ValueError("Frozen stress-package construction changed unexpectedly")
    report["memory_control"] = {"policy": "release_after_request", "wrapper_sha256": expected,
        "before": before, "after": memory_snapshot(), "batch_size_unchanged": 4}
    report["stress_budget"] = {"policy": STRESS_POLICY, "critic_max_tokens": BUDGET,
        "original_json_tokens": token_count(serialized(full_stress_package())),
        "tested_json_tokens": tokens, "filler_characters_per_field": size,
        "tested_package": bounded, "tested_package_sha256": hashlib.sha256(serialized(bounded).encode()).hexdigest(),
        "substituted_probe_rows": len(substitutions), "formal_predictions_modified": False,
        "scope": "Synthetic preflight only; real Critic outputs and generation budgets unchanged"}
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
