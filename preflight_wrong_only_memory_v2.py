#!/usr/bin/env python3
"""Run the frozen evidence-only preflight and verify both CUDA memory policies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import requests


def main():
    app = Path(__file__).resolve().parent
    args = sys.argv[1:]
    output = Path(args[args.index("--output") + 1])
    expected = hashlib.sha256((app / "serve_benchmark_memory_control.py").read_bytes()).hexdigest()
    before = {}
    for port in (8001, 8002):
        response = requests.get(f"http://127.0.0.1:{port}/v1/benchmark_memory", timeout=15)
        response.raise_for_status()
        memory = response.json()
        if memory.get("memory_policy") != "release_after_request" or memory.get("wrapper_sha256") != expected:
            raise ValueError("Wrong memory wrapper or CUDA reservation lifecycle")
        before[str(port)] = memory
    subprocess.run([sys.executable, "-u", str(app / "preflight_wrong_only.py"), *args], check=True)
    after = {}
    for port in (8001, 8002):
        response = requests.get(f"http://127.0.0.1:{port}/v1/benchmark_memory", timeout=15)
        response.raise_for_status()
        memory = response.json()
        if (memory["memory_policy"] != "release_after_request" or memory["wrapper_sha256"] != expected
                or memory["reserved_bytes"] - memory["allocated_bytes"] > 1024**3):
            raise ValueError("Unused CUDA reservations remained after completed requests")
        after[str(port)] = memory
    report = json.loads(output.read_text())
    report["memory_control"] = {"policy": "release_after_request", "wrapper_sha256": expected,
                                "before": before, "after": after, "batch_size_unchanged": 4}
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
