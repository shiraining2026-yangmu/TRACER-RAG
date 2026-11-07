#!/usr/bin/env python3
"""Apply the frozen wrong-only audit to the separately versioned memory fix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_ter_wrong_only as original

PROTOCOL = "ter-wrong-only-batch4-memory-v2"


def summarize(work, slug):
    manifest = original.load(work / "manifest.json")
    report = original.load(work / "refinement_deployment_preflight.json")
    memory = report.get("memory_control", {})
    if (manifest.get("protocol") != PROTOCOL or manifest.get("cuda_memory_policy") != "release_after_request"
            or memory.get("policy") != "release_after_request"
            or memory.get("wrapper_sha256") != manifest["code_sha256"]["serve_benchmark_memory_control.py"]
            or set(memory.get("after", {})) != {"8001", "8002"}):
        raise ValueError("Missing versioned CUDA memory lifecycle contract")
    for value in memory["after"].values():
        if (value.get("memory_policy") != "release_after_request"
                or value.get("wrapper_sha256") != memory["wrapper_sha256"]
                or value["reserved_bytes"] - value["allocated_bytes"] > 1024**3):
            raise ValueError("CUDA reservation lifecycle check failed")
    saved = original.PROTOCOL
    try:
        original.PROTOCOL = PROTOCOL
        return original.summarize(work, slug)
    finally:
        original.PROTOCOL = saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--merge-year", choices=("2021", "2022"))
    args = parser.parse_args()
    if args.merge_year:
        original.merge_year(args.work_dir, args.merge_year)
    else:
        summarize(args.work_dir, args.slug)


if __name__ == "__main__":
    main()
