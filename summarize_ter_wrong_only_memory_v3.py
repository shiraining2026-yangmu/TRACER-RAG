#!/usr/bin/env python3
"""Audit memory-v3 plus the Critic-budget-bounded synthetic preflight."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import summarize_ter_wrong_only_memory_v2 as memory_v2
from preflight_wrong_only_memory_v3 import STRESS_POLICY, serialized

PROTOCOL = "ter-wrong-only-batch4-memory-v3"


def summarize(work, slug):
    manifest = memory_v2.original.load(work / "manifest.json")
    report = memory_v2.original.load(work / "refinement_deployment_preflight.json")
    stress = report.get("stress_budget", {})
    package = stress.get("tested_package")
    if (manifest.get("preflight_stress_policy") != STRESS_POLICY or stress.get("policy") != STRESS_POLICY
            or stress.get("critic_max_tokens") != 640 or not 0 < stress.get("tested_json_tokens", 0) <= 640
            or stress.get("formal_predictions_modified") is not False
            or stress.get("substituted_probe_rows") != sum(report["ter_wrong_counts"].values())
            or hashlib.sha256(serialized(package).encode()).hexdigest() != stress.get("tested_package_sha256")):
        raise ValueError("Missing reachable synthetic stress preflight contract")
    saved = memory_v2.PROTOCOL
    try:
        memory_v2.PROTOCOL = PROTOCOL
        return memory_v2.summarize(work, slug)
    finally:
        memory_v2.PROTOCOL = saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--merge-year", choices=("2021", "2022"))
    args = parser.parse_args()
    if args.merge_year:
        memory_v2.original.merge_year(args.work_dir, args.merge_year)
    else:
        summarize(args.work_dir, args.slug)


if __name__ == "__main__":
    main()
