#!/usr/bin/env python3
"""Answer-free paired correctness analysis with whole-cluster bootstrap.

CSV columns: id, year, before_correct, after_correct, cluster_id (optional).
Use only after inference is frozen; this program never makes model requests.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


def exact_mcnemar(gains, losses):
    n = gains + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)


def analyze(rows, draws=10000, seed=20261107):
    if not rows or draws < 100:
        raise ValueError("Need nonempty paired data and at least 100 bootstrap draws")
    groups = {}
    gains = losses = both_correct = both_wrong = 0
    for row in rows:
        before, after = int(row["before_correct"]), int(row["after_correct"])
        if before not in (0, 1) or after not in (0, 1):
            raise ValueError("Correctness must be 0 or 1")
        gains += before == 0 and after == 1
        losses += before == 1 and after == 0
        both_correct += before == after == 1
        both_wrong += before == after == 0
        cluster = row.get("cluster_id", "").strip() or "singleton:" + row["id"]
        groups.setdefault((row["year"], cluster), []).append(after - before)
    clusters = [(sum(v), len(v)) for v in groups.values()]
    singleton = [v[0] for v in groups.values() if len(v) == 1]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        estimates.append(100 * sum(x[0] for x in sampled) / sum(x[1] for x in sampled))
    estimates.sort()
    return {
        "n": len(rows), "n_clusters": len(clusters),
        "before_correct": both_correct + losses, "after_correct": both_correct + gains,
        "paired_2x2": {"both_correct": both_correct, "correct_to_wrong": losses,
                       "wrong_to_correct": gains, "both_wrong": both_wrong},
        "accuracy_difference_percentage_points": 100 * (gains - losses) / len(rows),
        "paired_cluster_bootstrap_ci95_percentage_points": [estimates[int(draws * .025)], estimates[min(draws - 1, math.ceil(draws * .975) - 1)]],
        "bootstrap_draws": draws, "seed": seed,
        "singleton_mcnemar_exact_two_sided_p": exact_mcnemar(singleton.count(1), singleton.count(-1)),
        "singleton_mcnemar_n": len(singleton),
        "unclustered_mcnemar_sensitivity_p": exact_mcnemar(gains, losses),
        "inference": "Primary interval resamples whole clusters. Unclustered McNemar is a sensitivity statistic when questions share cases/options; singleton McNemar excludes non-singleton clusters.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "year", "before_correct", "after_correct"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required paired CSV columns")
        if set(reader.fieldnames) - required - {"cluster_id"}:
            raise ValueError("Unexpected CSV columns; provide correctness-only data")
        rows = list(reader)
    ids = [(r["year"], r["id"]) for r in rows]
    if len(set(ids)) != len(ids) or any(not a or not b for a, b in ids):
        raise ValueError("Missing or duplicate year/ID")
    result = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "combined": analyze(rows, args.draws, args.seed),
              "by_year": {year: analyze([r for r in rows if r["year"] == year], args.draws, args.seed)
                          for year in sorted({r["year"] for r in rows})}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
