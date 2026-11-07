#!/usr/bin/env python3
"""Combine reciprocal, blind AI ratings for paired open-ended QA answers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


DIMS = ("correctness", "completeness", "safety")
BOOTSTRAP_DRAWS = 10_000


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["row_id"])] = row
    return rows


def percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def bootstrap_ci(values: list[float], seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = [sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_DRAWS)]
    estimates.sort()
    return [round(percentile(estimates, 0.025), 4), round(percentile(estimates, 0.975), 4)]


def sign_test_p(positive: int, negative: int) -> float:
    n = positive + negative
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def model_scores(row: dict, model: str) -> dict[str, float]:
    slot = "response_a" if row["model_mapping"]["A"] == model else "response_b"
    return {dim: float(row["judgement"][slot][dim]) for dim in DIMS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-a-pairs", type=Path, required=True)
    parser.add_argument("--judge-a-map", type=Path, required=True)
    parser.add_argument("--judge-b-pairs", type=Path, required=True)
    parser.add_argument("--judge-b-map", type=Path, required=True)
    parser.add_argument("--judge-a-name", required=True)
    parser.add_argument("--judge-b-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    pairs_a = read_jsonl(args.judge_a_pairs)
    pairs_b = read_jsonl(args.judge_b_pairs)
    map_a = json.loads(args.judge_a_map.read_text(encoding="utf-8"))
    map_b = json.loads(args.judge_b_map.read_text(encoding="utf-8"))
    models = map_a["models"]
    if map_b["models"] != models or set(pairs_a) != set(pairs_b):
        raise ValueError("judge runs must use the same candidate models and question IDs")

    model_a, model_b = models["A"], models["B"]
    complete = []
    for row_id in sorted(pairs_a, key=lambda value: (0, int(value)) if value.isdigit() else (1, value)):
        left, right = pairs_a[row_id], pairs_b[row_id]
        if left.get("model_mapping") != right.get("model_mapping"):
            raise ValueError(f"candidate order differs between judges for row {row_id}")
        if left.get("parse_status") != "valid" or right.get("parse_status") != "valid":
            continue
        scores_a = model_scores(left, model_a)
        scores_b = model_scores(left, model_b)
        scores_a_second = model_scores(right, model_a)
        scores_b_second = model_scores(right, model_b)
        per_model = {
            model_a: {dim: (scores_a[dim] + scores_a_second[dim]) / 2 for dim in DIMS},
            model_b: {dim: (scores_b[dim] + scores_b_second[dim]) / 2 for dim in DIMS},
        }
        complete.append({"row_id": row_id, "scores": per_model,
                         "judge_a_preference": left["judgement"]["preference"],
                         "judge_b_preference": right["judgement"]["preference"]})

    differences = {dim: [row["scores"][model_b][dim] - row["scores"][model_a][dim]
                         for row in complete] for dim in DIMS}
    composite = [sum(row["scores"][model_b][dim] - row["scores"][model_a][dim]
                     for dim in DIMS) / len(DIMS) for row in complete]
    positive = sum(value > 0 for value in composite)
    negative = sum(value < 0 for value in composite)
    preference_counts = {args.judge_a_name: {model_a: 0, model_b: 0, "tie": 0, "neither": 0},
                         args.judge_b_name: {model_a: 0, model_b: 0, "tie": 0, "neither": 0}}
    for row in complete:
        for judge, preference in ((args.judge_a_name, row["judge_a_preference"]),
                                  (args.judge_b_name, row["judge_b_preference"])):
            mapped = model_a if preference == "A" else model_b if preference == "B" else preference
            preference_counts[judge][mapped] += 1

    mean_scores = {model: {dim: round(sum(row["scores"][model][dim] for row in complete) / len(complete), 4)
                           if complete else None for dim in DIMS}
                   for model in (model_a, model_b)}
    summary = {
        "version": "open_qa_reciprocal_ai_judges_v1",
        "analysis_type": "exploratory_reciprocal_ai_judgement_not_expert_annotation",
        "judge_models": [args.judge_a_name, args.judge_b_name],
        "candidate_models": {"A": model_a, "B": model_b},
        "n_questions_in_both_judges": len(complete),
        "n_questions_in_input": len(pairs_a),
        "n_invalid_or_missing_pairs": len(pairs_a) - len(complete),
        "mean_dimension_scores_0_to_2_averaged_across_judges": mean_scores,
        "paired_difference_B_minus_A_0_to_2": {
            dim: {"mean": round(sum(values) / len(values), 4) if values else None,
                  "bootstrap_ci95": bootstrap_ci(values, 20261023 + index)}
            for index, (dim, values) in enumerate(differences.items())
        },
        "paired_composite_B_minus_A_0_to_2": {
            "mean": round(sum(composite) / len(composite), 4) if composite else None,
            "bootstrap_ci95": bootstrap_ci(composite, 20262023),
            "exact_two_sided_sign_test_p_excluding_ties": sign_test_p(positive, negative),
            "positive": positive, "negative": negative,
            "ties": len(composite) - positive - negative,
        },
        "preference_counts_by_judge": preference_counts,
        "bootstrap": {"unit": "question", "draws": BOOTSTRAP_DRAWS,
                      "interval": "percentile 95%", "seed": 20261023},
        "limitations": [
            "Both judges are also candidate generators; reciprocal scoring reduces but does not remove self-preference bias.",
            "AI ratings are exploratory and are not expert or clinical assessments.",
            "Judges saw the public reference answer; reference incompleteness may affect ratings.",
            "Public MedQuAD pretraining exposure cannot be ruled out; no patient-specific advice should be inferred.",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "paired_open_qa_ai_scores.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "model", *DIMS, "composite_mean_0_to_2",
                         "judge_a_preference", "judge_b_preference"])
        for row in complete:
            for model in (model_a, model_b):
                scores = row["scores"][model]
                writer.writerow([row["row_id"], model, *(scores[dim] for dim in DIMS),
                                 sum(scores.values()) / len(DIMS), row["judge_a_preference"],
                                 row["judge_b_preference"]])
    (args.output_root / "reciprocal_ai_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
