#!/usr/bin/env python3
"""Replace named inference rows with corrected reruns while preserving Test order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_integrity import atomic_jsonl, csv_rows, merge_wrong_only


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grading-csv", type=Path,
                        help="Require exactly every valid-key wrong ID and preserve all others")
    args = parser.parse_args()

    original = read_jsonl(args.original)
    if args.grading_csv:
        replacement_rows = read_jsonl(args.replacement)
        output = merge_wrong_only(original, replacement_rows, csv_rows(args.grading_csv))
        atomic_jsonl(args.output, output)
        print(json.dumps({"rows": len(output), "replaced": len(replacement_rows), "validated_wrong_only": True}))
        return
    replacements = {str(row.get("application_id", "")): row for row in read_jsonl(args.replacement)}
    if not replacements or any(not application_id for application_id in replacements):
        raise ValueError("Replacement output has missing application IDs")
    output = []
    replaced = set()
    for row in original:
        application_id = str(row.get("application_id", ""))
        if application_id in replacements:
            output.append(replacements[application_id])
            replaced.add(application_id)
        else:
            output.append(row)
    if replaced != set(replacements):
        raise ValueError("At least one replacement ID was absent from original output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(output), "replaced_application_ids": sorted(replaced)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
