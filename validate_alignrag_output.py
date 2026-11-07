#!/usr/bin/env python3
"""Fail closed when a smoke inference contains an invalid Critic fallback."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                try:
                    critique = json.loads(row["critic"])
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{args.input}:{number} has an invalid normalized Critic") from exc
                if not isinstance(critique, dict) or not str(row.get("answer", "")).strip():
                    raise ValueError(f"{args.input}:{number} has an empty answer or invalid Critic object")
                rows.append(row)
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(rows)}")
    statuses = Counter(str(row.get("critic_status", "missing")) for row in rows)
    if statuses.get("fallback", 0):
        raise RuntimeError(f"Smoke inference used fallback Critic output: {dict(statuses)}")
    print(json.dumps({"validated_rows": len(rows), "critic_statuses": dict(statuses)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
