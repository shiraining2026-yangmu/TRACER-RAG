#!/usr/bin/env python3
"""Check public file boundaries, Python syntax, credential patterns and hashes.

Only tracked and nonignored files are scanned. Findings identify locations, not
matching values. This is a static release check, not a privacy certification.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = "SHA256SUMS.txt"
BLOCKED_SUFFIXES = {".jsonl", ".xlsx", ".xls", ".pdf", ".tex", ".docx", ".drawio",
                    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".npy", ".npz",
                    ".sqlite", ".db", ".key", ".pem", ".zip", ".7z", ".gz", ".tar"}
BLOCKED_PARTS = {"data", "models", "models_benchmark", "outputs", "runs", "results", "checkpoints", "indexes", "questionbank_eval", "__pycache__"}
PATTERNS = {
    "API credential": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential in URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
}
REQUIRED = {"README.md", "README.zh-CN.md", "RIGHTS.md", "DATA_AVAILABILITY.md",
            "docs/REPRODUCIBILITY.md", "docs/DATA_CONTRACTS.md", "docs/RELEASE_SCOPE.md",
            "reproducibility/SOURCE_MANIFEST.json", "run_evicr.py", "prepare_ter_corpus.py",
            "export_evicr_qd_sft.py", "paired_statistics.py", "examples/offline_demo.py"}
REVIEWED_SUPPORTING_CSVS = {
    "reproducibility/supporting_data/" + name for name in (
        "archived_exam_correctness.csv", "archived_paired_statistics_primary.csv",
        "archived_paired_statistics_sensitivity.csv", "archived_no_rag_vs_dense.csv",
        "structure_assessments.csv", "open_qa_ai_scores.csv", "strict_ablation_aggregates.csv",
        "evaluation_query_membership.csv", "critic_seed_build_membership.csv",
    )
}
REQUIRED |= REVIEWED_SUPPORTING_CSVS | {
    "scripts/check_supporting_data.py", "reproducibility/supporting_data/README.md",
    "reproducibility/supporting_data/DATA_DICTIONARY.md",
    "reproducibility/supporting_data/PROVENANCE.json",
    "reproducibility/supporting_data/numeric_summary.json",
}


def public_files():
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, capture_output=True)
    names = sorted(set(n for n in result.stdout.decode("utf-8").split("\0") if n))
    return {name: ROOT / name for name in names}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-checksums", action="store_true", help="Regenerate checksum list after reviewing the release content")
    args = parser.parse_args()
    files = public_files()
    issues = ["Missing required file: " + name for name in sorted(REQUIRED - set(files))]
    for name, path in files.items():
        if path.is_symlink() or not path.is_file():
            issues.append("Missing file or unexpected symlink: " + name)
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            issues.append("Unexpected large file: " + name)
        if path.suffix.lower() in BLOCKED_SUFFIXES or BLOCKED_PARTS.intersection(Path(name).parts):
            issues.append("Excluded artifact category: " + name)
        if path.name.startswith(".env") and path.name != ".env.example":
            issues.append("Environment file: " + name)
        if path.suffix.lower() == ".csv" and not (name.startswith("reproducibility/weight_search/") or name in REVIEWED_SUPPORTING_CSVS):
            issues.append("Unreviewed CSV: " + name)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeError, OSError):
            issues.append("Unexpected unreadable/nontext artifact: " + name)
            continue
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError:
                issues.append("Python syntax error: " + name)
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                issues.append(f"{label}: {name}:{text.count(chr(10), 0, match.start()) + 1}")
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    entries = {name: digest(path) for name, path in files.items() if name != CHECKSUMS}
    if args.write_checksums:
        (ROOT / CHECKSUMS).write_text("".join(f"{value}  {name}\n" for name, value in sorted(entries.items())), encoding="utf-8", newline="\n")
    if not (ROOT / CHECKSUMS).exists():
        print("Checksum file is missing", file=sys.stderr)
        return 1
    recorded = {}
    for line in (ROOT / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        if name in recorded:
            print("Duplicate checksum entry: " + name, file=sys.stderr)
            return 1
        recorded[name] = value
    if recorded != entries:
        print("Checksum mismatch; review changes before running --write-checksums", file=sys.stderr)
        return 1
    print(f"PASS: {len(files)} public files, Python syntax, selected credential patterns and {len(entries)} SHA-256 entries checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
