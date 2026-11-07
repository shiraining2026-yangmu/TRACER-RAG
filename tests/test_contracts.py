"""Offline contract tests for the TCM AlignRAG preparation scripts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "critic_seed_3000.jsonl"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


supervision = load_module("generate_critic_supervision", ROOT / "generate_critic_supervision.py")


def test_seed_contract() -> None:
    if not SEEDS.exists():
        import unittest
        raise unittest.SkipTest("Private 3,000-record dataset is not distributed; synthetic contracts are tested separately")
    rows = [json.loads(line) for line in SEEDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3000
    assert all(row["split"] == "build" for row in rows)
    assert all("answer" not in row for row in rows)
    assert len({row["sample_id"] for row in rows}) == len(rows)
    assert len({"".join(row["question"].split()) for row in rows}) == len(rows)


def test_prompt_separation() -> None:
    question = "某症状如何理解？"
    evidence = "[E1 | E-1 | explanatory_support]\n历史问题：示例\n历史回答：示例\nK 知识：证据"
    weak = "WEAK_SECRET"
    strong = "STRONG_SECRET"
    teacher_prompt = "\n".join(message["content"] for message in supervision.teacher_messages(question, evidence))
    critique_prompt = "\n".join(message["content"] for message in supervision.critique_messages(question, evidence, weak, strong))
    assert weak not in teacher_prompt
    assert strong not in teacher_prompt
    assert weak in critique_prompt and strong in critique_prompt


def test_export_and_training_dry_run() -> None:
    critique = {
        "verdict": "needs_revision",
        "issues": [{
            "type": "unsupported_claim",
            "claim_or_location": "第 1 句",
            "why": "E1 未支持该断言",
            "evidence_ids": ["E1"],
            "fix": "删除或改为证据不足",
        }],
        "revision_instructions": ["删除未支持断言", "说明证据边界"],
        "evidence_boundary": "仅能根据 E1 作有限概括。",
    }
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        raw = temp / "critique.jsonl"
        with raw.open("w", encoding="utf-8") as handle:
            for index in range(80):
                row = {
                    "sample_id": f"T-{index:03d}",
                    "application_id": f"A-{index:03d}",
                    "split": "build",
                    "generation_phase": "critique",
                    "question": f"问题 {index}",
                    "evidence": "[E1 | E-1 | explanatory_support]\n历史问题：示例\n历史回答：示例\nK 知识：证据",
                    "weak_answer": "存在未支持断言。",
                    "strong_answer": "仅用于教师构建，不应进入学生输入。",
                    "teacher_critique": json.dumps(critique, ensure_ascii=False),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_dir = temp / "sft"
        subprocess.run(
            [sys.executable, str(ROOT / "export_critic_sft.py"), "--input", str(raw), "--output-dir", str(output_dir), "--min-accepted", "80"],
            check=True,
            capture_output=True,
            text=True,
        )
        train = output_dir / "critic_sft_train.jsonl"
        validation = output_dir / "critic_sft_validation.jsonl"
        all_rows = [json.loads(line) for path in (train, validation) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(all_rows) == 80
        for row in all_rows:
            student_input = "\n".join(message["content"] for message in row["messages"][:2])
            assert "y_strong" not in student_input
            assert "强教师的独立参考回答" not in student_input
        subprocess.run(
            [
                sys.executable, str(ROOT / "train_critic_lora.py"),
                "--model-name-or-path", "/not/loaded/by-dry-run",
                "--train-file", str(train),
                "--validation-file", str(validation),
                "--output-dir", str(temp / "adapter"),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    test_seed_contract()
    test_prompt_separation()
    test_export_and_training_dry_run()
    print("contract tests passed")
