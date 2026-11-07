#!/usr/bin/env python3
"""Exercise public data contracts with entirely synthetic symbols; no API/GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_evicr_qd_labels import qd_hash


def dump(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def package():
    return {"evidence_status": "partial", "evidence_actions": [
        {"evidence_id": "E1", "action": "keep", "reason": "Synthetic symbol correspondence only"}],
        "knowledge_supplements": [], "change_support": "none",
        "answer_guidance": "State only the supplied synthetic symbol relationship; make no medical claims."}


def run(*args):
    subprocess.run([sys.executable, str(ROOT / args[0]), *map(str, args[1:])], check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh demo directory")
    out.mkdir(parents=True, exist_ok=True)
    payload = {"e_id": "E-SYNTHETIC-1", "application_id": "A-SYNTHETIC-HISTORY",
               "knowledge_id": "K-SYNTHETIC-1", "relation": "explanatory_support",
               "application_question": "Synthetic historical symbol A?",
               "application_answer": "Synthetic symbol A relates to B only under condition Z.",
               "knowledge_text": "Synthetic relation: A -> B if Z. No clinical meaning."}
    training = [{"sample_id": f"SYNTHETIC-{i}", "application_id": f"A-SYNTHETIC-{i}",
                 "question": f"Explain synthetic relation {i}.", "split": "build",
                 "retrieved": [{"payload": payload}]} for i in range(120)]
    dump(out / "synthetic_build_retrieval.jsonl", training)
    run("prepare_evicr_qd_supervision.py", "--source", out / "synthetic_build_retrieval.jsonl", "--output-dir", out / "seeds")
    seeds = [json.loads(s) for s in (out / "seeds/build_qd_seeds.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = [{**r, "schema_version": "evicr-qd-labels-v1", "status": "valid",
               "label": package(), "synthetic_demo_only": True} for r in seeds]
    dump(out / "synthetic_labels.jsonl", labels)
    run("export_evicr_qd_sft.py", "--labels", out / "synthetic_labels.jsonl", "--output-dir", out / "sft")
    run("train_critic_lora.py", "--model-name-or-path", "MODEL_NOT_LOADED",
        "--train-file", out / "sft/critic_sft_train.jsonl", "--validation-file", out / "sft/critic_sft_validation.jsonl",
        "--output-dir", out / "not_trained", "--dry-run")
    test = [{"application_id": "TEST-SYNTHETIC-1", "question": "Does synthetic relation A -> B require Z?",
             "split": "test", "retrieved": [{"payload": payload}]}]
    dump(out / "synthetic_test.jsonl", test)
    dump(out / "synthetic_y0.jsonl", [{"application_id": test[0]["application_id"], "question": test[0]["question"], "answer": "Synthetic baseline answer"}])
    run("run_evicr.py", "--frozen-test", out / "synthetic_test.jsonl", "--ter-y0", out / "synthetic_y0.jsonl", "--output-dir", out / "inference_preview", "--dry-run")
    print("PASS: 120 synthetic Build records -> q,D-only seeds -> SFT validation -> training dry-run -> inference prompt preview. No model training, scientific results, or API calls.")


if __name__ == "__main__":
    main()
