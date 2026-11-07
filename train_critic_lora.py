#!/usr/bin/env python3
"""QLoRA/SFT training entry point for the evidence-sensitive Qwen Critic.

Input JSONL must be produced by export_critic_sft.py. Only assistant critique
tokens contribute to loss; question, evidence, and weak answer are masked.
The script is designed for a single 24 GB CUDA GPU and saves a PEFT adapter,
not a merged base model.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingSummary:
    model_name_or_path: str
    train_file: str
    validation_file: str
    output_dir: str
    quantization: str
    max_length: int
    epochs: float
    learning_rate: float
    lora_r: int
    lora_alpha: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and tokenization without loading a model.")
    return parser.parse_args()


def load_runtime():
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, transformers, datasets, peft, "
            "accelerate and bitsandbytes in the remote environment."
        ) from exc
    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def inspect_raw_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_split") != "build":
                raise ValueError(f"{path}:{number} is not Build-only")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(f"{path}:{number} must contain system, user, assistant messages")
            if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{number} has an unexpected message order")
            if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
                raise ValueError(f"{path}:{number} contains blank message content")
            # The student data builder must have removed privileged y_strong.
            combined = "\n".join(message["content"] for message in messages[:2])
            if "y_strong" in combined or "强教师的独立参考回答" in combined:
                raise ValueError(f"{path}:{number} leaks y_strong into critic input")
            count += 1
    if not count:
        raise ValueError(f"No records found in {path}")
    return count


def make_collator(torch_module: Any, pad_token_id: int):
    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
            "attention_mask": torch_module.tensor(attention_mask, dtype=torch_module.long),
            "labels": torch_module.tensor(labels, dtype=torch_module.long),
        }

    return collate


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory {args.output_dir}")
    if args.max_length < 1024:
        raise ValueError("--max-length must be at least 1024")
    train_count = inspect_raw_records(args.train_file)
    validation_count = inspect_raw_records(args.validation_file)
    summary = TrainingSummary(
        model_name_or_path=args.model_name_or_path,
        train_file=str(args.train_file),
        validation_file=str(args.validation_file),
        output_dir=str(args.output_dir),
        quantization=args.quantization,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "train_records": train_count, "validation_records": validation_count, **asdict(summary)}, ensure_ascii=False))
        return

    runtime = load_runtime()
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA training entry point")
    runtime["set_seed"](args.seed)
    random.seed(args.seed)

    raw = runtime["load_dataset"](
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )
    tokenizer = runtime["AutoTokenizer"].from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dropped_overlength = {"train": 0, "validation": 0}

    def tokenize_record(record: dict[str, Any]) -> dict[str, list[int]]:
        messages = record["messages"]
        prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        assistant_ids = tokenizer(messages[2]["content"], add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            assistant_ids.append(tokenizer.eos_token_id)
        if not assistant_ids:
            raise ValueError("No assistant supervision tokens found")
        full_ids = prompt_ids + assistant_ids
        if len(full_ids) > args.max_length:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        labels = [-100] * len(prompt_ids) + assistant_ids
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    tokenized = {}
    for split in ("train", "validation"):
        mapped = raw[split].map(tokenize_record, remove_columns=raw[split].column_names)
        before = len(mapped)
        mapped = mapped.filter(lambda row: len(row["input_ids"]) > 0)
        dropped_overlength[split] = before - len(mapped)
        if not len(mapped):
            raise RuntimeError(f"All {split} records exceed max length {args.max_length}")
        tokenized[split] = mapped

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
        "device_map": "auto",
    }
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = runtime["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = runtime["AutoModelForCausalLM"].from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if args.quantization == "4bit":
        model = runtime["prepare_model_for_kbit_training"](model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = runtime["LoraConfig"](
        task_type=runtime["TaskType"].CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = runtime["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    argument_names = inspect.signature(runtime["TrainingArguments"].__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"
    training_kwargs[evaluation_key] = "steps"
    training_kwargs["save_strategy"] = "steps"
    training_args = runtime["TrainingArguments"](**training_kwargs)
    trainer = runtime["Trainer"](
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=make_collator(torch, tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    metrics.update({f"eval_{key}": value for key, value in trainer.evaluate().items()})
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    manifest = {
        **asdict(summary),
        "train_records_raw": train_count,
        "validation_records_raw": validation_count,
        "train_records_used": len(tokenized["train"]),
        "validation_records_used": len(tokenized["validation"]),
        "dropped_overlength": dropped_overlength,
        "cuda_device": torch.cuda.get_device_name(0),
        "metrics": metrics,
        "adapter_only": True,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
