#!/usr/bin/env python3
"""QLoRA supervised fine-tuning for CFPB complaint classification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]
SYSTEM_PROMPT = (
    "You are an AI assistant specializing in consumer complaint classification. "
    "Classify each complaint into exactly one of these categories: "
    f"{LABELS}. Return only valid JSON in this format: "
    '{"product": "chosen_category_name"}. Do not provide explanations.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--train-file", type=Path, default=PROJECT_ROOT / "dataset/splits/train.csv"
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=PROJECT_ROOT / "dataset/splits/validation.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qwen3_8b_qlora_pilot",
    )
    parser.add_argument("--max-train-samples", type=int, default=10_000)
    parser.add_argument("--max-eval-samples", type=int, default=2_000)
    parser.add_argument(
        "--train-sampling",
        choices=["natural", "balanced"],
        default="natural",
        help="Sampling distribution used only for the training subset.",
    )
    parser.add_argument("--max-narrative-chars", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def natural_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    fractions = frame["product"].value_counts(normalize=True)
    allocations = (fractions * limit).round().astype(int)
    while allocations.sum() > limit:
        allocations.loc[allocations.idxmax()] -= 1
    while allocations.sum() < limit:
        allocations.loc[fractions.idxmax()] += 1
    pieces = []
    for label, count in allocations.items():
        group = frame.loc[frame["product"].eq(label)]
        pieces.append(group.sample(n=min(count, len(group)), random_state=seed))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def balanced_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        per_class = min(frame["product"].value_counts())
        limit = per_class * len(LABELS)
    base, remainder = divmod(limit, len(LABELS))
    allocations = {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(LABELS)
    }
    pieces = []
    for index, label in enumerate(LABELS):
        group = frame.loc[frame["product"].eq(label)]
        count = allocations[label]
        if count > len(group):
            raise ValueError(
                f"Balanced sampling requests {count} rows for {label}, "
                f"but only {len(group)} are available"
            )
        pieces.append(group.sample(n=count, random_state=seed + index))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def load_frame(
    path: Path, limit: int, seed: int, sampling: str = "natural"
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"product", "narrative"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} missing columns: {required - set(frame.columns)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing labels or narratives")
    unknown = sorted(set(frame["product"]) - set(LABELS))
    if unknown:
        raise ValueError(f"{path} contains unknown labels: {unknown}")
    if sampling == "balanced":
        return balanced_sample(frame, limit, seed)
    return natural_sample(frame, limit, seed)


def to_prompt_completion(
    frame: pd.DataFrame, tokenizer: AutoTokenizer, max_narrative_chars: int
) -> Dataset:
    records = []
    for row in frame.itertuples(index=False):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Complaint:\n"
                        f"{str(row.narrative)[:max_narrative_chars]}"
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        records.append(
            {
                "prompt": prompt,
                "completion": json.dumps(
                    {"product": row.product}, ensure_ascii=False
                )
                + tokenizer.eos_token,
            }
        )
    return Dataset.from_list(records)


def write_selection_manifest(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in ["record_id", "product", "narrative_hash"]
        if column in frame.columns
    ]
    frame[columns].to_csv(path, index=False)


def write_run_metadata(
    args: argparse.Namespace, train_frame: pd.DataFrame, eval_frame: pd.DataFrame
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pip_freeze = subprocess.check_output(
            [os.sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines()
    except Exception:
        pip_freeze = []
    metadata = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_rows": len(train_frame),
        "validation_rows": len(eval_frame),
        "train_class_counts": {
            label: int(count)
            for label, count in Counter(train_frame["product"]).items()
        },
        "validation_class_counts": {
            label: int(count)
            for label, count in Counter(eval_frame["product"]).items()
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pip_freeze": pip_freeze,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_selection_manifest(
        train_frame, args.output_dir / "train_selection_manifest.csv"
    )
    write_selection_manifest(
        eval_frame, args.output_dir / "validation_selection_manifest.csv"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this QLoRA training script")
    set_seed(args.seed)

    train_frame = load_frame(
        args.train_file.resolve(),
        args.max_train_samples,
        args.seed,
        sampling=args.train_sampling,
    )
    eval_frame = load_frame(
        args.validation_file.resolve(),
        args.max_eval_samples,
        args.seed,
        sampling="natural",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = to_prompt_completion(
        train_frame, tokenizer, args.max_narrative_chars
    )
    eval_dataset = to_prompt_completion(
        eval_frame, tokenizer, args.max_narrative_chars
    )
    write_run_metadata(args, train_frame, eval_frame)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        bf16=False,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=args.seed,
        data_seed=args.seed,
        dataset_num_proc=4,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "best_adapter"))
    tokenizer.save_pretrained(args.output_dir / "best_adapter")
    trainer.save_metrics("train", train_result.metrics)
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
