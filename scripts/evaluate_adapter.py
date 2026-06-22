#!/usr/bin/env python3
"""Generate CFPB labels with a QLoRA adapter and calculate metrics."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
PRODUCT_RE = re.compile(r'"product"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs/qwen3_8b_qlora_pilot/best_adapter",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=PROJECT_ROOT / "dataset/splits/validation.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qwen3_8b_qlora_pilot/validation_inference",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-narrative-chars", type=int, default=500)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-final-eval",
        action="store_true",
        help="Required for internal_test or legacy files.",
    )
    return parser.parse_args()


def protect_final_sets(path: Path, allow: bool) -> None:
    path_text = str(path.resolve())
    protected = (
        "internal_test.csv" in path_text
        or "legacy_subsets" in path_text
        or "legacy_test_manifest.csv" in path_text
    )
    if protected and not allow:
        raise RuntimeError(
            "Final/legacy evaluation is locked. Freeze the experiment first and "
            "rerun with --allow-final-eval."
        )


def parse_label(response: str) -> Optional[str]:
    matches = PRODUCT_RE.findall(response)
    if not matches:
        return None
    label = matches[-1].strip()
    return label if label in LABELS else None


def main() -> None:
    args = parse_args()
    protect_final_sets(args.data_file, args.allow_final_eval)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    frame = pd.read_csv(args.data_file)
    if args.max_samples > 0 and args.max_samples < len(frame):
        frame = frame.sample(n=args.max_samples, random_state=args.seed)
    frame = frame.reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = []
    for row in frame.itertuples(index=False):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Complaint:\n"
                    f"{str(row.narrative)[:args.max_narrative_chars]}"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to("cuda")
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        response = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        prediction = parse_label(response)
        rows.append(
            {
                "record_id": getattr(row, "record_id", None),
                "true_label": row.product,
                "prediction": prediction or "__off_label__",
                "is_off_label": prediction is None,
                "latency_seconds": latency,
                "raw_response": response,
            }
        )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    y_true = predictions["true_label"]
    y_pred = predictions["prediction"]
    metrics = {
        "n": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted")
        ),
        "weighted_precision": float(
            precision_score(
                y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
            )
        ),
        "off_label_count": int(predictions["is_off_label"].sum()),
        "mean_latency_seconds": float(predictions["latency_seconds"].mean()),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
