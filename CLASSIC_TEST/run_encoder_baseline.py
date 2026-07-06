#!/usr/bin/env python3
"""Fine-tuned encoder (BERT / DistilBERT / RoBERTa) baselines for CFPB classification.

Uses the same leakage-controlled split, 10,000-example training budget,
500-character narrative truncation, and evaluation sets as the QLoRA experiments.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "retail_banking", "credit_reporting", "credit_card",
    "mortgages_and_loans", "debt_collection",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class Config:
    model_name: str
    short_name: str
    max_train: int = 10_000
    max_chars: int = 500
    max_length: int = 512
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-5
    epochs: int = 2
    seed: int = 42


MODELS = [
    Config("bert-base-uncased",       "BERT-base"),
    Config("distilbert-base-uncased", "DistilBERT"),
    Config("roberta-base",            "RoBERTa-base"),
]


class TextDataset(Dataset):
    """Tokenises on-the-fly to avoid loading all tensors into RAM at once."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx])


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    return {
        "accuracy":    float(accuracy_score(labels_arr, preds_arr)),
        "macro_f1":    float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)),
        "per_class_f1": {
            LABELS[i]: float(f)
            for i, f in enumerate(
                f1_score(labels_arr, preds_arr, average=None, zero_division=0)
            )
        },
    }


def run_one(cfg: Config, train_df: pd.DataFrame, test_df: pd.DataFrame,
            legacy_dfs: dict[str, pd.DataFrame], out_dir: Path):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Model: {cfg.short_name}  device={device}")
    print(
        f"Config: max_train={cfg.max_train} max_chars={cfg.max_chars} "
        f"max_length={cfg.max_length} epochs={cfg.epochs} "
        f"batch_size={cfg.batch_size} grad_accum={cfg.grad_accum} "
        f"effective_batch={cfg.batch_size * cfg.grad_accum} "
        f"lr={cfg.lr} seed={cfg.seed}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    def make_loader(df: pd.DataFrame, shuffle: bool = False) -> DataLoader:
        texts  = df["narrative"].str[:cfg.max_chars].tolist()
        labels = [LABEL2ID[l] for l in df["true_label"]]
        ds = TextDataset(texts, labels, tokenizer, cfg.max_length)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_df, shuffle=True)
    test_loader  = make_loader(test_df)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = int(np.ceil(len(train_loader) / cfg.grad_accum)) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # Training
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (batch, labels) in enumerate(train_loader, start=1):
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            with autocast("cuda", enabled=use_amp):
                raw_loss = model(**batch, labels=labels).loss
                loss = raw_loss / cfg.grad_accum
            scaler.scale(loss).backward()
            if step % cfg.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += raw_loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}")

    # Internal test evaluation
    int_metrics = evaluate(model, test_loader, device)
    print(f"  Internal test: acc={int_metrics['accuracy']:.4f}  "
          f"macro-F1={int_metrics['macro_f1']:.4f}  "
          f"wF1={int_metrics['weighted_f1']:.4f}")

    # Legacy evaluation
    legacy_results = {}
    for subset_name, leg_df in legacy_dfs.items():
        leg_loader = make_loader(leg_df)
        m = evaluate(model, leg_loader, device)
        legacy_results[subset_name] = m

    legacy_macro_f1s = [v["macro_f1"] for v in legacy_results.values()]
    legacy_wf1s      = [v["weighted_f1"] for v in legacy_results.values()]
    print(f"  Legacy mean macro-F1={np.mean(legacy_macro_f1s):.4f}  "
          f"mean wF1={np.mean(legacy_wf1s):.4f}")

    # Free model memory before next run
    del model, optimizer, scheduler, train_loader, test_loader
    gc.collect()

    # Save results
    result = {
        "model": cfg.short_name,
        "model_name": cfg.model_name,
        "max_train": cfg.max_train,
        "max_chars": cfg.max_chars,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "internal_test": int_metrics,
        "legacy_mean_macro_f1": float(np.mean(legacy_macro_f1s)),
        "legacy_mean_weighted_f1": float(np.mean(legacy_wf1s)),
        "legacy_subsets": legacy_results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_{cfg.short_name.replace('-','_').replace('/','_')}.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file",    type=Path,
                        default=ROOT / "dataset/splits/train.csv")
    parser.add_argument("--test-file",     type=Path,
                        default=ROOT / "dataset/splits/internal_test.csv")
    parser.add_argument("--legacy-dir",    type=Path,
                        default=ROOT / "dataset/legacy_subsets")
    parser.add_argument("--out-dir",       type=Path,
                        default=Path(__file__).parent / "outputs/encoder_baselines_10k_len512")
    parser.add_argument("--max-train",     type=int, default=10_000)
    parser.add_argument("--max-chars",     type=int, default=500)
    parser.add_argument("--max-length",    type=int, default=512)
    parser.add_argument("--batch-size",    type=int, default=4)
    parser.add_argument("--grad-accum",    type=int, default=4)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--epochs",        type=int, default=2)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--models",        nargs="+",
                        default=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"])
    args = parser.parse_args()

    # Load data  (splits use columns: record_id, product, narrative, narrative_hash)
    print("Loading data...")
    train_df = pd.read_csv(args.train_file).rename(columns={"product": "true_label"})
    test_df  = pd.read_csv(args.test_file).rename(columns={"product": "true_label"})

    # Natural 10k sample
    train_df = train_df.dropna(subset=["true_label", "narrative"])
    train_df = train_df[train_df.true_label.isin(LABELS)]
    train_df = train_df.sample(n=min(args.max_train, len(train_df)), random_state=42)
    print(f"Train: {len(train_df)} rows")

    test_df = test_df.dropna(subset=["true_label", "narrative"])
    test_df = test_df[test_df.true_label.isin(LABELS)]
    print(f"Test:  {len(test_df)} rows")

    # Legacy subsets
    legacy_dfs = {}
    for f in sorted(args.legacy_dir.glob("*.csv")):
        df = pd.read_csv(f)
        legacy_dfs[f.stem] = df

    print(f"Legacy subsets: {len(legacy_dfs)}")

    all_results = []
    model_configs = [c for c in MODELS if c.model_name in args.models]

    for cfg in model_configs:
        cfg.max_train = args.max_train
        cfg.max_chars = args.max_chars
        cfg.max_length = args.max_length
        cfg.batch_size = args.batch_size
        cfg.grad_accum = args.grad_accum
        cfg.lr = args.lr
        cfg.epochs = args.epochs
        cfg.seed = args.seed
        out_dir = args.out_dir
        res = run_one(cfg, train_df, test_df, legacy_dfs, out_dir)
        all_results.append(res)

    # Summary CSV
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "model": r["model"],
            "max_train": r["max_train"],
            "max_chars": r["max_chars"],
            "max_length": r["max_length"],
            "batch_size": r["batch_size"],
            "grad_accum": r["grad_accum"],
            "effective_batch_size": r["effective_batch_size"],
            "epochs": r["epochs"],
            "lr": r["lr"],
            "seed": r["seed"],
            "internal_accuracy":    r["internal_test"]["accuracy"],
            "internal_macro_f1":    r["internal_test"]["macro_f1"],
            "internal_weighted_f1": r["internal_test"]["weighted_f1"],
            "legacy_macro_f1_mean": r["legacy_mean_macro_f1"],
            "legacy_weighted_f1_mean": r["legacy_mean_weighted_f1"],
            "debt_f1": r["internal_test"]["per_class_f1"].get("debt_collection", 0),
        })
    summary_df = pd.DataFrame(summary_rows)
    out_path = args.out_dir / "encoder_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
