"""
Fine-tune a transformer verifier on [claim + evidence].

- Default model: distilbert-base-uncased
- max_length=512 (claim kept at the start; tokenizer handles truncation)
- Early stopping (patience=2), weight decay
- FP16 on CUDA automatically
- Threshold tuned on validation (max F1), then evaluated on test

Inputs (JSONL):
  {"id": "...", "input_text": "...", "label": 0/1, "split": "train"|"validation"|"test"}

Outputs:
  - artifacts/verifier_bert/                 (HF model)
  - artifacts/verifier_bert/preds_verifier_bert.jsonl
  - reports/metrics_verifier_bert.json       (eval_* and tuned test metrics)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
    EarlyStoppingCallback
)


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _safe_auroc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else 0.0


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    y = labels
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auroc": float(_safe_auroc(y, probs)),
    }


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return float(best_t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--tune_threshold", action="store_true")
    args = ap.parse_args()

    # Load dataset
    df = _load_jsonl(Path(args.dataset))
    assert {"id", "input_text", "label", "split"}.issubset(df.columns)

    ds = DatasetDict({
        "train": Dataset.from_pandas(df[df.split == "train"][["input_text", "label"]].reset_index(drop=True)),
        "validation": Dataset.from_pandas(df[df.split == "validation"][["input_text", "label"]].reset_index(drop=True)),
        "test": Dataset.from_pandas(df[df.split == "test"][["input_text", "label"]].reset_index(drop=True)),
    })

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok(batch):
        # We keep claim at the start (already in input_text); tokenizer truncates the tail (evidence) first.
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    ds = ds.map(tok, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(Path(args.out_dir) / "hf_runs"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    # Threshold tuning on validation
    val_out = trainer.predict(ds["validation"])
    test_out = trainer.predict(ds["test"])

    val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1)[:, 1].numpy()
    val_labels = val_out.label_ids
    if args.tune_threshold and len(val_labels) > 0:
        t_best = _best_threshold(val_labels, val_probs)
    else:
        t_best = 0.5

    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1)[:, 1].numpy()
    test_labels = test_out.label_ids
    test_preds = (test_probs >= t_best).astype(int)

    rep_tuned: Dict[str, Any] = {
        "accuracy": float(accuracy_score(test_labels, test_preds)),
        "precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "auroc": float(_safe_auroc(test_labels, test_probs)),
        "threshold": float(t_best),
        "n_test": int(len(test_labels)),
    }

    # Save model & preds
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)

    preds_path = out_dir / "preds_verifier_bert.jsonl"
    test_ids = df[df.split == "test"]["id"].tolist()
    with preds_path.open("w") as f:
        for _id, pr in zip(test_ids, test_probs.tolist()):
            f.write(json.dumps({"id": _id, "proba": float(pr), "pred": int(pr >= t_best)}) + "\n")

    # Write metrics report
    report = {
        "eval_loss": float(test_out.metrics.get("test_loss", 0.0)),
        "eval_accuracy": float(test_out.metrics.get("test_accuracy", 0.0)),
        "eval_precision": float(test_out.metrics.get("test_precision", 0.0)),
        "eval_recall": float(test_out.metrics.get("test_recall", 0.0)),
        "eval_f1": float(test_out.metrics.get("test_f1", 0.0)),
        "eval_auroc": float(test_out.metrics.get("test_auroc", 0.0)),
        "tuned": rep_tuned,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved model to {out_dir}")
    print(f"Wrote preds to {preds_path}")


if __name__ == "__main__":
    main()
