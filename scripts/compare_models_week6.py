#!/usr/bin/env python3
"""
Robust comparison on the *same claims* in pipeline_pred:

Models:
1) Baseline LR (TF-IDF + LogisticRegression) on claim text only
2) Standalone BERT (fine-tuned DistilBERT) on claim text only
3) Week-5 Pipeline (predictions read from week5_graph_run.jsonl)

Alignment strategy for ground-truth labels:
- Prefer reports/verifier_dataset.jsonl (by id; fallback by exact claim text)
- Fallback to data/WELFake.csv only if IDs overlap (rare for claim-level tasks)

Usage:
PYTHONPATH=. python scripts/compare_models_week6.py \
  --pipeline_pred reports/week5_graph_run.jsonl \
  --labels_jsonl reports/verifier_dataset.jsonl \
  --truth_csv data/WELFake.csv \
  --lr_artifact artifacts/verifier_lr/baseline_tfidf_lr.joblib \
  --bert_dir artifacts/verifier_bert \
  --out_dir reports/week6
"""
import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def _to_str_id(x):
    try:
        return str(int(x))
    except Exception:
        return str(x)


def load_pipeline_preds(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    need = {"id","claim","verdict","confidence"}
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Pipeline predictions missing column: {c}")
    # normalize id as str for safer joins
    df["id_str"] = df["id"].apply(_to_str_id)
    df["claim"] = df["claim"].astype(str)
    return df[["id","id_str","claim","verdict","confidence"]]


def load_labels_jsonl(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if "id" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"{path} must contain 'id' and 'label' fields.")
    df["id_str"] = df["id"].apply(_to_str_id)
    # prefer explicit 'claim' but fall back to 'text'
    claim_col = "claim" if "claim" in df.columns else ("text" if "text" in df.columns else None)
    if claim_col:
        df["claim_norm"] = df[claim_col].astype(str).str.strip()
    else:
        df["claim_norm"] = ""
    return df[["id","id_str","claim_norm","label"]]


def load_truth_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"{path} must contain 'id' and 'label' columns.")
    df["id_str"] = df["id"].apply(_to_str_id)
    return df[["id","id_str","label"]]


def evaluate(y_true, y_pred, y_prob):
    return dict(
        accuracy = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall = recall_score(y_true, y_pred, zero_division=0),
        f1 = f1_score(y_true, y_pred, zero_division=0),
        auroc = roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan"),
        n = int(len(y_true))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_pred", required=True, help="week5_graph_run.jsonl (id, claim, verdict, confidence)")
    ap.add_argument("--labels_jsonl", default="reports/verifier_dataset.jsonl",
                    help="JSONL with (id, label, claim/text). Preferred source for ground truth.")
    ap.add_argument("--truth_csv", default=None,
                    help="Optional: data/WELFake.csv (id,label). Used only if ids overlap (fallback).")
    ap.add_argument("--lr_artifact", default="artifacts/verifier_lr/baseline_tfidf_lr.joblib")
    ap.add_argument("--bert_dir", default="artifacts/verifier_bert")
    ap.add_argument("--out_dir", default="reports/week6")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- load pipeline predictions (defines evaluation set + claim texts) ---
    pipe = load_pipeline_preds(args.pipeline_pred)

    # --- build ground-truth mapping ---
    y_true_map = {}

    # 1) Prefer labels from JSONL (id match)
    used_jsonl = False
    if args.labels_jsonl and Path(args.labels_jsonl).exists():
        lab = load_labels_jsonl(args.labels_jsonl)
        # try id match
        merged = pipe.merge(lab[["id_str","label"]], on="id_str", how="left")
        id_hits = merged["label"].notna().sum()
        if id_hits > 0:
            used_jsonl = True
            for r in merged.itertuples():
                if pd.notna(r.label):
                    y_true_map[r.id_str] = int(r.label)
        else:
            # fallback: exact claim text match
            lab_claim_map = lab.set_index("claim_norm")["label"].to_dict()
            claim_hits = 0
            for r in pipe.itertuples():
                lbl = lab_claim_map.get(str(r.claim).strip())
                if lbl is not None:
                    y_true_map[r.id_str] = int(lbl); claim_hits += 1
            used_jsonl = claim_hits > 0

    # 2) If still empty and a CSV truth is provided, try id overlap with CSV
    if not y_true_map and args.truth_csv and Path(args.truth_csv).exists():
        csvtruth = load_truth_csv(args.truth_csv)
        csv_map = csvtruth.set_index("id_str")["label"].astype(int).to_dict()
        overlap = set(pipe["id_str"]) & set(csv_map.keys())
        if overlap:
            for sid in overlap:
                y_true_map[sid] = int(csv_map[sid])

    # Final check
    ids_eval = [sid for sid in pipe["id_str"].tolist() if sid in y_true_map]
    if not ids_eval:
        # Diagnostics
        sample_pipe_ids = list(pipe["id_str"].unique())[:5]
        msg = [
            "No overlapping IDs or claim-texts between predictions and labels.",
            f"- pipeline_pred: {args.pipeline_pred}",
            f"- labels_jsonl: {args.labels_jsonl} ({'found' if Path(args.labels_jsonl).exists() else 'missing'})",
            f"- truth_csv: {args.truth_csv} ({'found' if (args.truth_csv and Path(args.truth_csv).exists()) else 'missing'})",
            f"- sample pipeline id_str values: {sample_pipe_ids}",
            "If your pipeline had id=None, make sure reports/verifier_dataset.jsonl is the same source of claims used to generate week5_graph_run.jsonl.",
        ]
        raise RuntimeError("\n".join(msg))

    # slice to aligned set, and build arrays
    pipe_aligned = pipe[pipe["id_str"].isin(ids_eval)].copy()
    y_true = np.array([y_true_map[sid] for sid in pipe_aligned["id_str"]], dtype=int)

    # Pipeline metrics
    y_pred_pipe = pipe_aligned["verdict"].astype(int).to_numpy()
    y_prob_pipe = pipe_aligned["confidence"].astype(float).to_numpy()
    met_pipe = evaluate(y_true, y_pred_pipe, y_prob_pipe)

    # Baseline LR (TF-IDF + LR) on claim-only inputs
    lr_bundle = joblib.load(args.lr_artifact)
    if hasattr(lr_bundle, "predict_proba"):
        lr_pipeline = lr_bundle
        def lr_predict_proba(texts): return lr_pipeline.predict_proba(texts)[:,1]
    else:
        vec = lr_bundle["vectorizer"]; model = lr_bundle["model"]
        def lr_predict_proba(texts): return model.predict_proba(vec.transform(texts))[:,1]
    texts = pipe_aligned["claim"].astype(str).tolist()
    y_prob_lr = lr_predict_proba(texts)
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    met_lr = evaluate(y_true, y_pred_lr, y_prob_lr)

    # Standalone BERT on claim-only inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.bert_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.bert_dir).to(device).eval()
    pairs = [f"[CLAIM] {t}" for t in texts]
    y_prob_bert_chunks = []
    with torch.no_grad():
        bs = 32
        for i in range(0, len(pairs), bs):
            batch = tok(pairs[i:i+bs], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            logits = mdl(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
            y_prob_bert_chunks.append(probs)
    y_prob_bert = np.concatenate(y_prob_bert_chunks) if y_prob_bert_chunks else np.array([])
    y_pred_bert = (y_prob_bert >= 0.5).astype(int)
    met_bert = evaluate(y_true, y_pred_bert, y_prob_bert)

    # Save table + plot
    table = pd.DataFrame([
        dict(model="Baseline_LR", **met_lr),
        dict(model="Standalone_BERT", **met_bert),
        dict(model="Pipeline", **met_pipe),
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "model_comparison.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))

    # Bar chart (F1 & AUROC)
    plt.figure()
    xs = np.arange(len(table))
    plt.bar(xs-0.15, table["f1"], width=0.3, label="F1")
    plt.bar(xs+0.15, table["auroc"], width=0.3, label="AUROC")
    plt.xticks(xs, table["model"])
    title_src = "labels_jsonl" if (args.labels_jsonl and Path(args.labels_jsonl).exists()) else "truth_csv"
    plt.title(f"Model Comparison (F1 & AUROC) – aligned via {title_src}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close()

    # Also dump a quick alignment manifest for transparency
    manifest = {
        "used_labels_jsonl": bool(used_jsonl if 'used_jsonl' in locals() else False),
        "aligned_count": int(len(ids_eval)),
        "total_pipeline": int(len(pipe)),
        "out_dir": str(out_dir),
        "notes": "Alignment prefers verifier_dataset.jsonl by id; falls back to claim-text; then optional CSV by id."
    }
    with open(out_dir / "alignment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
