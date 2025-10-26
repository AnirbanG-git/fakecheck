"""
Train a classical verifier on [claim + evidence] strings.

Model:
  - TF-IDF (1–2 grams, max_features=200k, min_df=2)
  - Logistic Regression (class_weight='balanced', C=2.0, max_iter=2000)
  - Optional probability calibration on validation (isotonic if enough data)
  - Threshold tuned on validation to maximize F1

Inputs (JSONL):
  {"id": "...", "input_text": "...", "label": 0/1, "split": "train"|"validation"|"test"}

Outputs:
  - artifacts/verifier_lr/verifier_lr.joblib              (sklearn pipeline)
  - artifacts/verifier_lr/preds_verifier_lr.jsonl         (id, pred, proba) on test
  - reports/metrics_verifier_lr.json                      (metrics incl. tuned threshold)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    # Maximize F1 over a dense grid
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
    ap.add_argument("--dataset", required=True, help="Path to reports/verifier_dataset.jsonl")
    ap.add_argument("--out_dir", required=True, help="Where to save the model & preds")
    ap.add_argument("--report", required=True, help="Where to write metrics JSON")
    args = ap.parse_args()

    df = _load_jsonl(Path(args.dataset))
    assert {"input_text", "label", "split"}.issubset(df.columns), \
        "dataset must have input_text, label, split"

    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df   = df[df.split == "validation"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)

    X_tr, y_tr = train_df.input_text.tolist(), train_df.label.astype(int).to_numpy()
    X_va, y_va = val_df.input_text.tolist(),   val_df.label.astype(int).to_numpy()
    X_te, y_te = test_df.input_text.tolist(),  test_df.label.astype(int).to_numpy()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=200_000,
            min_df=2,
            dtype=np.float32
        )),
        ("clf", LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    pipe.fit(X_tr, y_tr)

    # Probability calibration on validation if there are enough samples
    try:
        if len(y_va) >= 30:
            calib = CalibratedClassifierCV(pipe.named_steps["clf"], method="isotonic", cv="prefit")
            clf_cal = calib.fit(pipe.named_steps["tfidf"].transform(X_va), y_va)
            pipe = Pipeline([("tfidf", pipe.named_steps["tfidf"]), ("clf", clf_cal)])
    except Exception:
        # Fall back to uncalibrated probabilities if isotonic fails
        pass

    # Tune decision threshold on validation to maximize F1
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba_va = pipe.predict_proba(X_va)[:, 1]
    else:
        proba_va = pipe.decision_function(X_va)
    t_best = _best_threshold(y_va, proba_va)

    # Test evaluation
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba_te = pipe.predict_proba(X_te)[:, 1]
    else:
        proba_te = pipe.decision_function(X_te)
    preds_te = (proba_te >= t_best).astype(int)

    def _safe_auroc(y_true, p):
        return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_te, preds_te)),
        "precision": float(precision_score(y_te, preds_te, zero_division=0)),
        "recall": float(recall_score(y_te, preds_te, zero_division=0)),
        "f1": float(f1_score(y_te, preds_te, zero_division=0)),
        "auroc": float(_safe_auroc(y_te, proba_te)),
        "n_test": int(len(y_te)),
        "threshold": float(t_best),
        "confusion_matrix": confusion_matrix(y_te, preds_te).tolist()
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "verifier_lr.joblib"
    joblib.dump(pipe, model_path)

    preds_path = out_dir / "preds_verifier_lr.jsonl"
    with preds_path.open("w") as f:
        for _id, p, pr in zip(test_df["id"].tolist(), preds_te.tolist(), proba_te.tolist()):
            f.write(json.dumps({"id": _id, "pred": int(p), "proba": float(pr)}) + "\n")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Wrote preds to {preds_path}")


if __name__ == "__main__":
    main()
