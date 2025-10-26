"""
Week-6 evaluation for the integrated (LangGraph-style) pipeline.
Reads reports/week5_graph_run.jsonl and computes metrics vs WELFake labels.
Also emits ROC curve, confusion matrix, confidence histogram, latency plot,
and coverage-vs-threshold curve.

Usage:
PYTHONPATH=. python scripts/evaluate_week6.py \
  --pred reports/week5_graph_run.jsonl \
  --truth_csv data/WELFake.csv \
  --out_dir reports/week6
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ----------------------------
# Helpers
# ----------------------------
_ID_NUM_RE = re.compile(r"\d+")

def _to_int_id_series(s: pd.Series) -> pd.Series:
    """
    Try hard to coerce IDs to int for joining:
    - If already numeric, astype(int)
    - If string like '123' or 'id-123', extract the first integer token
    - Otherwise, set to NaN and drop later
    """
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.astype("Int64")

    # strings or mixed
    def _coerce_one(x):
        if pd.isna(x):
            return None
        try:
            return int(x)
        except Exception:
            m = _ID_NUM_RE.search(str(x))
            return int(m.group()) if m else None

    coerced = s.map(_coerce_one)
    return pd.Series(coerced, index=s.index, dtype="Int64")

def _to_float_series(s: pd.Series) -> pd.Series:
    try:
        return s.astype(float)
    except Exception:
        return pd.to_numeric(s, errors="coerce")

def _verdict_to_int(s: pd.Series) -> pd.Series:
    """
    Map verdict to {0,1}. Accepts:
      - integers 0/1
      - booleans
      - strings: 'real','true','yes','1' -> 1; 'fake','false','no','0' -> 0
    Unknowns -> NaN (dropped).
    """
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.astype("Int64")

    def _map_one(x):
        if pd.isna(x):
            return None
        xs = str(x).strip().lower()
        if xs in {"1","true","yes","y","real","reliable","genuine"}:
            return 1
        if xs in {"0","false","no","n","fake","unreliable","bogus"}:
            return 0
        # sometimes people encode 'REAL'/'FAKE' titlecase:
        if xs.startswith("real"):
            return 1
        if xs.startswith("fake"):
            return 0
        # try numeric fallback
        try:
            v = int(float(xs))
            return 1 if v >= 1 else 0
        except Exception:
            return None

    mapped = s.map(_map_one)
    return pd.Series(mapped, index=s.index, dtype="Int64")

# ----------------------------
# IO
# ----------------------------
def load_preds(pred_path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(pred_path, "r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    keep = ["id","claim","verdict","confidence","evidence_internal","trace"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    # normalize types
    df["id"]        = _to_int_id_series(df["id"])
    df["verdict"]   = _verdict_to_int(df["verdict"])
    df["confidence"]= _to_float_series(df["confidence"])
    return df[keep]

def load_truth(truth_csv: str) -> pd.DataFrame:
    df = pd.read_csv(truth_csv)
    if "id" not in df.columns:
        # create 0..N-1 ids that match your Week-5 generator
        df = df.reset_index(drop=True).reset_index(names="id")
    out = df[["id","label"]].copy()
    out["id"]    = _to_int_id_series(out["id"])
    out["label"] = _verdict_to_int(out["label"])
    return out

# ----------------------------
# Metrics / Plots
# ----------------------------
def metrics_dict(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan")
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auroc=auroc)

def plot_roc(y_true, y_prob, out_png: Path, label_txt="Pipeline"):
    from sklearn.metrics import RocCurveDisplay
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=label_txt)
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_confusion(y_true, y_pred, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    plt.figure()
    plt.imshow(cmn, interpolation="nearest")
    for (i,j), v in np.ndenumerate(cmn):
        plt.text(j, i, f"{v:.2f}", ha="center", va="center")
    plt.xticks([0,1], ["Fake","Real"])
    plt.yticks([0,1], ["Fake","Real"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_hist_confidence(y_prob, out_png: Path):
    plt.figure()
    plt.hist(y_prob, bins=10)
    plt.xlabel("Confidence (p)")
    plt.ylabel("Count")
    plt.title("Confidence Histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_latency(ms_list, out_png: Path):
    plt.figure()
    plt.boxplot(ms_list, vert=True)
    plt.ylabel("Latency (ms)")
    plt.title("End-to-End Latency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_coverage_curve(y_true, y_prob, y_pred, out_png: Path):
    """
    Coverage vs threshold: fraction of predictions with confidence >= t,
    and accuracy among the covered subset.
    """
    probs = np.asarray(y_prob)
    preds = np.asarray(y_pred)
    y     = np.asarray(y_true)
    thresholds = np.linspace(0.5, 0.95, 10)

    cov, acc_cov = [], []
    for t in thresholds:
        mask = probs >= t
        cov.append(mask.mean())
        if mask.any():
            acc_cov.append(accuracy_score(y[mask], preds[mask]))
        else:
            acc_cov.append(np.nan)

    plt.figure()
    plt.plot(thresholds, cov, marker="o", label="Coverage (≥ t)")
    plt.plot(thresholds, acc_cov, marker="s", label="Accuracy among covered")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Fraction / Accuracy")
    plt.title("Coverage vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth_csv", required=True)
    ap.add_argument("--out_dir", default="reports/week6")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pred  = load_preds(args.pred)
    df_truth = load_truth(args.truth_csv)

    # drop rows with missing critical fields after normalization
    df_pred = df_pred.dropna(subset=["id","verdict","confidence"]).copy()
    df_truth = df_truth.dropna(subset=["id","label"]).copy()

    merged = pd.merge(df_pred, df_truth, on="id", how="inner")

    if merged.empty:
        # Give user some hints before erroring out
        print("ERROR: No overlap on 'id' between predictions and truth after normalization.", file=sys.stderr)
        print(f"pred unique ids (sample 10): {list(df_pred['id'].dropna().unique()[:10])}", file=sys.stderr)
        print(f"truth unique ids (sample 10): {list(df_truth['id'].dropna().unique()[:10])}", file=sys.stderr)
        raise RuntimeError("No overlap on 'id' between predictions and truth.")

    overlap_ratio = len(merged) / min(len(df_pred), len(df_truth))
    if overlap_ratio < 0.5:
        print(f"Warning: low ID overlap ({overlap_ratio:.2%}). Check ID schemes.", file=sys.stderr)

    y_true = merged["label"].astype(int).to_numpy()
    y_pred = merged["verdict"].astype(int).to_numpy()
    y_prob = merged["confidence"].astype(float).to_numpy()

    # main metrics
    m = metrics_dict(y_true, y_pred, y_prob)

    # latency (if available)
    latency_ms = []
    for tr in merged["trace"]:
        try:
            t = tr or {}
            latency_ms.append(int(t.get("retrieve_ms",0)) + int(t.get("verify_ms",0)))
        except Exception:
            pass

    # write metrics json
    out_json = out_dir / "final_metrics.json"
    payload = dict(
        n=len(y_true),
        **m,
        latency_ms_summary=dict(
            count=len(latency_ms),
            mean=float(np.mean(latency_ms)) if latency_ms else None,
            median=float(np.median(latency_ms)) if latency_ms else None,
            p90=float(np.percentile(latency_ms, 90)) if latency_ms else None,
        )
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))

    # plots
    plot_roc(y_true, y_prob, out_dir / "roc.png", label_txt="Pipeline")
    plot_confusion(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_hist_confidence(y_prob, out_dir / "confidence_hist.png")
    if latency_ms:
        plot_latency(latency_ms, out_dir / "latency_box.png")
    plot_coverage_curve(y_true, y_prob, y_pred, out_dir / "coverage_vs_threshold.png")

if __name__ == "__main__":
    main()
