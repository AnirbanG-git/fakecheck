#!/usr/bin/env python3
"""
Robust error analysis for Week 6.

Aligns predictions with ground truth using:
1) reports/verifier_dataset.jsonl (by id_str; fallback by exact claim text)
2) fallback to data/WELFake.csv (by id_str) if needed

Prints up to K misclassifications with top internal evidence.

Usage:
PYTHONPATH=. python scripts/error_analysis_week6.py \
  --pred reports/week5_graph_run.jsonl \
  --truth_csv data/WELFake.csv \
  --labels_jsonl reports/verifier_dataset.jsonl \
  --k 15
"""
import argparse, json, random
from textwrap import shorten
from pathlib import Path
import pandas as pd

def _to_str_id(x):
    try:
        return str(int(x))
    except Exception:
        return str(x)

def load_preds(pred_path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(pred_path, "r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    for c in ["id","claim","verdict","confidence"]:
        if c not in df.columns:
            df[c] = None
    df["id_str"] = df["id"].apply(_to_str_id)
    df["claim_norm"] = df["claim"].astype(str).str.strip()
    return df[["id","id_str","claim_norm","verdict","confidence","evidence_internal"]]

def load_labels_jsonl(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if "id" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"{path} must contain 'id' and 'label'.")
    df["id_str"] = df["id"].apply(_to_str_id)
    claim_col = "claim" if "claim" in df.columns else ("text" if "text" in df.columns else None)
    df["claim_norm"] = df[claim_col].astype(str).str.strip() if claim_col else ""
    return df[["id_str","claim_norm","label"]]

def load_truth_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"{path} must contain 'id' and 'label' columns.")
    df["id_str"] = df["id"].apply(_to_str_id)
    return df[["id_str","label"]]

def align_truth(preds: pd.DataFrame, labels_jsonl: str | None, truth_csv: str | None) -> pd.DataFrame:
    # 1) Prefer verifier_dataset.jsonl (id_str match)
    if labels_jsonl and Path(labels_jsonl).exists():
        lab = load_labels_jsonl(labels_jsonl)
        merged = preds.merge(lab[["id_str","label"]], on="id_str", how="left")
        if merged["label"].notna().any():
            return merged

        # Fallback: exact claim text match
        merged2 = preds.merge(lab[["claim_norm","label"]].drop_duplicates(), on="claim_norm", how="left")
        if merged2["label"].notna().any():
            return merged2

    # 2) Fallback to CSV (id_str)
    if truth_csv and Path(truth_csv).exists():
        truth = load_truth_csv(truth_csv)
        merged3 = preds.merge(truth, on="id_str", how="left")
        if merged3["label"].notna().any():
            return merged3

    raise RuntimeError(
        "Could not align predictions with labels.\n"
        f"- labels_jsonl found: {Path(labels_jsonl).exists() if labels_jsonl else False}\n"
        f"- truth_csv found: {Path(truth_csv).exists() if truth_csv else False}\n"
        "Tip: Ensure week5_graph_run.jsonl claims come from reports/verifier_dataset.jsonl."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="reports/week5_graph_run.jsonl")
    ap.add_argument("--truth_csv", default="data/WELFake.csv")
    ap.add_argument("--labels_jsonl", default="reports/verifier_dataset.jsonl")
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    preds = load_preds(args.pred)
    merged = align_truth(preds, args.labels_jsonl, args.truth_csv)

    # keep only aligned rows
    merged = merged[merged["label"].notna()].copy()
    merged["label"] = merged["label"].astype(int)
    merged["verdict"] = merged["verdict"].astype(int)

    merged["correct"] = (merged["verdict"] == merged["label"])
    errs = merged[~merged["correct"]].copy()

    total = len(merged)
    wrong = int((~merged["correct"]).sum())
    print(f"Total evaluated: {total} | Errors: {wrong}\n")

    # Shuffle deterministically and print up to K
    errs = errs.sample(n=min(args.k, len(errs)), random_state=42) if len(errs) > args.k else errs

    for _, r in errs.iterrows():
        claim = r["claim_norm"]
        pred = int(r["verdict"]); gold = int(r["label"])
        conf = float(r["confidence"]) if r["confidence"] is not None else float("nan")
        top_ev = ""
        try:
            evs = r["evidence_internal"] or []
            if isinstance(evs, list) and evs:
                # evidence from week5 JSONL is usually dicts
                top_ev = evs[0].get("text", "") if isinstance(evs[0], dict) else str(evs[0])[:500]
        except Exception:
            pass

        print("-" * 100)
        print("ID:", r["id_str"])
        print("CLAIM:", shorten(str(claim), width=160, placeholder="…"))
        print(f"PRED={pred}  GOLD={gold}  p={conf:.2f}")
        if top_ev:
            print("EVIDENCE:", shorten(top_ev, width=160, placeholder="…"))

if __name__ == "__main__":
    main()
