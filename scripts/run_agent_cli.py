#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

from src.agent.state import AgentState, ClaimInput
from src.agent.nodes import RetrieveNode, VerifyNode, ExplainNode

def load_corpus(csv_path: str, limit: int | None = None):
    """Return (corpus_texts: list[str], corpus_meta: list[dict])."""
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        df = df.reset_index(drop=True).reset_index(names="id")
    # combined text for retrieval
    if "title" in df.columns and "text" in df.columns:
        text_series = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    elif "text" in df.columns:
        text_series = df["text"].fillna("").astype(str)
    else:
        raise ValueError("CSV must have a 'text' column or 'title' + 'text' columns")
    if limit:
        df = df.head(limit)
        text_series = text_series.head(limit)

    corpus_texts = text_series.tolist()
    corpus_meta = [{"id": int(df.iloc[i]["id"])} for i in range(len(df))]
    return corpus_texts, corpus_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True, help="Claim text to verify")
    ap.add_argument("--mode", choices=["bert","lr"], default="bert", help="Verifier mode")
    ap.add_argument("--csv", default="data/WELFake.csv", help="Corpus CSV path")
    ap.add_argument("--out", default="-", help="Output path (json). Use '-' for stdout")
    ap.add_argument("--topk", type=int, default=5, help="Top-k internal hits")
    ap.add_argument("--dense_index_dir", default=None, help="Directory with a persisted FAISS index (loads if present)")
    ap.add_argument("--emb_model", default="intfloat/e5-base-v2", help="Sentence embedding model name")
    ap.add_argument("--device", default=None, help="Force device (e.g., 'cuda' or 'cpu'); default auto")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for REAL class")
    ap.add_argument("--evidence_dir", default="evidence_cache", help="Directory with {id}.json external caches")
    ap.add_argument("--external_cache_dir", default="external_cache", help="Optional SHA1-based external cache dir")
    ap.add_argument("--bert_dir", default="artifacts/verifier_bert", help="Directory of fine-tuned BERT verifier")
    ap.add_argument("--lr_primary", default="artifacts/verifier_lr/verifier_lr.joblib", help="Primary LR artifact")
    ap.add_argument("--lr_fallback", default="artifacts/verifier_lr/baseline_tfidf_lr.joblib", help="Fallback LR artifact")
    ap.add_argument("--limit_corpus", type=int, default=None, help="(Optional) limit corpus rows for quick runs")
    args = ap.parse_args()

    # 1) Load corpus expected by RetrieveNode
    corpus_texts, corpus_meta = load_corpus(args.csv, limit=args.limit_corpus)

    # 2) Build Retrieve → Verify → Explain nodes
    retrieve = RetrieveNode(
        corpus_texts=corpus_texts,
        corpus_meta=corpus_meta,
        dense_index_dir=args.dense_index_dir if args.dense_index_dir else "artifacts/e5_base",
        emb_model=args.emb_model,
        device=args.device,
        top_k_internal=args.topk,
        top_k_external=3,
        evidence_dir=args.evidence_dir,
        external_cache_dir=args.external_cache_dir
    )

    verify = VerifyNode(
        mode=args.mode,
        lr_artifact_primary=args.lr_primary,
        lr_artifact_fallback=args.lr_fallback,
        bert_dir=args.bert_dir,
        max_len=256,
        threshold=args.threshold
    )
    explain = ExplainNode()

    # 3) Prepare initial state
    state = AgentState(claim=ClaimInput(id=None, text=args.claim))

    # 4) Orchestrate
    state = retrieve(state)
    state = verify(state)
    state = explain(state)

    # 5) Emit output
    rec = {
        "claim": state.claim.text,
        "verdict": state.verdict.label,
        "confidence": state.verdict.proba,
        "explanation": state.verdict.explanation,
        "evidence_internal": [
            {"text": ev.text, "score": ev.score, "meta": ev.meta} for ev in state.evidence.internal
        ],
        "evidence_external": [
            {"snippet": ex.snippet, "url": ex.url, "source": ex.source} for ex in state.evidence.external
        ],
        "trace": state.trace
    }

    if args.out == "-" or args.out.strip() == "":
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
