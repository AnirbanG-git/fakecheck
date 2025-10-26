"""
Batch runner for Week-5/6:
Reads a dataset JSONL (with claim ids/splits), runs Retrieve→Verify→Explain
over N items, and writes a JSONL of results.

Compatible with current src/agent/nodes.py and retrievers.
"""

import argparse, json, random
from pathlib import Path
import pandas as pd

from src.agent.state import AgentState, ClaimInput
from src.agent.nodes import RetrieveNode, VerifyNode, ExplainNode


def load_corpus(csv_path: str):
    """Return (corpus_texts, corpus_meta) from WELFake CSV."""
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        df = df.reset_index(drop=True).reset_index(names="id")
    # combine title + text if available, else use text
    if {"title", "text"} <= set(df.columns):
        texts = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    elif "text" in df.columns:
        texts = df["text"].fillna("").astype(str)
    else:
        raise ValueError("CSV must have 'text' or 'title'+'text' columns")
    corpus_texts = texts.tolist()
    corpus_meta = [{"id": int(df.iloc[i]["id"])} for i in range(len(df))]
    return corpus_texts, corpus_meta


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_rows(rows, split: str | None, n: int, seed: int):
    if split:
        rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
    # prefer 'claim' but fall back to 'text'
    rows = [r for r in rows if (r.get("claim") or r.get("text"))]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True, help="Path to dataset JSONL (with id/claim/split)")
    ap.add_argument("--csv", required=True, help="Corpus CSV path (e.g., data/WELFake.csv)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=25, help="Number of samples")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--split", default="test", help="Which split to use (e.g., test/val/train); empty=all")
    # retrieval / dense
    ap.add_argument("--topk", type=int, default=5, help="Top-k internal docs")
    ap.add_argument("--emb_model", default="intfloat/e5-base-v2", help="Sentence embedding model")
    ap.add_argument("--device", default=None, help="Force device (cuda/cpu); default auto")
    ap.add_argument("--dense_index_dir", default=None, help="Persisted dense index dir (loads if present)")
    # verifier
    ap.add_argument("--verifier", choices=["bert", "lr"], default="bert")
    ap.add_argument("--bert_model_dir", default="artifacts/verifier_bert")
    ap.add_argument("--lr_primary", default="artifacts/verifier_lr/verifier_lr.joblib")
    ap.add_argument("--lr_fallback", default="artifacts/verifier_lr/baseline_tfidf_lr.joblib")
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()

    rows = load_jsonl(args.dataset_jsonl)
    todo = select_rows(rows, args.split, args.n, args.seed)

    corpus_texts, corpus_meta = load_corpus(args.csv)

    retrieve = RetrieveNode(
        corpus_texts=corpus_texts,
        corpus_meta=corpus_meta,
        dense_index_dir=args.dense_index_dir if args.dense_index_dir else "artifacts/e5_base",
        emb_model=args.emb_model,
        device=args.device,
        top_k_internal=args.topk,
        top_k_external=3,
        evidence_dir="evidence_cache",
        external_cache_dir="external_cache",
    )

    verify = VerifyNode(
        mode=args.verifier,
        lr_artifact_primary=args.lr_primary,
        lr_artifact_fallback=args.lr_fallback,
        bert_dir=args.bert_model_dir,
        max_len=256,
        threshold=args.threshold,
    )

    explain = ExplainNode()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as w:
        for r in todo:
            claim_text = r.get("claim") or r.get("text") or ""
            cid = r.get("id")
            st = AgentState(claim=ClaimInput(id=cid, text=claim_text))
            st = retrieve(st)
            st = verify(st)
            st = explain(st)
            rec = {
                "id": cid,
                "claim": st.claim.text,
                "verdict": st.verdict.label,
                "confidence": st.verdict.proba,
                "explanation": st.verdict.explanation,
                "evidence_internal": [
                    {"text": e.text, "score": e.score, "meta": e.meta} for e in st.evidence.internal
                ],
                "evidence_external": [
                    {"snippet": ex.snippet, "url": ex.url, "source": ex.source} for ex in st.evidence.external
                ],
                "trace": st.trace,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {args.out} ({written} lines)")


if __name__ == "__main__":
    main()
