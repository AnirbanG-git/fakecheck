#!/usr/bin/env python3
"""
Run Retrieve -> Verify -> Explain on 25 claims drawn from WELFake CSV,
using the article TITLE as the claim. This avoids empty-claim issues.

Output: reports/week5_graph_run.jsonl (overwrites previous)
"""

import json, random
from pathlib import Path
import pandas as pd

from src.agent.state import AgentState, ClaimInput
from src.agent.nodes import RetrieveNode, VerifyNode, ExplainNode

def load_corpus(csv_path: str):
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        df = df.reset_index(drop=True).reset_index(names="id")
    # retrieval corpus = title + text
    if {"title","text"} <= set(df.columns):
        texts = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    elif "text" in df.columns:
        texts = df["text"].fillna("").astype(str)
    else:
        raise ValueError("CSV must have 'text' or 'title'+'text' columns")
    corpus_texts = texts.tolist()
    corpus_meta = [{"id": int(df.iloc[i]["id"])} for i in range(len(df))]
    return df, corpus_texts, corpus_meta

def pick_25(df: pd.DataFrame, seed: int = 42):
    # prefer 'test' split if present in your repo's csv (often not). Otherwise random 25.
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    take = []
    for i in idxs:
        row = df.iloc[i]
        title = str(row.get("title", "") or "").strip()
        if title:
            take.append(int(row["id"]))
        if len(take) == 25:
            break
    return df[df["id"].isin(take)].copy()

def main():
    csv_path = "data/WELFake.csv"
    out_path = "reports/week5_graph_run.jsonl"
    bert_dir = "artifacts/verifier_bert"
    dense_index_dir = "indexes/welfake_dense"

    df_all, corpus_texts, corpus_meta = load_corpus(csv_path)
    df25 = pick_25(df_all, seed=42)

    retrieve = RetrieveNode(
        corpus_texts=corpus_texts,
        corpus_meta=corpus_meta,
        dense_index_dir=dense_index_dir,
        emb_model="intfloat/e5-base-v2",
        top_k_internal=5,
        top_k_external=3,
    )
    verify = VerifyNode(mode="bert", bert_dir=bert_dir, threshold=0.55)  # slightly stricter than 0.50
    explain = ExplainNode()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for r in df25.itertuples():
            claim = str(getattr(r, "title", "") or "").strip()
            # guard: skip empties just in case
            if not claim:
                continue
            cid = int(getattr(r, "id"))
            st = AgentState(claim=ClaimInput(id=cid, text=claim))
            st = retrieve(st); st = verify(st); st = explain(st)
            rec = {
                "id": cid,
                "claim": st.claim.text,
                "verdict": st.verdict.label,
                "confidence": st.verdict.proba,
                "explanation": st.verdict.explanation,
                "evidence_internal": [{"text": e.text, "score": e.score, "meta": e.meta} for e in st.evidence.internal],
                "evidence_external": [{"snippet": ex.snippet, "url": ex.url, "source": ex.source} for ex in st.evidence.external],
                "trace": st.trace,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
