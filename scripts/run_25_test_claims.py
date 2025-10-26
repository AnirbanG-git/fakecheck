#!/usr/bin/env python3
import json, random
from pathlib import Path
import pandas as pd

from src.agent.state import AgentState, ClaimInput
from src.agent.nodes import RetrieveNode, VerifyNode, ExplainNode

def load_corpus(csv_path: str):
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        df = df.reset_index(drop=True).reset_index(names="id")
    if {"title","text"} <= set(df.columns):
        texts = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    else:
        texts = df["text"].fillna("").astype(str)
    corpus_texts = texts.tolist()
    corpus_meta = [{"id": int(df.iloc[i]["id"])} for i in range(len(df))]
    return corpus_texts, corpus_meta

def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    dataset_jsonl = "reports/verifier_dataset.jsonl"
    csv_path = "data/WELFake.csv"
    out_path = "reports/week5_graph_run.jsonl"
    dense_index_dir = "indexes/welfake_dense"
    bert_dir = "artifacts/verifier_bert"

    rows = load_jsonl(dataset_jsonl)
    test = [r for r in rows if r.get("split") == "test"]
    random.Random(42).shuffle(test)
    test = test[:25]

    corpus_texts, corpus_meta = load_corpus(csv_path)

    retrieve = RetrieveNode(
        corpus_texts=corpus_texts,
        corpus_meta=corpus_meta,
        dense_index_dir=dense_index_dir,
        emb_model="intfloat/e5-base-v2",
        top_k_internal=5,
        top_k_external=3,
    )
    verify = VerifyNode(mode="bert", bert_dir=bert_dir, threshold=0.50)
    explain = ExplainNode()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as w:
        for r in test:
            claim = r.get("claim") or r.get("text") or ""
            cid = r.get("id")
            st = AgentState(claim=ClaimInput(id=cid, text=claim))
            st = retrieve(st); st = verify(st); st = explain(st)
            rec = {
                "id": cid,
                "claim": st.claim.text,
                "verdict": st.verdict.label,
                "confidence": st.verdict.proba,
                "explanation": st.verdict.explanation,
                "evidence_internal": [{"text": e.text, "score": e.score, "meta": e.meta} for e in st.evidence.internal],
                "evidence_external": [{"snippet": e.snippet, "url": e.url, "source": e.source} for e in st.evidence.external],
                "trace": st.trace,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
