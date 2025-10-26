import argparse, json, random
from pathlib import Path
import pandas as pd
import torch

from src.utils.data_io import load_welfake
from src.retriever.bm25_retriever import BM25Retriever, Doc as BDoc
from src.retriever.embedding_retriever import DenseRetriever, Doc as EDoc
from src.retriever.hybrid import rrf_fuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--claim_id", type=int, default=None, help="Row id in the CSV to use as a query")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_dir", default="evidence_cache")

    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--index_dir", default="indexes/welfake_dense")
    ap.add_argument("--reuse_index", action="store_true")
    ap.add_argument("--exclude_self", action="store_true")

    args = ap.parse_args()

    df = load_welfake(args.csv)
    df = df.reset_index(drop=True).reset_index(names="id")
    df["_text"] = (df.get("title", "") + " " + df.get("text", "")).astype(str)

    docs_b = [BDoc(doc_id=str(i), text=t, meta={"id": str(i), "label": int(l)})
              for i, t, l in zip(df["id"], df["_text"], df["label"])]
    docs_e = [EDoc(doc_id=str(i), text=t, meta={"id": str(i), "label": int(l)})
              for i, t, l in zip(df["id"], df["_text"], df["label"])]

    bm25 = BM25Retriever(docs_b)

    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":  # user forced auto but no GPU available
        device = "cpu"

    dense = DenseRetriever(
        model_name=args.dense_model,
        device=device,
        batch_size=args.batch_size,
        index_dir=args.index_dir,      # <- persists here
        use_gpu_faiss=(device == "cuda"),
        use_amp=(device == "cuda")
    )

    if args.reuse_index and dense.exists():
        dense.load()                 # load saved embeddings/FAISS
    else:
        dense.build(docs_e)          # encode once
        dense.save()                 # persist to disk for reuse

    # pick a claim
    if args.claim_id is None:
        args.claim_id = random.randint(0, len(df)-1)
    claim_row = df.iloc[args.claim_id]
    claim_text = str(claim_row["_text"])
    claim_meta = {"id": str(claim_row["id"]), "label": int(claim_row["label"])}

    bm25_hits = bm25.query(claim_text, top_k=args.top_k)
    dense_hits = dense.query(claim_text, top_k=args.top_k)

    if args.exclude_self:
        claim_id_str = str(claim_meta["id"])
        bm25_hits  = [(t,m,s) for (t,m,s) in bm25_hits  if m.get("id") != claim_id_str]
        dense_hits = [(t,m,s) for (t,m,s) in dense_hits if m.get("id") != claim_id_str]

    fused = rrf_fuse(bm25_hits, dense_hits, top_k=args.top_k)

    # write cache
    out = {
        "claim_id": claim_meta["id"],
        "claim_label": claim_meta["label"],
        "claim_text": claim_text,
        "retrieval": {
            "bm25": [{"text": t, "meta": m, "score": s} for (t,m,s) in bm25_hits],
            "dense": [{"text": t, "meta": m, "score": s} for (t,m,s) in dense_hits],
            "hybrid": [{"text": t, "meta": m, "score": s} for (t,m,s) in fused]
        },
        "external": {"wikipedia": [], "fact_check": [], "google_cse": []}
    }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{claim_meta['id']}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved evidence for claim_id={claim_meta['id']} → {out_path}")

if __name__ == "__main__":
    main()
