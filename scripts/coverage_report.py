import argparse, json, random
from pathlib import Path
import pandas as pd
import torch
import time
im

from src.utils.data_io import load_welfake
from src.retriever.bm25_retriever import BM25Retriever, Doc as BDoc
from src.retriever.embedding_retriever import DenseRetriever, Doc as EDoc
from src.retriever.hybrid import rrf_fuse
from src.evaluation.coverage import compute_coverage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out", default="reports/retrieval_coverage.json")

    # NEW: align with hybrid_retrieve.py
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--index_dir", default="indexes/welfake_dense")
    ap.add_argument("--reuse_index", action="store_true", help="Load persisted dense index if available")
    ap.add_argument("--exclude_self", action="store_true", help="Drop the claim row from retrieved hits")
    args = ap.parse_args()

    df = load_welfake(args.csv)
    df = df.reset_index(drop=True).reset_index(names="id")
    df["_text"] = (df.get("title", "") + " " + df.get("text", "")).astype(str)

    ids = list(df["id"].values)
    random.seed(42)
    sample_ids = random.sample(ids, min(args.sample, len(ids)))

    # Build docs once
    docs_b = [BDoc(doc_id=str(i), text=t, meta={"id": str(i), "label": int(l)})
              for i, t, l in zip(df["id"], df["_text"], df["label"])]
    docs_e = [EDoc(doc_id=str(i), text=t, meta={"id": str(i), "label": int(l)})
              for i, t, l in zip(df["id"], df["_text"], df["label"])]

    bm25 = BM25Retriever(docs_b)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    dense = DenseRetriever(
        model_name=args.dense_model,
        device=device,
        batch_size=args.batch_size,
        index_dir=args.index_dir,
        use_gpu_faiss=(device == "cuda"),
        use_amp=(device == "cuda")
    )

    if args.reuse_index and dense.exists():
        dense.load()
    else:
        dense.build(docs_e)
        dense.save()

    # Run retrieval for each claim in sample
    claim_texts = {}
    hybrid_results = {}

    t0 = time.time()
    counter = 0
    for cid in sample_ids:
        t0_cid = time.time()
        row = df.iloc[cid]
        claim_texts[str(cid)] = str(row["_text"])

        bm25_hits  = bm25.query(claim_texts[str(cid)], top_k=args.top_k)
        dense_hits = dense.query(claim_texts[str(cid)], top_k=args.top_k)

        # ---- exclude the claim itself to avoid inflated coverage
        if args.exclude_self:
            cid_str = str(cid)
            bm25_hits  = [(t,m,s) for (t,m,s) in bm25_hits  if m.get("id") != cid_str]
            dense_hits = [(t,m,s) for (t,m,s) in dense_hits if m.get("id") != cid_str]

        fused = rrf_fuse(bm25_hits, dense_hits, top_k=args.top_k)
        hybrid_results[str(cid)] = [{"text": t, "meta": m, "score": s} for (t,m,s) in fused]

        t1_cid = time.time() 
        counter = counter + 1
        print(f"{counter} Processed {cid} in {t1_cid - t0_cid:.2f}s ")

    t1 = time.time()    
    print(f"Processed {len(sample_ids)} claims in {t1 - t0:.2f}s "
            f"(~{(t1 - t0)/max(1,len(sample_ids)):.4f}s/claim)")

    coverage = compute_coverage(claim_texts, hybrid_results, min_overlap=5)

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"sample": len(sample_ids), "top_k": args.top_k, "coverage": coverage}, f, indent=2)

    print(json.dumps({"sample": len(sample_ids), "top_k": args.top_k, "coverage": coverage}, indent=2))

if __name__ == "__main__":
    main()
