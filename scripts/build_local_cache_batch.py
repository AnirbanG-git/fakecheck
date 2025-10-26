import argparse, json, random
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from src.utils.data_io import load_welfake
from src.retriever.bm25_retriever import BM25Retriever, Doc as BDoc
from src.retriever.embedding_retriever import DenseRetriever, Doc as EDoc
from src.retriever.hybrid import rrf_fuse


def gather_cache_ids(cache_dir: Path) -> List[int]:
    ids = []
    for p in cache_dir.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except Exception:
            pass
    return sorted(ids)


def pick_ids_from_dataset(n_rows: int, sample: Optional[int], seed: int) -> List[int]:
    N = n_rows if (sample is None or sample <= 0) else min(sample, n_rows)
    rng = random.Random(seed)
    ids = list(range(n_rows))
    rng.shuffle(ids)
    return ids[:N]


def pick_ids_from_cache(cache_dir: Path, sample: Optional[int], seed: int) -> List[int]:
    cache_ids = gather_cache_ids(cache_dir)
    if not cache_ids:
        return []
    if sample is None or sample <= 0:
        return cache_ids
    rng = random.Random(seed)
    rng.shuffle(cache_ids)
    return cache_ids[: min(sample, len(cache_ids))]


def main():
    ap = argparse.ArgumentParser(description="Build local (BM25 + Dense + Hybrid) evidence cache in bulk.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="evidence_cache")
    ap.add_argument("--sample", type=int, default=200, help="How many claims to cache (<=0 means all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--from_cache_ids", action="store_true",
                    help="If set, sample ids from existing cache_dir files instead of the dataset rows.")
    ap.add_argument("--cache_dir", default="evidence_cache",
                    help="When --from_cache_ids is set, read ids from here (usually same as out_dir).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing <id>.json. Otherwise skip and resume.")

    # Dense retriever / FAISS knobs (aligned with hybrid_retrieve.py)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--index_dir", default="indexes/welfake_dense")
    ap.add_argument("--reuse_index", action="store_true")

    # Retrieval hygiene
    ap.add_argument("--exclude_self", action="store_true", help="Remove the query doc from retrieved hits.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_welfake(args.csv).reset_index(drop=True).reset_index(names="id")
    df["_text"] = (df.get("title", "") + " " + df.get("text", "")).astype(str)
    n_rows = len(df)

    # Decide which ids to process
    if args.from_cache_ids:
        ids = pick_ids_from_cache(Path(args.cache_dir), args.sample, args.seed)
        if not ids:
            print(f"[WARN] --from_cache_ids was set, but no files found in {args.cache_dir}. "
                  f"Falling back to dataset sampling.")
            ids = pick_ids_from_dataset(n_rows, args.sample, args.seed)
    else:
        ids = pick_ids_from_dataset(n_rows, args.sample, args.seed)

    if not ids:
        print("[ERROR] No ids to process. Exiting.")
        return

    # Prepare documents for BM25 & Dense
    docs_b = [BDoc(doc_id=str(i), text=t, meta={"id": str(i), "label": int(l)})
              for i, t, l in zip(df["id"], df["_text"], df["label"])]
    docs_e = [EDoc(doc_id=str(i), text=t, meta={"id": str(i), "label": int(l)})
              for i, t, l in zip(df["id"], df["_text"], df["label"])]

    # BM25 builds instantly in-memory
    bm25 = BM25Retriever(docs_b)

    # Dense retriever setup (GPU if available and --device=auto)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dense = DenseRetriever(
        model_name=args.dense_model,
        device=device,
        batch_size=args.batch_size,
        index_dir=args.index_dir,  # persisted FAISS location (exists/load/save handled in class)
    )

    # Build or reuse the dense index
    if args.reuse_index and dense.exists():
        dense.load()
        print(f"✅ Loaded FAISS index from {args.index_dir} (dim={dense.dim() if hasattr(dense, 'dim') else 'unknown'})")
    else:
        print("Encoding corpus for dense retrieval… (one-time)")
        dense.build(docs_e)
        dense.save()
        print(f"✅ Saved FAISS index to {args.index_dir}")

    # Process each selected claim id
    processed, skipped, errors = 0, 0, 0
    pbar = tqdm(ids, desc="Building local cache", ncols=100)
    for cid in pbar:
        out_path = out_dir / f"{cid}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            row = df.iloc[cid]
            claim_text = str(row["_text"])
            claim_meta = {"id": str(row["id"]), "label": int(row["label"])}

            bm25_hits = bm25.query(claim_text, top_k=args.top_k)
            dense_hits = dense.query(claim_text, top_k=args.top_k)

            if args.exclude_self:
                cid_str = str(claim_meta["id"])
                bm25_hits  = [(t, m, s) for (t, m, s) in bm25_hits  if m.get("id") != cid_str]
                dense_hits = [(t, m, s) for (t, m, s) in dense_hits if m.get("id") != cid_str]

            fused = rrf_fuse(bm25_hits, dense_hits, top_k=args.top_k)

            rec = {
                "claim_id": claim_meta["id"],
                "claim_label": claim_meta["label"],
                "claim_text": claim_text,
                "retrieval": {
                    "bm25":   [{"text": t, "meta": m, "score": s} for (t, m, s) in bm25_hits],
                    "dense":  [{"text": t, "meta": m, "score": s} for (t, m, s) in dense_hits],
                    "hybrid": [{"text": t, "meta": m, "score": s} for (t, m, s) in fused],
                },
                "external": {"wikipedia": [], "fact_check": [], "google_cse": []}  # filled later by augment_external_evidence.py
            }

            out_path.write_text(json.dumps(rec, indent=2))
            processed += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] id={cid}: {e}")

        pbar.set_postfix(proc=processed, skip=skipped, err=errors)

    print(f"\nDone. Processed={processed}, Skipped={skipped}, Errors={errors}, "
          f"OutDir={str(out_dir)}")


if __name__ == "__main__":
    main()
