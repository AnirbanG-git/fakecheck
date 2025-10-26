import argparse, json
from pathlib import Path
import pandas as pd

from src.utils.data_io import load_welfake
from src.retriever.bm25_retriever import BM25Retriever, Doc as BDoc
from src.retriever.embedding_retriever import DenseRetriever, Doc as EDoc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--catalog_out", default="artifacts/doc_catalog.json")
    ap.add_argument("--dense_model", default="intfloat/e5-base-v2")
    ap.add_argument("--save_dense_index", action="store_true")  # for future (optional)
    args = ap.parse_args()

    df = load_welfake(args.csv)
    df = df.reset_index(drop=True).reset_index(names="id")
    df["_text"] = (df.get("title", "") + " " + df.get("text", "")).astype(str)

    # Build a small catalog: id -> text/meta
    catalog = []
    docs_b = []
    docs_e = []

    for _, row in df.iterrows():
        meta = {"id": str(row["id"]), "label": int(row["label"])}
        text = str(row["_text"])
        catalog.append({"id": str(row["id"]), "label": int(row["label"])})
        docs_b.append(BDoc(doc_id=str(row["id"]), text=text, meta=meta))
        docs_e.append(EDoc(doc_id=str(row["id"]), text=text, meta=meta))

    # BM25
    bm25 = BM25Retriever(docs_b)

    # Dense
    dense = DenseRetriever(model_name=args.dense_model)
    dense.build(docs_e)

    # Save a lightweight catalog (indexes live in memory for now)
    Path(Path(args.catalog_out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.catalog_out, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Indexed {len(catalog)} docs.")
    print("Tip: keep this process running in a notebook if you want the in-memory indexes handy.")

if __name__ == "__main__":
    main()
