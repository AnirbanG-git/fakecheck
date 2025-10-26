from typing import List, Tuple, Dict

# hits: List[(text, meta, score)] from each retriever
def rrf_fuse(bm25_hits, dense_hits, k: int = 60, top_k: int = 10):
    ranks: Dict[Tuple[str, str], float] = {}

    def add_ranked(hits):
        for r, (text, meta, _score) in enumerate(hits, start=1):
            key = (text, meta.get("id", ""))
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (k + r)

    add_ranked(bm25_hits)
    add_ranked(dense_hits)

    fused = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)
    # convert back to (text, meta, fused_score)
    out = []
    for (text, doc_id), fused_score in fused[:top_k]:
        out.append((text, {"id": doc_id}, float(fused_score)))
    return out
