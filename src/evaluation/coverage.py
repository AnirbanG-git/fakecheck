from typing import Dict, List

def naive_overlap(a: str, b: str, min_overlap: int = 5) -> bool:
    at = {w for w in a.lower().split() if len(w) > 3}
    bt = {w for w in b.lower().split() if len(w) > 3}
    return len(at & bt) >= min_overlap

def compute_coverage(claim_texts: Dict[str, str], hybrid_results: Dict[str, List[dict]], min_overlap: int = 5) -> float:
    covered = 0
    for cid, ctext in claim_texts.items():
        docs = hybrid_results.get(cid, [])
        ok = any(naive_overlap(ctext, d["text"], min_overlap=min_overlap) for d in docs)
        covered += int(ok)
    return covered / max(1, len(claim_texts))
