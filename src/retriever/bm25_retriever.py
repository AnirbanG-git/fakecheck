from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from typing import List, Dict
import re
TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")  # keep digits

@dataclass
class Doc:
    doc_id: str
    text: str
    meta: Dict

def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25, preserving numbers (e.g., '141') and hyphenated words.
    """
    if not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]    

class BM25Retriever:
    def __init__(self, docs: List[Doc]):
        self.docs = docs
        tokenized = [d.text.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, q: str, top_k: int = 10):
        scores = self.bm25.get_scores(q.lower().split())
        # Top-k indices by score
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for i in idxs:
            d = self.docs[i]
            hits.append((d.text, d.meta, float(scores[i])))
        return hits


