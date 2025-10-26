"""
Generate short, citation-backed explanations for verifier predictions.

Inputs:
  --in_dir  : cache dir with <id>.json (claim_text, retrieval.bm25, external.ranked, etc.)
  --preds   : JSONL with rows {"id": <id>, "pred": 0/1, "proba": float}
  --out     : JSONL with rows:
              {
                "id": "...",
                "verdict": "SUPPORTED"|"NOT_SUPPORTED",
                "proba": 0.93,
                "explanation": "… [1] … [2]",
                "citations": [{"tag":"[1]","source":"ranked","url":"..."}, ...]
              }

Options:
  --encoder : sentence-transformers model or LOCAL DIR (default all-MiniLM-L6-v2)
  --device  : auto|cpu|cuda
  --batch_size
  --sentences_per_source
  --max_total_sentences
  --min_sim : drop candidates below this cosine (optional)

Only uses local cache; does not fetch anything from the web.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _concat_title_snippet(it: Dict[str, Any]) -> Tuple[str, str]:
    title = (it.get("title") or "").strip()
    snip = (it.get("snippet") or it.get("text") or it.get("content") or "").strip()
    txt = (title + " " + snip).strip()
    url = (it.get("url") or it.get("source_url") or "").strip()
    return txt, url


def _collect_candidates(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    # external.ranked (URL-friendly first)
    ext = rec.get("external") or {}
    ranked = ext.get("ranked") or []
    for it in ranked:
        txt, url = _concat_title_snippet(it)
        if txt:
            cands.append({"text": txt, "url": url, "source": "ranked"})
    # internal retrieval (bm25)
    bm25 = ((rec.get("retrieval") or {}).get("bm25") or [])
    for it in bm25:
        txt = (it.get("text") or "").strip()
        if txt:
            cands.append({"text": txt, "url": "", "source": "bm25"})
    return cands


def _load_st_model(name_or_path: str, device: str):
    from sentence_transformers import SentenceTransformer
    p = Path(name_or_path)
    if p.exists() and p.is_dir():
        return SentenceTransformer(str(p.resolve()), device=device)
    return SentenceTransformer(name_or_path, device=device)


def _encode(model, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    with torch.inference_mode():
        if device == "cuda":
            from torch import amp
            with amp.autocast("cuda"):
                X = model.encode(
                    texts, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False, batch_size=batch_size,
                )
        else:
            X = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, batch_size=batch_size,
            )
    return np.asarray(X, dtype=np.float32)


def _rank_candidates(claim: str, items: List[Dict[str, Any]], model_name: str, device: str, batch_size: int) -> List[Dict[str, Any]]:
    st = _load_st_model(model_name, device=device)
    q = _encode(st, [claim], device, batch_size)                   # [1, d]
    X = _encode(st, [it["text"] for it in items], device, batch_size)  # [N, d]
    sims = (X @ q.T).reshape(-1)
    ranked = [{"text": it["text"], "url": it["url"], "source": it["source"], "sim": float(s)} for it, s in zip(items, sims)]
    ranked.sort(key=lambda r: r["sim"], reverse=True)
    return ranked


def _pick_top(ranked: List[Dict[str, Any]], sentences_per_source: int, max_total: int, min_sim: float) -> List[Dict[str, Any]]:
    per_src_count: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for r in ranked:
        if r["sim"] < min_sim:
            continue
        src = r["source"]
        per_src_count[src] = per_src_count.get(src, 0) + 1
        if per_src_count[src] > sentences_per_source:
            continue
        out.append(r)
        if len(out) >= max_total:
            break
    # Prefer URL-backed evidence earlier
    out.sort(key=lambda x: (0 if x["url"] else 1, -x["sim"]))
    return out


def _make_explanation(picks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    cites: List[Dict[str, str]] = []
    parts: List[str] = []
    for i, p in enumerate(picks, start=1):
        tag = f"[{i}]"
        cites.append({"tag": tag, "source": p["source"], "url": p["url"]})
        snippet = p["text"]
        if len(snippet) > 300:
            snippet = snippet[:297].rsplit(" ", 1)[0] + "..."
        parts.append(f"{snippet} {tag}")
    body = " ".join(parts) if parts else "No high-similarity evidence available in cache."
    return body, cites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Cache directory with <id>.json files")
    ap.add_argument("--preds", required=True, help="JSONL predictions from a verifier")
    ap.add_argument("--out", required=True, help="Where to write explanations JSONL")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name OR local directory")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--sentences_per_source", type=int, default=2)
    ap.add_argument("--max_total_sentences", type=int, default=5)
    ap.add_argument("--min_sim", type=float, default=0.25)
    args = ap.parse_args()

    cache_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    # Load predictions
    preds = []
    with Path(args.preds).open() as f:
        for line in f:
            preds.append(json.loads(line))

    with out_path.open("w") as fout:
        for row in preds:
            cid = str(row["id"])
            pred = int(row.get("pred", 1 if row.get("proba", 0.5) >= 0.5 else 0))
            proba = float(row.get("proba", 0.5))

            rec = _read_json(cache_dir / f"{cid}.json")
            claim = (rec.get("claim_text") or "").strip()

            items = _collect_candidates(rec)
            ranked = _rank_candidates(claim, items, args.encoder, device, args.batch_size) if (claim and items) else []
            picks = _pick_top(ranked, args.sentences_per_source, args.max_total_sentences, args.min_sim)

            explanation, citations = _make_explanation(picks)
            verdict = "SUPPORTED" if pred == 1 else "NOT_SUPPORTED"

            fout.write(json.dumps({
                "id": cid,
                "verdict": verdict,
                "proba": proba,
                "explanation": explanation,
                "citations": citations
            }, ensure_ascii=False) + "\n")

    print(f"Wrote explanations to {out_path}")


if __name__ == "__main__":
    main()
