# scripts/augment_external_evidence.py
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.utils.data_io import load_welfake
from src.utils.query_prep import expand_queries_general, expand_queries_fc
from src.external.wikipedia import search_wikipedia
from src.external.factcheck import search_fact_checks
from src.external.google_cse import search_cse

###############################################################################
# Credibility/Diversity utils
###############################################################################

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "aljazeera.com",
    "nytimes.com", "theguardian.com", "ft.com", "washingtonpost.com",
    "economist.com", "wikipedia.org", "who.int", "un.org",
}
LOWCRED_DOMAINS = {
    # Extend with any blogs you observed in week 2 that you want to down-weight.
    # Keep this conservative; we don't *block*, we just reduce rank weight.
    # e.g., "some-hyperpartisan-site.com",
}

def norm_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        d = urlparse(url).netloc.lower()
        if d.startswith("www."):
            d = d[4:]
        return d
    except Exception:
        return ""

def domain_weight(domain: str, boost_trusted: float, penalize_lowcred: float) -> float:
    if not domain:
        return 1.0
    if domain in TRUSTED_DOMAINS:
        return float(boost_trusted)
    if domain in LOWCRED_DOMAINS:
        return float(penalize_lowcred)
    return 1.0

def soft_time_weight(published_at: Optional[str], center_date: Optional[str], window_days: int) -> float:
    """
    Returns a smooth weight in [0.6, 1.2] favoring dates near center_date within a window.
    If no dates, return neutral 1.0.
    """
    if not published_at or not center_date or window_days <= 0:
        return 1.0
    try:
        from datetime import datetime
        p = datetime.fromisoformat(published_at[:10])
        c = datetime.fromisoformat(center_date[:10])
        delta = abs((p - c).days)
        if delta <= window_days:
            # within window → upweight up to 1.2
            return 1.2 - 0.6 * (delta / max(1, window_days))
        # outside window → slight downweight, floored at 0.8
        return max(0.8, 1.0 - 0.002 * (delta - window_days))
    except Exception:
        return 1.0

def extract_first_yyyy_mm_from_text(text: str) -> Optional[str]:
    """
    Heuristic: find a YYYY-MM pattern to anchor an approximate event window (used if caller doesn't supply).
    Returns 'YYYY-MM-15' as center_date for soft weighting.
    """
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", text)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-15"

###############################################################################
# Dedup / MMR
###############################################################################

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, device: str) -> np.ndarray:
    # AMP only on CUDA; SentenceTransformers handles batching
    if device == "cuda":
        from torch import amp
        with torch.inference_mode(), amp.autocast("cuda"):
            X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=batch_size)
    else:
        with torch.inference_mode():
            X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False, batch_size=batch_size)
    return np.asarray(X, dtype=np.float32)

def dedup_by_cosine(texts: List[str], metas: List[Dict[str, Any]], scores: List[float],
                    model: SentenceTransformer, device: str, batch_size: int, cos_thresh: float) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    if len(texts) <= 1:
        return texts, metas, scores
    X = encode_texts(model, texts, batch_size, device)
    keep_idx = []
    used = np.zeros(len(texts), dtype=bool)
    # keep items in order of current score (highest first)
    order = np.argsort(-np.asarray(scores))
    for idx in order:
        if used[idx]:
            continue
        keep_idx.append(idx)
        sims = X[idx] @ X.T
        dupes = np.where(sims >= cos_thresh)[0]
        used[dupes] = True
    keep_idx.sort()
    return [texts[i] for i in keep_idx], [metas[i] for i in keep_idx], [scores[i] for i in keep_idx]

def mmr_rank(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_: float, k: int) -> List[int]:
    """
    Simple MMR: balances relevance to query (cosine) vs novelty w.r.t. selected set.
    """
    N = cand_vecs.shape[0]
    if N <= k:
        return list(range(N))
    rel = (cand_vecs @ query_vec.T).reshape(-1)  # [N]
    selected = []
    remaining = set(range(N))
    while remaining and len(selected) < k:
        if not selected:
            i = int(np.argmax(rel[list(remaining)]))
            pick = list(remaining)[i]
        else:
            S = cand_vecs[selected]            # [s, d]
            max_sim = (S @ cand_vecs.T).max(axis=0)  # [N]
            score = lambda_ * rel + (1 - lambda_) * (1 - max_sim)
            # forbid already selected
            score[selected] = -1e9
            idxs = list(remaining)
            local = score[idxs]
            pick = idxs[int(np.argmax(local))]
        selected.append(pick)
        remaining.remove(pick)
    return selected

###############################################################################
# Uniform schema + fusion
###############################################################################

def to_uniform_items(block: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Normalize external items to a shared schema the Verifier/Explainer can use.
    We keep originals in external.{source}, and also produce a fused ranked list.
    """
    items = []
    for x in block or []:
        title = x.get("title") or x.get("page_title") or x.get("name") or ""
        url = x.get("url") or x.get("link") or ""
        snippet = x.get("snippet") or x.get("summary") or x.get("content") or x.get("extract") or ""
        publisher = x.get("publisher") or x.get("site") or ""
        published_at = x.get("published_at") or x.get("date") or x.get("reviewDate") or x.get("published") or None
        items.append({
            "source": source,
            "title": title,
            "url": url,
            "publisher": publisher,
            "published_at": published_at,
            "snippet": snippet
        })
    return items

def fuse_and_rank(
    claim_text: str,
    blocks: Dict[str, List[Dict[str, Any]]],
    model: SentenceTransformer,
    device: str,
    batch_size: int,
    k: int,
    max_per_domain: int,
    dedup_cos: float,
    boost_trusted: float,
    penalize_lowcred: float,
    center_date: Optional[str],
    time_window_days: int,
    mmr_lambda: float = 0.7,
) -> List[Dict[str, Any]]:
    # Flatten and keep source label
    cand = []
    for src, block in blocks.items():
        cand.extend(to_uniform_items(block, src))

    if not cand:
        return []

    # Prepare text used for scoring (title + snippet is usually best)
    texts = [(c.get("title","") + " " + c.get("snippet","")).strip() for c in cand]
    # Domain weights
    domains = [norm_domain(c.get("url", "")) for c in cand]
    d_weights = [domain_weight(d, boost_trusted, penalize_lowcred) for d in domains]
    # Time weights
    t_weights = [soft_time_weight(c.get("published_at"), center_date, time_window_days) for c in cand]

    # Embed query + candidates
    q_vec = encode_texts(model, [claim_text], batch_size, device)  # [1, d]
    X = encode_texts(model, texts, batch_size, device)             # [N, d]
    cos = (X @ q_vec.T).reshape(-1)                                # [N]

    # Base final score = cosine * domain * time
    base_scores = cos * np.asarray(d_weights) * np.asarray(t_weights)

    # Deduplicate near-duplicates by cosine on candidate embeddings
    texts_dedup, metas_dedup, scores_dedup = dedup_by_cosine(
        texts, cand, base_scores.tolist(), model, device, batch_size, cos_thresh=dedup_cos
    )
    # Recompute vectors for deduped items (to align indices)
    if len(texts_dedup) != len(texts):
        X = encode_texts(model, texts_dedup, batch_size, device)
        cos = (X @ q_vec.T).reshape(-1)
        cand = metas_dedup
        base_scores = np.asarray(scores_dedup)

    # MMR diversify
    order = mmr_rank(q_vec, X, lambda_=mmr_lambda, k=min(k*4, len(cand)))  # get a larger pool, then cap per-domain
    cand = [cand[i] for i in order]
    cos = cos[order]
    X = X[order]
    base_scores = base_scores[order]

    # Per-domain cap & take top-k
    final = []
    per_domain_count: Dict[str, int] = {}
    for i in range(len(cand)):
        d = norm_domain(cand[i].get("url",""))
        if d and per_domain_count.get(d, 0) >= max_per_domain:
            continue
        final.append((cand[i], float(base_scores[i])))
        per_domain_count[d] = per_domain_count.get(d, 0) + 1
        if len(final) >= k:
            break

    # Attach final_score for transparency
    ranked = []
    for item, score in final:
        it = dict(item)
        it["final_score"] = score
        it["domain"] = norm_domain(it.get("url",""))
        ranked.append(it)
    return ranked

###############################################################################
# External fetch wrapper (concurrent)
###############################################################################

def fetch_all_external(
    queries: List[str],
    top_k: int,
    api_key_cse: str,
    cse_cx: str,
    api_key_fact: str,
    max_concurrency: int = 5,
    fc_language: str = "en",
    fc_max_pages: int = 3,
    fc_retries: int = 3,
    fc_timeout_s: int = 12,
    fc_min_results_to_stop: int = 6,
    queries_fc: Optional[List[str]] = None,   # NEW
) -> Dict[str, List[Dict[str, Any]]]:
    results = {"wikipedia": [], "fact_check": [], "google_cse": []}
    fc_qs = queries_fc or queries  # prefer FC-specific compact queries

    def _run(name: str):
        try:
            if name == "wikipedia":
                return name, search_wikipedia(queries, top_k=top_k)
            if name == "fact_check":
                if not api_key_fact:
                    return name, []
                fact = search_fact_checks(
                    queries=fc_qs,
                    api_key=api_key_fact,
                    top_k=top_k,
                    language_code=fc_language,
                    max_pages=fc_max_pages,
                    retries=fc_retries,
                    timeout_s=fc_timeout_s,
                    min_results_to_stop=fc_min_results_to_stop,
                )
                return name, fact
            if name == "google_cse":
                if not (api_key_cse and cse_cx):
                    return name, []
                return name, search_cse(queries, api_key=api_key_cse, cx=cse_cx, top_k=top_k)
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            return name, []
        return name, []

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = [ex.submit(_run, n) for n in ["wikipedia", "fact_check", "google_cse"]]
        for f in as_completed(futs):
            name, payload = f.result()
            results[name] = payload or []
    return results


###############################################################################
# Main
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--claim_id", type=int, default=None, help="If set, only augment this claim id")
    ap.add_argument("--cache_dir", default="evidence_cache")
    ap.add_argument("--out_dir", default="evidence_cache")  # in-place update
    ap.add_argument("--top_k", type=int, default=10)

    # GPU + ranking controls
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)

    # credibility / diversity / time
    ap.add_argument("--max_per_domain", type=int, default=2)
    ap.add_argument("--dedup_cos", type=float, default=0.95)
    ap.add_argument("--boost_trusted", type=float, default=1.2)
    ap.add_argument("--penalize_lowcred", type=float, default=0.6)
    ap.add_argument("--time_window_days", type=int, default=45)
    ap.add_argument("--center_date", type=str, default=None, help="ISO date (YYYY-MM-DD); if not set we try to infer from claim text")

    # concurrency controls (external fetch)
    ap.add_argument("--max_concurrency", type=int, default=5)

    # FactCheck robustness
    ap.add_argument("--fc_retries", type=int, default=3)
    ap.add_argument("--fc_timeout_s", type=int, default=12)
    ap.add_argument("--fc_max_pages", type=int, default=3)
    ap.add_argument("--fc_min_results_to_stop", type=int, default=6)
    ap.add_argument("--fc_language", type=str, default="en")    

    args = ap.parse_args()

    # Load API keys
    load_dotenv("configs/.env")
    API_KEY = os.getenv("GOOGLE_API_KEY", "")
    CSE_CX = os.getenv("GOOGLE_CSE_CX", "")
    FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

    # Dataset (to get claim text if cache lacks it)
    df = load_welfake(args.csv).reset_index(drop=True).reset_index(names="id")
    df["_text"] = (df.get("title", "") + " " + df.get("text", "")).astype(str)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Shared encoder for ranking/dedup on GPU/CPU
    st_model = SentenceTransformer(args.dense_model, device=device)

    # Determine which claim(s) to augment
    to_process: List[int] = []
    if args.claim_id is not None:
        to_process = [args.claim_id]
    else:
        # augment everything that already exists in cache
        for p in Path(args.cache_dir).glob("*.json"):
            try:
                cid = int(p.stem)
                to_process.append(cid)
            except Exception:
                pass
    to_process.sort()

    for cid in to_process:
        path = Path(args.cache_dir) / f"{cid}.json"
        if not path.exists():
            print(f"Skip: no local evidence for claim_id={cid}")
            continue

        data = json.loads(path.read_text())
        # claim text
        claim_text = data.get("claim_text") or str(df.iloc[cid]["_text"])

        # Prepare queries (proposal-aligned: entity/time enriched)
        queries_general = expand_queries_general(claim_text)
        queries_fc = expand_queries_fc(claim_text)


        # Optional: infer center date from claim text if not provided
        center_date = args.center_date or extract_first_yyyy_mm_from_text(claim_text)

        # --- Fetch external evidence concurrently
        ext = fetch_all_external(
            queries=queries_general,              # used by wiki + CSE
            queries_fc=queries_fc,                # dedicated compact set for FactCheck
            top_k=args.top_k,
            api_key_cse=API_KEY,
            cse_cx=CSE_CX,
            api_key_fact=FACTCHECK_API_KEY,
            max_concurrency=args.max_concurrency,
            fc_language=args.fc_language,
            fc_max_pages=args.fc_max_pages,
            fc_retries=args.fc_retries,
            fc_timeout_s=args.fc_timeout_s,
            fc_min_results_to_stop=args.fc_min_results_to_stop,
        )


        if not ext.get("fact_check"):
            print(f"[INFO] FactCheck zero hits. Queries tried: {queries_fc[:3]} ...")

        # --- Fusion: credibility-aware, time-aware, diversified, deduped, GPU re-ranked
        fused_ranked = fuse_and_rank(
            claim_text=claim_text,
            blocks=ext,
            model=st_model,
            device=device,
            batch_size=args.batch_size,
            k=args.top_k,
            max_per_domain=args.max_per_domain,
            dedup_cos=args.dedup_cos,
            boost_trusted=args.boost_trusted,
            penalize_lowcred=args.penalize_lowcred,
            center_date=center_date,
            time_window_days=args.time_window_days,
            mmr_lambda=0.7,
        )

        # --- Write back (preserve original per-source blocks + add fused)
        data.setdefault("external", {})
        data["external"]["wikipedia"] = ext.get("wikipedia", [])
        data["external"]["fact_check"] = ext.get("fact_check", [])
        data["external"]["google_cse"] = ext.get("google_cse", [])
        # New: ranked + diversified list the Verifier/Explainer can consume directly
        data["external"]["ranked"] = fused_ranked

        # save
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out_dir) / f"{cid}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Augmented claim_id={cid} → {out_path}")

if __name__ == "__main__":
    main()
