"""
Compute external coverage over cached claims.

Modes:
1) Legacy single-threshold mode (backward compatible):
   - If --semantic_filter is ON: semantic coverage at --cos_thresh
   - If --semantic_filter is OFF: coverage = presence of any items (no cosine)
   Output fields (same as before): coverage_external, cos_thresh, etc.

2) Threshold sweep mode (new):
   - Use --cos_sweep "0.20,0.25,0.30,0.35"
   - Always computes semantic coverage (cosine) per threshold
   - Writes coverage_by_threshold (and optionally per_source_by_threshold)

Other features preserved:
- CSV fallback for claim text via src.utils.data_io.load_welfake
- Sampling from cache or whole dataset
- prefer_ranked / per_source breakdown
- GPU acceleration with torch + SentenceTransformers
- Efficient batched encoding with autocast on CUDA
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.data_io import load_welfake


# --------------------------
# Helpers (IO, sampling, etc.)
# --------------------------

def _gather_cache_ids(cache_dir: Path) -> List[int]:
    ids: List[int] = []
    for p in cache_dir.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except Exception:
            pass
    ids.sort()
    return ids


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_claim_text(cid: int, rec: Dict[str, Any], df_row_fallback: Optional[str]) -> str:
    # Prefer claim_text in cache; else fallback to dataframe row (title + text).
    ct = rec.get("claim_text")
    if isinstance(ct, str) and ct.strip():
        return ct
    return df_row_fallback or ""


def _concat_title_snippet(it: Dict[str, Any]) -> str:
    return ((it.get("title", "") or "") + " " + (it.get("snippet", "") or "")).strip()


def _collect_items(rec: Dict[str, Any], prefer_ranked: bool) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Returns:
      - items (either ranked-only or union across sources)
      - per-source dict {source_name: list_of_items}
    """
    ext = rec.get("external") or {}
    ranked = ext.get("ranked") or []
    wikipedia = ext.get("wikipedia") or []
    google_cse = ext.get("google_cse") or []
    fact_check = ext.get("fact_check") or []

    per_src = {
        "wikipedia": wikipedia,
        "google_cse": google_cse,
        "fact_check": fact_check,
        "ranked": ranked,
    }

    if prefer_ranked:
        items = ranked
    else:
        items = ranked + wikipedia + google_cse + fact_check

    return items, per_src


# --------------------------
# Encoding / Similarity utils
# --------------------------

def _encode(model, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # dimension isn't used if empty
    with torch.inference_mode():
        if device == "cuda":
            from torch import amp
            with amp.autocast("cuda"):
                X = model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
        else:
            X = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
    return np.asarray(X, dtype=np.float32)


def _covered_at_threshold(q_vec: np.ndarray, X: np.ndarray, th: float) -> bool:
    if X.shape[0] == 0:
        return False
    sims = (X @ q_vec.T).reshape(-1)
    return bool((sims >= th).any())


# --------------------------
# Main computation
# --------------------------

def _prepare_ids_and_fallbacks(
    args,
    cache_dir: Path,
    df: "pd.DataFrame",
) -> Tuple[List[int], Dict[int, str], int]:
    """
    Returns:
      - ids to examine
      - fallback text per id (title+text)
      - count of missing cache files (initialized at 0; updated later)
    """
    # Determine which IDs to evaluate
    if args.sample_from_cache:
        all_cache_ids = _gather_cache_ids(cache_dir)
        rng = random.Random(args.seed)
        rng.shuffle(all_cache_ids)
        ids = all_cache_ids[: min(args.sample, len(all_cache_ids))]
        # Fallbacks only for ids within df range
        fallback_text = {}
        if len(df) > 0:
            df_fallback = (df.get("title", "") + " " + df.get("text", "")).astype(str)
            for cid in ids:
                fallback_text[cid] = df_fallback.iloc[cid] if cid < len(df_fallback) else ""
        return ids, fallback_text, 0
    else:
        # Sample from the CSV indices
        n_rows = len(df)
        ids = list(range(n_rows))
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        ids = ids[: args.sample]
        df_fallback = (df.get("title", "") + " " + df.get("text", "")).astype(str)
        fallback_text = {i: df_fallback.iloc[i] for i in ids}
        return ids, fallback_text, 0


def _single_threshold_coverage(
    args,
    ids: List[int],
    fallback_text: Dict[int, str],
    cache_dir: Path,
) -> Dict[str, Any]:
    """
    Legacy single-threshold (or non-semantic) mode.
    Matches your previous JSON structure to avoid breaking callers.
    """
    examined = 0
    missing_files = 0
    covered = 0

    # per-source coverage (optional)
    per_src_names = ("wikipedia", "google_cse", "fact_check")
    per_src_cov = {s: 0 for s in per_src_names} if args.per_source else None

    # Optional ST encoder
    st_model = None
    device = None
    if args.semantic_filter:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
        st_model = SentenceTransformer(args.dense_model, device=device)

    for cid in ids:
        p = cache_dir / f"{cid}.json"
        if not p.exists():
            missing_files += 1
            continue

        rec = _read_json(p)
        claim_text = _load_claim_text(cid, rec, fallback_text.get(cid, ""))

        items, per_src = _collect_items(rec, args.prefer_ranked)

        if args.semantic_filter:
            # Build candidate sentence list: title + snippet (like your existing logic)
            cand_texts = [_concat_title_snippet(it) for it in items if _concat_title_snippet(it)]
            if not cand_texts or not claim_text.strip():
                ok = False
            else:
                q = _encode(st_model, [claim_text], device, args.batch_size)  # [1, d]
                X = _encode(st_model, cand_texts, device, args.batch_size)    # [N, d]
                ok = _covered_at_threshold(q, X, args.cos_thresh)
        else:
            # Non-semantic: coverage if any item exists
            ok = len(items) > 0

        covered += int(ok)
        examined += 1

        if args.per_source:
            for s in per_src_names:
                s_items = per_src.get(s) or []
                if args.semantic_filter:
                    s_texts = [_concat_title_snippet(it) for it in s_items if _concat_title_snippet(it)]
                    if s_texts and claim_text.strip():
                        q = _encode(st_model, [claim_text], device, args.batch_size)
                        X = _encode(st_model, s_texts, device, args.batch_size)
                        ps_ok = _covered_at_threshold(q, X, args.cos_thresh)
                    else:
                        ps_ok = False
                else:
                    ps_ok = len(s_items) > 0
                per_src_cov[s] += int(ps_ok)

    coverage = (covered / examined) if examined else 0.0
    out: Dict[str, Any] = {
        "sample": args.sample,
        "examined": examined,
        "missing_cache_files": missing_files,
        "prefer_ranked": bool(args.prefer_ranked),
        "semantic_filter": bool(args.semantic_filter),
        "cos_thresh": args.cos_thresh if args.semantic_filter else None,
        "coverage_external": round(coverage, 4),
        "device": device if args.semantic_filter else None,
        "dense_model": args.dense_model if args.semantic_filter else None,
        "sample_from_cache": bool(args.sample_from_cache),
    }
    if args.per_source and examined:
        out["per_source"] = {s: round(per_src_cov[s] / examined, 4) for s in per_src_names}
    return out


def _sweep_thresholds(
    args,
    ids: List[int],
    fallback_text: Dict[int, str],
    cache_dir: Path,
    thresholds: List[float],
) -> Dict[str, Any]:
    """
    Threshold sweep mode:
    - Forces semantic evaluation (cosine) for each threshold.
    - Produces overall coverage_by_threshold and optional per_source_by_threshold.
    """
    from sentence_transformers import SentenceTransformer

    # Init encoder
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    st_model = SentenceTransformer(args.dense_model, device=device)

    # Accumulators
    examined = 0
    missing_files = 0
    covered_counts = {str(t): 0 for t in thresholds}

    per_src_names = ("wikipedia", "google_cse", "fact_check")
    per_src_counts = {s: {str(t): 0 for t in thresholds} for s in per_src_names} if args.per_source else None

    for cid in ids:
        p = cache_dir / f"{cid}.json"
        if not p.exists():
            missing_files += 1
            continue

        rec = _read_json(p)
        claim_text = _load_claim_text(cid, rec, fallback_text.get(cid, ""))

        items, per_src = _collect_items(rec, args.prefer_ranked)
        cand_texts = [_concat_title_snippet(it) for it in items if _concat_title_snippet(it)]

        if not cand_texts or not claim_text.strip():
            continue

        q = _encode(st_model, [claim_text], device, args.batch_size)  # [1, d]
        X = _encode(st_model, cand_texts, device, args.batch_size)    # [N, d]
        examined += 1

        for th in thresholds:
            if _covered_at_threshold(q, X, float(th)):
                covered_counts[str(th)] += 1

        if args.per_source:
            for s in per_src_names:
                s_items = per_src.get(s) or []
                s_texts = [_concat_title_snippet(it) for it in s_items if _concat_title_snippet(it)]
                if not s_texts:
                    continue
                Xs = _encode(st_model, s_texts, device, args.batch_size)
                for th in thresholds:
                    if _covered_at_threshold(q, Xs, float(th)):
                        per_src_counts[s][str(th)] += 1

    # Normalize to coverage
    coverage_by_threshold = {k: (v / examined if examined else 0.0) for k, v in covered_counts.items()}
    out: Dict[str, Any] = {
        "sample": args.sample,
        "examined": examined,
        "missing_cache_files": missing_files,
        "prefer_ranked": bool(args.prefer_ranked),
        "semantic_filter": True,  # sweep always uses semantics
        "dense_model": args.dense_model,
        "device": device,
        "sample_from_cache": bool(args.sample_from_cache),
        "thresholds": thresholds,
        "coverage_by_threshold": {k: round(v, 4) for k, v in coverage_by_threshold.items()},
        "count_thresholds": len(thresholds),
    }
    if args.per_source and examined:
        out["per_source_by_threshold"] = {
            s: {k: round(v / examined, 4) for k, v in per_src_counts[s].items()} for s in per_src_names
        }
    return out


# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute external evidence coverage over cached claims.")
    ap.add_argument("--csv", required=True, help="Path to WELFake CSV (used for fallback claim texts).")
    ap.add_argument("--sample", type=int, default=200, help="How many claims to evaluate.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    ap.add_argument("--cache_dir", default="evidence_cache", help="Directory with <id>.json cache files.")
    ap.add_argument("--out", default="reports/coverage_external.json", help="Where to write the JSON report.")
    ap.add_argument("--prefer_ranked", action="store_true",
                    help="Count coverage using external.ranked only; otherwise union of all external sources.")
    ap.add_argument("--sample_from_cache", action="store_true",
                    help="Sample IDs from existing cache files instead of the full dataset.")

    # Semantic / model options
    ap.add_argument("--semantic_filter", action="store_true",
                    help="Single-threshold mode only: require cosine >= cos_thresh.")
    ap.add_argument("--cos_thresh", type=float, default=0.25, help="Cosine threshold for semantic_filter.")
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model for semantic filtering.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)

    # Per-source breakdowns
    ap.add_argument("--per_source", action="store_true",
                    help="Also report per-source coverage (wikipedia / google_cse / fact_check).")

    # NEW: threshold sweep
    ap.add_argument("--cos_sweep", type=str, default=None,
                    help="Comma-separated thresholds to sweep, e.g., '0.20,0.25,0.30,0.35'.")

    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset for fallback texts
    df = load_welfake(args.csv).reset_index(drop=True)

    # Prepare IDs + fallback texts
    ids, fallback_text, _ = _prepare_ids_and_fallbacks(args, cache_dir, df)

    # Decide mode
    if args.cos_sweep:
        thresholds = [float(x) for x in args.cos_sweep.split(",") if x.strip()]
        result = _sweep_thresholds(args, ids, fallback_text, cache_dir, thresholds)
    else:
        result = _single_threshold_coverage(args, ids, fallback_text, cache_dir)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
