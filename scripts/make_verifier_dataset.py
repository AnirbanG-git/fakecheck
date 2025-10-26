"""
Verifier dataset builder — compatible with cache schema:
  - claim_id / claim_label / claim_text
  - retrieval.bm25[*].text  (used as evidence)

It writes JSONL rows: { "id", "input_text", "label", "split" }
where input_text := "[CLAIM] {claim} [SEP] ev1 [SEP] ev2 ..."

Features
- Recursively reads --cache_dir for .json and .jsonl
- Accepts legacy keys too, but optimized for:
    claim_id / claim_label / claim_text
    retrieval.bm25[*].text
- Optional semantic filtering with --min_semantic_cos and --encoder
- Stratified train/val/test splits
- Verbose drop diagnostics

Author: you + Week-4
"""
import argparse, json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

SEP = " [SEP] "


# ---------------- I/O helpers ---------------- #
def _iter_cache_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
            yield p


def _read_json_any(p: Path) -> List[Dict[str, Any]]:
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows
    else:
        try:
            obj = json.loads(p.read_text())
        except Exception:
            return []
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        return []


# ---------------- Schema adapters (your cache) ---------------- #
def _get(entry: Dict[str, Any], key: str, default=None):
    """Dot-path get."""
    cur = entry
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _extract_claim(entry: Dict[str, Any]) -> str:
    # Your schema first
    claim = entry.get("claim_text")
    if isinstance(claim, str) and claim.strip():
        return claim.strip()

    # Fallbacks for robustness
    for k in ("claim", "statement", "text", "title"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_label(entry: Dict[str, Any]) -> Optional[int]:
    # schema first: claim_label
    v = entry.get("claim_label")
    if v in (0, 1):
        return int(v)
    if isinstance(v, bool):
        return int(v)
    try:
        iv = int(v)
        if iv in (0, 1):
            return iv
    except Exception:
        pass

    # Fallbacks
    v = entry.get("label")
    if v in (0, 1):
        return int(v)
    if isinstance(v, bool):
        return int(v)
    try:
        iv = int(v)
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    return None


def _extract_id(entry: Dict[str, Any], fp: Path) -> str:
    v = entry.get("claim_id") or entry.get("id") or fp.stem
    return str(v)


def _extract_evidence(entry: Dict[str, Any], top_k_bm25: int) -> List[str]:
    """Use retrieval.bm25[*].text as primary evidence."""
    texts: List[str] = []

    # Your schema: retrieval.bm25[*].text
    bm25_list = _get(entry, "retrieval.bm25", default=[])
    if isinstance(bm25_list, list):
        for it in bm25_list:
            if isinstance(it, dict):
                tv = it.get("text") or it.get("snippet") or it.get("passage") or it.get("content")
                if isinstance(tv, str) and tv.strip():
                    texts.append(tv.strip())
                    if len(texts) >= top_k_bm25:
                        break

    # Fallbacks (just in case)
    if not texts:
        # external.ranked style
        ext_ranked = _get(entry, "external.ranked", default=[]) or entry.get("external_ranked") or _get(entry, "external.ranked[]", default=[])
        if isinstance(ext_ranked, list):
            for it in ext_ranked:
                if isinstance(it, dict):
                    tv = it.get("snippet") or it.get("text") or it.get("content") or it.get("passage")
                    if isinstance(tv, str) and tv.strip():
                        texts.append(tv.strip())

    # Deduplicate keep order
    seen, out = set(), []
    for t in texts:
        tt = t.strip()
        if tt and tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out


# --------------- Semantic filtering --------------- #
def _semantic_filter(claim: str, evidences: List[str], encoder_name: str, min_cos: float) -> List[str]:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np  # noqa
    except Exception:
        # If encoder not available, return original list
        return evidences
    model = SentenceTransformer(encoder_name)
    ce = model.encode([claim], normalize_embeddings=True)
    ee = model.encode(evidences, normalize_embeddings=True)
    import numpy as np
    cos = (ee @ ce.T).reshape(-1)
    keep = [t for t, c in zip(evidences, cos) if float(c) >= float(min_cos)]
    return keep


# ---------------- Main ---------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)

    ap.add_argument("--top_k", type=int, default=5, help="max bm25 evidence to include")
    ap.add_argument("--top_m_external", type=int, default=0, help="ignored (backward compat)")

    ap.add_argument("--max_chars", type=int, default=4000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)

    ap.add_argument("--min_semantic_cos", type=float, default=None)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--strict_semantic_filter", action="store_true")

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    cache_root = Path(args.cache_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(_iter_cache_files(cache_root))
    if args.verbose:
        print(f"[INFO] scanning {len(files)} files under {cache_root}")

    rows: List[Dict[str, Any]] = []
    drops = {"bad_json":0, "no_claim":0, "no_label":0, "no_evidence":0, "empty_after_filter":0}

    for fp in files:
        docs = _read_json_any(fp)
        if not docs:
            drops["bad_json"] += 1
            continue
        for entry in docs:
            cid = _extract_id(entry, fp)
            claim = _extract_claim(entry)
            if not claim:
                drops["no_claim"] += 1
                continue

            label = _extract_label(entry)
            if label is None:
                drops["no_label"] += 1
                continue

            ev = _extract_evidence(entry, args.top_k)

            if args.min_semantic_cos is not None and ev:
                kept = _semantic_filter(claim, ev, args.encoder, args.min_semantic_cos)
                if kept:
                    ev = kept
                else:
                    if args.strict_semantic_filter:
                        drops["empty_after_filter"] += 1
                        continue
                    # fallback to unfiltered if strict not requested

            if not ev:
                drops["no_evidence"] += 1
                continue

            text = f"[CLAIM] {claim}"
            if ev:
                text += SEP + SEP.join(ev)
            if args.max_chars and args.max_chars > 0:
                text = text[: args.max_chars]

            rows.append({"id": str(cid), "input_text": text, "label": int(label)})

    if not rows:
        if args.verbose:
            print("[ERROR] Built 0 rows.")
            print("[DROPS]", drops)
            print("[HINT] Since your schema is claim_text/claim_label + retrieval.bm25, "
                  "ensure those keys exist in each JSON file. Try running without semantic filter.")
        raise SystemExit("No rows built — check cache_dir and inputs.")

    df = pd.DataFrame(rows)

    # Splits (stratified if possible)
    strat = df["label"] if df["label"].nunique() > 1 else None
    test_size = max(0.05, min(0.4, float(args.test_size)))
    val_size  = max(0.05, min(0.4, float(args.val_size)))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=args.seed, stratify=strat)
    strat_tr = train_df["label"] if train_df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=args.seed, stratify=strat_tr)

    train_df = train_df.assign(split="train")
    val_df   = val_df.assign(split="validation")
    test_df  = test_df.assign(split="test")
    out_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    with out_path.open("w") as f:
        for _, r in out_df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(out_df)} rows → {out_path}")
    print("[SPLITS]", out_df["split"].value_counts().to_dict())
    print("[LABELS]", out_df["label"].value_counts().to_dict())
    if args.verbose:
        print("[DROPS]", drops)


if __name__ == "__main__":
    main()
