"""
Tiny domain adaptation of a SentenceTransformer encoder using the verifier dataset.
"""

import argparse, json, random
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer, losses, InputExample, datasets


def _load_pairs(jsonl_path: Path, max_pairs: int = 20000) -> List[InputExample]:
    ex: List[InputExample] = []
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            text = row.get("input_text") or ""
            parts = text.split(" [SEP] ")
            if not parts:
                continue
            claim = parts[0].replace("[CLAIM]", "").strip()
            if not claim:
                continue
            evs = [p.strip() for p in parts[1:3] if p.strip()]
            for e in evs:
                ex.append(InputExample(texts=[claim, e], label=1.0))
    random.shuffle(ex)
    return ex[:max_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out_dir", default="artifacts/miniLM_welfake")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    pairs = _load_pairs(Path(args.dataset))
    if not pairs:
        raise SystemExit("No pairs built — verify dataset contents and format.")

    model = SentenceTransformer(args.base_model)
    # ✅  Remove shuffle — not supported anymore
    train_dl = datasets.NoDuplicatesDataLoader(pairs, batch_size=args.batch_size)
    loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dl, loss)], epochs=args.epochs, warmup_steps=100)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save(args.out_dir)
    print(f"Saved adapted encoder to: {args.out_dir}")


if __name__ == "__main__":
    main()
