import argparse, json
from collections import Counter
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Basic summary
    n_rows = len(df)
    cols = list(df.columns)
    label_counts = Counter(df.get("label", []))

    text_lens = df.get("text", pd.Series([], dtype=str)).fillna("").map(lambda s: len(str(s).split()))
    title_lens = df.get("title", pd.Series([], dtype=str)).fillna("").map(lambda s: len(str(s).split()))

    out = {
        "rows": n_rows,
        "columns": cols,
        "label_counts": {str(k): int(v) for k,v in label_counts.items()},
        "text_tokens_mean": float(text_lens.mean()) if len(text_lens)>0 else None,
        "text_tokens_median": float(text_lens.median()) if len(text_lens)>0 else None,
        "title_tokens_mean": float(title_lens.mean()) if len(title_lens)>0 else None,
        "title_tokens_median": float(title_lens.median()) if len(title_lens)>0 else None,
    }

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
