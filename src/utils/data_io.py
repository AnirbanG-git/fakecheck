from pathlib import Path
import pandas as pd

def load_welfake(csv_path: str) -> pd.DataFrame:
    """
    Load WELFake CSV and normalize label to int {0,1}.
    Expected columns: at least ["title","text","label"].
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset at {csv_path}")
    df = pd.read_csv(p)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")
    df["label"] = df["label"].astype(int)
    # Fill NaNs for text fields
    for col in ("title","text"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df
