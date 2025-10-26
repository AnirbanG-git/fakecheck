from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_splits(df: pd.DataFrame, seed: int = 42, test_size=0.1, val_size=0.1):
    """
    Returns train, val, test DataFrames using stratified splits.
    """
    # First split off test
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    # Now split train/val
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_ratio, random_state=seed, stratify=df_train_val["label"]
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
