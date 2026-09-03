"""
Temporal train/validation/test split.

Random splits leak information in sequential data: a model trained on a
random 80% of rows can see events chronologically AFTER events in its
test set, effectively "seeing the future" through population-level drift
signals it shouldn't have access to yet. We split strictly by time
instead: train on the earliest 60%, validate on the next 20%, test on
the most recent 20%. The concept-drift window (see generate_data.py)
deliberately falls late in the timeline so it appears mostly in
validation/test, not training -- this is what lets us evaluate how the
system behaves on drift it has NOT seen.
"""
import pandas as pd


def temporal_split(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    return train, val, test
