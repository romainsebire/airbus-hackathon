"""
Data cleaning pipeline for the Airbus Hackathon dataset.

Steps:
  1. Remove constant columns (variance = 0) from raw CSV data
  2. Remove highly correlated features (>98% Pearson correlation)
  3. Save cleaned data as memory-efficient Parquet files
"""

import os
import pandas as pd
import numpy as np
import gc


def load_smart_types(filepath: str) -> pd.DataFrame:
    """Load CSV with optimized dtypes for memory efficiency."""
    cols = pd.read_csv(filepath, nrows=0).columns
    dtype_map = {}
    for col in cols:
        if col in ["wip", "investissement", "satisfaction"]:
            dtype_map[col] = "float32"
        elif col == "id":
            dtype_map[col] = "string"
        else:
            dtype_map[col] = "int8"
    return pd.read_csv(filepath, dtype=dtype_map, engine="c")


def remove_constant_columns(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Remove features with zero variance (constant across all samples)."""
    targets = ["wip", "investissement", "satisfaction"]
    features = [c for c in df_train.columns if c not in targets and c != "id"]

    const_cols = [col for col in features if df_train[col].min() == df_train[col].max()]
    print(f"Found {len(const_cols)} constant columns to remove.")

    df_train.drop(columns=const_cols, inplace=True)
    cols_to_drop_test = [c for c in const_cols if c in df_test.columns]
    df_test.drop(columns=cols_to_drop_test, inplace=True)

    return df_train, df_test, const_cols


def remove_correlated_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: float = 0.98,
    sample_size: int = 10_000,
    corr_cache_path: str = None,
):
    """Remove features with Pearson correlation above threshold (computed on sample)."""
    targets = ["wip", "investissement", "satisfaction"]
    features = [c for c in df_train.columns if c not in targets and c != "id"]

    if corr_cache_path and os.path.exists(corr_cache_path):
        print("Loading cached correlation matrix...")
        corr_matrix = pd.read_parquet(corr_cache_path)
    else:
        print(f"Computing correlation on {sample_size:,} sample rows...")
        sample_df = df_train[features].sample(n=min(sample_size, len(df_train)), random_state=42)
        corr_matrix = sample_df.corr().abs()
        if corr_cache_path:
            corr_matrix.to_parquet(corr_cache_path)
        del sample_df
        gc.collect()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Found {len(to_drop)} features correlated > {threshold:.0%}.")

    if to_drop:
        df_train.drop(columns=[c for c in to_drop if c in df_train.columns], inplace=True)
        df_test.drop(columns=[c for c in to_drop if c in df_test.columns], inplace=True)

    return df_train, df_test, to_drop


def clean_dataset(
    train_csv: str,
    test_csv: str,
    output_dir: str = "data",
    corr_threshold: float = 0.98,
):
    """Full cleaning pipeline: load → remove constants → remove correlated → save."""
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)
    df_train = load_smart_types(train_csv)
    df_test = load_smart_types(test_csv)
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Removing constant columns")
    print("=" * 60)
    df_train, df_test, const_cols = remove_constant_columns(df_train, df_test)
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}")

    # Save intermediate cleaned files
    df_train.to_parquet(os.path.join(output_dir, "train_cleaned.parquet"), index=False)
    df_test.to_parquet(os.path.join(output_dir, "test_cleaned.parquet"), index=False)

    print("\n" + "=" * 60)
    print("STEP 3: Removing correlated features")
    print("=" * 60)
    corr_cache = os.path.join(output_dir, "corr_matrix_sample.parquet")
    df_train, df_test, corr_cols = remove_correlated_features(
        df_train, df_test, threshold=corr_threshold, corr_cache_path=corr_cache
    )
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}")

    # Save final uncorrelated files
    df_train.to_parquet(os.path.join(output_dir, "train_uncorrelated.parquet"), index=False)
    df_test.to_parquet(os.path.join(output_dir, "test_uncorrelated.parquet"), index=False)

    print("\nCleaning complete!")
    return df_train, df_test


if __name__ == "__main__":
    clean_dataset("data/train.csv", "data/test.csv")
