"""
Weighted Ensemble Averaging — the winning strategy.

Computes a (weighted) average of predictions from multiple model submissions.
This was the key to winning the hackathon: combining diverse models
(LightGBM, XGBoost, CatBoost, MLP) reduces individual model variance
and produces more robust predictions.

Usage:
    python src/ensemble/average_predictions.py
"""

import pandas as pd
import numpy as np
import sys
import os


def average_submissions(
    filepaths: list[str],
    weights: list[float] = None,
    output_path: str = "submissions/submission_ensemble_avg.csv",
) -> pd.DataFrame:
    """
    Compute weighted average of multiple submission CSV files.

    Args:
        filepaths: list of paths to submission CSVs
        weights: optional weights for each file (uniform if None)
        output_path: where to save the averaged submission

    Returns:
        Averaged submission DataFrame
    """
    if weights is None:
        weights = [1.0 / len(filepaths)] * len(filepaths)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    assert len(filepaths) == len(weights), "Number of files must match number of weights"

    pred_cols = ["wip", "investissement", "satisfaction"]

    # Load first file for structure
    df_ensemble = pd.read_csv(filepaths[0])

    # Ensure consistent column order
    for col in pred_cols:
        if col not in df_ensemble.columns:
            raise ValueError(f"Column '{col}' not found in {filepaths[0]}")

    # Initialize prediction columns to 0
    for col in pred_cols:
        df_ensemble[col] = 0.0

    # Weighted sum
    for filepath, weight in zip(filepaths, weights):
        print(f"  + {filepath} (weight={weight:.3f})")
        df_temp = pd.read_csv(filepath)

        if not df_temp["id"].equals(df_ensemble["id"]):
            raise ValueError(f"IDs in {filepath} don't match the first file!")

        for col in pred_cols:
            df_ensemble[col] += df_temp[col] * weight

    # Save
    df_ensemble.to_csv(output_path, index=False)
    print(f"\nEnsemble saved to: {output_path}")
    print(df_ensemble.head())

    return df_ensemble


if __name__ == "__main__":
    # These are the submissions used in the winning ensemble
    files = [
        "submissions/submission_ensemble_4006.csv",
        "submissions/submission_chained_xgboost_10000rounds.csv",
        "submissions/submission_chained_lgbm_10000estimators.csv",
        "submissions/submission_chained_lgbm_3params.csv",
    ]

    # Uniform averaging (can be adjusted to weighted)
    average_submissions(files)
