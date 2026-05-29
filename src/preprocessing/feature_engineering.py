"""
Advanced Feature Engineering (V2) for the Airbus Hackathon.

Creates ~150+ engineered features from raw demand and param columns:
  - Demand statistics: sum, mean, std, quantiles, skew, kurtosis, etc.
  - Param statistics: global and per-year (year0, year1, year2)
  - Temporal dynamics: year-over-year deltas and acceleration
  - Cross-feature interactions: demand/param ratios and products

Used by the best-performing gradient boosting models.
"""

import re
import numpy as np
import pandas as pd


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply advanced feature engineering to demand/param columns."""
    df = df.copy()

    demand_cols = [c for c in df.columns if c.startswith("demand")]
    param_cols = [c for c in df.columns if c.startswith("param")]

    print(f"Feature Engineering: {len(demand_cols)} demand cols & {len(param_cols)} param cols...")

    # ── 1. Demand Statistics ──
    df["demand_sum"] = df[demand_cols].sum(axis=1).astype(np.float32)
    df["demand_mean"] = df[demand_cols].mean(axis=1).astype(np.float32)
    df["demand_std"] = df[demand_cols].std(axis=1).astype(np.float32)
    df["demand_max"] = df[demand_cols].max(axis=1).astype(np.float32)
    df["demand_min"] = df[demand_cols].min(axis=1).astype(np.float32)
    df["demand_q25"] = df[demand_cols].quantile(0.25, axis=1).astype(np.float32)
    df["demand_median"] = df[demand_cols].median(axis=1).astype(np.float32)
    df["demand_q75"] = df[demand_cols].quantile(0.75, axis=1).astype(np.float32)
    df["demand_range"] = (df["demand_max"] - df["demand_min"]).astype(np.float32)
    df["demand_skew"] = df[demand_cols].skew(axis=1).astype(np.float32)
    df["demand_kurt"] = df[demand_cols].kurt(axis=1).astype(np.float32)
    df["demand_cv"] = (df["demand_std"] / (df["demand_mean"] + 1e-5)).astype(np.float32)
    df["demand_zeros"] = (df[demand_cols] == 0).sum(axis=1).astype(np.float32)

    # ── 2. Param Statistics ──
    # Global stats
    df["param_mean"] = df[param_cols].mean(axis=1).astype(np.float32)
    df["param_std"] = df[param_cols].std(axis=1).astype(np.float32)
    df["param_max"] = df[param_cols].max(axis=1).astype(np.float32)
    df["param_min"] = df[param_cols].min(axis=1).astype(np.float32)

    # Per-year stats
    for year in ["year0", "year1", "year2"]:
        y_cols = [c for c in param_cols if year in c]
        if y_cols:
            df[f"param_{year}_mean"] = df[y_cols].mean(axis=1).astype(np.float32)
            df[f"param_{year}_std"] = df[y_cols].std(axis=1).astype(np.float32)
            df[f"param_{year}_sum"] = df[y_cols].sum(axis=1).astype(np.float32)
            df[f"param_{year}_max"] = df[y_cols].max(axis=1).astype(np.float32)

    # ── 3. Temporal Dynamics (Param year-over-year) ──
    for i in range(17):  # f0 to f16
        f_name = f"f{i}"
        col_y0 = f"param_year0_{f_name}"
        col_y1 = f"param_year1_{f_name}"
        col_y2 = f"param_year2_{f_name}"

        if col_y0 in df.columns and col_y1 in df.columns:
            df[f"delta_y1_y0_{f_name}"] = (df[col_y1] - df[col_y0]).astype(np.float32)

        if col_y1 in df.columns and col_y2 in df.columns:
            df[f"delta_y2_y1_{f_name}"] = (df[col_y2] - df[col_y1]).astype(np.float32)

        delta1 = f"delta_y1_y0_{f_name}"
        delta2 = f"delta_y2_y1_{f_name}"
        if delta1 in df.columns and delta2 in df.columns:
            df[f"accel_{f_name}"] = (df[delta2] - df[delta1]).astype(np.float32)

    # ── 4. Cross-feature Interactions ──
    df["ratio_demand_param_mean"] = df["demand_mean"] / (df["param_mean"] + 1e-5)
    df["prod_demand_param_max"] = df["demand_max"] * df["param_max"]

    print(f"Features generated. Total columns: {df.shape[1]}")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove special characters from column names (required by LightGBM)."""
    df.columns = [re.sub(r'[",:{}\[\]]', "", str(col)) for col in df.columns]
    return df
