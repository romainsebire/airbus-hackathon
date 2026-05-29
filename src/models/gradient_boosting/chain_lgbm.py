"""
Chained LightGBM Regression for multi-output KPI prediction.

Strategy (Regressor Chain):
  Step 1: Predict WIP from base features
  Step 2: Predict Investment from base features + pred_wip
  Step 3: Predict Satisfaction from base features + pred_wip + pred_inv

This captures the natural dependency: Satisfaction depends on WIP & Investment.

Best configuration (used in winning submission):
  - 12,000 iterations, learning_rate=0.005
  - Advanced feature engineering applied
  - 5-Fold CV with test prediction averaging
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import gc
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing.feature_engineering import advanced_feature_engineering, clean_column_names


# ═══════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 12000,
    "learning_rate": 0.005,
    "num_leaves": 63,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "n_jobs": -1,
    "verbosity": 1,
    "force_col_wise": True,
}

N_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING = 100


def train_chain_lgbm_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict = None,
    use_feature_engineering: bool = True,
    n_folds: int = N_FOLDS,
):
    """
    Train chained LightGBM with K-Fold CV and return test predictions.

    Returns:
        submission: pd.DataFrame with columns [id, wip, investissement, satisfaction]
        scores: list of per-fold satisfaction accuracy scores
    """
    if params is None:
        params = PARAMS.copy()

    targets = ["wip", "investissement", "satisfaction"]
    features_base = [c for c in df_train.columns if c not in targets and c != "id"]

    test_ids = df_test["id"] if "id" in df_test.columns else df_test.index

    # Apply feature engineering
    if use_feature_engineering:
        print("Applying advanced feature engineering...")
        len_train = len(df_train)
        X_all = pd.concat([df_train[features_base], df_test[features_base]], axis=0)
        X_all = advanced_feature_engineering(X_all)
        X_all = clean_column_names(X_all)

        X_train_full = X_all.iloc[:len_train].reset_index(drop=True)
        X_test_full = X_all.iloc[len_train:].reset_index(drop=True)
        del X_all
        gc.collect()
    else:
        X_train_full = clean_column_names(df_train[features_base].copy())
        X_test_full = clean_column_names(df_test[features_base].copy())

    y_wip = df_train["wip"]
    y_inv = df_train["investissement"]
    y_sat = df_train["satisfaction"]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    oof_preds = np.zeros(len(X_train_full))

    pred_wip_test = np.zeros(len(X_test_full))
    pred_inv_test = np.zeros(len(X_test_full))
    pred_sat_test = np.zeros(len(X_test_full))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_tr = X_train_full.iloc[train_idx]
        X_val = X_train_full.iloc[val_idx]

        # Step 1: WIP
        dtrain = lgb.Dataset(X_tr, label=y_wip.iloc[train_idx])
        dval = lgb.Dataset(X_val, label=y_wip.iloc[val_idx], reference=dtrain)
        model_wip = lgb.train(
            params, dtrain, valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(2000)],
        )
        wip_pred_tr = model_wip.predict(X_tr)
        wip_pred_val = model_wip.predict(X_val)
        wip_pred_test_fold = model_wip.predict(X_test_full)
        pred_wip_test += wip_pred_test_fold / n_folds
        del dtrain, dval, model_wip
        gc.collect()

        # Step 2: Investment
        dtrain = lgb.Dataset(X_tr, label=y_inv.iloc[train_idx])
        dval = lgb.Dataset(X_val, label=y_inv.iloc[val_idx], reference=dtrain)
        model_inv = lgb.train(
            params, dtrain, valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(2000)],
        )
        inv_pred_tr = model_inv.predict(X_tr)
        inv_pred_val = model_inv.predict(X_val)
        inv_pred_test_fold = model_inv.predict(X_test_full)
        pred_inv_test += inv_pred_test_fold / n_folds
        del dtrain, dval, model_inv
        gc.collect()

        # Step 3: Satisfaction (with chained predictions)
        X_tr_aug = X_tr.copy()
        X_val_aug = X_val.copy()
        X_test_fold = X_test_full.copy()

        X_tr_aug["pred_wip"] = wip_pred_tr
        X_tr_aug["pred_inv"] = inv_pred_tr
        X_val_aug["pred_wip"] = wip_pred_val
        X_val_aug["pred_inv"] = inv_pred_val
        X_test_fold["pred_wip"] = wip_pred_test_fold
        X_test_fold["pred_inv"] = inv_pred_test_fold

        dtrain = lgb.Dataset(X_tr_aug, label=y_sat.iloc[train_idx])
        dval = lgb.Dataset(X_val_aug, label=y_sat.iloc[val_idx], reference=dtrain)
        model_sat = lgb.train(
            params, dtrain, valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(2000)],
        )
        val_preds = model_sat.predict(X_val_aug)
        oof_preds[val_idx] = val_preds
        pred_sat_test += model_sat.predict(X_test_fold) / n_folds

        score = np.mean(np.abs(y_sat.iloc[val_idx] - val_preds) < 0.05)
        scores.append(score)
        print(f"Fold {fold + 1} Satisfaction Accuracy: {score:.4%}")

        del X_tr_aug, X_val_aug, X_test_fold, dtrain, dval, model_sat
        gc.collect()

    print(f"\nFINAL CV SCORE (Chained LightGBM): {np.mean(scores):.4%}")

    submission = pd.DataFrame({
        "id": test_ids,
        "wip": pred_wip_test,
        "investissement": pred_inv_test,
        "satisfaction": pred_sat_test,
    })
    return submission, scores


if __name__ == "__main__":
    print("Loading data...")
    df_train = pd.read_csv("data/clean_train.csv")
    df_test = pd.read_csv("data/clean_test.csv")

    submission, scores = train_chain_lgbm_cv(df_train, df_test)

    output_file = "submissions/submission_chain_lgbm.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
