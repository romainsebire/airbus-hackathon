"""
Chained CatBoost Regression for multi-output KPI prediction.

Strategy (Regressor Chain):
  Step 1: Predict WIP from base features
  Step 2: Predict Investment from base features + pred_wip
  Step 3: Predict Satisfaction from base features + pred_wip + pred_inv

Best configuration:
  - 12,000 iterations, learning_rate=0.005, depth=8
  - Advanced feature engineering applied
  - 5-Fold CV with early stopping (200 rounds) and test averaging
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import gc
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing.feature_engineering import advanced_feature_engineering


# ═══════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════

CAT_PARAMS = {
    "iterations": 12000,
    "learning_rate": 0.005,
    "depth": 8,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "task_type": "CPU",
    "thread_count": -1,
    "verbose": 1000,
}

N_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING = 200


def train_chain_catboost_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict = None,
    use_feature_engineering: bool = True,
    n_folds: int = N_FOLDS,
):
    """
    Train chained CatBoost with K-Fold CV and return test predictions.

    Returns:
        submission: pd.DataFrame with columns [id, wip, investissement, satisfaction]
        scores: list of per-fold satisfaction accuracy scores
    """
    if params is None:
        params = CAT_PARAMS.copy()

    targets = ["wip", "investissement", "satisfaction"]
    features_base = [c for c in df_train.columns if c not in targets and c != "id"]
    test_ids = df_test["id"] if "id" in df_test.columns else df_test.index

    # Prepare features
    X = df_train[features_base].copy()
    y_wip = df_train["wip"]
    y_inv = df_train["investissement"]
    y_sat = df_train["satisfaction"]
    X_test = df_test[features_base].copy()

    if use_feature_engineering:
        print("Applying advanced feature engineering...")
        len_train = len(X)
        X_all = pd.concat([X, X_test], axis=0)
        X_all = advanced_feature_engineering(X_all)
        X = X_all.iloc[:len_train].reset_index(drop=True)
        X_test = X_all.iloc[len_train:].reset_index(drop=True)
        del X_all
        gc.collect()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    oof_sat = np.zeros(len(X))
    test_pred_wip = np.zeros(len(X_test))
    test_pred_inv = np.zeros(len(X_test))
    test_pred_sat = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        X_te_fold = X_test.copy()

        # Step 1: WIP
        print("  > Training WIP...")
        model_wip = CatBoostRegressor(**params)
        model_wip.fit(
            X_tr, y_wip.iloc[train_idx],
            eval_set=(X_val, y_wip.iloc[val_idx]),
            early_stopping_rounds=EARLY_STOPPING,
            verbose=False,
        )
        wip_tr = model_wip.predict(X_tr)
        wip_val = model_wip.predict(X_val)
        wip_te = model_wip.predict(X_te_fold)

        X_tr["pred_wip"] = wip_tr
        X_val["pred_wip"] = wip_val
        X_te_fold["pred_wip"] = wip_te
        test_pred_wip += wip_te / n_folds

        # Step 2: Investment
        print("  > Training Investment...")
        model_inv = CatBoostRegressor(**params)
        model_inv.fit(
            X_tr, y_inv.iloc[train_idx],
            eval_set=(X_val, y_inv.iloc[val_idx]),
            early_stopping_rounds=EARLY_STOPPING,
            verbose=False,
        )
        inv_tr = model_inv.predict(X_tr)
        inv_val = model_inv.predict(X_val)
        inv_te = model_inv.predict(X_te_fold)

        X_tr["pred_inv"] = inv_tr
        X_val["pred_inv"] = inv_val
        X_te_fold["pred_inv"] = inv_te
        test_pred_inv += inv_te / n_folds

        # Step 3: Satisfaction
        print("  > Training Satisfaction...")
        model_sat = CatBoostRegressor(**params)
        model_sat.fit(
            X_tr, y_sat.iloc[train_idx],
            eval_set=(X_val, y_sat.iloc[val_idx]),
            early_stopping_rounds=EARLY_STOPPING,
            verbose=1000,
        )
        val_preds = model_sat.predict(X_val)
        oof_sat[val_idx] = val_preds

        score = np.mean(np.abs(y_sat.iloc[val_idx] - val_preds) < 0.05)
        scores.append(score)
        print(f"  Fold {fold + 1} Satisfaction Accuracy: {score:.4%}")

        test_pred_sat += model_sat.predict(X_te_fold) / n_folds

        del X_tr, X_val, X_te_fold, model_wip, model_inv, model_sat
        gc.collect()

    print(f"\nFINAL CV SCORE (Chained CatBoost): {np.mean(scores):.4%}")

    submission = pd.DataFrame({
        "id": test_ids,
        "wip": test_pred_wip,
        "investissement": test_pred_inv,
        "satisfaction": test_pred_sat,
    })
    return submission, scores


if __name__ == "__main__":
    print("Loading data...")
    df_train = pd.read_csv("data/clean_train.csv")
    df_test = pd.read_csv("data/clean_test.csv")

    submission, scores = train_chain_catboost_cv(df_train, df_test)

    output_file = "submissions/submission_chain_catboost.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
