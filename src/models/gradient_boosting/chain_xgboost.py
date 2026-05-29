"""
Chained XGBoost Regression for multi-output KPI prediction.

Strategy (Regressor Chain):
  Step 1: Predict WIP from base features
  Step 2: Predict Investment from base features + pred_wip
  Step 3: Predict Satisfaction from base features + pred_wip + pred_inv

Best configuration:
  - 12,000 rounds, learning_rate=0.01, max_depth=8
  - Advanced feature engineering applied
  - 5-Fold CV with early stopping (100 rounds) and test averaging
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import gc
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing.feature_engineering import advanced_feature_engineering, clean_column_names


# ═══════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════

PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "n_jobs": -1,
    "verbosity": 1,
    "tree_method": "hist",
}

NUM_ROUNDS = 12000
N_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING = 100


def train_chain_xgb_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict = None,
    num_rounds: int = NUM_ROUNDS,
    use_feature_engineering: bool = True,
    n_folds: int = N_FOLDS,
):
    """
    Train chained XGBoost with K-Fold CV and return test predictions.

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

        X_train_full = X_all.iloc[:len_train].reset_index(drop=True)
        X_test_full = X_all.iloc[len_train:].reset_index(drop=True)
        del X_all
        gc.collect()
    else:
        X_train_full = df_train[features_base].copy()
        X_test_full = df_test[features_base].copy()

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

        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]

        # Step 1: WIP
        dtrain = xgb.DMatrix(X_tr, label=y_wip.iloc[train_idx])
        dval = xgb.DMatrix(X_val, label=y_wip.iloc[val_idx])
        dtest = xgb.DMatrix(X_test_full)

        model_wip = xgb.train(
            params, dtrain, num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=EARLY_STOPPING, verbose_eval=2000,
        )
        wip_pred_tr = model_wip.predict(dtrain)
        wip_pred_val = model_wip.predict(dval)
        wip_pred_test_fold = model_wip.predict(dtest)
        pred_wip_test += wip_pred_test_fold / n_folds
        del dtrain, dval, dtest, model_wip
        gc.collect()

        # Step 2: Investment
        dtrain = xgb.DMatrix(X_tr, label=y_inv.iloc[train_idx])
        dval = xgb.DMatrix(X_val, label=y_inv.iloc[val_idx])
        dtest = xgb.DMatrix(X_test_full)

        model_inv = xgb.train(
            params, dtrain, num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=EARLY_STOPPING, verbose_eval=2000,
        )
        inv_pred_tr = model_inv.predict(dtrain)
        inv_pred_val = model_inv.predict(dval)
        inv_pred_test_fold = model_inv.predict(dtest)
        pred_inv_test += inv_pred_test_fold / n_folds
        del dtrain, dval, dtest, model_inv
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

        dtrain = xgb.DMatrix(X_tr_aug, label=y_sat.iloc[train_idx])
        dval = xgb.DMatrix(X_val_aug, label=y_sat.iloc[val_idx])
        dtest = xgb.DMatrix(X_test_fold)

        model_sat = xgb.train(
            params, dtrain, num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=EARLY_STOPPING, verbose_eval=2000,
        )
        val_preds = model_sat.predict(dval)
        oof_preds[val_idx] = val_preds
        pred_sat_test += model_sat.predict(dtest) / n_folds

        score = np.mean(np.abs(y_sat.iloc[val_idx] - val_preds) < 0.05)
        scores.append(score)
        print(f"Fold {fold + 1} Satisfaction Accuracy: {score:.4%}")

        del X_tr_aug, X_val_aug, X_test_fold, dtrain, dval, dtest, model_sat
        gc.collect()

    print(f"\nFINAL CV SCORE (Chained XGBoost): {np.mean(scores):.4%}")

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

    submission, scores = train_chain_xgb_cv(df_train, df_test)

    output_file = "submissions/submission_chain_xgboost.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
