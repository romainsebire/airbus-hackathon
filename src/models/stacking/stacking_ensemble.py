"""
Two-level Stacking Ensemble for multi-output KPI prediction.

Architecture:
  Level 1 (Experts):
    - Expert 1: CatBoost trained on demand-only features
    - Expert 2: CatBoost trained on param-only features
    Both produce out-of-fold predictions via 5-Fold CV.

  Level 2 (Meta-learner):
    - LightGBM/XGBoost RegressorChain trained on expert predictions
    - Combines demand and param pathway outputs

CatBoost config: iterations=5000, lr=0.025, depth=8, early_stopping=150
Meta-learner config: LightGBM n_est=3000, lr=0.015, num_leaves=200
"""

import pandas as pd
import numpy as np
import warnings
import time

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.multioutput import RegressorChain

warnings.filterwarnings("ignore")


TARGET_COLS = ["wip", "investissement", "satisfaction"]
TOL = 0.05


def accuracy_tolerance_score(y_true, y_pred, tol=TOL):
    """Percentage of predictions within ±tol of true value."""
    return np.mean(np.abs(y_true - y_pred) <= tol)


def make_base_estimator(seed=42):
    """Build a strong meta-learner (LightGBM preferred, XGBoost fallback)."""
    if _HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.015, num_leaves=200,
            max_depth=-1, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.2, reg_lambda=1.5, min_child_samples=8,
            random_state=seed, objective="mae", n_jobs=-1,
        )
    return xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.018, max_depth=8,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
        reg_alpha=0.15, reg_lambda=1.3, objective="reg:absoluteerror",
        n_jobs=-1, random_state=seed,
    )


def train_stacking(
    X_demand, X_param, y, X_test_demand, X_test_param, test_ids,
    n_folds=5, cat_params=None,
):
    """
    Train two-level stacking ensemble.

    Returns:
        submission: pd.DataFrame with columns [id, wip, investissement, satisfaction]
        final_score: estimated CV accuracy score
    """
    if cat_params is None:
        cat_params = {
            "iterations": 5000,
            "learning_rate": 0.025,
            "depth": 8,
            "loss_function": "MAE",
            "random_strength": 0.8,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 0.8,
            "task_type": "CPU",
            "early_stopping_rounds": 150,
            "verbose": 200,
            "random_seed": 42,
        }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds_demand = np.zeros(y.shape)
    oof_preds_param = np.zeros(y.shape)
    test_preds_demand = np.zeros((len(test_ids), 3))
    test_preds_param = np.zeros((len(test_ids), 3))

    base_estimator = make_base_estimator(seed=42)
    scores_demand, scores_param = [], []

    print("Training Level-1 Experts...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_demand, y)):
        fold_start = time.time()
        print(f"  Fold {fold + 1}/{n_folds}...")

        if _HAS_CAT:
            # Expert 1: Demand
            for t, col in enumerate(TARGET_COLS):
                model = CatBoostRegressor(**{**cat_params, "random_seed": 42 + fold * 10 + t})
                model.fit(
                    X_demand.iloc[train_idx], y.iloc[train_idx][col],
                    eval_set=(X_demand.iloc[val_idx], y.iloc[val_idx][col]),
                    verbose_eval=False,
                )
                oof_preds_demand[val_idx, t] = model.predict(X_demand.iloc[val_idx])
                test_preds_demand[:, t] += model.predict(X_test_demand) / n_folds

            # Expert 2: Param
            for t, col in enumerate(TARGET_COLS):
                model = CatBoostRegressor(**{**cat_params, "random_seed": 84 + fold * 10 + t})
                model.fit(
                    X_param.iloc[train_idx], y.iloc[train_idx][col],
                    eval_set=(X_param.iloc[val_idx], y.iloc[val_idx][col]),
                    verbose_eval=False,
                )
                oof_preds_param[val_idx, t] = model.predict(X_param.iloc[val_idx])
                test_preds_param[:, t] += model.predict(X_test_param) / n_folds
        else:
            model_d = RegressorChain(base_estimator, order=[0, 1, 2])
            model_d.fit(X_demand.iloc[train_idx], y.iloc[train_idx])
            oof_preds_demand[val_idx] = model_d.predict(X_demand.iloc[val_idx])
            test_preds_demand += model_d.predict(X_test_demand) / n_folds

            model_p = RegressorChain(base_estimator, order=[0, 1, 2])
            model_p.fit(X_param.iloc[train_idx], y.iloc[train_idx])
            oof_preds_param[val_idx] = model_p.predict(X_param.iloc[val_idx])
            test_preds_param += model_p.predict(X_test_param) / n_folds

        s_d = accuracy_tolerance_score(y.iloc[val_idx]["satisfaction"], oof_preds_demand[val_idx, 2])
        s_p = accuracy_tolerance_score(y.iloc[val_idx]["satisfaction"], oof_preds_param[val_idx, 2])
        scores_demand.append(s_d)
        scores_param.append(s_p)
        elapsed = time.time() - fold_start
        print(f"    Demand Acc={s_d:.2%} | Param Acc={s_p:.2%} | {elapsed / 60:.1f} min")

    print(f"  Avg Demand Expert: {np.mean(scores_demand):.2%}")
    print(f"  Avg Param Expert:  {np.mean(scores_param):.2%}")

    # Level 2: Meta-learner
    print("\nTraining Level-2 Meta-learner...")
    X_meta_train = np.hstack([oof_preds_demand, oof_preds_param])
    X_meta_test = np.hstack([test_preds_demand, test_preds_param])

    meta_estimator = make_base_estimator(seed=2024)
    meta_model = RegressorChain(meta_estimator, order=[0, 1, 2])
    meta_model.fit(X_meta_train, y)
    final_predictions = meta_model.predict(X_meta_test)
    final_predictions[:, 2] = np.clip(final_predictions[:, 2], 0.0, 1.0)

    # CV score for meta
    meta_scores = []
    kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf_meta.split(X_meta_train, y):
        temp = RegressorChain(make_base_estimator(seed=2024), order=[0, 1, 2])
        temp.fit(X_meta_train[train_idx], y.iloc[train_idx])
        val_pred = temp.predict(X_meta_train[val_idx])
        meta_scores.append(accuracy_tolerance_score(y.iloc[val_idx]["satisfaction"], val_pred[:, 2]))

    final_score = np.mean(meta_scores)
    print(f"  Meta-learner CV Score: {final_score:.2%}")

    submission = pd.DataFrame({
        "id": test_ids,
        "wip": final_predictions[:, 0],
        "investissement": final_predictions[:, 1],
        "satisfaction": final_predictions[:, 2],
    })
    return submission, final_score
