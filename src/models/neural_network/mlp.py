"""
MLP Neural Network for multi-output KPI prediction.

Architecture: MLPDemandParam
  - Separate input pathways for demand features (7,540 binary) and param features (51 continuous)
  - Gated fusion mechanism to combine both pathways
  - Multi-output head predicting [wip, investissement, satisfaction]

Custom Loss Functions:
  - WeightedMAELoss: MAE with per-target weights (emphasizes satisfaction)
  - SatisfactionThresholdLoss: MAE + penalty for predictions outside ±0.05 tolerance zone

Hyperparameter search over 960+ YAML configurations:
  - learning_rates: [1e-5, 5e-4, 1e-4, 1e-3, 5e-3]
  - batch_sizes: [256, 512]
  - dropout: [0.1, 0.2, 0.3]
  - weight_decay: [1e-4, 1e-5, 1e-6]
  - hidden_dims: [[256, 128]]
  - loss_weights: [[1.0, 1.0, 2.0], [1.0, 1.0, 3.0]]
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════


class WeightedMAELoss(nn.Module):
    """Weighted Mean Absolute Error loss for multi-output regression.

    Allows giving more importance to specific targets (e.g., satisfaction).
    Default weights: [1.0, 1.0, 2.0] → satisfaction weighted 2x.
    """

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, preds, targets):
        return (torch.abs(preds - targets) * self.weights).mean()


class SatisfactionThresholdLoss(nn.Module):
    """MAE + penalty for satisfaction predictions outside ±0.05 tolerance.

    Directly optimizes for the competition metric: % of predictions within tolerance.
    """

    def __init__(self, base_weights, tol=0.05, lambda_sat=1.0):
        super().__init__()
        self.base_loss = WeightedMAELoss(base_weights)
        self.tol = tol
        self.lambda_sat = lambda_sat

    def forward(self, preds, targets):
        loss = self.base_loss(preds, targets)

        # Penalty only for satisfaction (column index 2)
        sat_pred = preds[:, 2]
        sat_true = targets[:, 2]
        sat_error = torch.abs(sat_pred - sat_true)
        sat_penalty = torch.relu(sat_error - self.tol).mean()

        return loss + self.lambda_sat * sat_penalty


# ═══════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════


def load_and_prepare_data(cfg, train_data, param_demand_model_list=[]):
    """Prepare data for MLP training with appropriate scaling.

    For param-demand models:
      - Demand features (binary 0/1): no scaling
      - Param features (continuous): StandardScaler
      - Targets: StandardScaler

    For standard models:
      - All features: StandardScaler
      - Targets: StandardScaler
    """
    if cfg["model"]["type"] not in param_demand_model_list:
        X_train = train_data.drop(columns=["id", "wip", "investissement", "satisfaction"])
        y_train = train_data[["wip", "investissement", "satisfaction"]]

        X_train2, X_val, y_train2, y_val = train_test_split(
            X_train, y_train,
            test_size=cfg["data"]["val_split"],
            random_state=cfg["data"]["random_state"],
            shuffle=True,
        )

        x_scaler = StandardScaler()
        X_train2_scaled = x_scaler.fit_transform(X_train2)
        X_val_scaled = x_scaler.transform(X_val)

        y_scaler = StandardScaler()
        y_train2_scaled = y_scaler.fit_transform(y_train2)
        y_val_scaled = y_scaler.transform(y_val)

        return X_train2_scaled, y_train2_scaled, X_val_scaled, y_val_scaled, x_scaler, y_scaler

    else:
        demand_cols = [c for c in train_data.columns if c.startswith("demand")]
        param_cols = [c for c in train_data.columns if c.startswith("param")]

        X_demand = train_data[demand_cols].values
        X_param = train_data[param_cols].values
        y = train_data[["wip", "investissement", "satisfaction"]].values

        Xd_train, Xd_val, Xp_train, Xp_val, y_train, y_val = train_test_split(
            X_demand, X_param, y,
            test_size=cfg["data"]["val_split"],
            random_state=cfg["data"]["random_state"],
            shuffle=True,
        )

        # Demand features are binary (0/1) → no scaling needed
        x_param_scaler = StandardScaler()
        Xp_train_scaled = x_param_scaler.fit_transform(Xp_train)
        Xp_val_scaled = x_param_scaler.transform(Xp_val)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)

        return (
            Xd_train, Xp_train_scaled, y_train_scaled,
            Xd_val, Xp_val_scaled, y_val_scaled,
            x_param_scaler, y_scaler,
        )


# ═══════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════


def count_parameters(model):
    """Count total, trainable, and frozen parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen
