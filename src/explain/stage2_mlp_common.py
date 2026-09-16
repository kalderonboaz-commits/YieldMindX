"""Shared utilities for the Stage 2 Run B complementary MLP checkpoint.

This is an ADDITIONAL evidence source, not a replacement for LightGBM,
spline/GAM, cross-stage analysis, or the v3/v4/v5 interaction machinery --
none of those are touched by this module. Same locked Run B universe
(1,425 predictors: 1,405 numeric + 20 categorical one-hot dummies, zero
engineered features), same holdout split, same GroupKFold folds as every
other Stage 2 checkpoint (reused from stage2_v2_common.load_locked_split).

Architecture is fixed at (16, 8) hidden units per the project instruction
-- not tuned, not grid-searched. With 1,425 inputs this is ~23,000
trainable parameters against ~1,200 training rows (80 lots) -- a severely
overparameterized regime by classical standards, which is exactly why
sections 5/6 of this checkpoint exist as a mandatory gate before any
relationship-discovery work is attempted.

Early stopping is implemented manually (via partial_fit + a lot-grouped
inner validation split) rather than via MLPRegressor's own
early_stopping=True, because sklearn's built-in mechanism uses a random
(non-grouped) internal split, which would let wafers from the same lot
appear in both the internal "train" and "validation" partitions --
violating this project's mandatory grouped-validation rule.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HIDDEN_LAYERS = (16, 8)
ALPHA = 10.0                 # strong L2 -- deliberately conservative given input_dim >> n_samples
LEARNING_RATE_INIT = 0.001   # conservative, per instruction
RANDOM_STATE = 42
MAX_EPOCHS = 500
PATIENCE = 20
INNER_VAL_FRACTION = 0.2


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def count_trainable_params(hidden_layers: tuple[int, ...], n_inputs: int, n_outputs: int = 1) -> int:
    layer_sizes = [n_inputs] + list(hidden_layers) + [n_outputs]
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1]  # weights
        total += layer_sizes[i + 1]                    # biases
    return total


@dataclasses.dataclass
class MlpFitResult:
    model: MLPRegressor
    scaler: StandardScaler
    best_epoch: int
    n_epochs_run: int
    train_curve: list
    val_curve: list
    inner_train_rmse: float
    inner_val_rmse: float


def fit_mlp_grouped_early_stopping(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    hidden_layers: tuple[int, ...] = HIDDEN_LAYERS, alpha: float = ALPHA,
    learning_rate_init: float = LEARNING_RATE_INIT, random_state: int = RANDOM_STATE,
    max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, inner_val_fraction: float = INNER_VAL_FRACTION,
) -> MlpFitResult:
    """Fits ONE MLP with manual, lot-grouped early stopping: an inner
    GroupShuffleSplit carves out a validation slice (no lot overlap with
    the inner-training rows), and partial_fit is called epoch-by-epoch,
    snapshotting the best-validation-RMSE weights (sklearn does not do
    this automatically) and stopping after `patience` epochs with no
    improvement."""
    gss = GroupShuffleSplit(n_splits=1, test_size=inner_val_fraction, random_state=random_state)
    inner_tr_idx, inner_va_idx = next(gss.split(X, y, groups=groups))

    scaler = StandardScaler()
    X_inner_tr = scaler.fit_transform(X[inner_tr_idx])
    X_inner_va = scaler.transform(X[inner_va_idx])
    y_inner_tr, y_inner_va = y[inner_tr_idx], y[inner_va_idx]

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers, activation="relu", solver="adam",
        alpha=alpha, learning_rate_init=learning_rate_init, random_state=random_state,
        max_iter=1, warm_start=True, early_stopping=False, batch_size=min(64, len(inner_tr_idx)),
    )

    best_val_rmse = np.inf
    best_epoch = 0
    best_coefs, best_intercepts = None, None
    train_curve, val_curve = [], []
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.partial_fit(X_inner_tr, y_inner_tr)
        tr_pred = model.predict(X_inner_tr)
        va_pred = model.predict(X_inner_va)
        tr_rmse, va_rmse = rmse(y_inner_tr, tr_pred), rmse(y_inner_va, va_pred)
        train_curve.append(tr_rmse)
        val_curve.append(va_rmse)
        if va_rmse < best_val_rmse - 1e-6:
            best_val_rmse = va_rmse
            best_epoch = epoch
            best_coefs = copy.deepcopy(model.coefs_)
            best_intercepts = copy.deepcopy(model.intercepts_)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.coefs_ = best_coefs
    model.intercepts_ = best_intercepts

    return MlpFitResult(
        model=model, scaler=scaler, best_epoch=best_epoch, n_epochs_run=len(train_curve),
        train_curve=train_curve, val_curve=val_curve,
        inner_train_rmse=train_curve[best_epoch], inner_val_rmse=val_curve[best_epoch],
    )


def predict_scaled(fit: MlpFitResult, X: np.ndarray) -> np.ndarray:
    return fit.model.predict(fit.scaler.transform(X))
