"""Shared utilities for the Stage 2 Run B spline/GAM-style single-parameter
discovery method. This is a COMPLEMENTARY method to the existing LightGBM
(v1) and model-free quantile-binning (v2) single-parameter analyses -- it
does not touch, retrain, or replace either. No tree model, no neural
network. Nothing here reads the ground-truth document or references any
specific parameter name.

Method: for a single numeric predictor x, fit
    SortingYield ~ smooth_function(x)
via a regularized spline basis (SplineTransformer + RidgeCV), compared
against a plain linear null (LinearRegression) under grouped (by LotName)
cross-validation. The out-of-fold response curve is then classified into a
relationship shape using a significance-filtered turning-point count, where
"significant" is judged against the curve's own cross-fold variability
(the spline analogue of the standard-error filter already used by
classify_binned_shape in stage2_v2_common.py -- same flat_threshold_frac
and se_multiple constants, reused rather than re-tuned).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

FLAT_THRESHOLD_FRAC = 0.15   # reused from stage2_v2_common.classify_binned_shape
SE_MULTIPLE = 1.3            # reused from stage2_v2_common.classify_binned_shape
EDGE_FRAC = 0.2               # a turning point within the outer 20% of the grid is treated as a boundary effect, not an interior extremum
THRESHOLD_WINDOW_FRAC = 0.3   # a monotonic change concentrated in <=30% of the grid width counts as a sharp transition, not a uniform trend
WINDOW_MIN_FRAC = 0.3         # an interior maximum whose near-peak plateau spans >=30% of the grid is called a process window rather than a narrow peak
PLATEAU_TOL_FRAC = 0.15       # "near the extremum" = within 15% of the curve's total range of the extremum value
ALPHAS = np.logspace(-2, 3, 12)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Simple linear null: unregularized ordinary least squares on the raw feature."""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def fit_spline(x: np.ndarray, y: np.ndarray, n_knots: int = 6, degree: int = 3) -> Pipeline:
    """Regularized spline/GAM-style single-feature model: cubic B-spline basis
    + ridge regression with internally cross-validated alpha. extrapolation='linear'
    so the fitted curve behaves sensibly slightly beyond a given fold's own
    training range (needed when reusing one shared grid across per-fold curves)."""
    n_knots = min(n_knots, max(4, np.unique(x).size // 3))
    pipe = Pipeline([
        ("spline", SplineTransformer(n_knots=n_knots, degree=degree, extrapolation="linear", include_bias=True)),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])
    pipe.fit(x.reshape(-1, 1), y)
    return pipe


def cv_metrics(x: np.ndarray, y: np.ndarray, folds) -> dict | None:
    """Grouped-fold out-of-fold comparison of the linear null vs the spline
    model. Fits ONLY on each fold's training rows, evaluates on that fold's
    held-out rows -- the final holdout set is never touched here."""
    lin_rmse, spl_rmse, lin_r2, spl_r2 = [], [], [], []
    spl_better = 0
    for tr, va in folds:
        x_tr, y_tr = x[tr], y[tr]
        x_va, y_va = x[va], y[va]
        if np.unique(x_tr).size < 8 or len(va) < 10:
            continue
        lin_model = fit_linear(x_tr, y_tr)
        spl_model = fit_spline(x_tr, y_tr)
        p_lin = lin_model.predict(x_va.reshape(-1, 1))
        p_spl = spl_model.predict(x_va.reshape(-1, 1))
        r_lin, r_spl = _rmse(y_va, p_lin), _rmse(y_va, p_spl)
        lin_rmse.append(r_lin)
        spl_rmse.append(r_spl)
        lin_r2.append(r2_score(y_va, p_lin))
        spl_r2.append(r2_score(y_va, p_spl))
        if r_spl < r_lin:
            spl_better += 1
    if len(lin_rmse) < 3:
        return None
    n = len(lin_rmse)
    mean_lin_rmse, mean_spl_rmse = float(np.mean(lin_rmse)), float(np.mean(spl_rmse))
    rel_improve = (mean_lin_rmse - mean_spl_rmse) / mean_lin_rmse if mean_lin_rmse > 0 else 0.0
    mean_lin_r2, mean_spl_r2 = float(np.mean(lin_r2)), float(np.mean(spl_r2))
    nonlinear_advantage = (spl_better >= max(4, int(round(0.8 * n)))) and (rel_improve > 0.01) and (mean_spl_r2 > mean_lin_r2)
    return dict(
        n_folds_used=n, linear_cv_rmse=mean_lin_rmse, spline_cv_rmse=mean_spl_rmse,
        rmse_improvement_pct=round(100 * rel_improve, 3), linear_cv_r2=mean_lin_r2, spline_cv_r2=mean_spl_r2,
        folds_spline_better=spl_better, nonlinear_advantage=bool(nonlinear_advantage),
    )


def predict_grid(model, x_ref: np.ndarray, n_points: int = 40, lo_pct: float = 2, hi_pct: float = 98,
                  grid: np.ndarray | None = None):
    if grid is None:
        lo, hi = np.percentile(x_ref, [lo_pct, hi_pct])
        if lo == hi:
            lo, hi = float(np.min(x_ref)), float(np.max(x_ref))
        grid = np.linspace(lo, hi, n_points)
    pred = model.predict(grid.reshape(-1, 1))
    return grid, pred


def _widest_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, -1)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - 1 - i) > (best[1] - best[0]):
                best = (i, j - 1)
            i = j
        else:
            i += 1
    return best


def _largest_significant_window(sig_mask: np.ndarray, mags: np.ndarray) -> tuple[int, int, float]:
    best_start, best_end, best_sum = -1, -1, 0.0
    i, n = 0, len(sig_mask)
    while i < n:
        if sig_mask[i]:
            j = i
            s = 0.0
            while j < n and sig_mask[j]:
                s += mags[j]
                j += 1
            if s > best_sum:
                best_start, best_end, best_sum = i, j - 1, s
            i = j
        else:
            i += 1
    return best_start, best_end, best_sum


def classify_curve_shape(grid_x: np.ndarray, curve_y: np.ndarray, se_curve: np.ndarray, y_std: float) -> dict:
    """Classify a smooth out-of-fold response curve into one of:
    MONOTONIC_POSITIVE, MONOTONIC_NEGATIVE, U_SHAPED, INVERTED_U,
    PROCESS_WINDOW, THRESHOLD, SATURATION, COMPLEX_NONLINEAR, FLAT,
    INCONCLUSIVE. se_curve is the cross-fold standard deviation of the curve
    at each grid point (from independently-fit per-fold spline curves on the
    SAME grid) -- used as the significance yardstick for turning points,
    analogous to classify_binned_shape's within-bin sampling SE filter."""
    range_ = float(curve_y.max() - curve_y.min())
    empty = dict(shape=None, direction="n/a", turning_points=[], effect_range=round(range_, 4),
                 beneficial_range=None, harmful_range=None)

    if y_std < 1e-9 or range_ < FLAT_THRESHOLD_FRAC * y_std:
        return {**empty, "shape": "FLAT"}

    mean_se = float(np.mean(se_curve)) if len(se_curve) else 0.0
    if mean_se > 0.6 * range_:
        return {**empty, "shape": "INCONCLUSIVE"}

    diffs = np.diff(curve_y)
    se_diff = np.sqrt(se_curve[:-1] ** 2 + se_curve[1:] ** 2)
    se_diff = np.where(se_diff < 1e-9, np.inf, se_diff)
    significant = np.abs(diffs) > (SE_MULTIPLE * se_diff)

    if not significant.any():
        return {**empty, "shape": "FLAT"}

    signs = np.where(significant, np.sign(diffs), 0)
    sig_idx = np.where(significant)[0]
    sig_signs = signs[sig_idx]
    change_positions = sig_idx[1:][np.diff(sig_signs) != 0]
    sign_changes = len(change_positions)
    n_diffs = len(diffs)

    shape, direction, turning_points = None, "n/a", []

    if sign_changes == 0:
        direction = "positive" if sig_signs[0] > 0 else "negative"
        s, e, _ = _largest_significant_window(significant, np.abs(diffs))
        span_frac = (e - s + 1) / n_diffs
        center_rel = ((s + e) / 2) / max(n_diffs - 1, 1)
        if span_frac <= THRESHOLD_WINDOW_FRAC:
            tp = (grid_x[s] + grid_x[e + 1]) / 2
            turning_points = [round(float(tp), 6)]
            shape = "SATURATION" if (center_rel <= EDGE_FRAC or center_rel >= 1 - EDGE_FRAC) else "THRESHOLD"
        else:
            shape = "MONOTONIC_POSITIVE" if direction == "positive" else "MONOTONIC_NEGATIVE"

    elif sign_changes == 1:
        p = int(change_positions[0])
        turn_x = (grid_x[p] + grid_x[p + 1]) / 2
        rel_pos = p / max(n_diffs - 1, 1)
        if EDGE_FRAC <= rel_pos <= 1 - EDGE_FRAC:
            extremum_val = curve_y[p + 1]
            is_min = extremum_val <= (curve_y[0] + curve_y[-1]) / 2
            near = np.abs(curve_y - extremum_val) <= PLATEAU_TOL_FRAC * range_
            plateau_frac = near.sum() / len(curve_y)
            if is_min:
                shape = "U_SHAPED"
            else:
                shape = "PROCESS_WINDOW" if plateau_frac >= WINDOW_MIN_FRAC else "INVERTED_U"
            direction = "n/a"
            turning_points = [round(float(turn_x), 6)]
        else:
            shape = "THRESHOLD"
            direction = "positive" if sig_signs[0] > 0 else "negative"
            turning_points = [round(float(turn_x), 6)]

    else:
        shape = "COMPLEX_NONLINEAR"
        direction = "n/a"
        turning_points = [round(float((grid_x[p] + grid_x[p + 1]) / 2), 6) for p in change_positions[:2]]

    mid = float(curve_y.mean())
    above, below = curve_y > mid, curve_y < mid
    bi = _widest_run(above)
    hi_ = _widest_run(below)
    beneficial_range = f"{grid_x[bi[0]]:.4g} to {grid_x[bi[1]]:.4g}" if bi[1] >= bi[0] else None
    harmful_range = f"{grid_x[hi_[0]]:.4g} to {grid_x[hi_[1]]:.4g}" if hi_[1] >= hi_[0] else None

    return dict(shape=shape, direction=direction, turning_points=turning_points, effect_range=round(range_, 4),
                beneficial_range=beneficial_range, harmful_range=harmful_range)


def evidence_label_spline(shape: str, shape_matches: int, n_fold_curves: int, mean_spl_r2_oof: float) -> tuple[str, str]:
    """Fold-stability + out-of-fold-predictiveness gate. A feature is never
    labeled STRONG/MODERATE purely because a fold-averaged curve LOOKS
    curved -- it must also show positive out-of-fold R^2 (genuine
    predictive contribution, per project requirement) and the same shape
    must reproduce across the majority of independently-fit fold curves."""
    if shape in ("FLAT", "INCONCLUSIVE") or n_fold_curves < 3:
        return "INCONCLUSIVE", f"Shape='{shape}' or too few usable fold curves ({n_fold_curves})."
    frac = shape_matches / n_fold_curves
    predictive = mean_spl_r2_oof > 0
    if frac >= 0.8 and predictive:
        return "STRONG", f"Shape reproduced in {shape_matches}/{n_fold_curves} fold curves; positive out-of-fold R^2 ({mean_spl_r2_oof:.4f})."
    if frac >= 0.6 and predictive:
        return "MODERATE", f"Shape reproduced in {shape_matches}/{n_fold_curves} fold curves; positive out-of-fold R^2 ({mean_spl_r2_oof:.4f}) -- majority but not unanimous."
    if frac <= 0.2 and not predictive:
        return "INCONCLUSIVE", f"Shape reproduced in only {shape_matches}/{n_fold_curves} fold curves and out-of-fold R^2 is non-positive ({mean_spl_r2_oof:.4f})."
    return "WEAK", f"Shape reproduced in {shape_matches}/{n_fold_curves} fold curves; out-of-fold R^2={mean_spl_r2_oof:.4f} -- inconsistent or weak."
