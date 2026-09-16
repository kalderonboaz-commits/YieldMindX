"""Shared utilities for the Stage 2 Run B v2 (coverage-improved) discovery
pipeline. Generic methods only -- nothing here references any specific
parameter name, threshold, or planted relationship. The ground-truth
document is not read anywhere in this module.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold, make_holdout_split
from src.features.stage2_matrix import load_stage2_dataset

RANDOM_STATE = 42
R = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports"


@dataclasses.dataclass
class Stage2Split:
    ds: object
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    folds: list  # list of (train_idx, val_idx) arrays, positions into X_train


def load_locked_split() -> Stage2Split:
    ds = load_stage2_dataset(include_review_columns=False)
    holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=RANDOM_STATE)
    X_train = ds.X.iloc[holdout.train_idx].reset_index(drop=True)
    X_test = ds.X.iloc[holdout.test_idx].reset_index(drop=True)
    y_train = ds.y.iloc[holdout.train_idx].reset_index(drop=True)
    y_test = ds.y.iloc[holdout.test_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)
    _, folds = make_group_kfold(groups_train, n_splits=5)
    return Stage2Split(ds=ds, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                        groups_train=groups_train, folds=folds)


def classify_binned_shape(x: np.ndarray, y: np.ndarray, n_bins: int = 8, min_bin_size: int = 15,
                           flat_threshold_frac: float = 0.15, se_multiple: float = 1.3
                           ) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Model-free relationship-shape classifier: bins x into quantiles,
    computes the mean of y per bin, and classifies the shape from the sign
    pattern of first differences between consecutive bin means -- the same
    logic family as src/explain/interactions.py's classify_pdp_shape, but
    operating directly on raw (x, y) data with NO model involved. A feature
    LightGBM never splits on is still fully testable here.

    Statistical noise control (important): a naive sign-change count on raw
    bin means is highly sensitive to sampling noise -- with ~150-240 rows
    per bin, consecutive bin means fluctuate by chance alone, which without
    control massively over-classifies flat/monotonic relationships as
    'complex_nonmonotonic'. Each consecutive-bin difference is only counted
    as a real direction if it exceeds se_multiple * its own standard error
    (SE_diff = sqrt(SE_i^2 + SE_{i+1}^2), from the bin's own y-std/sqrt(n));
    non-significant differences are treated as flat (no directional
    evidence) rather than contributing a spurious sign flip. se_multiple=1.3
    is a deliberately lenient SCREENING threshold (not a strict p<0.05
    test) -- this is a first-pass filter, with expensive confirmation
    (SHAP/H-statistic/fold-stability) applied downstream to whatever
    survives.

    Returns (shape_label, direction, bin_centers, bin_means).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < min_bin_size * 3 or np.unique(x).size < n_bins:
        return "insufficient_data", "n/a", np.array([]), np.array([])

    try:
        bins = pd.qcut(x, n_bins, duplicates="drop")
    except ValueError:
        return "insufficient_data", "n/a", np.array([]), np.array([])

    df = pd.DataFrame({"bin": bins, "x": x, "y": y})
    grp = df.groupby("bin", observed=True)
    counts = grp.size()
    if (counts < min_bin_size).any():
        if n_bins > 4:
            return classify_binned_shape(x, y, n_bins=max(4, n_bins - 2), min_bin_size=min_bin_size,
                                          flat_threshold_frac=flat_threshold_frac, se_multiple=se_multiple)
        return "insufficient_data", "n/a", np.array([]), np.array([])

    bin_means = grp["y"].mean().to_numpy()
    bin_stds = grp["y"].std(ddof=1).fillna(0.0).to_numpy()
    bin_ns = grp.size().to_numpy()
    bin_centers = grp["x"].mean().to_numpy()

    y_std = y.std()
    rng = bin_means.max() - bin_means.min()
    flat_threshold = flat_threshold_frac * y_std
    if rng < flat_threshold or y_std < 1e-9:
        return "flat", "n/a", bin_centers, bin_means

    diffs = np.diff(bin_means)
    se = bin_stds / np.sqrt(np.maximum(bin_ns, 1))
    se_diff = np.sqrt(se[:-1] ** 2 + se[1:] ** 2)
    se_diff = np.where(se_diff < 1e-9, np.inf, se_diff)  # avoid false "significant" on degenerate bins
    significant = np.abs(diffs) > (se_multiple * se_diff)

    signs = np.where(significant, np.sign(diffs), 0)
    signs_nz = signs[signs != 0]
    step_sizes = np.abs(diffs)

    if len(signs_nz) == 0:
        return "flat", "n/a", bin_centers, bin_means
    sign_changes = int(np.sum(np.diff(signs_nz) != 0))

    if sign_changes == 0:
        direction = "positive" if signs_nz[0] > 0 else "negative"
        if step_sizes.max() > 0.5 * rng:
            return "threshold", direction, bin_centers, bin_means
        return "monotonic", direction, bin_centers, bin_means

    if sign_changes == 1:
        change_idx = int(np.where(np.diff(signs_nz) != 0)[0][0]) + 1
        rel_pos = change_idx / max(len(signs_nz) - 1, 1)
        if 0.2 <= rel_pos <= 0.8:
            shape = "U_shape" if bin_means.argmin() not in (0, len(bin_means) - 1) else "inverted_U"
            return shape, "n/a", bin_centers, bin_means
        return "threshold", "n/a", bin_centers, bin_means

    return "complex_nonmonotonic", "n/a", bin_centers, bin_means


def evidence_label_from_consistency(shape_consistency: int, direction_consistency: int, n_folds: int,
                                     main_shape: str) -> tuple[str, str]:
    if main_shape in ("flat", "insufficient_data"):
        return "INCONCLUSIVE", f"Main-split shape is '{main_shape}' -- no stable evidence."
    if shape_consistency >= 4 and direction_consistency >= 4:
        return "STRONG", f"Shape consistent in {shape_consistency}/{n_folds} folds, direction in {direction_consistency}/{n_folds}."
    if shape_consistency >= 3 or direction_consistency >= 3:
        return "MODERATE", f"Shape consistent in {shape_consistency}/{n_folds} folds, direction in {direction_consistency}/{n_folds} -- majority but not unanimous."
    if shape_consistency <= 1 and direction_consistency <= 1:
        return "INCONCLUSIVE", f"Shape/direction agree in only {shape_consistency}/{n_folds} and {direction_consistency}/{n_folds} folds."
    return "WEAK", f"Shape consistent in {shape_consistency}/{n_folds} folds, direction in {direction_consistency}/{n_folds} -- inconsistent."
