"""Shared utilities for the Stage 2 Run B complementary EBM (Explainable
Boosting Machine / GA2M) checkpoint.

ADDITIONAL evidence source -- does not replace LightGBM, spline/GAM,
cross-stage analysis, or the v3/v4/v5 interaction machinery. Same locked
Run B universe (1,425 predictors, zero engineered features), same holdout
split, same GroupKFold folds as every other Stage 2 checkpoint.

Configuration is DELIBERATELY conservative given ~1,200 training rows / 80
lots against 1,425 predictors:
  max_bins=128            (default 1024 -- fewer bins per feature, less
                            opportunity to fit noise in sparse bins)
  max_interaction_bins=16 (default 64 -- coarser, smoother pairwise
                            surfaces)
  interactions=15         (a small FIXED integer, not the "5x" adaptive
                            default, explicitly bounding how many pairwise
                            terms the model is allowed to select at all)
  outer_bags=8             (default 14 -- still enough for a meaningful
                            bagging-based stability estimate, at lower
                            compute cost)
  learning_rate=0.04       (default, already conservative)
  min_samples_leaf=10      (default 4 -- larger minimum leaf size, more
                            resistant to fitting single-lot idiosyncrasies)
  reg_lambda=1.0           (default 0.0 -- adds explicit L2-style
                            regularization, absent by default)
  random_state=42          (deterministic)

Grouped validation: EBM's own internal per-bag early-stopping validation
split is normally RANDOM (not lot-aware). This is fixed by constructing
the `bags` array ourselves via GroupShuffleSplit (one lot-respecting
train/validation assignment per outer bag) and passing it explicitly to
`fit(..., bags=bags)` -- the officially supported mechanism for exactly
this purpose (see ExplainableBoostingRegressor.fit docstring). This
project's own outer grouped 5-fold CV (reusing the SAME folds as every
other Stage 2 method) is a separate, additional layer on top of this.
"""
from __future__ import annotations

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

EBM_CONFIG = dict(
    max_bins=128, max_interaction_bins=16, interactions=15, outer_bags=8,
    learning_rate=0.04, min_samples_leaf=10, reg_lambda=1.0, random_state=42, n_jobs=-2,
)
BAG_VALIDATION_FRACTION = 0.15


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def build_grouped_bags(groups: np.ndarray, n_bags: int = None, val_fraction: float = BAG_VALIDATION_FRACTION,
                        random_state: int = 42) -> np.ndarray:
    """Builds the (n_samples, n_bags) bags array EBM expects, with each
    bag's train/validation split respecting LotName grouping (no lot
    split between a bag's +1 training rows and -1 validation rows)."""
    n_bags = n_bags or EBM_CONFIG["outer_bags"]
    n = len(groups)
    bags = np.zeros((n, n_bags), dtype=np.int8)
    for b in range(n_bags):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state + b)
        tr, va = next(gss.split(np.zeros(n), groups=groups))
        bags[tr, b] = 1
        bags[va, b] = -1
    return bags


def fit_ebm(X, y, groups: np.ndarray, config: dict | None = None, random_state: int = 42) -> ExplainableBoostingRegressor:
    cfg = dict(config or EBM_CONFIG)
    cfg["random_state"] = random_state
    bags = build_grouped_bags(groups, n_bags=cfg["outer_bags"], random_state=random_state)
    ebm = ExplainableBoostingRegressor(**cfg)
    ebm.fit(X, y, bags=bags)
    return ebm


SE_MULTIPLE = 1.3         # reused from stage2_v2_common.classify_binned_shape
FLAT_THRESHOLD_FRAC = 0.15
EDGE_FRAC = 0.2
THRESHOLD_WINDOW_FRAC = 0.3
WINDOW_MIN_FRAC = 0.3
PLATEAU_TOL_FRAC = 0.15


def _widest_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, -1)
    i, n = 0, len(mask)
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


def classify_ebm_main_shape(centers: np.ndarray, scores: np.ndarray, stds: np.ndarray, flat_ref_scale: float) -> dict:
    """Same significance-filtered turning-point-count shape classifier as
    src/explain/stage2_spline_common.py::classify_curve_shape (same
    FLAT_THRESHOLD_FRAC/SE_MULTIPLE/EDGE_FRAC/etc. constants), but with an
    EBM-appropriate reference scale for the initial "is there any range at
    all" flat pre-filter.

    WHY a separate function instead of reusing classify_curve_shape
    directly: that function's flat gate compares a curve's range to the
    FULL TARGET's standard deviation -- correct when the curve comes from
    a model that uses ONE feature to explain the WHOLE target (spline/GAM
    single-parameter case). EBM's main-effect terms are each just one of
    1,425 ADDITIVE pieces of a sum that collectively explains the target
    (EBM here reaches holdout R^2=0.996) -- so any INDIVIDUAL term's own
    range is naturally a small fraction of the target's total spread even
    when it is a perfectly real, important, well-fitted term (verified:
    even the single highest-importance term has range/y_std ~= 0.047,
    already below spline's 0.15 flat cutoff). Reusing that cutoff here
    mislabels every single term FLAT regardless of true importance -- a
    genuine scale-mismatch bug caught and fixed here, not a new
    ground-truth-informed method. flat_ref_scale should be a measure of
    the TYPICAL main-effect term's own range (e.g. the median range across
    all 1,425 terms), making this a relative-importance-among-terms
    criterion instead of a relative-to-target-scale one.
    """
    range_ = float(scores.max() - scores.min())
    empty = dict(shape=None, direction="n/a", turning_points=[], effect_range=round(range_, 6),
                 beneficial_range=None, harmful_range=None)

    if flat_ref_scale < 1e-9 or range_ < FLAT_THRESHOLD_FRAC * flat_ref_scale:
        return {**empty, "shape": "FLAT"}

    mean_se = float(np.mean(stds)) if len(stds) else 0.0
    if mean_se > 0.6 * range_:
        return {**empty, "shape": "INCONCLUSIVE"}

    diffs = np.diff(scores)
    se_diff = np.sqrt(stds[:-1] ** 2 + stds[1:] ** 2)
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
            tp = (centers[s] + centers[e + 1]) / 2
            turning_points = [round(float(tp), 6)]
            shape = "SATURATION" if (center_rel <= EDGE_FRAC or center_rel >= 1 - EDGE_FRAC) else "THRESHOLD"
        else:
            shape = "MONOTONIC_POSITIVE" if direction == "positive" else "MONOTONIC_NEGATIVE"

    elif sign_changes == 1:
        p = int(change_positions[0])
        turn_x = (centers[p] + centers[p + 1]) / 2
        rel_pos = p / max(n_diffs - 1, 1)
        if EDGE_FRAC <= rel_pos <= 1 - EDGE_FRAC:
            extremum_val = scores[p + 1]
            is_min = extremum_val <= (scores[0] + scores[-1]) / 2
            near = np.abs(scores - extremum_val) <= PLATEAU_TOL_FRAC * range_
            plateau_frac = near.sum() / len(scores)
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
        turning_points = [round(float((centers[p] + centers[p + 1]) / 2), 6) for p in change_positions[:2]]

    mid = float(scores.mean())
    above, below = scores > mid, scores < mid
    bi = _widest_run(above)
    hi_ = _widest_run(below)
    beneficial_range = f"{centers[bi[0]]:.4g} to {centers[bi[1]]:.4g}" if bi[1] >= bi[0] else None
    harmful_range = f"{centers[hi_[0]]:.4g} to {centers[hi_[1]]:.4g}" if hi_[1] >= hi_[0] else None

    return dict(shape=shape, direction=direction, turning_points=turning_points, effect_range=round(range_, 6),
                beneficial_range=beneficial_range, harmful_range=harmful_range)


def real_bin_layout(ebm: ExplainableBoostingRegressor, term_idx: int):
    """For a MAIN-EFFECT term, returns (bin_centers, scores, stds) for
    only the REAL value bins -- index 0 of term_scores_/standard_deviations_
    is always the 'missing' bin (score 0 in this leakage-free, no-missing-
    values dataset) and any indices beyond len(edges)+1 are unused padding
    up to max_bins; both are excluded here."""
    edges = np.asarray(ebm.bins_[term_idx][0], dtype=np.float64)
    scores = np.asarray(ebm.term_scores_[term_idx], dtype=np.float64)
    stds = np.asarray(ebm.standard_deviations_[term_idx], dtype=np.float64)
    n_real_bins = len(edges) + 1
    real_scores = scores[1:1 + n_real_bins]
    real_stds = stds[1:1 + n_real_bins]

    if len(edges) == 0:
        return np.array([0.0]), real_scores, real_stds
    centers = np.empty(n_real_bins)
    centers[0] = edges[0] - (edges[1] - edges[0]) / 2 if len(edges) > 1 else edges[0] - 1.0
    centers[-1] = edges[-1] + (edges[-1] - edges[-2]) / 2 if len(edges) > 1 else edges[-1] + 1.0
    if n_real_bins > 2:
        centers[1:-1] = (edges[:-1] + edges[1:]) / 2
    return centers, real_scores, real_stds
