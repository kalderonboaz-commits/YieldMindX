"""Shared utilities for the Stage 2 Run B RELATIONSHIP VALIDATION ENGINE.

This is a new, complementary Stage 2 component focused on directly
VALIDATING (not merely discovering) single-parameter, 2-way, and 3-way
relationships via genuine empirical (permutation-based) significance
testing and Benjamini-Hochberg FDR control. It is not a predictive model.

It deliberately REUSES existing, already-validated project machinery
rather than reinventing it:
  - single-parameter nonlinear fitting: fit_linear / fit_spline from
    src/explain/stage2_spline_common.py (the project's existing
    lightweight nonlinear method -- SplineTransformer + RidgeCV -- reused
    unmodified, per instruction not to use a large neural network)
  - 2-way additive-vs-joint and 3-way lower-order-vs-full binned-model
    machinery from src/explain/stage2_v3_common.py, reused unmodified
  - shape classification: classify_curve_shape from stage2_spline_common,
    reused unmodified for single-parameter shape/turning-point extraction
  - full-universe broad pair screening: compute_nonadditivity_screen
    (stage2_v3_common) and compute_pairwise_screen_scores
    (stage2_v2_pair_screen), reused unmodified for exhaustive coverage

What is NEW here (not present in any prior checkpoint): genuine
permutation-based empirical p-values (a pooled null built from B target
shuffles, shared across all candidates within a family so the null is
cheap to compute yet valid), and a from-scratch Benjamini-Hochberg FDR
procedure (implemented here with plain numpy -- no new dependency), used
as the PRIMARY acceptance gate for every relationship this engine reports,
in addition to (not instead of) grouped-fold stability.

Same locked Run B universe and same GroupKFold folds as every other Stage
2 checkpoint (reused from stage2_v2_common.load_locked_split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_spline_common import fit_linear, fit_spline, predict_grid
from src.explain.stage2_v2_common import R, load_locked_split  # noqa: F401 (re-exported)
from src.explain.stage2_v3_common import (
    compute_bin_index, fit_additive_joint, fit_threeway, predict_additive, predict_joint, predict_threeway,
)

RANDOM_STATE = 42
CACHE_DIR = f"{R}/relval_cache"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Section 1: single-parameter baseline (linear/"null") vs nonlinear (spline)
# ---------------------------------------------------------------------------

def single_param_cv(x: np.ndarray, y: np.ndarray, folds) -> dict | None:
    """BASELINE = plain linear regression ("simple/null relationship").
    NONLINEAR = spline+ridge ("existing lightweight nonlinear method").
    Reuses fit_linear/fit_spline (raw model constructors) from
    stage2_spline_common -- does not reuse or modify that module's own
    cv_metrics/evidence-labeling logic, since this engine computes MAE
    (not part of the original spline checkpoint's metric set) and applies
    its own FDR-gated labeling framework (section 8 of this checkpoint),
    not the earlier checkpoint's threshold rule."""
    b_rmse, n_rmse, b_mae, n_mae, b_r2, n_r2 = [], [], [], [], [], []
    n_improved = 0
    for tr, va in folds:
        x_tr, y_tr = x[tr], y[tr]
        x_va, y_va = x[va], y[va]
        if np.unique(x_tr).size < 8 or len(va) < 10:
            continue
        base_model = fit_linear(x_tr, y_tr)
        nl_model = fit_spline(x_tr, y_tr)
        p_base = base_model.predict(x_va.reshape(-1, 1))
        p_nl = nl_model.predict(x_va.reshape(-1, 1))
        r_base, r_nl = rmse(y_va, p_base), rmse(y_va, p_nl)
        b_rmse.append(r_base); n_rmse.append(r_nl)
        b_mae.append(mae(y_va, p_base)); n_mae.append(mae(y_va, p_nl))
        b_r2.append(r2(y_va, p_base)); n_r2.append(r2(y_va, p_nl))
        if r_nl < r_base:
            n_improved += 1
    if len(b_rmse) < 3:
        return None
    n = len(b_rmse)
    mean_b_rmse, mean_n_rmse = float(np.mean(b_rmse)), float(np.mean(n_rmse))
    mean_b_mae, mean_n_mae = float(np.mean(b_mae)), float(np.mean(n_mae))
    mean_b_r2, mean_n_r2 = float(np.mean(b_r2)), float(np.mean(n_r2))
    return dict(
        n_folds=n, baseline_cv_rmse=mean_b_rmse, nonlinear_cv_rmse=mean_n_rmse,
        delta_rmse=mean_b_rmse - mean_n_rmse,
        baseline_cv_mae=mean_b_mae, nonlinear_cv_mae=mean_n_mae, delta_mae=mean_b_mae - mean_n_mae,
        baseline_cv_r2=mean_b_r2, nonlinear_cv_r2=mean_n_r2, delta_r2=mean_n_r2 - mean_b_r2,
        folds_improved=n_improved,
    )


# ---------------------------------------------------------------------------
# Section 3: 2-way additive-vs-joint, extended with MAE (v3_common's own
# additive_vs_joint_cv does not compute MAE) -- reuses fit_additive_joint /
# predict_additive / predict_joint (v3_common's raw binned-model
# constructors) unmodified.
# ---------------------------------------------------------------------------

def pair_cv(xa: np.ndarray, xb: np.ndarray, y: np.ndarray, folds, n_bins: int = 5) -> dict | None:
    a_rmse, j_rmse, a_mae, j_mae, a_r2, j_r2 = [], [], [], [], [], []
    n_improved = 0
    for tr, va in folds:
        xa_tr, xb_tr, y_tr = xa[tr], xb[tr], y[tr]
        xa_va, xb_va, y_va = xa[va], xb[va], y[va]
        if len(va) < 10:
            continue
        fit = fit_additive_joint(xa_tr, xb_tr, y_tr, n_bins=n_bins)
        p_add = predict_additive(fit, xa_va, xb_va)
        p_joint = predict_joint(fit, xa_va, xb_va)
        r_add, r_joint = rmse(y_va, p_add), rmse(y_va, p_joint)
        a_rmse.append(r_add); j_rmse.append(r_joint)
        a_mae.append(mae(y_va, p_add)); j_mae.append(mae(y_va, p_joint))
        a_r2.append(r2(y_va, p_add)); j_r2.append(r2(y_va, p_joint))
        if r_joint < r_add:
            n_improved += 1
    if len(a_rmse) < 3:
        return None
    n = len(a_rmse)
    mean_a_rmse, mean_j_rmse = float(np.mean(a_rmse)), float(np.mean(j_rmse))
    mean_a_mae, mean_j_mae = float(np.mean(a_mae)), float(np.mean(j_mae))
    mean_a_r2, mean_j_r2 = float(np.mean(a_r2)), float(np.mean(j_r2))
    return dict(
        n_folds=n, additive_cv_rmse=mean_a_rmse, joint_cv_rmse=mean_j_rmse, delta_rmse=mean_a_rmse - mean_j_rmse,
        additive_cv_mae=mean_a_mae, joint_cv_mae=mean_j_mae, delta_mae=mean_a_mae - mean_j_mae,
        additive_cv_r2=mean_a_r2, joint_cv_r2=mean_j_r2, delta_r2=mean_j_r2 - mean_a_r2,
        folds_improved=n_improved,
    )


# ---------------------------------------------------------------------------
# Section 5: 3-way lower-order-vs-full, extended with MAE.
# ---------------------------------------------------------------------------

def triple_cv(xa: np.ndarray, xb: np.ndarray, xc: np.ndarray, y: np.ndarray, folds, n_bins: int = 2) -> dict | None:
    lo_rmse, fu_rmse, lo_mae, fu_mae, lo_r2, fu_r2 = [], [], [], [], [], []
    n_improved = 0
    for tr, va in folds:
        if len(va) < 10:
            continue
        fit = fit_threeway(xa[tr], xb[tr], xc[tr], y[tr], n_bins=n_bins)
        p_lo = predict_threeway(fit, xa[va], xb[va], xc[va], "lower")
        p_fu = predict_threeway(fit, xa[va], xb[va], xc[va], "full")
        r_lo, r_fu = rmse(y[va], p_lo), rmse(y[va], p_fu)
        lo_rmse.append(r_lo); fu_rmse.append(r_fu)
        lo_mae.append(mae(y[va], p_lo)); fu_mae.append(mae(y[va], p_fu))
        lo_r2.append(r2(y[va], p_lo)); fu_r2.append(r2(y[va], p_fu))
        if r_fu < r_lo:
            n_improved += 1
    if len(lo_rmse) < 3:
        return None
    n = len(lo_rmse)
    mean_lo_rmse, mean_fu_rmse = float(np.mean(lo_rmse)), float(np.mean(fu_rmse))
    mean_lo_mae, mean_fu_mae = float(np.mean(lo_mae)), float(np.mean(fu_mae))
    mean_lo_r2, mean_fu_r2 = float(np.mean(lo_r2)), float(np.mean(fu_r2))
    return dict(
        n_folds=n, lower_cv_rmse=mean_lo_rmse, full_cv_rmse=mean_fu_rmse, delta_rmse=mean_lo_rmse - mean_fu_rmse,
        lower_cv_mae=mean_lo_mae, full_cv_mae=mean_fu_mae, delta_mae=mean_lo_mae - mean_fu_mae,
        lower_cv_r2=mean_lo_r2, full_cv_r2=mean_fu_r2, delta_r2=mean_fu_r2 - mean_lo_r2,
        folds_improved=n_improved,
    )


# ---------------------------------------------------------------------------
# Section 6/7: empirical permutation-based significance + Benjamini-Hochberg
# ---------------------------------------------------------------------------

def empirical_pvalue(observed: float, null_pool: np.ndarray) -> float:
    """One-sided empirical p-value with the standard add-one continuity
    correction (Davison & Hinkley) -- never exactly 0 no matter how
    extreme `observed` is relative to a finite null pool."""
    null_pool = np.asarray(null_pool)
    n = len(null_pool)
    if n == 0:
        return 1.0
    count_ge = int(np.sum(null_pool >= observed))
    return float((1 + count_ge) / (1 + n))


def group_block_shuffle(y: np.ndarray, groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Group-preserving (LotName-block) permutation: reassigns each lot's
    INTACT y-block (in original within-lot row order) to a different lot's
    row positions, via a random permutation of lot identities. This
    destroys the X-Y association (what the null hypothesis requires) while
    preserving each lot's own internal y structure -- unlike a flat
    row-level shuffle, which additionally destroys any real between-lot
    variance in y itself (e.g. lot-to-lot batch effects), understating the
    natural noise floor that grouped-CV out-of-fold evaluation must live
    with under the REAL data. Since every fold split in this project groups
    by LotName, the null must be permuted the same way for the permutation
    test to be valid for a grouped-CV statistic.

    Requires every group to have equal size (true for this project's
    locked training split: 80 lots x 15 wafers each) -- raises otherwise,
    rather than silently falling back to a flat shuffle."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups, first_seen = np.unique(groups, return_index=True)
    order = np.argsort(first_seen)
    unique_groups = unique_groups[order]
    positions = {g: np.where(groups == g)[0] for g in unique_groups}
    sizes = {g: len(p) for g, p in positions.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"group_block_shuffle requires equal-size groups; got sizes {set(sizes.values())}")
    perm = rng.permutation(unique_groups)
    y_shuf = np.empty_like(y)
    for g_dest, g_src in zip(unique_groups, perm):
        y_shuf[positions[g_dest]] = y[positions[g_src]]
    return y_shuf


# ---------------------------------------------------------------------------
# Broad-screen scoring improvement (per stage2_runB_relval_broadscreen_
# scoring_audit.txt): a SHRUNK variant of v3_common's compute_nonadditivity_
# screen (score_2). NOT a modification of stage2_v3_common.py (reused
# unmodified elsewhere, including inside this engine's own confirmatory
# fit_additive_joint) -- a fresh reimplementation here, in this engine's own
# module, so v3/v4/v5's frozen historical outputs remain exactly reproducible.
#
# Audit finding: the original score_2 uses RAW, UNSHRUNK per-cell joint
# means -- noisy for sparsely-populated cells (common for correlated
# feature pairs, which are common in this dataset), which can dilute a
# real but moderate non-additive effect below any promotion cutoff. This
# variant applies the EXACT SAME cell-count-weighted shrinkage principle
# already used and audited in this engine's own confirmatory test
# (fit_additive_joint's shrink_joint parameter) -- shrinking each cell's
# contribution toward "no departure from additive" in proportion to how
# few samples support it, rather than trusting a noisy small-cell mean at
# full weight. Algebraically: shrunk_residual = raw_residual *
# count/(count+shrink_k) -- a pure elementwise reweighting of the existing
# vectorized computation, so this remains fully vectorized across all
# C(1405,2) pairs (no per-pair loop, no new expensive step).
# ---------------------------------------------------------------------------

def compute_nonadditivity_screen_shrunk(X: np.ndarray, y: np.ndarray, n_bins: int = 4,
                                         shrink_k: float = 8.0) -> np.ndarray:
    """Shrunk variant of stage2_v3_common.compute_nonadditivity_screen.
    Same inputs/outputs/shape; same underlying cell-mean/additive-baseline
    construction; the only change is applying a cell-count-weighted
    shrinkage factor to each cell's (JOINT-ADDITIVE) departure before
    squaring and aggregating, exactly mirroring fit_additive_joint's own
    shrink_joint mechanism (default shrink_k=8.0, same default value)."""
    n, F = X.shape
    Q = n_bins
    X32 = X.astype(np.float32)
    y32 = y.astype(np.float32)

    bin_idx = np.zeros((n, F), dtype=np.int32)
    for f in range(F):
        bin_idx[:, f], _ = compute_bin_index(X32[:, f], Q)

    Onehot2D = np.zeros((n, F * Q), dtype=np.float32)
    rows = np.arange(n)
    for f in range(F):
        Onehot2D[rows, f * Q + bin_idx[:, f]] = 1.0

    y_c = y32 - y32.mean()
    YOnehot2D = Onehot2D * y_c[:, None]

    CellSumFull = YOnehot2D.T @ Onehot2D
    CellCountFull = Onehot2D.T @ Onehot2D

    CellSum4 = CellSumFull.reshape(F, Q, F, Q)
    CellCount4 = CellCountFull.reshape(F, Q, F, Q)
    with np.errstate(invalid="ignore", divide="ignore"):
        CellMean4 = np.where(CellCount4 > 0, CellSum4 / np.maximum(CellCount4, 1), 0.0)

    marg_count = np.zeros((F, Q), dtype=np.float64)
    marg_sum = np.zeros((F, Q), dtype=np.float64)
    for f in range(F):
        for q in range(Q):
            mask = bin_idx[:, f] == q
            marg_count[f, q] = mask.sum()
            marg_sum[f, q] = y_c[mask].sum() if mask.any() else 0.0
    marg_mean = np.divide(marg_sum, np.maximum(marg_count, 1), out=np.zeros_like(marg_sum))

    additive = marg_mean[:, :, None, None] + marg_mean[None, None, :, :]
    residual = CellMean4 - additive

    # NEW: cell-count-weighted shrinkage toward "no departure" (residual=0),
    # identical in spirit to fit_additive_joint's shrink_joint -- a cell
    # with few supporting samples has its apparent departure shrunk toward
    # zero rather than trusted at full weight.
    shrink_factor = CellCount4 / (CellCount4 + shrink_k)
    residual = residual * shrink_factor

    weighted_sq = (residual ** 2) * CellCount4
    total_count = CellCount4.sum(axis=(1, 3))
    total_count = np.maximum(total_count, 1)
    score = weighted_sq.sum(axis=(1, 3)) / total_count
    np.fill_diagonal(score, 0.0)
    return score


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Standard Benjamini-Hochberg step-up FDR procedure, implemented from
    scratch with plain numpy (no scipy/statsmodels dependency). Returns
    (adjusted_q_values, accept_mask) aligned to the INPUT order of pvals."""
    pvals = np.asarray(pvals, dtype=np.float64)
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    q_ranked = ranked * n / ranks
    # enforce monotonicity: q-value at rank i cannot exceed q-value at rank i+1
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty(n)
    q[order] = q_ranked
    accept = q <= alpha
    return q, accept
