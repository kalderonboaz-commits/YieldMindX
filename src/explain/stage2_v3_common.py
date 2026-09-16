"""Shared utilities for Stage 2 Run B v3 -- interaction discovery focused
specifically on NON-ADDITIVITY: does the joint response f(A,B) explain
Yield behavior beyond the additive sum f(A) + f(B)? Generic methods only;
nothing here reads the ground-truth document or references a specific
planted parameter name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split  # noqa: F401 (re-exported for v3 scripts)

RANDOM_STATE = 42


def compute_bin_index(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-width-in-quantile-rank binning via searchsorted (always returns
    exactly n_bins possible bins, robust to duplicate quantile edges from
    skewed/discrete data -- unlike pd.qcut, which can silently drop bins).
    Returns (bin_idx, interior_edges) so the SAME edges can be reused to bin
    a held-out fold/validation set without leaking its own distribution."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right"), 0, n_bins - 1)
    return bin_idx.astype(np.int32), edges


def assign_bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) + 1
    return np.clip(np.searchsorted(edges, x, side="right"), 0, n_bins - 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Broad screen, score 2: full-universe 2D-binned non-additivity energy
# ---------------------------------------------------------------------------

def compute_nonadditivity_screen(X: np.ndarray, y: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Vectorized, FULL-COVERAGE non-additivity energy for every pair of
    columns in X (n_samples, n_features). Conceptually different from
    stage2_v2_pair_screen's residual-vs-bin-index correlation proxy: this
    directly computes, for every pair, the classic Tukey one-degree-of-
    freedom-for-non-additivity decomposition on a coarse 2D binned grid --
    ADDITIVE[q1,q2] = marginal_mean_A[q1] + marginal_mean_B[q2] - grand_mean
    JOINT[q1,q2]    = actual cell mean of y for (bin q1 of A, bin q2 of B)
    score(A,B) = count-weighted mean of (JOINT - ADDITIVE)^2 over all cells
               = the variance in y explained by the joint response beyond
                 what the two marginals already explain additively.

    Computed for ALL C(F,2) pairs via two dense matmuls of shape
    (n_bins*F, n) @ (n, n_bins*F) -- exact same "reduce to linear algebra"
    trick as v2's pair screen, but on one-hot bin memberships instead of
    centered bin indices, which is what makes this test genuine 2D
    non-additivity rather than a linear interaction-basis correlation.
    """
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

    CellSumFull = YOnehot2D.T @ Onehot2D      # (F*Q, F*Q)
    CellCountFull = Onehot2D.T @ Onehot2D     # (F*Q, F*Q)

    CellSum4 = CellSumFull.reshape(F, Q, F, Q)
    CellCount4 = CellCountFull.reshape(F, Q, F, Q)
    with np.errstate(invalid="ignore", divide="ignore"):
        CellMean4 = np.where(CellCount4 > 0, CellSum4 / np.maximum(CellCount4, 1), 0.0)

    # per-feature marginal bin means (independent of partner feature, see
    # module docstring derivation -- summing a fixed feature's cell counts
    # over the partner's bin dimension always recovers that feature's own
    # unconditional per-bin count).
    marg_count = np.zeros((F, Q), dtype=np.float64)
    marg_sum = np.zeros((F, Q), dtype=np.float64)
    for f in range(F):
        for q in range(Q):
            mask = bin_idx[:, f] == q
            marg_count[f, q] = mask.sum()
            marg_sum[f, q] = y_c[mask].sum() if mask.any() else 0.0
    marg_mean = np.divide(marg_sum, np.maximum(marg_count, 1), out=np.zeros_like(marg_sum))

    # AdditivePred[f, q1, g, q2] = marg_mean[f,q1] + marg_mean[g,q2] (already
    # y-centered, so "grand_mean - grand_mean" cancels)
    additive = marg_mean[:, :, None, None] + marg_mean[None, None, :, :]
    residual = CellMean4 - additive
    weighted_sq = (residual ** 2) * CellCount4
    total_count = CellCount4.sum(axis=(1, 3))
    total_count = np.maximum(total_count, 1)
    score = weighted_sq.sum(axis=(1, 3)) / total_count  # (F, F)
    np.fill_diagonal(score, 0.0)
    return score


# ---------------------------------------------------------------------------
# Confirmation stage: leakage-safe additive-vs-joint 2D binned comparison
# ---------------------------------------------------------------------------

def _shrunk_bin_means(bin_idx: np.ndarray, y: np.ndarray, n_bins: int, grand_mean: float, shrink_k: float) -> np.ndarray:
    means = np.full(n_bins, grand_mean)
    for q in range(n_bins):
        mask = bin_idx == q
        n_q = mask.sum()
        if n_q > 0:
            means[q] = (y[mask].sum() + shrink_k * grand_mean) / (n_q + shrink_k)
    return means


def fit_additive_joint(xa_tr: np.ndarray, xb_tr: np.ndarray, y_tr: np.ndarray,
                        n_bins: int = 5, shrink_marginal: float = 5.0, shrink_joint: float = 8.0):
    """Fits, on TRAINING data only: (1) two marginal binned response curves
    (for the additive model), (2) one 2D joint binned response surface
    (shrunk toward the additive prediction by cell count, so sparse cells
    don't overfit). Returns a dict of fitted state that can predict on new
    (held-out) x via predict_additive/predict_joint below."""
    grand_mean = float(y_tr.mean())
    bin_a, edges_a = compute_bin_index(xa_tr, n_bins)
    bin_b, edges_b = compute_bin_index(xb_tr, n_bins)

    marg_a = _shrunk_bin_means(bin_a, y_tr, n_bins, grand_mean, shrink_marginal)
    marg_b = _shrunk_bin_means(bin_b, y_tr, n_bins, grand_mean, shrink_marginal)

    joint_sum = np.zeros((n_bins, n_bins))
    joint_count = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (bin_a == i) & (bin_b == j)
            joint_count[i, j] = mask.sum()
            joint_sum[i, j] = y_tr[mask].sum()

    additive_grid = marg_a[:, None] + marg_b[None, :] - grand_mean
    # shrink the joint cell mean toward the ADDITIVE prediction for that
    # cell (not toward the grand mean) -- a sparse cell falls back to "no
    # interaction assumed" rather than "no effect assumed."
    joint_grid = (joint_sum + shrink_joint * additive_grid) / (joint_count + shrink_joint)

    return dict(edges_a=edges_a, edges_b=edges_b, marg_a=marg_a, marg_b=marg_b,
                grand_mean=grand_mean, additive_grid=additive_grid, joint_grid=joint_grid,
                joint_count=joint_count, n_bins=n_bins)


def predict_additive(fit: dict, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    ba = assign_bin_index(xa, fit["edges_a"])
    bb = assign_bin_index(xb, fit["edges_b"])
    return fit["marg_a"][ba] + fit["marg_b"][bb] - fit["grand_mean"]


def predict_joint(fit: dict, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    ba = assign_bin_index(xa, fit["edges_a"])
    bb = assign_bin_index(xb, fit["edges_b"])
    return fit["joint_grid"][ba, bb]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def additive_vs_joint_cv(xa: np.ndarray, xb: np.ndarray, y: np.ndarray, folds, n_bins: int = 5) -> dict | None:
    """Grouped-fold out-of-fold comparison: fit ONLY on each fold's training
    rows, evaluate on held-out rows. Returns aggregated additive/joint
    RMSE & R^2, and per-fold "joint beats additive" reproducibility."""
    add_rmse, joint_rmse, add_r2, joint_r2 = [], [], [], []
    joint_better = 0
    for tr, va in folds:
        xa_tr, xb_tr, y_tr = xa[tr], xb[tr], y[tr]
        xa_va, xb_va, y_va = xa[va], xb[va], y[va]
        if len(va) < 10:
            continue
        fit = fit_additive_joint(xa_tr, xb_tr, y_tr, n_bins=n_bins)
        p_add = predict_additive(fit, xa_va, xb_va)
        p_joint = predict_joint(fit, xa_va, xb_va)
        r_add, r_joint = rmse(y_va, p_add), rmse(y_va, p_joint)
        add_rmse.append(r_add)
        joint_rmse.append(r_joint)
        add_r2.append(r2(y_va, p_add))
        joint_r2.append(r2(y_va, p_joint))
        if r_joint < r_add:
            joint_better += 1
    if len(add_rmse) < 3:
        return None
    n = len(add_rmse)
    mean_add_rmse, mean_joint_rmse = float(np.mean(add_rmse)), float(np.mean(joint_rmse))
    rel_improve = (mean_add_rmse - mean_joint_rmse) / mean_add_rmse if mean_add_rmse > 0 else 0.0
    mean_add_r2, mean_joint_r2 = float(np.mean(add_r2)), float(np.mean(joint_r2))
    return dict(
        n_folds_used=n, additive_cv_rmse=mean_add_rmse, joint_cv_rmse=mean_joint_rmse,
        rmse_improvement_pct=round(100 * rel_improve, 3), additive_cv_r2=mean_add_r2, joint_cv_r2=mean_joint_r2,
        incremental_r2=round(mean_joint_r2 - mean_add_r2, 4), folds_joint_better=joint_better,
        joint_beats_additive=bool(joint_better >= max(4, int(round(0.8 * n)))) and (rel_improve > 0.005) and (mean_joint_r2 > mean_add_r2),
    )


# ---------------------------------------------------------------------------
# 3-way: lower-order (sum of 3 pairwise joint surfaces) vs. full 3-way table
# ---------------------------------------------------------------------------

def fit_threeway(xa_tr: np.ndarray, xb_tr: np.ndarray, xc_tr: np.ndarray, y_tr: np.ndarray,
                  n_bins: int = 2, shrink_marginal: float = 5.0, shrink_pairwise: float = 6.0,
                  shrink_full: float = 6.0) -> dict:
    """LOWER-ORDER model = main effects of A,B,C + all three pairwise joint
    surfaces (AB, AC, BC), combined via inclusion-exclusion so main effects
    are not double/triple counted:
        lower_pred(a,b,c) = jointAB(a,b) + jointAC(a,c) + jointBC(b,c)
                             - mainA(a) - mainB(b) - mainC(c) + grand_mean
    (Derivation: with centered main effects mainX = grand+alphaX and
    pairwise joints jointXY = grand+alphaX+alphaY+gammaXY, this formula
    reduces exactly to grand+alphaA+alphaB+alphaC+gammaAB+gammaAC+gammaBC --
    main effects plus all three 2-way interactions, nothing more.)

    FULL 3-way model = an 8-cell (2x2x2 by default) binned table, shrunk
    TOWARD the lower-order prediction for that cell (not toward the grand
    mean) -- a sparse cell falls back to "no 3-way effect assumed" rather
    than "no effect at all," exactly mirroring the 2-way joint-vs-additive
    shrinkage design in fit_additive_joint above.
    """
    grand_mean = float(y_tr.mean())
    bin_a, edges_a = compute_bin_index(xa_tr, n_bins)
    bin_b, edges_b = compute_bin_index(xb_tr, n_bins)
    bin_c, edges_c = compute_bin_index(xc_tr, n_bins)

    main_a = _shrunk_bin_means(bin_a, y_tr, n_bins, grand_mean, shrink_marginal)
    main_b = _shrunk_bin_means(bin_b, y_tr, n_bins, grand_mean, shrink_marginal)
    main_c = _shrunk_bin_means(bin_c, y_tr, n_bins, grand_mean, shrink_marginal)

    def pairwise_joint(bin_x, bin_y_, main_x, main_y_):
        add = main_x[:, None] + main_y_[None, :] - grand_mean
        jsum = np.zeros((n_bins, n_bins))
        jcnt = np.zeros((n_bins, n_bins))
        for i in range(n_bins):
            for j in range(n_bins):
                mask = (bin_x == i) & (bin_y_ == j)
                jcnt[i, j] = mask.sum()
                jsum[i, j] = y_tr[mask].sum()
        return (jsum + shrink_pairwise * add) / (jcnt + shrink_pairwise)

    joint_ab = pairwise_joint(bin_a, bin_b, main_a, main_b)
    joint_ac = pairwise_joint(bin_a, bin_c, main_a, main_c)
    joint_bc = pairwise_joint(bin_b, bin_c, main_b, main_c)

    lower_grid = (joint_ab[:, :, None] + joint_ac[:, None, :] + joint_bc[None, :, :]
                  - main_a[:, None, None] - main_b[None, :, None] - main_c[None, None, :] + grand_mean)

    full_sum = np.zeros((n_bins, n_bins, n_bins))
    full_cnt = np.zeros((n_bins, n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                mask = (bin_a == i) & (bin_b == j) & (bin_c == k)
                full_cnt[i, j, k] = mask.sum()
                full_sum[i, j, k] = y_tr[mask].sum()
    full_grid = (full_sum + shrink_full * lower_grid) / (full_cnt + shrink_full)

    return dict(edges_a=edges_a, edges_b=edges_b, edges_c=edges_c,
                lower_grid=lower_grid, full_grid=full_grid, full_cnt=full_cnt, n_bins=n_bins)


def predict_threeway(fit: dict, xa, xb, xc, which: str) -> np.ndarray:
    ba = assign_bin_index(xa, fit["edges_a"])
    bb = assign_bin_index(xb, fit["edges_b"])
    bc = assign_bin_index(xc, fit["edges_c"])
    grid = fit["lower_grid"] if which == "lower" else fit["full_grid"]
    return grid[ba, bb, bc]


def lower_vs_full_cv(xa: np.ndarray, xb: np.ndarray, xc: np.ndarray, y: np.ndarray, folds, n_bins: int = 2) -> dict | None:
    lower_rmse, full_rmse, lower_r2, full_r2 = [], [], [], []
    full_better = 0
    for tr, va in folds:
        if len(va) < 10:
            continue
        fit = fit_threeway(xa[tr], xb[tr], xc[tr], y[tr], n_bins=n_bins)
        p_lower = predict_threeway(fit, xa[va], xb[va], xc[va], "lower")
        p_full = predict_threeway(fit, xa[va], xb[va], xc[va], "full")
        r_lower, r_full = rmse(y[va], p_lower), rmse(y[va], p_full)
        lower_rmse.append(r_lower)
        full_rmse.append(r_full)
        lower_r2.append(r2(y[va], p_lower))
        full_r2.append(r2(y[va], p_full))
        if r_full < r_lower:
            full_better += 1
    if len(lower_rmse) < 3:
        return None
    n = len(lower_rmse)
    mean_lower_rmse, mean_full_rmse = float(np.mean(lower_rmse)), float(np.mean(full_rmse))
    rel_improve = (mean_lower_rmse - mean_full_rmse) / mean_lower_rmse if mean_lower_rmse > 0 else 0.0
    mean_lower_r2, mean_full_r2 = float(np.mean(lower_r2)), float(np.mean(full_r2))
    return dict(
        n_folds_used=n, lower_cv_rmse=mean_lower_rmse, full_cv_rmse=mean_full_rmse,
        rmse_improvement_pct=round(100 * rel_improve, 3), lower_cv_r2=mean_lower_r2, full_cv_r2=mean_full_r2,
        incremental_r2=round(mean_full_r2 - mean_lower_r2, 4), folds_full_better=full_better,
        full_beats_lower=bool(full_better >= max(4, int(round(0.8 * n)))) and (rel_improve > 0.005) and (mean_full_r2 > mean_lower_r2),
    )


def conditional_effect_text(fit: dict, param_a: str, param_b: str, n_bins_text: int = 3) -> str:
    """Human-readable description of how A's effect changes across B's
    low/medium/high range, from a joint grid re-fit at coarse (3x3)
    resolution on the SAME data the fit dict was built from (call with a
    3-bin fit for readability)."""
    grid = fit["joint_grid"]
    nb = fit["n_bins"]
    if nb != n_bins_text:
        return "n/a (requires a 3-bin fit)"
    labels = ["low", "medium", "high"]
    effect_at_b = {}
    for j in range(3):
        effect_at_b[labels[j]] = grid[2, j] - grid[0, j]  # A: high - low, at each B level
    biggest = max(effect_at_b, key=lambda k: abs(effect_at_b[k]))
    smallest = min(effect_at_b, key=lambda k: abs(effect_at_b[k]))
    direction = lambda v: "an increase" if v > 0 else "a decrease" if v < 0 else "no change"
    return (
        f"Effect of {param_a} (low->high) on Yield: when {param_b} is low = {effect_at_b['low']:+.3f} pt, "
        f"medium = {effect_at_b['medium']:+.3f} pt, high = {effect_at_b['high']:+.3f} pt. "
        f"The effect of {param_a} is largest when {param_b} is {biggest} ({direction(effect_at_b[biggest])} of "
        f"{abs(effect_at_b[biggest]):.3f} pt) and smallest when {param_b} is {smallest} "
        f"({abs(effect_at_b[smallest]):.3f} pt)."
    )
