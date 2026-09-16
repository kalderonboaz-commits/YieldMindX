"""Stage 2 Run B v2, item 4 Stage A: broad, cheap, FULLY VECTORIZED
two-way interaction screen across all numeric-predictor pairs.

Method (generic, no ground-truth tuning, no LightGBM dependency):
  1. For every feature i, fit a cheap quantile-bin-mean "marginal" model
     g_i(x) = E[y | bin(x_i)] directly from raw data (no ML model).
  2. Residualize: r_i = y_centered - g_i -- what's left of y after removing
     feature i's own marginal effect.
  3. Compute each feature's centered quantile-bin index c_i (an
     interaction "basis" -- literally x_i's own rank-position, centered).
  4. Interaction score for pair (i, j) = |corr(r_i, c_j)|, symmetrized with
     |corr(r_j, c_i)| -- does j's value predict systematic structure in y
     AFTER i's own marginal effect is removed (and vice versa)? This
     specifically targets "interaction-only" variables: j can score high
     here even if corr(y, x_j) alone is ~0, because the test is against a
     RESIDUAL, not against y directly.
  5. This is computed for ALL C(1405,2) ~= 986,010 pairs simultaneously via
     ONE matrix multiplication (R_std^T @ C_std), not a per-pair Python
     loop -- tractable at this scale specifically because it reduces to
     dense linear algebra.

This is a screening heuristic, not a confirmed finding -- Stage B applies
expensive, rigorous confirmation (SHAP interaction values + H-statistic +
fold-stability) to only the top-scoring candidates from this stage.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split


def build_marginal_fit_matrix(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str], n_bins: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Returns (G, C): G[:, k] = per-row quantile-bin mean of y for feature
    k (the cheap marginal fit); C[:, k] = per-row centered bin index for
    feature k (the interaction basis)."""
    n = len(X)
    p = len(feature_cols)
    G = np.zeros((n, p), dtype=np.float64)
    C = np.zeros((n, p), dtype=np.float64)
    y_s = pd.Series(y)
    for k, col in enumerate(feature_cols):
        x = X[col]
        try:
            bin_idx, bin_edges = pd.qcut(x, n_bins, duplicates="drop", retbins=True, labels=False)
        except ValueError:
            G[:, k] = y.mean()
            C[:, k] = 0.0
            continue
        bin_idx = pd.Series(bin_idx).fillna(-1).astype(int)
        actual_bins = bin_idx.max() + 1
        if actual_bins <= 1:
            G[:, k] = y.mean()
            C[:, k] = 0.0
            continue
        means = y_s.groupby(bin_idx).mean()
        G[:, k] = bin_idx.map(means).to_numpy()
        centered = bin_idx.to_numpy().astype(np.float64) - (actual_bins - 1) / 2.0
        C[:, k] = centered
    return G, C


def compute_pairwise_screen_scores(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    n = len(X)
    t0 = time.time()
    G, C = build_marginal_fit_matrix(X, y, feature_cols)
    print(f"  marginal-fit matrix built in {time.time()-t0:.1f}s, shape={G.shape}")

    y_c = y - y.mean()
    Resid = y_c[:, None] - G  # (n, p)

    def standardize(M):
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, keepdims=True)
        sd_safe = np.where(sd < 1e-12, 1.0, sd)
        out = (M - mu) / sd_safe
        out[:, (sd < 1e-12).ravel()] = 0.0
        return out

    R_std = standardize(Resid)
    C_std = standardize(C)

    t0 = time.time()
    M = (R_std.T @ C_std) / n  # (p, p): M[i,j] = corr(resid_i, basis_j)
    print(f"  vectorized pairwise correlation matrix computed in {time.time()-t0:.1f}s -- shape={M.shape}, "
          f"{M.shape[0]*(M.shape[0]-1)//2} unique pairs covered")

    score = np.maximum(np.abs(M), np.abs(M.T))
    np.fill_diagonal(score, 0.0)
    return score


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original
    model_cols = [ds.name_map[f] for f in numeric_features]
    n_pairs_total = len(numeric_features) * (len(numeric_features) - 1) // 2
    print(f"Broad pairwise interaction screen: {len(numeric_features)} numeric predictors -> {n_pairs_total} unique pairs")

    y_arr = split.y_train.to_numpy()
    X_num = split.X_train[model_cols]
    X_num.columns = numeric_features  # original names for clarity

    score_matrix = compute_pairwise_screen_scores(X_num, y_arr, numeric_features)

    iu = np.triu_indices(len(numeric_features), k=1)
    scores_flat = score_matrix[iu]
    order = np.argsort(-scores_flat)

    top_k = 200
    rows = []
    for idx in order[:top_k]:
        i, j = iu[0][idx], iu[1][idx]
        rows.append({
            "parameter_A": numeric_features[i],
            "parameter_B": numeric_features[j],
            "broad_screen_score": float(scores_flat[idx]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/stage2_runB_v2_pair_screen.csv", index=False)

    print(f"\nSaved {R}/stage2_runB_v2_pair_screen.csv (top {top_k} of {n_pairs_total} pairs, "
          f"100% of pairs received a score)")
    print(f"\nScore distribution: min={scores_flat.min():.4f} median={np.median(scores_flat):.4f} "
          f"p99={np.percentile(scores_flat,99):.4f} max={scores_flat.max():.4f}")
    print("\nTop 20 candidate pairs:")
    print(df.head(20).to_string(index=False))

    with open(f"{R}/_stage2_v2_pair_screen_coverage.txt", "w") as f:
        f.write(f"numeric_predictors_screened={len(numeric_features)}\n")
        f.write(f"total_unique_pairs={n_pairs_total}\n")
        f.write(f"pairs_with_broad_score=100%\n")
        f.write(f"top_k_selected_for_confirmation={top_k}\n")


if __name__ == "__main__":
    main()
