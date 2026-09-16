"""Stage 2 Run B v3 -- broad 2-way interaction screen, TWO conceptually
different full-coverage scores:

  score_1 (reused from v2): residual-vs-bin-index correlation proxy --
    |corr(y - marginal_fit(A), bin_index(B))|, symmetrized. A linear-in-
    rank interaction-basis test.
  score_2 (new): full 2D-binned Tukey non-additivity energy -- count-
    weighted mean squared deviation of the actual 2D joint cell mean from
    the additive (marginal_A + marginal_B) prediction. Directly measures
    "does the joint response depart from the additive sum," independent of
    any linear/rank-based assumption -- a materially different test from
    score_1 (score_1 can be near-zero for a symmetric, non-monotonic-in-
    either-marginal interaction that score_2 would still catch, since
    score_2 makes no linearity assumption about the interaction basis).

Both are computed for all C(1405,2) = 986,310 pairs via dense matmuls
(<2 seconds combined) -- 100% coverage, not a sample.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v2_pair_screen import compute_pairwise_screen_scores
from src.explain.stage2_v3_common import compute_nonadditivity_screen


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original
    model_cols = [ds.name_map[f] for f in numeric_features]
    n_pairs_total = len(numeric_features) * (len(numeric_features) - 1) // 2
    print(f"v3 broad 2-way screen: {len(numeric_features)} numeric predictors -> {n_pairs_total} unique pairs, two scores")

    y_arr = split.y_train.to_numpy()
    X_num = split.X_train[model_cols]
    X_num.columns = numeric_features
    X_arr = X_num.to_numpy(dtype=np.float64)

    t0 = time.time()
    score1_matrix = compute_pairwise_screen_scores(X_num, y_arr, numeric_features)
    print(f"score_1 (marginal-residual proxy, reused from v2) computed in {time.time()-t0:.2f}s")

    t0 = time.time()
    score2_matrix = compute_nonadditivity_screen(X_arr, y_arr, n_bins=4)
    print(f"score_2 (2D-binned non-additivity energy, NEW) computed in {time.time()-t0:.2f}s")

    n = len(numeric_features)
    iu = np.triu_indices(n, k=1)
    s1_flat, s2_flat = score1_matrix[iu], score2_matrix[iu]

    rank1 = pd.Series(-s1_flat).rank(method="min").astype(int).to_numpy()
    rank2 = pd.Series(-s2_flat).rank(method="min").astype(int).to_numpy()

    top_k = 300
    top1_idx = set(np.argsort(-s1_flat)[:top_k])
    top2_idx = set(np.argsort(-s2_flat)[:top_k])
    union_idx = sorted(top1_idx | top2_idx)

    rows = []
    for idx in union_idx:
        i, j = iu[0][idx], iu[1][idx]
        rows.append({
            "parameter_A": numeric_features[i],
            "parameter_B": numeric_features[j],
            "broad_screen_score_1": float(s1_flat[idx]),
            "broad_screen_score_2": float(s2_flat[idx]),
            "rank_score_1": int(rank1[idx]),
            "rank_score_2": int(rank2[idx]),
            "in_top300_score_1": idx in top1_idx,
            "in_top300_score_2": idx in top2_idx,
            "both_top300": (idx in top1_idx) and (idx in top2_idx),
        })
    df = pd.DataFrame(rows).sort_values(["both_top300", "broad_screen_score_2"], ascending=[False, False])
    out_path = f"{R}/stage2_runB_v3_pair_screen.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows -- union of top-{top_k} by either score, out of {n_pairs_total} 100%-covered pairs)")
    print(f"Pairs in top-{top_k} of BOTH scores (agreement): {int(df['both_top300'].sum())}")
    print(f"Pairs in top-{top_k} of score_1 only: {int((df['in_top300_score_1'] & ~df['in_top300_score_2']).sum())}")
    print(f"Pairs in top-{top_k} of score_2 only: {int((df['in_top300_score_2'] & ~df['in_top300_score_1']).sum())}")

    rank_corr = pd.Series(s1_flat).corr(pd.Series(s2_flat), method="spearman")
    print(f"\nFull-universe Spearman rank correlation between score_1 and score_2: {rank_corr:.4f}")

    with open(f"{R}/_stage2_v3_pair_screen_coverage.txt", "w") as f:
        f.write(f"numeric_predictors_screened={len(numeric_features)}\n")
        f.write(f"total_unique_pairs={n_pairs_total}\n")
        f.write(f"pairs_with_both_broad_scores=100%\n")
        f.write(f"top_k_per_score={top_k}\n")
        f.write(f"union_size={len(df)}\n")
        f.write(f"both_top{top_k}_agreement={int(df['both_top300'].sum())}\n")
        f.write(f"score1_score2_spearman_rank_corr={rank_corr:.4f}\n")

    print("\nTop 15 by score_2 (non-additivity energy):")
    print(df.sort_values('broad_screen_score_2', ascending=False).head(15)[
        ["parameter_A", "parameter_B", "broad_screen_score_1", "broad_screen_score_2", "rank_score_1", "rank_score_2"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
