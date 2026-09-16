"""Stage 2 Run B v2, item 8: shuffle-based negative control for the CHEAP
screening stages (univariate screen + pairwise broad screen). This
estimates the false-discovery baseline of the fast, model-free methods --
i.e. how many "STRONG"/high-scoring findings would appear purely by chance
on a target with no real relationship to any predictor.

Scoped to the cheap stages only (explicitly per instruction: "a limited,
computationally reasonable subset") -- the expensive LightGBM-based
confirmation and 3-way search are NOT re-run under shuffling here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, classify_binned_shape, evidence_label_from_consistency, load_locked_split
from src.explain.stage2_v2_pair_screen import compute_pairwise_screen_scores

RANDOM_STATE = 42


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original

    rng = np.random.default_rng(RANDOM_STATE)
    y_shuffled = rng.permutation(split.y_train.to_numpy())

    print("=== Negative control 1: univariate screen on SHUFFLED target ===")
    strong_count = 0
    n_test = len(numeric_features)
    shape_counts = {}
    for f in numeric_features:
        mcol = ds.name_map[f]
        x_full = split.X_train[mcol].to_numpy()
        main_shape, main_dir, _, _ = classify_binned_shape(x_full, y_shuffled, n_bins=8)

        per_fold_shapes, per_fold_dirs = [], []
        for tr, va in split.folds:
            x_fold = split.X_train.iloc[va][mcol].to_numpy()
            y_fold = y_shuffled[va]
            shp, drc, _, _ = classify_binned_shape(x_fold, y_fold, n_bins=6, min_bin_size=8)
            per_fold_shapes.append(shp)
            per_fold_dirs.append(drc)
        sc = sum(1 for s in per_fold_shapes if s == main_shape)
        dc = sum(1 for d in per_fold_dirs if d == main_dir) if main_dir != "n/a" else sc
        label, _ = evidence_label_from_consistency(sc, dc, 5, main_shape)
        if label == "STRONG":
            strong_count += 1
        shape_counts[main_shape] = shape_counts.get(main_shape, 0) + 1

    print(f"Univariate: {strong_count}/{n_test} ({100*strong_count/n_test:.1f}%) features labeled STRONG under a SHUFFLED target")
    print(f"  (real-target result was 180/1405 = 12.8% STRONG -- compare directly)")
    print(f"  shuffled shape distribution: {shape_counts}")

    print("\n=== Negative control 2: pairwise broad-screen score distribution on SHUFFLED target ===")
    model_cols = [ds.name_map[f] for f in numeric_features]
    X_num = split.X_train[model_cols]
    X_num.columns = numeric_features
    score_matrix_shuffled = compute_pairwise_screen_scores(X_num, y_shuffled, numeric_features)
    iu = np.triu_indices(len(numeric_features), k=1)
    scores_shuffled = score_matrix_shuffled[iu]
    print(f"Shuffled pairwise score distribution: min={scores_shuffled.min():.4f} "
          f"median={np.median(scores_shuffled):.4f} p99={np.percentile(scores_shuffled,99):.4f} "
          f"p99.98={np.percentile(scores_shuffled,99.98):.4f} max={scores_shuffled.max():.4f}")

    real_pool = pd.read_csv(f"{R}/stage2_runB_v2_pair_screen.csv")
    real_top200_min_score = real_pool["broad_screen_score"].min()
    frac_shuffled_exceeding = float((scores_shuffled >= real_top200_min_score).mean())
    n_shuffled_exceeding = int((scores_shuffled >= real_top200_min_score).sum())
    print(f"\nReal-target top-200 cutoff score = {real_top200_min_score:.4f}")
    print(f"Fraction of ALL {len(scores_shuffled)} shuffled-target pair scores that would exceed this cutoff: "
          f"{frac_shuffled_exceeding:.4%} ({n_shuffled_exceeding} pairs)")
    print("This is the empirical false-discovery baseline for the broad-screen cutoff used to select "
          "the top-200 candidates that fed Stage B confirmation.")

    with open(f"{R}/stage2_runB_v2_negative_control_summary.txt", "w") as f:
        f.write("STAGE 2 RUN B v2 -- NEGATIVE CONTROL (TARGET-SHUFFLE) SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("Scope: cheap screening stages only (univariate + pairwise broad screen).\n")
        f.write("Expensive LightGBM-based confirmation and 3-way search were NOT re-run\n")
        f.write("under shuffling (computationally reasonable subset, per instruction).\n\n")
        f.write(f"Univariate screen: {strong_count}/{n_test} ({100*strong_count/n_test:.1f}%) features STRONG "
                f"under shuffled target (real-target result: 180/1405 = 12.8%)\n\n")
        f.write(f"Pairwise broad screen: real-target top-200 cutoff score = {real_top200_min_score:.4f}\n")
        f.write(f"Fraction of shuffled-target pair scores exceeding this cutoff: "
                f"{frac_shuffled_exceeding:.4%} ({n_shuffled_exceeding} of {len(scores_shuffled)} pairs)\n\n")
        f.write("Interpretation: if the shuffled-target STRONG rate / cutoff-exceedance rate is small relative\n")
        f.write("to the real-target rate, this supports (but does not prove) that the real-target findings\n")
        f.write("reflect genuine structure rather than the screening methods' inherent false-positive rate.\n")
        f.write("This is NOT a formal multiple-testing correction (e.g. no FDR q-values are computed) --\n")
        f.write("it is a single-shuffle empirical baseline, explicitly scoped to the cheap stages only.\n")

    print(f"\nSaved {R}/stage2_runB_v2_negative_control_summary.txt")


if __name__ == "__main__":
    main()
