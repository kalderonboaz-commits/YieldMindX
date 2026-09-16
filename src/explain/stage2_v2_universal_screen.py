"""Stage 2 Run B v2, item 3: full univariate nonlinear screen.

Applies a MODEL-FREE quantile-binned shape classifier (classify_binned_shape)
to every one of the 1,405 numeric original predictors (categorical one-hot
dummies are out of scope for a continuous-shape screen by design, not a
coverage gap -- a binary indicator has no "shape" to classify). This does
NOT depend on LightGBM having used the feature in any tree -- a feature
with zero split importance is still fully testable here, directly
addressing the coverage gap identified in the v1 checkpoint (Gate CD,
Rc_an_ct2, TOPSIN_Gm_max all had exactly zero LightGBM importance and were
invisible to the v1 screening-based approach).

Stability: the shape/direction classification is recomputed independently
on each of 5 GroupKFold (by LotName) training folds, using ONLY that
fold's own rows -- consistent with every other stability check in this
project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import (
    R, classify_binned_shape, evidence_label_from_consistency, load_locked_split,
)


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original  # 1,405 original names, category B only
    print(f"Full univariate nonlinear screen: {len(numeric_features)} numeric predictors (100% of category-B universe)")

    y_train_arr = split.y_train.to_numpy()

    rows = []
    for i, f in enumerate(numeric_features):
        mcol = ds.name_map[f]
        x_full = split.X_train[mcol].to_numpy()
        main_shape, main_dir, centers, means = classify_binned_shape(x_full, y_train_arr, n_bins=8)

        per_fold_shapes, per_fold_dirs = [], []
        for tr, va in split.folds:
            x_fold = split.X_train.iloc[va][mcol].to_numpy()
            y_fold = split.y_train.iloc[va].to_numpy()
            shp, drc, _, _ = classify_binned_shape(x_fold, y_fold, n_bins=6, min_bin_size=8)
            per_fold_shapes.append(shp)
            per_fold_dirs.append(drc)

        shape_consistency = sum(1 for s in per_fold_shapes if s == main_shape)
        direction_consistency = sum(1 for d in per_fold_dirs if d == main_dir) if main_dir != "n/a" else shape_consistency
        label, reason = evidence_label_from_consistency(shape_consistency, direction_consistency, 5, main_shape)

        rng = float(means.max() - means.min()) if len(means) else 0.0
        rows.append({
            "raw_csv_parameter_name": f,
            "shape": main_shape,
            "direction": main_dir,
            "effect_range_yield_pct": round(rng, 4),
            "shape_fold_consistency": f"{shape_consistency}/5",
            "direction_fold_consistency": f"{direction_consistency}/5",
            "evidence_label": label,
            "reason": reason,
            "observed_x_min": round(float(np.nanmin(x_full)), 6) if len(x_full) else None,
            "observed_x_max": round(float(np.nanmax(x_full)), 6) if len(x_full) else None,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(numeric_features)} screened...", flush=True)

    df = pd.DataFrame(rows).sort_values(["evidence_label", "effect_range_yield_pct"],
                                          key=lambda s: s.map({"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}) if s.name == "evidence_label" else s,
                                          ascending=[True, False])
    df.to_csv(f"{R}/stage2_runB_v2_univariate_all.csv", index=False)
    print(f"\nSaved {R}/stage2_runB_v2_univariate_all.csv ({len(df)} rows -- {len(df)}/{len(numeric_features)} coverage = 100%)")
    print("\nEvidence label counts:")
    print(df["evidence_label"].value_counts().to_string())
    print("\nShape counts (main split):")
    print(df["shape"].value_counts().to_string())

    print("\nTop 15 STRONG findings by effect range:")
    print(df[df["evidence_label"] == "STRONG"].head(15)[
        ["raw_csv_parameter_name", "shape", "direction", "effect_range_yield_pct"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
