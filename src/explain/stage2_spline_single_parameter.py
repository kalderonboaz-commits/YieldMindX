"""Stage 2 Run B -- complementary spline/GAM-style single-parameter
relationship discovery.

Purpose (per project instruction): a SECOND, complementary method for
smooth/nonlinear single-parameter effects, targeted at shapes the existing
LightGBM+SHAP/PDP (v1) and model-free quantile-binning (v2) methods may
miss or characterize poorly. This does NOT replace, retrain, or modify
either existing method, Stage 1, or the Run B feature universe. No tree
model, no neural network -- SplineTransformer + Ridge/RidgeCV only, from
the existing scikit-learn install.

Covers all 1,405 original numeric Run B predictors (category B; excludes
the 20 categorical one-hot dummies, which have no continuous "shape" to
fit a spline to, and excludes all 307 engineered predictors and the 4
review columns, consistent with the locked Run B universe).

Ground-truth document is never read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_spline_common import (
    classify_curve_shape, cv_metrics, evidence_label_spline, fit_spline, predict_grid,
)
from src.explain.stage2_v2_common import R, load_locked_split


def analyze_feature(f: str, ds, split, y_train_arr: np.ndarray, y_std: float) -> dict:
    mcol = ds.name_map[f]
    x_full = split.X_train[mcol].to_numpy(dtype=float)
    n_used = int(np.sum(~np.isnan(x_full)))

    if np.unique(x_full[~np.isnan(x_full)]).size < 10 or np.nanstd(x_full) < 1e-12:
        return dict(raw_csv_parameter_name=f, relationship_class="INCONCLUSIVE", direction="n/a",
                    turning_point_1=None, turning_point_2=None, beneficial_range=None, harmful_range=None,
                    effect_size_yield_pct=0.0, shape_fold_consistency="0/0", n_fold_curves_used=0,
                    linear_cv_rmse=None, spline_cv_rmse=None, rmse_improvement_pct=None,
                    linear_cv_r2=None, spline_cv_r2=None, folds_spline_better=None, nonlinear_advantage=False,
                    evidence_label="INCONCLUSIVE", reason="Fewer than 10 unique values or near-constant -- not a fittable continuous predictor.",
                    n_train_rows_used=n_used)

    cvres = cv_metrics(x_full, y_train_arr, split.folds)
    if cvres is None:
        return dict(raw_csv_parameter_name=f, relationship_class="INCONCLUSIVE", direction="n/a",
                    turning_point_1=None, turning_point_2=None, beneficial_range=None, harmful_range=None,
                    effect_size_yield_pct=0.0, shape_fold_consistency="0/0", n_fold_curves_used=0,
                    linear_cv_rmse=None, spline_cv_rmse=None, rmse_improvement_pct=None,
                    linear_cv_r2=None, spline_cv_r2=None, folds_spline_better=None, nonlinear_advantage=False,
                    evidence_label="INCONCLUSIVE", reason="Fewer than 3 usable grouped folds for this feature.",
                    n_train_rows_used=n_used)

    main_model = fit_spline(x_full, y_train_arr)
    grid_x, main_curve = predict_grid(main_model, x_full)

    fold_curves = []
    for tr, va in split.folds:
        x_tr, y_tr = x_full[tr], y_train_arr[tr]
        if np.unique(x_tr).size < 8:
            continue
        m = fit_spline(x_tr, y_tr)
        _, c = predict_grid(m, x_full, grid=grid_x)
        fold_curves.append(c)
    se_curve = np.std(np.array(fold_curves), axis=0, ddof=1) if len(fold_curves) >= 2 else np.zeros_like(grid_x)

    main_result = classify_curve_shape(grid_x, main_curve, se_curve, y_std)

    shape_matches, dir_matches = 0, 0
    for c in fold_curves:
        fc_result = classify_curve_shape(grid_x, c, se_curve, y_std)
        if fc_result["shape"] == main_result["shape"]:
            shape_matches += 1
        if fc_result["direction"] == main_result["direction"]:
            dir_matches += 1
    n_fc = len(fold_curves)

    label, reason = evidence_label_spline(main_result["shape"], shape_matches, n_fc, cvres["spline_cv_r2"])

    tps = main_result["turning_points"]
    return dict(
        raw_csv_parameter_name=f,
        relationship_class=main_result["shape"],
        direction=main_result["direction"],
        turning_point_1=tps[0] if len(tps) > 0 else None,
        turning_point_2=tps[1] if len(tps) > 1 else None,
        beneficial_range=main_result["beneficial_range"],
        harmful_range=main_result["harmful_range"],
        effect_size_yield_pct=main_result["effect_range"],
        shape_fold_consistency=f"{shape_matches}/{n_fc}" if n_fc else "0/0",
        n_fold_curves_used=n_fc,
        linear_cv_rmse=round(cvres["linear_cv_rmse"], 4),
        spline_cv_rmse=round(cvres["spline_cv_rmse"], 4),
        rmse_improvement_pct=cvres["rmse_improvement_pct"],
        linear_cv_r2=round(cvres["linear_cv_r2"], 4),
        spline_cv_r2=round(cvres["spline_cv_r2"], 4),
        folds_spline_better=f"{cvres['folds_spline_better']}/{cvres['n_folds_used']}",
        nonlinear_advantage=cvres["nonlinear_advantage"],
        evidence_label=label,
        reason=reason,
        n_train_rows_used=n_used,
    )


def run_all(y_override: np.ndarray | None = None) -> pd.DataFrame:
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original
    y_train_arr = y_override if y_override is not None else split.y_train.to_numpy()
    y_std = float(np.std(y_train_arr))

    rows = []
    for i, f in enumerate(numeric_features):
        rows.append(analyze_feature(f, ds, split, y_train_arr, y_std))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(numeric_features)} features analyzed...", flush=True)

    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = pd.DataFrame(rows)
    df = df.sort_values(["evidence_label", "effect_size_yield_pct"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s,
                         ascending=[True, False])
    return df


def main():
    print("Spline/GAM-style single-parameter screen: 1,405 original numeric Run B predictors "
          "(SplineTransformer + RidgeCV vs. LinearRegression null, GroupKFold by LotName).")
    df = run_all()
    out_path = f"{R}/stage2_runB_spline_single_parameter.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows -- {len(df)}/1405 coverage = 100%)")

    print("\nEvidence label counts:")
    print(df["evidence_label"].value_counts().to_string())
    print("\nRelationship class counts (main split):")
    print(df["relationship_class"].value_counts().to_string())
    print("\nNonlinear-advantage counts (spline genuinely beats linear, reproducibly):")
    print(df["nonlinear_advantage"].value_counts().to_string())

    print("\nTop 15 STRONG findings by effect size:")
    print(df[df["evidence_label"] == "STRONG"].head(15)[
        ["raw_csv_parameter_name", "relationship_class", "direction", "effect_size_yield_pct", "nonlinear_advantage"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
