"""Relationship Validation Engine -- single-parameter validation.

For all 1,405 eligible numeric Run B predictors: BASELINE (linear) vs
NONLINEAR (spline) grouped 5-fold CV comparison (reusing the project's
existing spline/GAM machinery unmodified), with a genuine permutation-based
empirical p-value (pooled null from B_PERM target shuffles shared across
all 1,405 features per shuffle -- cheap, valid) and Benjamini-Hochberg FDR
control as the PRIMARY acceptance gate, on top of (not instead of)
grouped-fold stability. Shape/turning-point characterization (reusing
classify_curve_shape unmodified) is computed only for the real target, not
for the null permutations (irrelevant under a null).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_relval_common import (
    CACHE_DIR, R, RANDOM_STATE, benjamini_hochberg, empirical_pvalue, group_block_shuffle, load_locked_split,
    single_param_cv,
)
from src.explain.stage2_spline_common import classify_curve_shape, fit_spline, predict_grid

B_PERM = 8
ALPHA_FDR = 0.05
MIN_FOLD_FRACTION_STRONG = 1.0    # was: absolute count 5 (dormant bug -- see audit; fraction is correct for candidates with <5 valid folds)
MIN_FOLD_FRACTION_MODERATE = 0.8  # was: absolute count 4


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original
    n_features = len(numeric_features)
    y_train_arr = split.y_train.to_numpy()
    y_std = float(np.std(y_train_arr))
    print(f"Single-parameter validation: {n_features} eligible numeric Run B predictors (100% target coverage)")

    t0 = time.time()
    real_rows = []
    real_stat = {}  # feature -> delta_r2 (primary test statistic)
    for i, f in enumerate(numeric_features):
        mcol = ds.name_map[f]
        x_full = split.X_train[mcol].to_numpy(dtype=float)
        cv = single_param_cv(x_full, y_train_arr, split.folds)
        if cv is None:
            real_rows.append({"raw_csv_parameter_name": f, "_skip": True})
            real_stat[f] = -np.inf
            continue

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
        shape_result = classify_curve_shape(grid_x, main_curve, se_curve, y_std)

        shape_matches = 0
        for c in fold_curves:
            fc = classify_curve_shape(grid_x, c, se_curve, y_std)
            if fc["shape"] == shape_result["shape"]:
                shape_matches += 1
        n_fc = len(fold_curves)

        tps = shape_result["turning_points"]
        real_rows.append({
            "raw_csv_parameter_name": f, "_skip": False,
            "baseline_cv_rmse": round(cv["baseline_cv_rmse"], 4), "nonlinear_cv_rmse": round(cv["nonlinear_cv_rmse"], 4),
            "delta_rmse": round(cv["delta_rmse"], 4),
            "baseline_cv_mae": round(cv["baseline_cv_mae"], 4), "nonlinear_cv_mae": round(cv["nonlinear_cv_mae"], 4),
            "delta_mae": round(cv["delta_mae"], 4),
            "baseline_cv_r2": round(cv["baseline_cv_r2"], 4), "nonlinear_cv_r2": round(cv["nonlinear_cv_r2"], 4),
            "delta_r2": round(cv["delta_r2"], 6),
            "folds_improved": f"{cv['folds_improved']}/{cv['n_folds']}",
            "relationship_class": shape_result["shape"], "direction": shape_result["direction"],
            "turning_point_1": tps[0] if len(tps) > 0 else None,
            "turning_point_2": tps[1] if len(tps) > 1 else None,
            "beneficial_range": shape_result["beneficial_range"], "harmful_range": shape_result["harmful_range"],
            "effect_size_yield_pct": shape_result["effect_range"],
            "shape_fold_consistency": f"{shape_matches}/{n_fc}" if n_fc else "0/0",
        })
        real_stat[f] = cv["delta_r2"]
        if (i + 1) % 200 == 0:
            print(f"  real-target: {i+1}/{n_features} ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"Real-target pass done in {time.time()-t0:.1f}s")

    print(f"\nRunning {B_PERM} shuffled-target permutations (pooled empirical null, cheap CV-only path)...")
    null_pool = []
    t0 = time.time()
    groups_train = split.groups_train.to_numpy()
    for b in range(B_PERM):
        rng = np.random.RandomState(RANDOM_STATE + 1000 + b)
        y_shuf = group_block_shuffle(y_train_arr, groups_train, rng)
        for f in numeric_features:
            mcol = ds.name_map[f]
            x_full = split.X_train[mcol].to_numpy(dtype=float)
            cv = single_param_cv(x_full, y_shuf, split.folds)
            if cv is not None:
                null_pool.append(cv["delta_r2"])
        print(f"  permutation {b+1}/{B_PERM} done ({time.time()-t0:.0f}s elapsed, pool size={len(null_pool)})", flush=True)
    null_pool = np.array(null_pool)
    print(f"Permutation null pass done in {time.time()-t0:.1f}s -- pooled null size={len(null_pool)}")

    effect_size_95th = float(np.percentile(null_pool, 95))
    print(f"Empirical null 95th percentile of delta_r2 (meaningful-effect-size reference): {effect_size_95th:.6f}")

    df = pd.DataFrame(real_rows)
    df = df[~df["_skip"]].drop(columns=["_skip"]).reset_index(drop=True)

    pvals = np.array([empirical_pvalue(real_stat[f], null_pool) for f in df["raw_csv_parameter_name"]])
    qvals, accept = benjamini_hochberg(pvals, alpha=ALPHA_FDR)
    df["raw_statistic_delta_r2"] = [round(real_stat[f], 6) for f in df["raw_csv_parameter_name"]]
    df["empirical_p_value"] = np.round(pvals, 6)
    df["bh_adjusted_q_value"] = np.round(qvals, 6)
    df["survives_fdr"] = accept

    def label_row(r):
        fi_str = r["folds_improved"]
        fi, fn = (int(x) for x in fi_str.split("/"))
        meaningful = r["raw_statistic_delta_r2"] > effect_size_95th
        if r["relationship_class"] in ("FLAT", "INCONCLUSIVE"):
            return "INCONCLUSIVE"
        if not r["survives_fdr"]:
            return "INCONCLUSIVE"
        if fn > 0 and (fi / fn) >= MIN_FOLD_FRACTION_STRONG and meaningful:
            return "STRONG"
        if fn > 0 and (fi / fn) >= MIN_FOLD_FRACTION_MODERATE and meaningful:
            return "MODERATE"
        if fi >= 2 or meaningful:
            return "WEAK"
        return "INCONCLUSIVE"

    df["evidence_label"] = df.apply(label_row, axis=1)

    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "raw_statistic_delta_r2"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s, ascending=[True, False])

    out_path = f"{R}/stage2_runB_relationship_validation_single.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)}/{n_features} predictors, {100*len(df)/n_features:.1f}% coverage)")
    print(df["evidence_label"].value_counts().to_string())
    print(f"\nHypotheses tested: {len(df)} | Survive FDR (q<={ALPHA_FDR}): {int(df['survives_fdr'].sum())}")

    with open(f"{CACHE_DIR}/single_null_summary.json", "w") as f:
        json.dump({
            "family": "single_parameter", "n_hypotheses": int(len(df)), "b_perm": B_PERM,
            "null_pool_size": int(len(null_pool)),
            "null_delta_r2_mean": float(np.mean(null_pool)), "null_delta_r2_95th_pctile": effect_size_95th,
            "null_delta_r2_99th_pctile": float(np.percentile(null_pool, 99)),
            "real_strong_count": int((df["evidence_label"] == "STRONG").sum()),
            "real_moderate_count": int((df["evidence_label"] == "MODERATE").sum()),
            "real_survives_fdr_count": int(df["survives_fdr"].sum()),
        }, f, indent=2)
    np.save(f"{CACHE_DIR}/single_null_pool.npy", null_pool)
    print(f"Saved null-pool summary to {CACHE_DIR}/single_null_summary.json")

    print("\nTop 15 STRONG by delta_r2:")
    print(df[df["evidence_label"] == "STRONG"].head(15)[
        ["raw_csv_parameter_name", "relationship_class", "direction", "raw_statistic_delta_r2",
         "folds_improved", "bh_adjusted_q_value"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
