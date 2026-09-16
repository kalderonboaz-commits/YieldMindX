"""Part 2 + Part 3 experiment: feature-set-size ablation, and engineered-
features-help-or-not ablation.

Tier sizes are DATA-DRIVEN from the leakage-safe out-of-fold importance
cumulative curve (src/models/oof_feature_ranking.py), not chosen a priori:
  - moderate  = smallest unit count reaching 90% cumulative OOF importance
  - aggressive = smallest unit count reaching 80%
  - compact    = smallest unit count reaching 70%
  - (diagnostic) extreme = smallest unit count reaching 50%, included only
    to show where the performance-vs-size curve actually breaks, not
    proposed as a serious candidate

Models: ElasticNet + LightGBM only, for compute-budget reasons (Random
Forest is already established in prior checkpoints as dominated by
LightGBM on every metric at the full feature count, so its feature-count
sensitivity is not the open question this experiment needs to answer;
it re-enters the analysis for Part 4/5's final common-set benchmark).
"""
from __future__ import annotations

import pandas as pd

from src.data.load import load_dataset
from src.data.splits import make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.feature_set_eval import (
    cross_validate_on_columns,
    evaluate_holdout_on_columns,
    summarize_cv,
)
from src.models.train import OUTPUTS_DIR


def build_tier_columns(units_df: pd.DataFrame, cumulative_cutoff: float) -> list[str]:
    n = int((units_df["cumulative_fraction"] <= cumulative_cutoff).sum()) + 1
    n = min(n, len(units_df))
    return units_df["unit"].head(n).tolist()


def main():
    ds = load_dataset()
    fm = build_feature_matrix(ds)
    holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=42)

    fm_train = FeatureMatrix(
        X=fm.X.iloc[holdout.train_idx].reset_index(drop=True),
        numeric_cols=fm.numeric_cols, engineered_cols=fm.engineered_cols,
        categorical_cols=fm.categorical_cols, clustering=fm.clustering,
        name_map=fm.name_map, inverse_name_map=fm.inverse_name_map,
    )
    fm_test = FeatureMatrix(
        X=fm.X.iloc[holdout.test_idx].reset_index(drop=True),
        numeric_cols=fm.numeric_cols, engineered_cols=fm.engineered_cols,
        categorical_cols=fm.categorical_cols, clustering=fm.clustering,
        name_map=fm.name_map, inverse_name_map=fm.inverse_name_map,
    )
    y_train = ds.y.iloc[holdout.train_idx].reset_index(drop=True)
    y_test = ds.y.iloc[holdout.test_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)

    units_df = pd.read_csv(OUTPUTS_DIR / "reports" / "oof_ranked_units.csv")

    all_numeric_and_engineered = fm.numeric_cols + fm.engineered_cols
    raw_numeric_only = fm.numeric_cols  # no engineered features at all (Part 3 "A")
    cluster_reps_all = units_df["unit"].tolist()  # 1294 units = de-collinearized "full" tier

    tiers = {
        "T0_full_safe_1732": all_numeric_and_engineered,
        "T1_decollinearized_1314": cluster_reps_all,
        "T2_moderate_90pct": build_tier_columns(units_df, 0.90),
        "T3_aggressive_80pct": build_tier_columns(units_df, 0.80),
        "T4_compact_70pct": build_tier_columns(units_df, 0.70),
        "T5_extreme_50pct_diagnostic_only": build_tier_columns(units_df, 0.50),
        "PartA_raw_only_no_engineered_1405": raw_numeric_only,
    }

    print("Tier sizes (excluding the 20 always-included categorical dummy columns):")
    for name, cols in tiers.items():
        print(f"  {name}: {len(cols)} predictor columns (+20 categorical = {len(cols)+20} total)")

    all_cv_results = []
    all_holdout_results = []
    for tier_name, cols in tiers.items():
        print(f"\n=== TIER: {tier_name} ({len(cols)} + 20 categorical) ===")
        cv_df = cross_validate_on_columns(fm_train, y_train, groups_train, cols, models=("elasticnet", "lightgbm"))
        cv_df["tier"] = tier_name
        all_cv_results.append(cv_df)

        hold_df = evaluate_holdout_on_columns(fm_train, y_train, fm_test, y_test, cols, models=("elasticnet", "lightgbm"))
        hold_df["tier"] = tier_name
        all_holdout_results.append(hold_df)

        summary = summarize_cv(cv_df)
        print(f"CV summary for {tier_name}:")
        print(summary.round(4).to_string())
        print(f"Holdout for {tier_name}:")
        print(hold_df.round(4).to_string(index=False))

    cv_all = pd.concat(all_cv_results, ignore_index=True)
    holdout_all = pd.concat(all_holdout_results, ignore_index=True)

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    cv_all.to_csv(OUTPUTS_DIR / "reports" / "feature_ablation_cv_results.csv", index=False)
    holdout_all.to_csv(OUTPUTS_DIR / "reports" / "feature_ablation_holdout_results.csv", index=False)

    print("\n=== FINAL SUMMARY TABLE (CV mean/std by tier x model) ===")
    summary_all = cv_all.groupby(["tier", "model"])[["rmse", "mae", "r2"]].agg(["mean", "std"])
    print(summary_all.round(4).to_string())

    print("\n=== HOLDOUT TABLE (tier x model) ===")
    print(holdout_all.round(4).to_string(index=False))

    print("\nSaved outputs/reports/feature_ablation_cv_results.csv")
    print("Saved outputs/reports/feature_ablation_holdout_results.csv")


if __name__ == "__main__":
    main()
