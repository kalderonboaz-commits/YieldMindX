"""Stage 1 feature-set-size ablation, restricted to original raw-CSV
predictors only (no engineered features anywhere in this experiment).

Tier sizes are data-driven from the Stage 1 OOF cumulative-importance
curve (src/models/stage1_oof_ranking.py). Models: ElasticNet + LightGBM
(Random Forest re-enters at the final common-set fair benchmark, as in the
prior Stage 1 checkpoints, for compute-budget reasons).
"""
from __future__ import annotations

import pandas as pd

from src.data.load import load_dataset
from src.data.splits import make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.feature_set_eval import cross_validate_on_columns, evaluate_holdout_on_columns, summarize_cv
from src.models.stage1_universe import assert_no_engineered_features, get_stage1_predictors
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

    units_df = pd.read_csv(OUTPUTS_DIR / "reports" / "stage1_oof_ranked_units.csv")
    stage1_full = get_stage1_predictors(fm)  # 1,425 -- full original safe set (numeric + categorical, no engineered)

    tiers = {
        "S1_full_original_1425": [c for c in stage1_full if c not in fm.categorical_cols],  # numeric-only part; categoricals auto-added by eval fn
        "S1_moderate_90pct": build_tier_columns(units_df, 0.90),
        "S1_aggressive_80pct": build_tier_columns(units_df, 0.80),
        "S1_compact_70pct": build_tier_columns(units_df, 0.70),
        "S1_extreme_50pct_diagnostic": build_tier_columns(units_df, 0.50),
    }

    for name, cols in tiers.items():
        assert_no_engineered_features(cols, fm)

    print("Stage 1 tier sizes (numeric-cluster units; +20 categorical auto-added by feature_set_eval):")
    for name, cols in tiers.items():
        print(f"  {name}: {len(cols)} + 20 categorical = {len(cols)+20} total")

    all_cv, all_holdout = [], []
    for tier_name, cols in tiers.items():
        print(f"\n=== STAGE 1 TIER: {tier_name} ===")
        cv_df = cross_validate_on_columns(fm_train, y_train, groups_train, cols, models=("elasticnet", "lightgbm"))
        cv_df["tier"] = tier_name
        all_cv.append(cv_df)

        hold_df = evaluate_holdout_on_columns(fm_train, y_train, fm_test, y_test, cols, models=("elasticnet", "lightgbm"))
        hold_df["tier"] = tier_name
        all_holdout.append(hold_df)

        print(f"CV summary:\n{summarize_cv(cv_df).round(4).to_string()}")
        print(f"Holdout:\n{hold_df.round(4).to_string(index=False)}")

    cv_all = pd.concat(all_cv, ignore_index=True)
    holdout_all = pd.concat(all_holdout, ignore_index=True)

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    cv_all.to_csv(OUTPUTS_DIR / "reports" / "stage1_ablation_cv_results.csv", index=False)
    holdout_all.to_csv(OUTPUTS_DIR / "reports" / "stage1_ablation_holdout_results.csv", index=False)

    print("\n=== FINAL STAGE 1 ABLATION SUMMARY (CV mean/std by tier x model) ===")
    print(cv_all.groupby(["tier", "model"])[["rmse", "mae", "r2"]].agg(["mean", "std"]).round(4).to_string())
    print("\n=== HOLDOUT TABLE ===")
    print(holdout_all.round(4).to_string(index=False))
    print("\nSaved stage1_ablation_cv_results.csv / stage1_ablation_holdout_results.csv")


if __name__ == "__main__":
    main()
