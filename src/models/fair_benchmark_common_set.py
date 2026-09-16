"""Part 4/5: fair same-input benchmark of ElasticNet, Random Forest, and
LightGBM on ONE common predictor set (all three models receive EXACTLY
the same columns, same folds, same holdout).

Common set choice (see feature_ablation_experiment.py results): T2
("moderate", 90% cumulative OOF importance, 336 total columns) -- the
smallest data-driven tier where NEITHER ElasticNet nor LightGBM suffers
severe degradation (both stay within ~20% relative RMSE of the full-set
ceiling). T3 and below cripple ElasticNet specifically (RMSE 3-11x worse),
which would make any "fair" comparison actually just a test of which
model tolerates aggressive feature reduction, not which predicts yield
best. T0 (full 1732) is also reported here as the reference ceiling.
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
    from src.models.feature_ablation_experiment import build_tier_columns

    common_set_T2 = build_tier_columns(units_df, 0.90)
    full_set_T0 = fm.numeric_cols + fm.engineered_cols

    models = ("elasticnet", "random_forest", "lightgbm")

    for tier_name, cols in [("T2_common_336", common_set_T2), ("T0_full_1732", full_set_T0)]:
        print(f"\n=== FAIR BENCHMARK: {tier_name} ({len(cols)} + 20 categorical = {len(cols)+20}) ===")
        cv_df = cross_validate_on_columns(fm_train, y_train, groups_train, cols, models=models)
        cv_df["tier"] = tier_name
        summary = summarize_cv(cv_df)
        print(f"CV summary:\n{summary.round(4).to_string()}")

        hold_df = evaluate_holdout_on_columns(fm_train, y_train, fm_test, y_test, cols, models=models)
        hold_df["tier"] = tier_name
        print(f"Holdout:\n{hold_df.round(4).to_string(index=False)}")

        cv_df.to_csv(OUTPUTS_DIR / "reports" / f"fair_benchmark_cv_{tier_name}.csv", index=False)
        hold_df.to_csv(OUTPUTS_DIR / "reports" / f"fair_benchmark_holdout_{tier_name}.csv", index=False)

    print("\nSaved fair_benchmark_cv_*.csv and fair_benchmark_holdout_*.csv")


if __name__ == "__main__":
    main()
