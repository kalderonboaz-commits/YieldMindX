"""Stage 1 target-shuffle sanity test on the final original-predictor
common set. All three models: ElasticNet, Random Forest, LightGBM.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.data.load import load_dataset
from src.data.splits import make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.feature_set_eval import cross_validate_on_columns, summarize_cv
from src.models.stage1_ablation_experiment import build_tier_columns
from src.models.stage1_universe import assert_no_engineered_features
from src.models.train import OUTPUTS_DIR

STAGE1_COMMON_CUTOFF = 0.90  # finalized after reviewing stage1_ablation results


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
    y_train = ds.y.iloc[holdout.train_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)

    units_df = pd.read_csv(OUTPUTS_DIR / "reports" / "stage1_oof_ranked_units.csv")
    common_set = build_tier_columns(units_df, STAGE1_COMMON_CUTOFF)
    assert_no_engineered_features(common_set, fm)
    print(f"Stage 1 common set: {len(common_set)} + 20 categorical = {len(common_set)+20} total (original predictors only)")

    rng = np.random.default_rng(42)
    y_shuffled = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index, name=y_train.name)

    print("\n=== Grouped 5-fold CV on SHUFFLED target, Stage 1 common set, all 3 models ===")
    cv_df = cross_validate_on_columns(fm_train, y_shuffled, groups_train, common_set,
                                       models=("elasticnet", "random_forest", "lightgbm"))
    summary = summarize_cv(cv_df)
    print("\n--- Shuffled-target CV summary ---")
    print(summary.round(4).to_string())

    print("\n=== VERDICT (threshold: shuffled-target CV R^2 must be < 0.15 to pass) ===")
    any_fail = False
    for model in summary.index:
        r2_mean = summary.loc[model, ("r2", "mean")]
        status = "PASS" if r2_mean < 0.15 else "FAIL -- INVESTIGATE"
        print(f"  {model}: shuffled R^2 = {r2_mean:.4f} -> {status}")
        if r2_mean >= 0.15:
            any_fail = True

    if any_fail:
        print("\n*** AT LEAST ONE MODEL FAILED -- STOP AND INVESTIGATE LEAKAGE ***")
    else:
        print("\nAll models collapsed toward/below chance -- no leakage evidence on the Stage 1 common set.")

    cv_df.to_csv(OUTPUTS_DIR / "reports" / "stage1_shuffle_test_cv_results.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "reports" / "stage1_shuffle_test_cv_summary.csv")
    print("\nSaved outputs/reports/stage1_shuffle_test_cv_summary.csv")


if __name__ == "__main__":
    main()
