"""Target-shuffle sanity test specifically on the final 336-feature common
predictor set (316 data-driven-ranked units + 20 categorical dummies,
correctly assembled via src/models/feature_set_eval.py's centralized
_with_categoricals fix), for all three models: ElasticNet, LightGBM,
Random Forest. Same GroupKFold(5) grouped-by-LotName structure as
everywhere else in this project.
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
from src.models.feature_ablation_experiment import build_tier_columns
from src.models.feature_set_eval import cross_validate_on_columns, summarize_cv
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
    y_train = ds.y.iloc[holdout.train_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)

    units_df = pd.read_csv(OUTPUTS_DIR / "reports" / "oof_ranked_units.csv")
    common_336 = build_tier_columns(units_df, 0.90)
    print(f"Common set (before auto-categorical add): {len(common_336)} -> +20 categorical = {len(common_336)+20} total")

    rng = np.random.default_rng(42)
    y_shuffled = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index, name=y_train.name)
    print(f"Real target std={y_train.std():.3f} | Shuffled target std={y_shuffled.std():.3f} (values preserved, X-association destroyed)")

    print("\n=== Grouped 5-fold CV on SHUFFLED target, 336-feature common set, all 3 models ===")
    cv_df = cross_validate_on_columns(fm_train, y_shuffled, groups_train, common_336,
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
        print("\n*** AT LEAST ONE MODEL FAILED -- STOP AND INVESTIGATE LEAKAGE BEFORE PROCEEDING ***")
    else:
        print("\nAll models collapsed toward/below chance on the shuffled target -- no leakage evidence on the 336-feature set.")

    cv_df.to_csv(OUTPUTS_DIR / "reports" / "shuffle_test_336_cv_results.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "reports" / "shuffle_test_336_cv_summary.csv")
    print("\nSaved outputs/reports/shuffle_test_336_cv_summary.csv")


if __name__ == "__main__":
    main()
