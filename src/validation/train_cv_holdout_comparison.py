"""Part 4: explicit train (in-sample) vs grouped-CV (out-of-fold, within
train portion) vs holdout (untouched lots) comparison, on the corrected
336-feature common set, random_state=42 split -- to check for classical
overfitting (large train-vs-CV/holdout gap), underfitting (all three
similarly poor), or unstable generalization (CV/holdout disagree sharply).
"""
from __future__ import annotations

import pandas as pd

from src.data.load import load_dataset
from src.data.splits import make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.feature_ablation_experiment import build_tier_columns
from src.models.feature_set_eval import (
    _fit_predict,
    _metrics,
    _with_categoricals,
    cross_validate_on_columns,
    evaluate_holdout_on_columns,
    summarize_cv,
)
from src.models.train import OUTPUTS_DIR


def compute_train_insample_metrics(fm_train, y_train, feature_cols_original, models):
    feature_cols_original = _with_categoricals(fm_train, feature_cols_original)
    model_cols = fm_train.to_model_cols(feature_cols_original)
    X = fm_train.X[model_cols]
    rows = []
    for m in models:
        pred = _fit_predict(m, X, y_train, X, random_state=42)  # fit and evaluate on the SAME rows -- in-sample
        rows.append({"model": m, **_metrics(y_train, pred)})
    return pd.DataFrame(rows)


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
    common_336 = build_tier_columns(units_df, 0.90)
    models = ("elasticnet", "random_forest", "lightgbm")

    print("=== TRAIN (in-sample) metrics, 336-feature set, seed=42 ===")
    train_df = compute_train_insample_metrics(fm_train, y_train, common_336, models)
    print(train_df.round(4).to_string(index=False))

    print("\n=== CV (out-of-fold within train portion) metrics, 336-feature set, seed=42 ===")
    cv_df = cross_validate_on_columns(fm_train, y_train, groups_train, common_336, models=models)
    cv_summary = summarize_cv(cv_df)
    print(cv_summary.round(4).to_string())

    print("\n=== HOLDOUT (untouched 20 lots) metrics, 336-feature set, seed=42 ===")
    hold_df = evaluate_holdout_on_columns(fm_train, y_train, fm_test, y_test, common_336, models=models)
    print(hold_df.round(4).to_string(index=False))

    print("\n=== GENERALIZATION GAP SUMMARY ===")
    for m in models:
        train_rmse = train_df[train_df["model"] == m]["rmse"].values[0]
        cv_rmse = cv_summary.loc[m, ("rmse", "mean")]
        hold_rmse = hold_df[hold_df["model"] == m]["rmse"].values[0]
        gap_cv = (cv_rmse - train_rmse) / train_rmse * 100
        gap_hold = (hold_rmse - train_rmse) / train_rmse * 100
        print(f"  {m}: train_rmse={train_rmse:.4f} cv_rmse={cv_rmse:.4f} (+{gap_cv:.0f}%) "
              f"holdout_rmse={hold_rmse:.4f} (+{gap_hold:.0f}%)")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUTS_DIR / "reports" / "train_insample_metrics_336.csv", index=False)
    cv_df.to_csv(OUTPUTS_DIR / "reports" / "cv_metrics_336_corrected.csv", index=False)
    hold_df.to_csv(OUTPUTS_DIR / "reports" / "holdout_metrics_336_corrected_seed42.csv", index=False)
    print("\nSaved train/CV/holdout CSVs for the corrected 336-feature set.")


if __name__ == "__main__":
    main()
