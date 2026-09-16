"""Repeated grouped (lot-level) holdout validation for the final 336-feature
common predictor set.

Design decision (stated explicitly): the 336-feature set itself is FIXED
across all repeated splits -- derived once from the original random_state=42
80/20 split's leakage-safe out-of-fold ranking (per the previous checkpoint,
now bug-fixed to correctly include the 20 categorical columns). This
experiment tests whether the MODEL-RANKING decision is robust across
different train/holdout partitions of the data, not whether feature
selection itself is stable -- re-deriving the ranking per split would cost
~2-3 minutes per split (5-fold LightGBM refit + SHAP) purely for feature
selection, on top of the model evaluation itself, and is not what this
checkpoint asks for ("all three models must receive exactly the same 336
predictors within each comparison").

For each of several deterministic random_state seeds, an independent 80/20
lot-level split is drawn (make_holdout_split with that seed) and all three
models are fit on the ~80-lot training portion and evaluated on that
split's ~20-lot untouched holdout, using the SAME 336 columns every time.
"""
from __future__ import annotations

import pandas as pd

from src.data.load import load_dataset
from src.data.splits import make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.feature_ablation_experiment import build_tier_columns
from src.models.feature_set_eval import evaluate_holdout_on_columns
from src.models.train import OUTPUTS_DIR

SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def main():
    ds = load_dataset()
    fm = build_feature_matrix(ds)

    units_df = pd.read_csv(OUTPUTS_DIR / "reports" / "oof_ranked_units.csv")
    common_336 = build_tier_columns(units_df, 0.90)
    print(f"Fixed common feature set for all repeated splits: {len(common_336)} + 20 categorical "
          f"(derived once from the random_state=42 OOF ranking)")

    all_rows = []
    for seed in SEEDS:
        holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=seed)
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

        n_train_lots = ds.groups.iloc[holdout.train_idx].nunique()
        n_test_lots = ds.groups.iloc[holdout.test_idx].nunique()

        hold_df = evaluate_holdout_on_columns(
            fm_train, y_train, fm_test, y_test, common_336,
            models=("elasticnet", "random_forest", "lightgbm"),
        )
        hold_df["seed"] = seed
        hold_df["n_train_lots"] = n_train_lots
        hold_df["n_test_lots"] = n_test_lots
        all_rows.append(hold_df)
        print(f"seed={seed} ({n_train_lots} train lots / {n_test_lots} test lots):")
        print(hold_df.round(4).to_string(index=False))

    results = pd.concat(all_rows, ignore_index=True)
    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS_DIR / "reports" / "repeated_holdout_results.csv", index=False)

    print("\n=== SUMMARY ACROSS ALL SEEDS ===")
    summary = results.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std", "min", "max"])
    print(summary.round(4).to_string())

    print("\n=== BEST / WORST SPLIT PER MODEL (by RMSE) ===")
    for model in results["model"].unique():
        sub = results[results["model"] == model]
        best = sub.loc[sub["rmse"].idxmin()]
        worst = sub.loc[sub["rmse"].idxmax()]
        print(f"  {model}: best seed={int(best['seed'])} rmse={best['rmse']:.4f} | "
              f"worst seed={int(worst['seed'])} rmse={worst['rmse']:.4f}")

    print("\n=== WIN RATE (rank #1 by RMSE per seed) ===")
    pivot = results.pivot(index="seed", columns="model", values="rmse")
    winners = pivot.idxmin(axis=1)
    win_counts = winners.value_counts()
    for model in results["model"].unique():
        n_wins = int(win_counts.get(model, 0))
        print(f"  {model}: {n_wins}/{len(SEEDS)} splits ({100*n_wins/len(SEEDS):.0f}%)")

    print("\n=== Was random_state=42 representative? ===")
    for model in results["model"].unique():
        sub = results[results["model"] == model]
        seed42_rmse = sub[sub["seed"] == 42]["rmse"].values[0]
        other_rmse = sub[sub["seed"] != 42]["rmse"]
        percentile = (other_rmse < seed42_rmse).mean() * 100
        print(f"  {model}: seed=42 rmse={seed42_rmse:.4f} | other-seeds mean={other_rmse.mean():.4f} std={other_rmse.std():.4f} | "
              f"seed=42 beats {100-percentile:.0f}% of other seeds (lower rmse)")

    summary.to_csv(OUTPUTS_DIR / "reports" / "repeated_holdout_summary.csv")
    print("\nSaved outputs/reports/repeated_holdout_results.csv and _summary.csv")


if __name__ == "__main__":
    main()
