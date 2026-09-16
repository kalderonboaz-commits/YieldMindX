"""Target-shuffle sanity test (gold-standard overfitting/leakage check).

Shuffles SortingYield (breaking any true association with the features)
and reruns the exact same group-aware CV protocol used for the real
pipeline (same GroupKFold splits by LotName, same models, same feature
matrix). If any model still achieves high R^2 on out-of-fold validation
data against a SHUFFLED target, that model/pipeline is capable of fitting
noise at this n/p ratio, and the real-target R^2 of ~0.98-0.996 reported
elsewhere cannot be trusted at face value without further investigation.
If R^2 collapses toward 0 (or goes negative), that is evidence AGAINST
overfitting/leakage explaining the real-target results.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# Expected and benign here: ElasticNetCV fitting a SHUFFLED (noise) target
# often fails to fully converge within max_iter because there is no real
# signal for coordinate descent to converge toward. This is not silencing a
# real-data concern -- it only suppresses log spam from this specific test.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.data.load import load_dataset
from src.data.splits import make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.train import cross_validate_models, summarize_cv


def run_shuffle_test(random_state: int = 42, n_splits: int = 5) -> pd.DataFrame:
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

    rng = np.random.default_rng(random_state)
    y_shuffled = pd.Series(
        rng.permutation(y_train.to_numpy()), index=y_train.index, name=y_train.name
    )

    print(f"Real target: mean={y_train.mean():.3f} std={y_train.std():.3f}")
    print(f"Shuffled target: mean={y_shuffled.mean():.3f} std={y_shuffled.std():.3f} "
          f"(same values, association with X destroyed)")

    print("\n=== Grouped 5-fold CV on SHUFFLED target (same protocol as real pipeline) ===")
    cv_df = cross_validate_models(fm_train, y_shuffled, groups_train, n_splits=n_splits, random_state=random_state)
    summary = summarize_cv(cv_df)
    print("\n--- Shuffled-target CV summary (mean +/- std across folds) ---")
    print(summary.round(4).to_string())
    return summary


def verdict(summary: pd.DataFrame, r2_collapse_threshold: float = 0.15) -> dict:
    """A model 'passes' the sanity check if its shuffled-target CV R^2 stays
    near/below the mean-baseline floor. Threshold is a heuristic, not a
    formal test: real-data R^2 on shuffled targets should hover near 0
    (a small positive value is expected/tolerable from in-sample noise
    fitting under regularization/CV selection; a LARGE positive R^2 is the
    red flag)."""
    out = {}
    for model in summary.index:
        r2_mean = summary.loc[model, ("r2", "mean")]
        out[model] = {
            "shuffled_r2_mean": float(r2_mean),
            "passes_sanity_check": bool(r2_mean < r2_collapse_threshold),
        }
    return out


if __name__ == "__main__":
    summary = run_shuffle_test()
    v = verdict(summary)
    print("\n=== VERDICT (threshold: shuffled-target CV R^2 must be < 0.15 to pass) ===")
    any_fail = False
    for model, res in v.items():
        status = "PASS" if res["passes_sanity_check"] else "FAIL -- INVESTIGATE"
        print(f"  {model}: shuffled R^2 = {res['shuffled_r2_mean']:.4f}  -> {status}")
        if not res["passes_sanity_check"]:
            any_fail = True

    if any_fail:
        print("\n*** AT LEAST ONE MODEL FAILED THE SHUFFLE SANITY CHECK. "
              "Do not trust the real-target R^2 for that model without further investigation. ***")
    else:
        print("\nAll models collapsed toward chance on the shuffled target -- "
              "no evidence of leakage/overfitting-driven inflation from this test.")

    from src.models.train import OUTPUTS_DIR
    summary.to_csv(OUTPUTS_DIR / "reports" / "shuffle_test_cv_summary.csv")
