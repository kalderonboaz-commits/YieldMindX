"""End-to-end model training & evaluation with group-aware cross-validation.

Protocol:
  1. Hold out 20% of LOTS (not rows) as a final untouched test set.
  2. On the remaining 80% of lots, run 5-fold GroupKFold (grouped by
     LotName) to get an honest, non-optimistic estimate of generalization
     error for each candidate model.
  3. Refit each model on the full training portion (all 80% of lots) and
     evaluate once on the held-out test lots.
  4. Persist metrics + fitted final models for the explainability stage.

No hyperparameter search is run inside the CV loop in this first pass
(models use fixed, reasonable defaults) -- per the approved plan, tuning is
only justified once baselines show it's worth the added complexity.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load import Dataset, load_dataset
from src.data.splits import make_group_kfold, make_holdout_split
from src.features.matrix import FeatureMatrix, build_feature_matrix
from src.models.baselines import fit_linear_baseline, reduced_feature_columns
from src.models.trees import make_lightgbm, make_random_forest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclasses.dataclass
class FoldMetrics:
    model: str
    fold: int
    rmse: float
    mae: float
    r2: float


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_models(
    fm: FeatureMatrix, y: pd.Series, groups: pd.Series, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    _, folds = make_group_kfold(groups.reset_index(drop=True), n_splits=n_splits)
    linear_cols = reduced_feature_columns(fm)

    records = []
    for fold_i, (tr, va) in enumerate(folds):
        X_tr, X_va = fm.X.iloc[tr], fm.X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # --- mean baseline (floor) ---
        pred_mean = np.full(len(va), y_tr.mean())
        m = _metrics(y_va, pred_mean)
        records.append({"model": "mean_baseline", "fold": fold_i, **m})
        print(f"[fold {fold_i}] mean_baseline rmse={m['rmse']:.3f}", flush=True)

        # --- linear (ElasticNet on de-collinearized features) ---
        t0 = time.time()
        lb = fit_linear_baseline(X_tr, y_tr, linear_cols)
        pred_lin = lb.pipeline.predict(X_va[linear_cols])
        m = _metrics(y_va, pred_lin)
        records.append({"model": "elasticnet", "fold": fold_i, **m})
        print(f"[fold {fold_i}] elasticnet rmse={m['rmse']:.3f} ({time.time()-t0:.1f}s)", flush=True)

        # --- random forest ---
        t0 = time.time()
        rf = make_random_forest(random_state=random_state)
        rf.fit(X_tr, y_tr)
        pred_rf = rf.predict(X_va)
        m = _metrics(y_va, pred_rf)
        records.append({"model": "random_forest", "fold": fold_i, **m})
        print(f"[fold {fold_i}] random_forest rmse={m['rmse']:.3f} ({time.time()-t0:.1f}s)", flush=True)

        # --- lightgbm ---
        t0 = time.time()
        gbm = make_lightgbm(random_state=random_state)
        gbm.fit(X_tr, y_tr)
        pred_gbm = gbm.predict(X_va)
        m = _metrics(y_va, pred_gbm)
        records.append({"model": "lightgbm", "fold": fold_i, **m})
        print(f"[fold {fold_i}] lightgbm rmse={m['rmse']:.3f} ({time.time()-t0:.1f}s)", flush=True)

    return pd.DataFrame.from_records(records)


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    return (
        cv_df.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .sort_values(("rmse", "mean"))
    )


def fit_final_models(fm: FeatureMatrix, y: pd.Series, random_state: int = 42):
    linear_cols = reduced_feature_columns(fm)
    lb = fit_linear_baseline(fm.X, y, linear_cols)
    rf = make_random_forest(random_state=random_state)
    rf.fit(fm.X, y)
    gbm = make_lightgbm(random_state=random_state)
    gbm.fit(fm.X, y)
    return {"elasticnet": lb, "random_forest": rf, "lightgbm": gbm}


def evaluate_on_holdout(models: dict, fm_test: FeatureMatrix, y_test: pd.Series, linear_cols: list[str]) -> pd.DataFrame:
    rows = []
    pred_lin = models["elasticnet"].pipeline.predict(fm_test.X[linear_cols])
    rows.append({"model": "elasticnet", **_metrics(y_test, pred_lin)})
    pred_rf = models["random_forest"].predict(fm_test.X)
    rows.append({"model": "random_forest", **_metrics(y_test, pred_rf)})
    pred_gbm = models["lightgbm"].predict(fm_test.X)
    rows.append({"model": "lightgbm", **_metrics(y_test, pred_gbm)})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    ds = load_dataset()
    fm = build_feature_matrix(ds)
    print(f"Feature matrix built: {fm.X.shape} ({time.time()-t0:.1f}s)")

    holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=42)
    X_train_idx, X_test_idx = holdout.train_idx, holdout.test_idx

    fm_train = FeatureMatrix(
        X=fm.X.iloc[X_train_idx].reset_index(drop=True),
        numeric_cols=fm.numeric_cols, engineered_cols=fm.engineered_cols,
        categorical_cols=fm.categorical_cols, clustering=fm.clustering,
        name_map=fm.name_map, inverse_name_map=fm.inverse_name_map,
    )
    fm_test = FeatureMatrix(
        X=fm.X.iloc[X_test_idx].reset_index(drop=True),
        numeric_cols=fm.numeric_cols, engineered_cols=fm.engineered_cols,
        categorical_cols=fm.categorical_cols, clustering=fm.clustering,
        name_map=fm.name_map, inverse_name_map=fm.inverse_name_map,
    )
    y_train = ds.y.iloc[X_train_idx].reset_index(drop=True)
    y_test = ds.y.iloc[X_test_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[X_train_idx].reset_index(drop=True)

    print(f"Train: {fm_train.X.shape[0]} rows / {groups_train.nunique()} lots | "
          f"Test: {fm_test.X.shape[0]} rows / {ds.groups.iloc[X_test_idx].nunique()} lots")

    print("\n=== Grouped 5-fold cross-validation on TRAIN portion ===")
    cv_df = cross_validate_models(fm_train, y_train, groups_train, n_splits=5)
    summary = summarize_cv(cv_df)
    print("\n--- CV summary (mean +/- std across folds) ---")
    print(summary.round(4).to_string())

    print("\n=== Fitting final models on full TRAIN portion ===")
    linear_cols = reduced_feature_columns(fm_train)
    models = fit_final_models(fm_train, y_train)

    print("\n=== Evaluating on untouched HOLD-OUT TEST lots ===")
    holdout_metrics = evaluate_on_holdout(models, fm_test, y_test, linear_cols)
    print(holdout_metrics.round(4).to_string())

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(OUTPUTS_DIR / "reports" / "cv_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "reports" / "cv_summary.csv")
    holdout_metrics.to_csv(OUTPUTS_DIR / "reports" / "holdout_metrics.csv", index=False)

    import joblib
    joblib.dump(models["random_forest"], OUTPUTS_DIR / "models" / "random_forest.joblib")
    joblib.dump(models["lightgbm"], OUTPUTS_DIR / "models" / "lightgbm.joblib")
    joblib.dump(models["elasticnet"], OUTPUTS_DIR / "models" / "elasticnet.joblib")
    joblib.dump({"train_idx": X_train_idx, "test_idx": X_test_idx}, OUTPUTS_DIR / "models" / "holdout_split.joblib")
    joblib.dump(linear_cols, OUTPUTS_DIR / "models" / "linear_feature_cols.joblib")

    print(f"\nDone in {time.time()-t0:.1f}s. Metrics + models written to outputs/")


if __name__ == "__main__":
    main()
