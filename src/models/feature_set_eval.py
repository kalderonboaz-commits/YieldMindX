"""Evaluate one or more models on an EXPLICIT, IDENTICAL set of feature
columns -- the core tool for this checkpoint's feature-set-size ablation
(Part 2/3) and the fair same-input model benchmark (Part 4/5).

Unlike src/models/train.py's cross_validate_models (which deliberately
gives ElasticNet a de-collinearized feature set and the trees the full
set -- the "best-practice-per-model" philosophy), everything here takes
ONE feature_cols_original list and feeds every requested model EXACTLY
those columns, nothing more, nothing less -- so any performance
difference between tiers or between models is attributable to the tier
size / model choice alone, not to a hidden preprocessing difference.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.splits import make_group_kfold
from src.features.matrix import FeatureMatrix
from src.models.baselines import fit_linear_baseline
from src.models.trees import make_lightgbm, make_random_forest


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_predict(model_name: str, X_tr, y_tr, X_va, random_state: int):
    if model_name == "elasticnet":
        lb = fit_linear_baseline(X_tr, y_tr, list(X_tr.columns))
        return lb.pipeline.predict(X_va)
    elif model_name == "random_forest":
        m = make_random_forest(random_state=random_state)
        m.fit(X_tr, y_tr)
        return m.predict(X_va)
    elif model_name == "lightgbm":
        m = make_lightgbm(random_state=random_state)
        m.fit(X_tr, y_tr)
        return m.predict(X_va)
    else:
        raise ValueError(model_name)


def _with_categoricals(fm: FeatureMatrix, feature_cols_original: list[str]) -> list[str]:
    """Categorical predictors are always included alongside whatever numeric/
    engineered tier is being evaluated -- centralized here (rather than left
    to each caller to remember) after a prior checkpoint's tier-definition
    code omitted them despite labeling tiers as if they were included (see
    checkpoint report: this was the actual cause of a reproducibility
    discrepancy, not floating-point/parallel non-determinism)."""
    return list(dict.fromkeys(list(feature_cols_original) + fm.categorical_cols))


def cross_validate_on_columns(
    fm: FeatureMatrix, y: pd.Series, groups: pd.Series, feature_cols_original: list[str],
    models: tuple[str, ...] = ("elasticnet", "random_forest", "lightgbm"),
    n_splits: int = 5, random_state: int = 42,
) -> pd.DataFrame:
    feature_cols_original = _with_categoricals(fm, feature_cols_original)
    model_cols = fm.to_model_cols(feature_cols_original)
    _, folds = make_group_kfold(groups.reset_index(drop=True), n_splits=n_splits)

    records = []
    for fold_i, (tr, va) in enumerate(folds):
        X_tr, X_va = fm.X.iloc[tr][model_cols], fm.X.iloc[va][model_cols]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        for m in models:
            t0 = time.time()
            pred = _fit_predict(m, X_tr, y_tr, X_va, random_state)
            metrics = _metrics(y_va, pred)
            records.append({"model": m, "fold": fold_i, **metrics})
            print(f"    [fold {fold_i}] {m}: rmse={metrics['rmse']:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame.from_records(records)


def evaluate_holdout_on_columns(
    fm_train: FeatureMatrix, y_train: pd.Series, fm_test: FeatureMatrix, y_test: pd.Series,
    feature_cols_original: list[str], models: tuple[str, ...] = ("elasticnet", "random_forest", "lightgbm"),
    random_state: int = 42,
) -> pd.DataFrame:
    feature_cols_original = _with_categoricals(fm_train, feature_cols_original)
    model_cols = fm_train.to_model_cols(feature_cols_original)
    X_tr, X_te = fm_train.X[model_cols], fm_test.X[model_cols]

    rows = []
    for m in models:
        pred = _fit_predict(m, X_tr, y_train, X_te, random_state)
        rows.append({"model": m, **_metrics(y_test, pred)})
    return pd.DataFrame(rows)


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    return cv_df.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std"]).sort_values(("rmse", "mean"))
