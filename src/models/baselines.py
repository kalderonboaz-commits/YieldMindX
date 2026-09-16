"""Interpretable linear baseline: ElasticNet on a de-collinearized feature
set (one representative feature per multicollinearity cluster, per
src/features/clustering.py), plus engineered + categorical features.

This model exists to (a) give a transparent, low-variance floor to compare
the tree models against, and (b) give a second, independent read on the
sign/direction of the strongest linear effects, since its coefficients are
directly interpretable (unlike raw SHAP importance on 1700+ correlated
columns).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.matrix import FeatureMatrix


@dataclasses.dataclass
class LinearBaseline:
    pipeline: Pipeline
    feature_cols: list[str]


def reduced_feature_columns(fm: FeatureMatrix) -> list[str]:
    """One representative raw-numeric column per correlation cluster, plus
    all engineered and categorical columns (engineered features are already
    low-cardinality derived signals; categoricals are not numeric-clustered).
    Returns SANITIZED column names, ready to index into fm.X directly."""
    representatives = sorted(set(fm.clustering.representatives.values()))
    original_cols = representatives + fm.engineered_cols + fm.categorical_cols
    return fm.to_model_cols(original_cols)


def fit_linear_baseline(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> LinearBaseline:
    # NOTE: with ~1300 candidate columns, a wide l1_ratio/alpha grid x 5-fold
    # inner CV is prohibitively slow (thousands of coordinate-descent fits).
    # Keep the grid small and parallelize -- this is an interpretable
    # baseline/reference point, not the primary predictive model, so it does
    # not need an exhaustive hyperparameter search.
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            alphas=20,
            cv=3,
            max_iter=5000,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipeline.fit(X[feature_cols], y)
    return LinearBaseline(pipeline=pipeline, feature_cols=feature_cols)


def linear_coefficients(baseline: LinearBaseline) -> pd.Series:
    model = baseline.pipeline.named_steps["model"]
    scaler = baseline.pipeline.named_steps["scale"]
    # coefficients are in standardized units -> already comparable across features
    coefs = pd.Series(model.coef_, index=baseline.feature_cols, name="coef_std_units")
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


if __name__ == "__main__":
    from src.data.load import load_dataset
    from src.features.matrix import build_feature_matrix

    ds = load_dataset()
    fm = build_feature_matrix(ds)
    cols = reduced_feature_columns(fm)
    print("Reduced (de-collinearized) feature count for linear baseline:", len(cols))

    baseline = fit_linear_baseline(fm.X, ds.y, cols)
    model = baseline.pipeline.named_steps["model"]
    print("Chosen alpha:", model.alpha_, "| l1_ratio:", model.l1_ratio_)
    n_nonzero = int((model.coef_ != 0).sum())
    print(f"Non-zero coefficients: {n_nonzero} / {len(cols)}")

    coefs = linear_coefficients(baseline)
    coefs.index = [fm.to_original(c) for c in coefs.index]
    print("\nTop 20 |standardized coefficients| (original names):")
    print(coefs.head(20).round(4).to_string())
