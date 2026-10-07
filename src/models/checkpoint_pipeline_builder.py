"""Lot-safe, fully-nested pipeline construction for the checkpoint model
comparison. Every learned step -- imputation, constant-feature removal,
redundancy clustering + representative selection, scaling, and
hyperparameter search -- is fit INSIDE each applicable training fold, never
once on the whole training set beforehand.

Nothing in this file is "frozen legacy code": it is new, checkpoint-pipeline
-specific code. src/models/baselines.py and src/models/feature_set_eval.py
(Stage 1) are not imported or modified here for the model-fitting path --
only src/features/clustering.py::cluster_features is reused (unmodified),
for the redundancy-clustering step, exactly as already validated for
Stage 1's own de-collinearization.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from src.features.clustering import cluster_features

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-8  # VarianceThreshold's own fit-on-train-fold-only guard


class RedundancyClusterSelector(BaseEstimator, TransformerMixin):
    """Fits src/features/clustering.py::cluster_features on whatever rows
    it is given (must be TRAIN-FOLD rows only, enforced by caller via the
    surrounding Pipeline/cross-validation machinery, not by this class),
    keeps one representative column per correlation cluster, and applies
    that SAME column subset at transform time. Because this is a proper
    sklearn Transformer, wrapping it in a Pipeline guarantees it is refit
    fresh on each fold's training rows and only ever applied (not refit)
    to that fold's validation/test rows.
    """

    def __init__(self, distance_threshold: float = 0.30):
        self.distance_threshold = distance_threshold

    def fit(self, X: pd.DataFrame, y=None):
        fc = cluster_features(X, distance_threshold=self.distance_threshold)
        self.selected_columns_ = sorted(set(fc.representatives.values()))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_columns_]


ELASTICNET_ALPHA_GRID = list(np.logspace(-3, 1, 10))   # 0.001 .. 10, log-spaced
ELASTICNET_L1_RATIO_GRID = [0.2, 0.5, 0.8]


def build_elasticnet_pipeline_and_grid(feature_mode: str, redundancy_threshold: float = 0.30):
    """feature_mode: 'full' or 'selected'. Returns (pipeline, param_grid)
    for use with make_grouped_search(), exactly like the tree builders --
    this is the fix for the preprocessing-leakage gap found this review
    round. Reasoning:

    ElasticNetCV's OWN internal cv only controls how it splits data for
    its alpha/l1_ratio search -- but if it sits as the LAST step of a
    Pipeline, every step BEFORE it (impute/dropconst/redundancy/scale) is
    fit ONCE on the pipeline's full input before ElasticNetCV ever runs,
    regardless of how that internal cv is grouped. Grouping the internal
    split fixes WHICH rows are paired; it does not fix WHEN preprocessing
    is fit relative to that split.

    The correct fix: use a plain (non-CV) ElasticNet as the pipeline's
    final step, and let GridSearchCV -- wrapped around the ENTIRE
    pipeline, with a grouped `cv=` -- drive the alpha/l1_ratio search.
    GridSearchCV refits the WHOLE pipeline (including every preprocessing
    step) fresh on each inner fold's training rows only, which is what
    "preprocessing fit inside the fold" actually requires.
    """
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    dropconst = VarianceThreshold(threshold=NEAR_ZERO_VARIANCE_THRESHOLD).set_output(transform="pandas")

    steps = [("impute", imputer), ("dropconst", dropconst)]
    if feature_mode == "selected":
        steps.append(("redundancy", RedundancyClusterSelector(distance_threshold=redundancy_threshold)))
    steps.append(("scale", StandardScaler()))
    steps.append(("model", ElasticNet(max_iter=5000, random_state=42)))

    param_grid = {
        "model__alpha": ELASTICNET_ALPHA_GRID,
        "model__l1_ratio": ELASTICNET_L1_RATIO_GRID,
    }
    return Pipeline(steps), param_grid


def build_tree_pipeline_and_grid(model_name: str, feature_mode: str, redundancy_threshold: float = 0.30):
    """Returns (pipeline, param_grid) for GridSearchCV. Trees need no
    scaling; imputation/constant-removal/redundancy selection are still
    fit per-fold like every other step, applying the same nesting
    principle to tree-model tuning as explicitly requested."""
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    dropconst = VarianceThreshold(threshold=NEAR_ZERO_VARIANCE_THRESHOLD).set_output(transform="pandas")
    steps = [("impute", imputer), ("dropconst", dropconst)]
    if feature_mode == "selected":
        steps.append(("redundancy", RedundancyClusterSelector(distance_threshold=redundancy_threshold)))

    if model_name == "random_forest":
        steps.append(("model", RandomForestRegressor(random_state=42, n_jobs=-1)))
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 2],
        }
    elif model_name == "lightgbm":
        steps.append(("model", lgb.LGBMRegressor(random_state=42, verbose=-1)))
        param_grid = {
            "model__n_estimators": [300, 600],
            "model__num_leaves": [15, 31],
            "model__learning_rate": [0.03, 0.06],
        }
    else:
        raise ValueError(model_name)

    return Pipeline(steps), param_grid


def make_grouped_search(pipeline: Pipeline, param_grid: dict, groups_this_fold: pd.Series,
                          n_inner_splits: int = 3) -> GridSearchCV:
    n_groups = groups_this_fold.nunique()
    n_splits = min(n_inner_splits, n_groups)
    gkf_splits = list(GroupKFold(n_splits=n_splits).split(
        np.zeros(len(groups_this_fold)), groups=groups_this_fold
    ))
    return GridSearchCV(pipeline, param_grid, cv=gkf_splits, scoring="neg_root_mean_squared_error", n_jobs=-1)


def retained_feature_count(fitted_pipeline: Pipeline, feature_mode: str, n_full_features: int) -> int:
    """Actual number of predictor columns the fitted pipeline's model step
    ended up seeing, read back from the fitted pipeline itself -- never
    assumed. For 'full' mode this is n_full_features minus whatever
    VarianceThreshold happened to drop on that fold's training rows (not
    necessarily zero, even though it was zero on the full synthetic
    training set in EDA -- a fold-local subset could behave differently).
    For 'selected' mode it is len(selected_columns_) from the fitted
    RedundancyClusterSelector step, which can differ fold to fold.
    """
    if "redundancy" in fitted_pipeline.named_steps:
        return len(fitted_pipeline.named_steps["redundancy"].selected_columns_)
    dropconst = fitted_pipeline.named_steps["dropconst"]
    return int(dropconst.get_support().sum())
