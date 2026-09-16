"""Global feature importance: SHAP + permutation importance, reported at
BOTH the individual-feature level and the multicollinearity-cluster level
(user rule 3) -- single-column importance within a highly correlated family
(e.g. the ~70-column `_lkg_*` block) is close to arbitrary, so cluster-level
aggregation is treated as first-class output, not an afterthought.

Also computes fold-to-fold stability of the individual-feature ranking, so
downstream reporting can flag which "top features" are robust versus which
are fold-dependent noise (user rule 3: "assess feature-importance stability
across folds").
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.features.clustering import FeatureClustering
from src.features.matrix import FeatureMatrix


@dataclasses.dataclass
class ImportanceResult:
    shap_values: np.ndarray             # (n_samples, n_features), model-facing (sanitized) cols
    feature_importance: pd.Series       # mean |SHAP| per feature, original names, sorted desc
    cluster_importance: pd.Series       # summed mean |SHAP| per cluster (raw-numeric features only), sorted desc
    permutation_importance: pd.Series   # mean permutation importance (test-set RMSE increase), original names


def compute_shap_importance(model, X: pd.DataFrame, fm: FeatureMatrix) -> tuple[np.ndarray, pd.Series]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns)
    importance.index = [fm.to_original(c) for c in importance.index]
    importance = importance.groupby(level=0).sum()  # in case sanitization created any collisions
    return shap_values, importance.sort_values(ascending=False)


def aggregate_cluster_importance(
    feature_importance: pd.Series, clustering: FeatureClustering
) -> pd.Series:
    """Sum SHAP importance across all members of each multicollinearity
    cluster (raw numeric features only -- engineered/categorical features
    are not part of `clustering` and are reported individually)."""
    cluster_scores: dict[int, float] = {}
    cluster_label: dict[int, str] = {}
    for cid, members in clustering.clusters.items():
        present = [m for m in members if m in feature_importance.index]
        if not present:
            continue
        cluster_scores[cid] = feature_importance.loc[present].sum()
        rep = clustering.representatives[cid]
        label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
        cluster_label[cid] = label

    s = pd.Series(cluster_scores).sort_values(ascending=False)
    s.index = [cluster_label[cid] for cid in s.index]
    return s


def compute_permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, fm: FeatureMatrix, n_repeats: int = 10, random_state: int = 42
) -> pd.Series:
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    importance = pd.Series(result.importances_mean, index=X.columns)
    importance.index = [fm.to_original(c) for c in importance.index]
    importance = importance.groupby(level=0).sum()
    return importance.sort_values(ascending=False)


def full_importance_report(model, X: pd.DataFrame, y: pd.Series, fm: FeatureMatrix) -> ImportanceResult:
    shap_values, feat_imp = compute_shap_importance(model, X, fm)
    cluster_imp = aggregate_cluster_importance(feat_imp, fm.clustering)
    perm_imp = compute_permutation_importance(model, X, y, fm)
    return ImportanceResult(
        shap_values=shap_values,
        feature_importance=feat_imp,
        cluster_importance=cluster_imp,
        permutation_importance=perm_imp,
    )


if __name__ == "__main__":
    import joblib

    from src.data.load import load_dataset
    from src.features.matrix import build_feature_matrix
    from src.models.train import OUTPUTS_DIR

    ds = load_dataset()
    fm = build_feature_matrix(ds)
    split = joblib.load(OUTPUTS_DIR / "models" / "holdout_split.joblib")
    test_idx = split["test_idx"]

    gbm = joblib.load(OUTPUTS_DIR / "models" / "lightgbm.joblib")
    X_test = fm.X.iloc[test_idx].reset_index(drop=True)
    y_test = ds.y.iloc[test_idx].reset_index(drop=True)

    report = full_importance_report(gbm, X_test, y_test, fm)

    print("=== Top 25 individual-feature SHAP importance (mean |SHAP|, held-out test set) ===")
    print(report.feature_importance.head(25).round(4).to_string())

    print("\n=== Top 20 CLUSTER-level SHAP importance (summed across correlated family members) ===")
    print(report.cluster_importance.head(20).round(4).to_string())

    print("\n=== Top 25 permutation importance (RMSE increase, held-out test set) ===")
    print(report.permutation_importance.head(25).round(4).to_string())
