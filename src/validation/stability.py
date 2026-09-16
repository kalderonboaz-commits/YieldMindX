"""Stability checks for discovered relationships (user rule 3 + assessment
section 13): a feature-importance ranking or interaction is only reported
as a real finding if it is consistent across folds / resamples, not an
artifact of one particular train split or random seed.

Two checks:
  1. Fold-level SHAP importance stability: refit on each GroupKFold training
     fold, compute SHAP importance each time, report rank-agreement
     (Spearman correlation of importance ranks between folds) plus which
     features are consistently in the top-K across ALL folds vs only some.
  2. Cluster-level stability: same, but aggregated to the multicollinearity
     cluster level, since individual-column rankings within a collinear
     family are expected to be unstable even when the family's combined
     signal is very stable (this is the whole point of rule 3).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.splits import make_group_kfold
from src.explain.global_importance import aggregate_cluster_importance, compute_shap_importance
from src.features.matrix import FeatureMatrix
from src.models.trees import make_lightgbm


@dataclasses.dataclass
class StabilityResult:
    per_fold_feature_importance: pd.DataFrame   # feature x fold
    per_fold_cluster_importance: pd.DataFrame    # cluster_label x fold
    pairwise_rank_correlation: pd.DataFrame       # fold x fold Spearman of feature ranks
    consistently_top_k_features: list[str]
    consistently_top_k_clusters: list[str]


def assess_importance_stability(
    fm: FeatureMatrix, y: pd.Series, groups: pd.Series, n_splits: int = 5, top_k: int = 20, random_state: int = 42
) -> StabilityResult:
    _, folds = make_group_kfold(groups.reset_index(drop=True), n_splits=n_splits)

    feat_imps = {}
    clust_imps = {}
    for fold_i, (tr, va) in enumerate(folds):
        X_tr, y_tr = fm.X.iloc[tr], y.iloc[tr]
        gbm = make_lightgbm(random_state=random_state)
        gbm.fit(X_tr, y_tr)
        _, feat_imp = compute_shap_importance(gbm, X_tr, fm)
        feat_imps[f"fold_{fold_i}"] = feat_imp
        clust_imps[f"fold_{fold_i}"] = aggregate_cluster_importance(feat_imp, fm.clustering)
        print(f"  stability fold {fold_i}: top feature = {feat_imp.index[0]}", flush=True)

    feat_df = pd.DataFrame(feat_imps).fillna(0.0)
    clust_df = pd.DataFrame(clust_imps).fillna(0.0)

    fold_cols = feat_df.columns
    rank_corr = pd.DataFrame(index=fold_cols, columns=fold_cols, dtype=float)
    for a in fold_cols:
        for b in fold_cols:
            rank_corr.loc[a, b] = spearmanr(feat_df[a], feat_df[b]).statistic

    top_sets = [set(feat_df[c].sort_values(ascending=False).head(top_k).index) for c in fold_cols]
    consistently_top = sorted(set.intersection(*top_sets))

    top_clust_sets = [set(clust_df[c].sort_values(ascending=False).head(top_k).index) for c in fold_cols]
    consistently_top_clusters = sorted(set.intersection(*top_clust_sets))

    return StabilityResult(
        per_fold_feature_importance=feat_df,
        per_fold_cluster_importance=clust_df,
        pairwise_rank_correlation=rank_corr,
        consistently_top_k_features=consistently_top,
        consistently_top_k_clusters=consistently_top_clusters,
    )


if __name__ == "__main__":
    from src.data.load import load_dataset
    from src.data.splits import make_holdout_split
    from src.features.matrix import build_feature_matrix

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

    result = assess_importance_stability(fm_train, y_train, groups_train, n_splits=5, top_k=20)

    print("\n=== Pairwise fold rank correlation (Spearman, feature importance) ===")
    print(result.pairwise_rank_correlation.round(3).to_string())

    print(f"\n=== Features in top-20 SHAP importance in ALL 5 folds ({len(result.consistently_top_k_features)}) ===")
    for f in result.consistently_top_k_features:
        print(" ", f)

    print(f"\n=== Clusters/families in top-20 SHAP importance in ALL 5 folds ({len(result.consistently_top_k_clusters)}) ===")
    for c in result.consistently_top_k_clusters:
        print(" ", c)
