"""Correlation-based clustering of numeric features into redundancy families.

Rationale (see assessment report / user rule 3): tree models do not solve
multicollinearity, they just hide it -- SHAP/permutation importance credit
gets split arbitrarily across near-duplicate columns (e.g. the 70-column
`_lkg_*` leakage-current family has a median pairwise |Spearman| of ~0.86).
Reporting importance only at the single-column level would let us
mis-attribute a family-level signal to one arbitrarily-chosen column.

This module builds feature clusters (hierarchical clustering on
1 - |Spearman correlation|) so downstream explainability code can report
BOTH individual-feature importance AND cluster/family-level importance, and
so a "reduced" representative-feature set can be produced for the
interpretable linear baseline.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata


@dataclasses.dataclass
class FeatureClustering:
    corr: pd.DataFrame                    # feature x feature Spearman correlation
    cluster_of: pd.Series                 # feature -> cluster id
    clusters: dict[int, list[str]]        # cluster id -> member features
    representatives: dict[int, str]       # cluster id -> representative feature (most-correlated-to-cluster-mean)
    distance_threshold: float


def _spearman_corr_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Fast Spearman correlation matrix via rank transform + np.corrcoef."""
    ranked = X.apply(lambda col: rankdata(col.to_numpy()), axis=0)
    corr = np.corrcoef(ranked.to_numpy(), rowvar=False)
    return pd.DataFrame(corr, index=X.columns, columns=X.columns)


def cluster_features(
    X: pd.DataFrame,
    distance_threshold: float = 0.30,  # clusters features with |spearman| >= 0.70
    linkage_method: str = "average",
) -> FeatureClustering:
    corr = _spearman_corr_matrix(X)
    corr_vals = corr.to_numpy()
    dist = 1.0 - np.abs(corr_vals)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry (float rounding)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    cluster_of = pd.Series(labels, index=X.columns, name="cluster_id")
    clusters: dict[int, list[str]] = {}
    for feat, cid in cluster_of.items():
        clusters.setdefault(int(cid), []).append(feat)

    representatives: dict[int, str] = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            representatives[cid] = members[0]
            continue
        sub = corr.loc[members, members].abs()
        mean_corr_to_others = (sub.sum(axis=1) - 1) / (len(members) - 1)
        representatives[cid] = mean_corr_to_others.idxmax()

    return FeatureClustering(
        corr=corr,
        cluster_of=cluster_of,
        clusters=clusters,
        representatives=representatives,
        distance_threshold=distance_threshold,
    )


def cluster_summary(fc: FeatureClustering) -> pd.DataFrame:
    sizes = {cid: len(members) for cid, members in fc.clusters.items()}
    rows = [
        {"cluster_id": cid, "size": size, "representative": fc.representatives[cid]}
        for cid, size in sorted(sizes.items(), key=lambda kv: -kv[1])
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data.load import load_dataset

    ds = load_dataset()
    X_num = ds.raw[ds.numeric_feature_cols]

    fc = cluster_features(X_num)
    summary = cluster_summary(fc)
    print("Total numeric features:", X_num.shape[1])
    print("Total clusters:", len(fc.clusters))
    print("Singleton clusters (unique signal):", (summary["size"] == 1).sum())
    print("Clusters with >=5 members:", (summary["size"] >= 5).sum())
    print("\nTop 15 largest clusters:")
    print(summary.head(15).to_string())

    # sanity check: the known lkg family should collapse into few clusters
    lkg_cols = [c for c in X_num.columns if "_lkg_" in c.lower()]
    lkg_cluster_ids = fc.cluster_of.loc[lkg_cols].value_counts()
    print(f"\n{len(lkg_cols)} lkg_* columns fall into {lkg_cluster_ids.shape[0]} cluster(s):")
    print(lkg_cluster_ids.to_string())
