"""Rigorous, dual-method EDA for the four-checkpoint pipeline, addressing
explicit review feedback:
  - both Pearson and Spearman correlation with target are computed and
    shown side by side (no single-method claim).
  - near-constant detection uses TWO independent methods (dominant-value
    fraction AND variance-ratio), both with explicit, stated thresholds.
  - outlier detection uses TWO independent methods (IQR multiplier AND
    z-score), both with explicit, stated thresholds; no claim that
    flagged points are "real" physical phenomena vs. data artifacts --
    that distinction is explicitly marked as an UNVERIFIED HYPOTHESIS,
    not a finding.
  - redundancy clustering threshold (|Spearman r| >= 0.70) is justified by
    showing the actual pairwise-correlation distribution, not asserted.

Train rows only. No model is fit anywhere in this file.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.features.clustering import cluster_features

NEAR_CONST_DOMINANT_FRACTION_THRESHOLD = 0.98   # method 1: one value covers >=98% of rows
NEAR_CONST_VARIANCE_RATIO_THRESHOLD = 0.01      # method 2: std / (max-min) < 1%
OUTLIER_IQR_K = 3.0                             # method 1: beyond Q1-3*IQR .. Q3+3*IQR
OUTLIER_ZSCORE_THRESHOLD = 3.0                  # method 2: |z| > 3
REDUNDANCY_ABS_CORR_THRESHOLD = 0.70            # matches features/clustering.py's distance_threshold=0.30


@dataclasses.dataclass
class CheckpointEDAv2:
    name: str
    n_candidate_features: int
    n_missing_cells: int
    quantile_table: pd.DataFrame
    near_constant_method1: list[tuple]
    near_constant_method2: list[tuple]
    outlier_method1_worst: list[tuple]
    outlier_method1_n_flagged_cols: int
    outlier_method2_worst: list[tuple]
    outlier_method2_n_flagged_cols: int
    pairwise_abs_corr_percentiles: dict
    n_clusters: int
    redundancy_ratio: float
    largest_clusters: list[tuple]
    pearson_buckets: dict
    spearman_buckets: dict
    top10_pearson: list[tuple]
    top10_spearman: list[tuple]


def quantile_table(X: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    return X[sample_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(4)


def near_constant_method1(X: pd.DataFrame, threshold: float = NEAR_CONST_DOMINANT_FRACTION_THRESHOLD):
    hits = []
    for col in X.columns:
        vc = X[col].value_counts(normalize=True, dropna=False)
        top_frac = float(vc.iloc[0]) if len(vc) else 1.0
        if top_frac >= threshold:
            hits.append((col, round(top_frac, 4)))
    return sorted(hits, key=lambda t: -t[1])


def near_constant_method2(X: pd.DataFrame, threshold: float = NEAR_CONST_VARIANCE_RATIO_THRESHOLD):
    hits = []
    for col in X.columns:
        s = X[col]
        rng = s.max() - s.min()
        if rng == 0:
            hits.append((col, 0.0))
            continue
        ratio = s.std() / rng
        if ratio < threshold:
            hits.append((col, round(float(ratio), 5)))
    return sorted(hits, key=lambda t: t[1])


def outliers_iqr(X: pd.DataFrame, k: float = OUTLIER_IQR_K):
    n = len(X)
    fractions = {}
    for col in X.columns:
        s = X[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        flagged = int(((s < lo) | (s > hi)).sum())
        if flagged > 0:
            fractions[col] = flagged / n
    worst = sorted(fractions.items(), key=lambda kv: -kv[1])[:10]
    return worst, len(fractions)


def outliers_zscore(X: pd.DataFrame, z: float = OUTLIER_ZSCORE_THRESHOLD):
    n = len(X)
    fractions = {}
    for col in X.columns:
        s = X[col]
        std = s.std()
        if std == 0:
            continue
        zscores = (s - s.mean()) / std
        flagged = int((zscores.abs() > z).sum())
        if flagged > 0:
            fractions[col] = flagged / n
    worst = sorted(fractions.items(), key=lambda kv: -kv[1])[:10]
    return worst, len(fractions)


def pairwise_abs_spearman_percentiles(X: pd.DataFrame, max_pairs_exact: int = 400) -> dict:
    """Distribution of |Spearman r| across ALL feature pairs -- used to
    justify (not assert) the 0.70 redundancy threshold. Computed exactly
    (full corr matrix) since even 317 features is cheap (~50k pairs)."""
    corr = X.corr(method="spearman").abs()
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = corr.values[iu]
    return {
        "n_pairs": len(vals),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "frac_ge_0.70": float((vals >= 0.70).mean()),
    }, vals


def corr_buckets(abs_corr: pd.Series) -> dict:
    return {
        ">=0.5": int((abs_corr >= 0.5).sum()),
        "0.3-0.5": int(((abs_corr >= 0.3) & (abs_corr < 0.5)).sum()),
        "0.1-0.3": int(((abs_corr >= 0.1) & (abs_corr < 0.3)).sum()),
        "<0.1": int((abs_corr < 0.1).sum()),
    }


def run_eda_v2(df: pd.DataFrame, train_idx: np.ndarray, checkpoint_name: str,
                predictor_cols: list[str], target: str) -> tuple[CheckpointEDAv2, pd.Series, pd.Series]:
    train_df = df.iloc[train_idx]
    X = train_df[predictor_cols]
    y = train_df[target]

    pearson = X.corrwith(y, method="pearson").abs()
    spearman = X.corrwith(y, method="spearman").abs()

    fc = cluster_features(X)
    n_clusters = len(fc.clusters)
    sizes = sorted(((cid, len(m)) for cid, m in fc.clusters.items()), key=lambda t: -t[1])
    largest = [(cid, size, fc.representatives[cid]) for cid, size in sizes[:5]]

    pct, pair_vals = pairwise_abs_spearman_percentiles(X)
    out1, n1 = outliers_iqr(X)
    out2, n2 = outliers_zscore(X)

    report = CheckpointEDAv2(
        name=checkpoint_name,
        n_candidate_features=len(predictor_cols),
        n_missing_cells=int(X.isna().sum().sum()),
        quantile_table=quantile_table(X, list(X.columns[:4])),
        near_constant_method1=near_constant_method1(X),
        near_constant_method2=near_constant_method2(X),
        outlier_method1_worst=out1, outlier_method1_n_flagged_cols=n1,
        outlier_method2_worst=out2, outlier_method2_n_flagged_cols=n2,
        pairwise_abs_corr_percentiles=pct,
        n_clusters=n_clusters,
        redundancy_ratio=round(1 - n_clusters / len(predictor_cols), 4),
        largest_clusters=largest,
        pearson_buckets=corr_buckets(pearson),
        spearman_buckets=corr_buckets(spearman),
        top10_pearson=[(c, round(float(pearson[c]), 4)) for c in pearson.sort_values(ascending=False).head(10).index],
        top10_spearman=[(c, round(float(spearman[c]), 4)) for c in spearman.sort_values(ascending=False).head(10).index],
    )
    return report, X, y
