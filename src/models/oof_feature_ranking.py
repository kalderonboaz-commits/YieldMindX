"""Leakage-safe feature ranking for the feature-selection/ablation
experiment.

Critical methodological point: this ranking must NEVER use the untouched
holdout lots, and must not use in-sample (fit-and-evaluate-on-same-rows)
importance, which is optimistically biased. So importance here is
computed OUT-OF-FOLD, strictly within the 80-lot TRAINING portion's
existing GroupKFold(5) split: for each fold, fit on that fold's training
rows and compute SHAP importance on that fold's HELD-OUT VALIDATION rows
(never the training rows it was fit on, never the final holdout). The five
per-fold out-of-fold importance vectors are then averaged.

This differs from src/explain/feature_strength.py (prior checkpoint),
which computed SHAP importance IN-SAMPLE per fold (fit on X_tr, evaluate
also on X_tr) for a different purpose (fold-STABILITY assessment, which
only needs internal consistency, not leakage-safety against the holdout).
That distinction matters here because this ranking's OUTPUT directly
determines which predictors enter the models later evaluated on the
holdout -- an in-sample or holdout-touching ranking would bias that
evaluation optimistically.

Feature CLUSTERING (unsupervised, X-only, no target/no labels) is fit on
the full 1500-row dataset elsewhere in the pipeline (src/features/matrix.py)
-- that is not a leakage concern per the user's framing (concerned with
supervised, label-driven selection decisions), since no yield information
crosses from holdout to train through an unsupervised X-only correlation
structure. This is called out explicitly below rather than silently
assumed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.global_importance import aggregate_cluster_importance, compute_shap_importance
from src.features.matrix import FeatureMatrix
from src.models.trees import make_lightgbm


@dataclasses.dataclass
class OOFRanking:
    feature_importance_oof: pd.Series      # original names, mean out-of-fold |SHAP| across 5 folds, sorted desc
    cluster_importance_oof: pd.Series       # cluster display-label -> summed OOF importance, sorted desc
    pearson_train: pd.Series                # |Pearson| computed on TRAIN portion only
    spearman_train: pd.Series               # |Spearman| computed on TRAIN portion only


def compute_oof_feature_ranking(
    fm_train: FeatureMatrix, y_train: pd.Series, groups_train: pd.Series, n_splits: int = 5, random_state: int = 42,
) -> OOFRanking:
    _, folds = make_group_kfold(groups_train.reset_index(drop=True), n_splits=n_splits)

    per_fold_oof = {}
    for fold_i, (tr, va) in enumerate(folds):
        X_tr, y_tr = fm_train.X.iloc[tr], y_train.iloc[tr]
        X_va = fm_train.X.iloc[va]

        gbm = make_lightgbm(random_state=random_state)
        gbm.fit(X_tr, y_tr)
        _, imp_va = compute_shap_importance(gbm, X_va, fm_train)  # evaluated on VALIDATION fold -- out-of-fold
        per_fold_oof[f"fold_{fold_i}"] = imp_va
        print(f"  OOF ranking fold {fold_i}: fit on {len(tr)} rows, SHAP-evaluated on {len(va)} held-out rows", flush=True)

    oof_df = pd.DataFrame(per_fold_oof).fillna(0.0)
    feature_importance_oof = oof_df.mean(axis=1).sort_values(ascending=False)

    cluster_importance_oof = aggregate_cluster_importance(feature_importance_oof, fm_train.clustering)

    numeric_and_engineered = fm_train.numeric_cols + fm_train.engineered_cols
    model_cols = fm_train.to_model_cols(numeric_and_engineered)
    X_corr = fm_train.X[model_cols].copy()
    X_corr.columns = numeric_and_engineered
    pearson = X_corr.apply(lambda s: s.corr(y_train)).abs()
    spearman = X_corr.apply(lambda s: s.corr(y_train, method="spearman")).abs()

    return OOFRanking(
        feature_importance_oof=feature_importance_oof,
        cluster_importance_oof=cluster_importance_oof,
        pearson_train=pearson,
        spearman_train=spearman,
    )


def build_ranked_units(ranking: OOFRanking, fm: FeatureMatrix) -> pd.DataFrame:
    """Builds one ranking over 'selectable units': one row per numeric
    correlation-cluster (represented by its representative column) plus one
    row per engineered feature (not clustered -- see note in
    src/features/clustering.py, engineered features were not run through
    the numeric clustering step). Ranked by OOF SHAP importance so the
    resulting cumulative-importance curve is a genuine, data-driven basis
    for choosing feature-set-size cutoffs (not an arbitrary count)."""
    rows = []
    seen_clusters = set()
    for f, imp in ranking.feature_importance_oof.items():
        cid = fm.clustering.cluster_of.get(f)
        if cid is not None:
            cid = int(cid)
            if cid in seen_clusters:
                continue
            members = fm.clustering.clusters[cid]
            rep = fm.clustering.representatives[cid]
            label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
            cluster_val = ranking.cluster_importance_oof.get(label, imp)
            rows.append({"unit": rep, "unit_type": "numeric_cluster", "cluster_size": len(members), "oof_importance": cluster_val})
            seen_clusters.add(cid)
        elif f in fm.engineered_cols:
            rows.append({"unit": f, "unit_type": "engineered", "cluster_size": 1, "oof_importance": imp})
        # categorical dummies are always kept in every tier -- not part of this ranking

    df = pd.DataFrame(rows).sort_values("oof_importance", ascending=False).reset_index(drop=True)
    total = df["oof_importance"].sum()
    df["cumulative_fraction"] = df["oof_importance"].cumsum() / total
    return df


if __name__ == "__main__":
    from src.data.load import load_dataset
    from src.data.splits import make_holdout_split
    from src.features.matrix import build_feature_matrix
    from src.models.train import OUTPUTS_DIR

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

    print("Computing leakage-safe out-of-fold feature ranking (TRAIN portion only, never touches holdout)...")
    ranking = compute_oof_feature_ranking(fm_train, y_train, groups_train, n_splits=5)

    print("\nTop 20 OOF individual feature importance:")
    print(ranking.feature_importance_oof.head(20).round(4).to_string())

    units_df = build_ranked_units(ranking, fm)
    print(f"\nTotal selectable units: {len(units_df)} "
          f"({(units_df['unit_type']=='numeric_cluster').sum()} numeric clusters + "
          f"{(units_df['unit_type']=='engineered').sum()} engineered)")

    for cutoff in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30]:
        n = int((units_df["cumulative_fraction"] <= cutoff).sum()) + 1
        print(f"  cumulative importance <= {cutoff:.0%}: {n} units needed")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    units_df.to_csv(OUTPUTS_DIR / "reports" / "oof_ranked_units.csv", index=False)
    ranking.feature_importance_oof.to_csv(OUTPUTS_DIR / "reports" / "oof_feature_importance.csv")
    print("\nSaved outputs/reports/oof_ranked_units.csv")
    print("Saved outputs/reports/oof_feature_importance.csv")
