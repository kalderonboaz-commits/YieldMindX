"""Stage 1 leakage-safe out-of-fold feature ranking, restricted to the
original-CSV-only predictor universe (src/models/stage1_universe.py).

Same methodology as src/models/oof_feature_ranking.py (5-fold GroupKFold
within the 80-lot training portion, fit on training rows, SHAP importance
evaluated on held-out validation rows, never the final holdout) -- but the
model fit at every fold uses ONLY the 1,425 original predictors. Engineered
features are never present in X during this ranking, so they cannot appear
in the resulting ranked-units list or influence the ranking in any way.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.global_importance import aggregate_cluster_importance, compute_shap_importance
from src.features.matrix import FeatureMatrix
from src.models.stage1_universe import assert_no_engineered_features, get_stage1_predictors
from src.models.trees import make_lightgbm


@dataclasses.dataclass
class Stage1OOFRanking:
    feature_importance_oof: pd.Series
    cluster_importance_oof: pd.Series


def compute_stage1_oof_ranking(
    fm_train: FeatureMatrix, y_train: pd.Series, groups_train: pd.Series, n_splits: int = 5, random_state: int = 42,
) -> Stage1OOFRanking:
    stage1_cols = get_stage1_predictors(fm_train)
    model_cols = fm_train.to_model_cols(stage1_cols)

    _, folds = make_group_kfold(groups_train.reset_index(drop=True), n_splits=n_splits)

    per_fold_oof = {}
    for fold_i, (tr, va) in enumerate(folds):
        X_tr, y_tr = fm_train.X.iloc[tr][model_cols], y_train.iloc[tr]
        X_va = fm_train.X.iloc[va][model_cols]

        gbm = make_lightgbm(random_state=random_state)
        gbm.fit(X_tr, y_tr)  # <-- fit ONLY on Stage 1 original predictors, no engineered columns present
        _, imp_va = compute_shap_importance(gbm, X_va, fm_train)
        per_fold_oof[f"fold_{fold_i}"] = imp_va
        print(f"  Stage1 OOF ranking fold {fold_i}: fit on {len(tr)} rows / {len(model_cols)} original predictors, "
              f"SHAP-evaluated on {len(va)} held-out rows", flush=True)

    oof_df = pd.DataFrame(per_fold_oof).fillna(0.0)
    feature_importance_oof = oof_df.mean(axis=1).sort_values(ascending=False)

    assert_no_engineered_features(feature_importance_oof.index.tolist(), fm_train)

    cluster_importance_oof = aggregate_cluster_importance(feature_importance_oof, fm_train.clustering)
    return Stage1OOFRanking(feature_importance_oof=feature_importance_oof, cluster_importance_oof=cluster_importance_oof)


def build_stage1_ranked_units(ranking: Stage1OOFRanking, fm: FeatureMatrix) -> pd.DataFrame:
    """One row per numeric correlation-cluster (by representative column).
    No engineered-feature branch at all -- this function only ever
    considers fm.clustering, which itself was only ever fit on numeric
    predictors, so engineered features cannot appear here by construction."""
    rows = []
    seen_clusters = set()
    for f, imp in ranking.feature_importance_oof.items():
        cid = fm.clustering.cluster_of.get(f)
        if cid is None:
            continue  # categorical dummies: always included separately, not ranked
        cid = int(cid)
        if cid in seen_clusters:
            continue
        members = fm.clustering.clusters[cid]
        rep = fm.clustering.representatives[cid]
        label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
        cluster_val = ranking.cluster_importance_oof.get(label, imp)
        rows.append({"unit": rep, "cluster_size": len(members), "oof_importance": cluster_val})
        seen_clusters.add(cid)

    df = pd.DataFrame(rows).sort_values("oof_importance", ascending=False).reset_index(drop=True)
    total = df["oof_importance"].sum()
    df["cumulative_fraction"] = df["oof_importance"].cumsum() / total

    assert_no_engineered_features(df["unit"].tolist(), fm)
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

    print("Computing Stage 1 leakage-safe OOF feature ranking (original predictors only)...")
    ranking = compute_stage1_oof_ranking(fm_train, y_train, groups_train, n_splits=5)

    print("\nTop 20 OOF individual feature importance (Stage 1, original predictors only):")
    print(ranking.feature_importance_oof.head(20).round(4).to_string())

    units_df = build_stage1_ranked_units(ranking, fm)
    print(f"\nTotal ranked numeric-cluster units: {len(units_df)}")

    for cutoff in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30]:
        n = int((units_df["cumulative_fraction"] <= cutoff).sum()) + 1
        print(f"  cumulative importance <= {cutoff:.0%}: {n} units needed")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    units_df.to_csv(OUTPUTS_DIR / "reports" / "stage1_oof_ranked_units.csv", index=False)
    ranking.feature_importance_oof.to_csv(OUTPUTS_DIR / "reports" / "stage1_oof_feature_importance.csv")
    print("\nSaved outputs/reports/stage1_oof_ranked_units.csv")
    print("Saved outputs/reports/stage1_oof_feature_importance.csv")
