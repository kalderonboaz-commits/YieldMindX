"""Defensible, multi-signal feature-strength classification.

Per the review checkpoint: a feature must NOT be judged "weak" from a
single signal (e.g. low Pearson correlation or low individual SHAP
importance alone) -- Gate CD optimum (Pearson r~-0.02) and the Rc/Rsh
interaction (individually weak, jointly strong) are concrete proof in this
very dataset that single-signal judgments are wrong here.

For every raw numeric predictor + engineered feature, this module combines
SEVEN signals, all computed on the HELD-OUT TEST portion where a held-out
computation is meaningful (permutation importance, SHAP importance), or
computed from the model/data structure directly (correlation, clustering,
tree co-occurrence), plus fold-stability from the TRAIN-portion grouped CV:

  1. Pearson |correlation| with target
  2. Spearman |correlation| with target
  3. held-out permutation importance
  4. held-out SHAP importance
  5. fold-to-fold stability of SHAP importance (5 GroupKFold folds, train portion)
  6. cluster/family SHAP importance (dilution-aware)
  7. tree co-occurrence with top anchor features (structural interaction proxy)

and classifies each feature into one of:
  - strong_individual_predictor
  - nonlinear_predictor          (SHAP/permutation importance far outranks correlation)
  - interaction_dependent_predictor  (weak individually, high tree co-occurrence)
  - cluster_family_supported_predictor (weak individually, cluster is important)
  - weak_unstable_predictor      (weak AND unstable on every signal -- removal CANDIDATE only)
  - currently_inconclusive       (doesn't cleanly fit any bucket above)

No features are removed in this module -- classification only.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.explain.global_importance import aggregate_cluster_importance
from src.explain.interactions import compute_tree_cooccurrence
from src.features.matrix import FeatureMatrix


@dataclasses.dataclass
class FeatureStrengthInputs:
    pearson: pd.Series             # original names
    spearman: pd.Series
    perm_imp: pd.Series
    shap_imp: pd.Series
    cluster_imp: pd.Series          # cluster display-label -> value
    fold_stability_cv: pd.Series    # original names -> coefficient of variation of per-fold SHAP importance (lower = more stable)
    fold_top20_count: pd.Series     # original names -> how many of 5 folds this feature was in the top-20 SHAP importance
    cooccurrence: pd.Series         # original names -> tree co-occurrence count with broad anchor set


def compute_correlations(X_numeric: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    pearson = X_numeric.apply(lambda s: s.corr(y))
    spearman = X_numeric.apply(lambda s: s.corr(y, method="spearman"))
    return pearson.abs(), spearman.abs()


def compute_fold_stability(fm_train: FeatureMatrix, y_train: pd.Series, groups_train: pd.Series,
                            n_splits: int = 5, top_k: int = 20, random_state: int = 42) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Reuses the GroupKFold protocol from src/validation/stability.py but
    persists the FULL per-fold importance matrix (not just the top-20 list)
    so every feature -- not only the already-obviously-important ones --
    gets a stability score."""
    from src.data.splits import make_group_kfold
    from src.explain.global_importance import compute_shap_importance
    from src.models.trees import make_lightgbm

    _, folds = make_group_kfold(groups_train.reset_index(drop=True), n_splits=n_splits)

    per_fold = {}
    for fold_i, (tr, va) in enumerate(folds):
        X_tr, y_tr = fm_train.X.iloc[tr], y_train.iloc[tr]
        gbm = make_lightgbm(random_state=random_state)
        gbm.fit(X_tr, y_tr)
        _, imp = compute_shap_importance(gbm, X_tr, fm_train)
        per_fold[f"fold_{fold_i}"] = imp
        print(f"  fold {fold_i} fit + SHAP importance computed", flush=True)

    feat_df = pd.DataFrame(per_fold).fillna(0.0)

    mean_ = feat_df.mean(axis=1)
    std_ = feat_df.std(axis=1)
    cv = (std_ / mean_.replace(0, np.nan)).fillna(np.inf)

    top20_masks = feat_df.apply(lambda col: col.rank(ascending=False) <= top_k)
    top20_count = top20_masks.sum(axis=1)

    return cv, top20_count, feat_df


def classify_features(inputs: FeatureStrengthInputs, clustering, n_features: int) -> pd.DataFrame:
    def rank_of(s: pd.Series) -> pd.Series:
        return s.rank(ascending=False, method="min")

    pearson_rank = rank_of(inputs.pearson)
    spearman_rank = rank_of(inputs.spearman)
    perm_rank = rank_of(inputs.perm_imp)
    shap_rank = rank_of(inputs.shap_imp)
    coocc_rank = rank_of(inputs.cooccurrence)

    all_index = inputs.pearson.index
    cluster_rank_by_feature = pd.Series(index=all_index, dtype=float)
    for cid, members in clustering.clusters.items():
        rep = clustering.representatives[cid]
        label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
        cluster_val = inputs.cluster_imp.get(label, np.nan)
        for m in members:
            if m in cluster_rank_by_feature.index:
                cluster_rank_by_feature[m] = cluster_val
    cluster_rank = rank_of(cluster_rank_by_feature.fillna(0.0))

    N = n_features
    TOP_STRICT = max(int(N * 0.03), 30)     # top ~3% (or >=30) = "strong" cutoff
    TOP_LOOSE = max(int(N * 0.15), 150)      # top ~15% (or >=150) = "notable" cutoff
    WEAK_CUTOFF = int(N * 0.6)                # bottom 40% on a signal counts as "weak" on that signal

    rows = []
    for f in all_index:
        pr, sr, pmr, shr, cr, cor = (pearson_rank.get(f, N), spearman_rank.get(f, N),
                                      perm_rank.get(f, N), shap_rank.get(f, N),
                                      cluster_rank.get(f, N), coocc_rank.get(f, N))
        fold_cv = inputs.fold_stability_cv.get(f, np.inf)
        fold_top20 = inputs.fold_top20_count.get(f, 0)

        strong_corr = pr <= TOP_STRICT or sr <= TOP_STRICT
        strong_model_signal = shr <= TOP_STRICT or pmr <= TOP_STRICT
        notable_model_signal = shr <= TOP_LOOSE or pmr <= TOP_LOOSE
        stable = fold_top20 >= 3  # in top-20 SHAP importance in a majority (>=3/5) of folds

        weak_corr = pr > WEAK_CUTOFF and sr > WEAK_CUTOFF
        weak_model_signal = shr > WEAK_CUTOFF and pmr > WEAK_CUTOFF
        weak_cluster = cr > WEAK_CUTOFF
        weak_coocc = cor > WEAK_CUTOFF

        if strong_corr and strong_model_signal and stable:
            label = "strong_individual_predictor"
            rationale = "top-ranked on correlation AND SHAP/permutation importance, stable across folds"
        elif (not strong_corr) and strong_model_signal:
            label = "nonlinear_predictor"
            rationale = ("high SHAP/permutation importance despite weak linear/monotonic correlation "
                         "-- signal is not capturable by Pearson/Spearman alone (e.g. process-window/threshold shape)")
        elif (not notable_model_signal) and cor <= TOP_LOOSE:
            label = "interaction_dependent_predictor"
            rationale = "weak individual importance but frequently co-occurs with important features in the same trees"
        elif (not notable_model_signal) and cr <= TOP_LOOSE:
            label = "cluster_family_supported_predictor"
            rationale = "weak individual importance but belongs to a correlation cluster with high aggregate importance"
        elif weak_corr and weak_model_signal and weak_cluster and weak_coocc and fold_top20 == 0:
            label = "weak_unstable_predictor"
            rationale = "weak on every signal (correlation, SHAP, permutation, cluster, co-occurrence) and never stable across folds"
        else:
            label = "currently_inconclusive"
            rationale = "does not cleanly satisfy any strong/nonlinear/interaction/cluster/weak criterion -- insufficient evidence either way"

        rows.append({
            "feature": f,
            "classification": label,
            "rationale": rationale,
            "pearson_abs": inputs.pearson.get(f, np.nan),
            "spearman_abs": inputs.spearman.get(f, np.nan),
            "permutation_importance": inputs.perm_imp.get(f, np.nan),
            "shap_importance": inputs.shap_imp.get(f, np.nan),
            "pearson_rank": pr, "spearman_rank": sr, "permutation_rank": pmr,
            "shap_rank": shr, "cluster_rank": cr, "cooccurrence_rank": cor,
            "fold_stability_cv": fold_cv,
            "fold_top20_count": int(fold_top20),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import joblib

    from src.data.load import load_dataset
    from src.data.splits import make_holdout_split
    from src.explain.global_importance import compute_permutation_importance, compute_shap_importance
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

    gbm = joblib.load(OUTPUTS_DIR / "models" / "lightgbm.joblib")
    test_idx = holdout.test_idx
    X_test = fm.X.iloc[test_idx].reset_index(drop=True)
    y_test = ds.y.iloc[test_idx].reset_index(drop=True)

    numeric_and_engineered = fm.numeric_cols + fm.engineered_cols
    model_cols_num_eng = fm.to_model_cols(numeric_and_engineered)
    X_test_num_eng = X_test[model_cols_num_eng].copy()
    X_test_num_eng.columns = numeric_and_engineered  # original names for corr computation

    print("=== Computing Pearson/Spearman correlations (test set) ===")
    pearson, spearman = compute_correlations(X_test_num_eng, y_test)

    print("=== Loading held-out SHAP + permutation importance (test set) ===")
    _, shap_imp = compute_shap_importance(gbm, X_test, fm)
    perm_imp = compute_permutation_importance(gbm, X_test, y_test, fm)
    cluster_imp = aggregate_cluster_importance(shap_imp, fm.clustering)

    print("=== Fold stability (5 GroupKFold folds, TRAIN portion, full feature set) ===")
    fold_cv, fold_top20, per_fold_matrix = compute_fold_stability(fm_train, y_train, groups_train, n_splits=5, top_k=20)
    fold_cv.index = [fm.to_original(c) for c in fold_cv.index]
    fold_top20.index = [fm.to_original(c) for c in fold_top20.index]

    print("=== Tree co-occurrence (broad anchor set = top 150 individual+cluster features) ===")
    top_individual_broad = shap_imp.head(120).index.tolist()
    top_cluster_broad = []
    for cid, members in fm.clustering.clusters.items():
        rep = fm.clustering.representatives[cid]
        label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
        if label in cluster_imp.head(40).index:
            top_cluster_broad.append(rep)
    anchors = list(dict.fromkeys(top_individual_broad + top_cluster_broad))
    anchor_model_cols = fm.to_model_cols(anchors)
    coocc = compute_tree_cooccurrence(gbm, anchor_model_cols)
    coocc_original = pd.Series(coocc.values, index=[fm.to_original(c) for c in coocc.index])
    # features never co-occurring with any anchor get 0, not NaN
    coocc_full = coocc_original.reindex(pearson.index).fillna(0.0)

    inputs = FeatureStrengthInputs(
        pearson=pearson, spearman=spearman, perm_imp=perm_imp.reindex(pearson.index).fillna(0.0),
        shap_imp=shap_imp.reindex(pearson.index).fillna(0.0), cluster_imp=cluster_imp,
        fold_stability_cv=fold_cv, fold_top20_count=fold_top20.reindex(pearson.index).fillna(0),
        cooccurrence=coocc_full,
    )

    print("=== Classifying features ===")
    result = classify_features(inputs, fm.clustering, n_features=len(pearson))

    print("\nClassification counts:")
    print(result["classification"].value_counts().to_string())

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUTS_DIR / "reports" / "feature_strength_classification.csv", index=False)
    per_fold_matrix.to_csv(OUTPUTS_DIR / "reports" / "shap_importance_per_fold_full.csv")
    print("\nSaved outputs/reports/feature_strength_classification.csv")
    print("Saved outputs/reports/shap_importance_per_fold_full.csv")
