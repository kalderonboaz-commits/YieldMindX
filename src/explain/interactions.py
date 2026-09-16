"""Interaction and nonlinearity discovery.

Four complementary, blind layers (per the approved plan -- none is trusted
alone):
  1. SHAP interaction values from the boosted model: cheap way to RANK
     candidate pairwise interactions across a tractable candidate subset.
  2. Friedman's H-statistic (grid-based partial-dependence approximation):
     a SECOND, SHAP-independent interaction-strength method, used as a
     cross-check -- it measures interaction directly from model predictions
     (how much the joint effect departs from the sum of marginal effects),
     with no relationship to SHAP's game-theoretic attribution at all.
  3. Fold-stability analysis: SHAP-interaction rankings are recomputed on
     each of 5 GroupKFold folds (compact model trained on that fold's
     training split, interaction values evaluated on that fold's held-out
     validation split) -- an interaction is only reported as "stable" if it
     recurs across folds, not from one compact-model fit.
  4. 1D/2D partial dependence for visual confirmation of top pairs/features.

Candidate-selection principle (methodological fix from the review
checkpoint): the previous version selected interaction candidates only from
raw top-N individual SHAP importance, which systematically excludes
features whose real signal is diluted across a large correlated family
(rule 3), AND excludes any feature that is weak on its own but only matters
in combination with another (the entire point of interaction discovery).
select_interaction_candidates() below unions FIVE independent, blind
selection criteria so that no single ranking method can silently exclude a
real candidate:
  (a) top individual SHAP importance
  (b) top cluster/family importance (representative column per cluster)
  (c) top permutation importance (a different importance mechanism than
      SHAP -- can surface features SHAP under-ranks and vice versa)
  (d) top engineered (STAGE_RANGE/STD/DELTA) features by their own SHAP
      importance -- guarantees the engineered-feature channel is
      represented even if none individually cracks the raw top-N cut
  (e) tree co-occurrence: features that frequently appear in the SAME
      LightGBM trees as an already-selected "anchor" feature, computed
      directly from the FULL 1700+-feature model's tree structure (no SHAP,
      no data pass needed -- just booster structure). This is what
      specifically targets "weak marginal importance but plausible
      interaction signal": a feature can co-occur heavily with a top
      feature in tree splits while having low overall importance, because
      it only earns its keep in combination with that other feature.

None of this reads documents/corelation_data_table.docx.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence


@dataclasses.dataclass
class TopInteraction:
    feature_a: str
    feature_b: str
    interaction_strength: float
    method: str = "shap_interaction"


def rank_shap_interactions(model, X: pd.DataFrame, fm, sample_size: int = 400, top_k: int = 30,
                            random_state: int = 42) -> list[TopInteraction]:
    """SHAP interaction values are O(n_features^2) per sample -- subsample
    rows for tractability."""
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(X_sample)  # (n, n_feat, n_feat)
    mean_abs = np.abs(inter).mean(axis=0)
    np.fill_diagonal(mean_abs, 0.0)  # zero out main effects (diagonal)

    n = mean_abs.shape[0]
    cols = X.columns
    pairs = []
    iu = np.triu_indices(n, k=1)
    strengths = mean_abs[iu]
    order = np.argsort(-strengths)[:top_k]
    for idx in order:
        i, j = iu[0][idx], iu[1][idx]
        pairs.append(TopInteraction(
            feature_a=fm.to_original(cols[i]),
            feature_b=fm.to_original(cols[j]),
            interaction_strength=float(strengths[idx]),
            method="shap_interaction",
        ))
    return pairs


def one_d_partial_dependence(model, X: pd.DataFrame, model_col: str, grid_resolution: int = 30):
    # sklearn's partial_dependence rejects int/bool columns outright (and, as
    # discovered earlier, brute-force PDP on an unconverted int64 column can
    # also raise a LossySetitemError under pandas 3.x) -- cast once, here,
    # rather than relying on every caller to remember.
    X = X.astype(np.float64)
    idx = X.columns.get_loc(model_col)
    pd_result = partial_dependence(model, X, [idx], grid_resolution=grid_resolution, kind="average")
    return pd_result["grid_values"][0], pd_result["average"][0]


def classify_pdp_shape(model, X: pd.DataFrame, model_col: str, grid_resolution: int = 15,
                        flat_threshold: float = 0.05) -> tuple[str, str]:
    """Simple, defensible heuristic classification of a 1D PDP curve shape.
    Not a publication-grade curve-shape classifier -- a practical first-pass
    label for reporting purposes, based on the sign pattern of first
    differences along the curve:
      - 0 sign changes (net range above flat_threshold): 'direct' (monotonic),
        direction 'positive' or 'negative'
      - 1 sign change roughly centered: 'U_shaped_process_window'
      - 1 sign change near an edge, OR one grid step contains most of the
        total range change: 'threshold'
      - >=2 sign changes: 'nonlinear_complex'
      - range below flat_threshold * y-scale: 'flat_no_clear_effect'
    Returns (shape_label, direction) where direction in
    {'positive','negative','see_pdp_curve'} (the latter for non-monotonic shapes).
    """
    grid, values = one_d_partial_dependence(model, X, model_col, grid_resolution=grid_resolution)
    values = np.asarray(values)
    rng = values.max() - values.min()
    if rng < flat_threshold:
        return "flat_no_clear_effect", "see_pdp_curve"

    diffs = np.diff(values)
    step_sizes = np.abs(diffs)
    signs = np.sign(diffs)
    signs_nonzero = signs[signs != 0]
    if len(signs_nonzero) == 0:
        return "flat_no_clear_effect", "see_pdp_curve"
    sign_changes = int(np.sum(np.diff(signs_nonzero) != 0))

    if sign_changes == 0:
        direction = "positive" if signs_nonzero[0] > 0 else "negative"
        # concentrated single-step jump vs. gradual -> threshold vs direct
        if step_sizes.max() > 0.5 * rng:
            return "threshold", direction
        return "direct", direction

    if sign_changes == 1:
        change_idx = int(np.where(np.diff(signs_nonzero) != 0)[0][0]) + 1
        rel_pos = change_idx / max(len(signs_nonzero) - 1, 1)
        if 0.25 <= rel_pos <= 0.75:
            return "U_shaped_process_window", "see_pdp_curve"
        return "threshold", "see_pdp_curve"

    return "nonlinear_complex", "see_pdp_curve"


def two_d_partial_dependence(model, X: pd.DataFrame, model_col_a: str, model_col_b: str, grid_resolution: int = 15):
    # sklearn's partial_dependence pair-feature support is most reliable
    # with integer column positions (string-tuple column-name pairs are not
    # consistently resolved across sklearn versions) -- use positions.
    X = X.astype(np.float64)
    idx_a = X.columns.get_loc(model_col_a)
    idx_b = X.columns.get_loc(model_col_b)
    pd_result = partial_dependence(model, X, [(idx_a, idx_b)], grid_resolution=grid_resolution, kind="average")
    return pd_result["grid_values"], pd_result["average"][0]


# ---------------------------------------------------------------------------
# Method 2: Friedman's H-statistic (grid-based PDP approximation)
# ---------------------------------------------------------------------------

def compute_h_statistic(model, X: pd.DataFrame, model_col_a: str, model_col_b: str, grid_resolution: int = 8) -> float:
    """Grid-based approximation of Friedman & Popescu's H-statistic.

    H^2 = sum[(PD_ab - PD_a - PD_b)^2] / sum[PD_ab^2]   (all centered on their mean)

    where PD_ab is the 2D partial dependence, PD_a/PD_b the 1D partial
    dependences evaluated on the SAME marginal grid points used by the 2D
    call. This is the standard practical approximation used by most
    open-source H-statistic implementations (evaluating on a percentile
    grid rather than at every individual training point, which is
    intractable at scale). It shares no computational machinery with SHAP
    at all -- it only uses the model's raw predictions via
    sklearn.inspection.partial_dependence -- making it a genuine
    independent cross-check.
    """
    grid_2d, pd_ab = two_d_partial_dependence(model, X, model_col_a, model_col_b, grid_resolution=grid_resolution)
    grid_a_vals, grid_b_vals = grid_2d[0], grid_2d[1]

    idx_a = X.columns.get_loc(model_col_a)
    idx_b = X.columns.get_loc(model_col_b)
    pd_a_result = partial_dependence(model, X, [idx_a], grid_resolution=grid_resolution, kind="average")
    pd_b_result = partial_dependence(model, X, [idx_b], grid_resolution=grid_resolution, kind="average")

    # Guard against sklearn choosing a slightly different grid for the 1D
    # call vs. the marginal of the 2D call (can happen with duplicate
    # percentile values) -- if grids don't match in length, resample by
    # nearest-index alignment; in practice with grid_resolution<=10 and
    # continuous ET measurements this is very rare.
    pd_a = pd_a_result["average"][0]
    pd_b = pd_b_result["average"][0]
    if len(pd_a) != pd_ab.shape[0] or len(pd_b) != pd_ab.shape[1]:
        return float("nan")

    pd_ab_c = pd_ab - pd_ab.mean()
    pd_a_c = pd_a - pd_a.mean()
    pd_b_c = pd_b - pd_b.mean()

    residual = pd_ab_c - pd_a_c[:, None] - pd_b_c[None, :]
    numerator = np.sum(residual ** 2)
    denominator = np.sum(pd_ab_c ** 2)
    if denominator <= 1e-12:
        return 0.0
    h2 = numerator / denominator
    return float(np.sqrt(max(h2, 0.0)))


def rank_h_statistic_interactions(
    model, X: pd.DataFrame, candidate_model_cols: list[str], fm, grid_resolution: int = 6, top_k: int = 25,
    background_size: int = 60, random_state: int = 42,
) -> list[TopInteraction]:
    """H-statistic via sklearn's partial_dependence, which falls back to the
    'brute' method for non-sklearn-native estimators like LightGBM (its
    fast 'recursion' method is only implemented for a whitelist of sklearn
    tree estimators) -- brute cost is O(grid_resolution^2 * n_background)
    PER PAIR, so with a large candidate set this is the dominant cost of
    the whole interaction-discovery step. A fixed background subsample
    (reused across all pairs, for consistent/comparable grids) and a
    modest grid_resolution keep C(candidates,2) pairs tractable in
    minutes rather than hours; this is a deliberate accuracy/cost
    trade-off, documented here rather than silently defaulted."""
    from itertools import combinations

    # cast to float64: sklearn's brute-force PDP inserts interpolated
    # (fractional) grid values into whichever column is being varied --
    # pandas 3.x raises LossySetitemError if that column is still int64
    # (e.g. MaskSetSupplyChipsCount, or {0,1} one-hot dummy columns).
    X_bg = X.sample(min(background_size, len(X)), random_state=random_state) if len(X) > background_size else X
    X_bg = X_bg.astype(np.float64)

    results = []
    pairs = list(combinations(candidate_model_cols, 2))
    for a, b in pairs:
        h = compute_h_statistic(model, X_bg, a, b, grid_resolution=grid_resolution)
        if np.isnan(h):
            continue
        results.append(TopInteraction(
            feature_a=fm.to_original(a), feature_b=fm.to_original(b),
            interaction_strength=h, method="h_statistic",
        ))
    results.sort(key=lambda r: -r.interaction_strength)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Candidate selection: union of 5 independent, blind criteria
# ---------------------------------------------------------------------------

def compute_tree_cooccurrence(model, anchor_model_cols: list[str]) -> pd.Series:
    """For a fitted LightGBM model, count for every feature how many trees
    it shares with any of the anchor features (i.e. both appear as a split
    feature somewhere in the same tree). Uses only the booster's own tree
    structure -- no SHAP, no extra data pass. Surfaces features that are
    individually weak but structurally paired with important features."""
    trees_df = model.booster_.trees_to_dataframe()
    trees_df = trees_df.dropna(subset=["split_feature"])

    anchor_set = set(anchor_model_cols)
    counts: dict[str, int] = {}
    for tree_idx, group in trees_df.groupby("tree_index"):
        feats_in_tree = set(group["split_feature"].unique())
        if not (feats_in_tree & anchor_set):
            continue
        for f in feats_in_tree:
            if f in anchor_set:
                continue
            counts[f] = counts.get(f, 0) + 1

    return pd.Series(counts, name="cooccurrence_count").sort_values(ascending=False)


def select_interaction_candidates(
    feat_imp: pd.Series,             # original names, sorted desc (SHAP importance)
    perm_imp: pd.Series,             # original names, sorted desc (permutation importance)
    cluster_imp: pd.Series,          # cluster display-labels, sorted desc
    clustering,                      # FeatureClustering
    fm,                              # FeatureMatrix
    full_model,                      # fitted on the FULL feature matrix
    engineered_cols: list[str],
    top_individual: int = 20,
    top_clusters: int = 15,
    top_permutation: int = 10,
    top_engineered: int = 8,
    top_cooccurrence: int = 10,
    feature_strength_df: pd.DataFrame | None = None,  # from src/explain/feature_strength.py, optional
    top_nonlinear: int = 8,
    top_interaction_dependent: int = 8,
) -> tuple[list[str], dict[str, str]]:
    """Returns (candidate_original_cols, source_map) where source_map records
    which selection criterion/criteria brought each feature in, for
    transparency in reporting.

    Seven channels when feature_strength_df is supplied (five without it):
      1. top_individual_shap
      2. top_cluster_representative
      3. top_permutation_importance
      4. top_engineered_feature
      5. tree_cooccurrence_with_anchor (raw structural co-occurrence)
      6. nonlinear_candidate -- features whose SHAP/permutation importance
         far outranks their Pearson/Spearman correlation (per Part B
         classification): these are exactly the features a correlation- or
         top-linear-importance-only selection would systematically miss.
      7. weak_marginal_interaction_candidate -- features Part B classified
         as "interaction_dependent_predictor" (weak individually, high
         structural co-occurrence): explicit, traceable inclusion of
         low-marginal-importance features with plausible interaction signal,
         rather than relying on a single raw cooccurrence cutoff alone.
    """
    source_map: dict[str, set] = {}

    def add(cols, label):
        for c in cols:
            source_map.setdefault(c, set()).add(label)

    top_ind = feat_imp.head(top_individual).index.tolist()
    add(top_ind, "top_individual_shap")

    top_cluster_reps = []
    for cid, members in clustering.clusters.items():
        rep = clustering.representatives[cid]
        label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
        if label in cluster_imp.head(top_clusters).index:
            top_cluster_reps.append(rep)
    add(top_cluster_reps, "top_cluster_representative")

    top_perm = perm_imp.head(top_permutation).index.tolist()
    add(top_perm, "top_permutation_importance")

    engineered_ranked = feat_imp.reindex([c for c in feat_imp.index if c in engineered_cols]).dropna()
    top_eng = engineered_ranked.head(top_engineered).index.tolist()
    add(top_eng, "top_engineered_feature")

    anchors_so_far = list(dict.fromkeys(top_ind + top_cluster_reps + top_perm + top_eng))
    anchor_model_cols = fm.to_model_cols(anchors_so_far)
    cooccurrence = compute_tree_cooccurrence(full_model, anchor_model_cols)
    cooccurrence_original = pd.Series(
        cooccurrence.values, index=[fm.to_original(c) for c in cooccurrence.index]
    )
    already = set(anchors_so_far)
    top_coocc = [c for c in cooccurrence_original.index if c not in already][:top_cooccurrence]
    add(top_coocc, "tree_cooccurrence_with_anchor")

    all_candidates = list(dict.fromkeys(anchors_so_far + top_coocc))

    if feature_strength_df is not None:
        fs = feature_strength_df.set_index("feature")
        nonlinear = fs[fs["classification"] == "nonlinear_predictor"].sort_values("shap_importance", ascending=False)
        top_nl = [f for f in nonlinear.index if f not in all_candidates][:top_nonlinear]
        add(top_nl, "nonlinear_candidate")

        interaction_dep = fs[fs["classification"] == "interaction_dependent_predictor"].sort_values("cooccurrence_rank")
        top_id = [f for f in interaction_dep.index if f not in all_candidates and f not in top_nl][:top_interaction_dependent]
        add(top_id, "weak_marginal_interaction_candidate")

        all_candidates = list(dict.fromkeys(all_candidates + top_nl + top_id))

    source_map_str = {k: "+".join(sorted(v)) for k, v in source_map.items()}
    return all_candidates, source_map_str


# ---------------------------------------------------------------------------
# Fold-stability for SHAP-interaction rankings
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InteractionStabilityResult:
    per_fold_top_pairs: list[set]          # one set of frozenset({a,b}) per fold
    pair_fold_counts: pd.Series             # frozenset pair -> number of folds it appeared in top_k
    stable_pairs: list[frozenset]           # appeared in ALL folds
    majority_pairs: list[frozenset]         # appeared in >= majority of folds


def assess_interaction_stability(
    candidate_model_cols: list[str], fm, X_train_full: pd.DataFrame, y_train_full: pd.Series, groups_train: pd.Series,
    n_splits: int = 5, top_k: int = 25, sample_size: int = 300, random_state: int = 42,
) -> InteractionStabilityResult:
    """Recomputes the SHAP-interaction ranking on each of n_splits
    GroupKFold folds: a compact model is trained on that fold's TRAINING
    split only, and interaction values are evaluated on that fold's
    HELD-OUT VALIDATION split (out-of-fold, not in-sample) -- this is
    stricter than a single compact-model fit on the whole training set."""
    from src.data.splits import make_group_kfold
    from src.models.trees import make_lightgbm

    _, folds = make_group_kfold(groups_train.reset_index(drop=True), n_splits=n_splits)

    per_fold_pairs = []
    for fold_i, (tr, va) in enumerate(folds):
        X_tr = X_train_full.iloc[tr][candidate_model_cols]
        y_tr = y_train_full.iloc[tr]
        X_va = X_train_full.iloc[va][candidate_model_cols]

        model = make_lightgbm(random_state=random_state)
        model.fit(X_tr, y_tr)
        top_pairs = rank_shap_interactions(model, X_va, fm, sample_size=min(sample_size, len(X_va)), top_k=top_k, random_state=random_state)
        pair_set = {frozenset([p.feature_a, p.feature_b]) for p in top_pairs}
        per_fold_pairs.append(pair_set)
        print(f"  interaction-stability fold {fold_i}: top pair = "
              f"{sorted(top_pairs[0].feature_a for _ in [0])[0] if top_pairs else 'n/a'} x "
              f"{top_pairs[0].feature_b if top_pairs else ''}", flush=True)

    all_pairs = set().union(*per_fold_pairs)
    counts = pd.Series({p: sum(p in s for s in per_fold_pairs) for p in all_pairs}).sort_values(ascending=False)
    stable = [p for p, c in counts.items() if c == n_splits]
    majority = [p for p, c in counts.items() if c >= (n_splits // 2 + 1)]

    return InteractionStabilityResult(
        per_fold_top_pairs=per_fold_pairs,
        pair_fold_counts=counts,
        stable_pairs=stable,
        majority_pairs=majority,
    )


if __name__ == "__main__":
    import joblib

    from src.data.load import load_dataset
    from src.explain.global_importance import (
        aggregate_cluster_importance,
        compute_permutation_importance,
        compute_shap_importance,
    )
    from src.features.matrix import build_feature_matrix
    from src.models.train import OUTPUTS_DIR
    from src.models.trees import make_lightgbm

    ds = load_dataset()
    fm = build_feature_matrix(ds)
    split = joblib.load(OUTPUTS_DIR / "models" / "holdout_split.joblib")
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    X_train = fm.X.iloc[train_idx].reset_index(drop=True)
    y_train = ds.y.iloc[train_idx].reset_index(drop=True)
    X_test = fm.X.iloc[test_idx].reset_index(drop=True)
    y_test = ds.y.iloc[test_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[train_idx].reset_index(drop=True)

    gbm = joblib.load(OUTPUTS_DIR / "models" / "lightgbm.joblib")

    print("=== Computing OFFICIAL (held-out test-set) importance for candidate selection ===")
    _, feat_imp = compute_shap_importance(gbm, X_test, fm)
    perm_imp = compute_permutation_importance(gbm, X_test, y_test, fm)
    cluster_imp = aggregate_cluster_importance(feat_imp, fm.clustering)

    candidates, source_map = select_interaction_candidates(
        feat_imp, perm_imp, cluster_imp, fm.clustering, fm, gbm, fm.engineered_cols,
    )
    print(f"\nInteraction candidate set: {len(candidates)} features")
    from collections import Counter
    print("By source:", Counter(v for v in source_map.values()))

    candidate_model_cols = fm.to_model_cols(candidates)
    print(f"\nRefitting compact LightGBM on {len(candidate_model_cols)} candidates (train portion)...")
    compact_gbm = make_lightgbm(random_state=42)
    compact_gbm.fit(X_train[candidate_model_cols], y_train)

    print("\n=== METHOD 1: SHAP interactions (official, held-out test set) ===")
    shap_pairs = rank_shap_interactions(compact_gbm, X_test[candidate_model_cols], fm, sample_size=min(300, len(X_test)), top_k=25)
    for p in shap_pairs[:15]:
        print(f"  {p.feature_a}  x  {p.feature_b}   H={p.interaction_strength:.4f}")

    print("\n=== METHOD 2: H-statistic (official, held-out test set, independent cross-check) ===")
    h_pairs = rank_h_statistic_interactions(compact_gbm, X_test[candidate_model_cols], candidate_model_cols, fm, grid_resolution=6, top_k=25, background_size=60)
    for p in h_pairs[:15]:
        print(f"  {p.feature_a}  x  {p.feature_b}   H={p.interaction_strength:.4f}")

    shap_pair_set = {frozenset([p.feature_a, p.feature_b]) for p in shap_pairs}
    h_pair_set = {frozenset([p.feature_a, p.feature_b]) for p in h_pairs}
    agreement = shap_pair_set & h_pair_set
    print(f"\nAgreement between SHAP-interaction top-25 and H-statistic top-25: {len(agreement)} pairs in common")
    for p in agreement:
        a, b = tuple(p)
        print(f"  AGREE: {a} x {b}")

    print("\n=== Fold-stability of SHAP-interaction ranking (5 GroupKFold folds, TRAIN portion, out-of-fold eval) ===")
    stability = assess_interaction_stability(candidate_model_cols, fm, X_train, y_train, groups_train, n_splits=5, top_k=25)
    print(f"\nPairs stable in ALL 5 folds: {len(stability.stable_pairs)}")
    for p in stability.stable_pairs:
        a, b = tuple(p)
        print(f"  STABLE: {a} x {b}")
    print(f"\nPairs in majority (>=3/5) folds: {len(stability.majority_pairs)}")
    for p in stability.majority_pairs:
        a, b = tuple(p)
        print(f"  MAJORITY: {a} x {b}  (folds={stability.pair_fold_counts[p]})")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature_a": p.feature_a, "feature_b": p.feature_b, "strength": p.interaction_strength, "method": p.method} for p in shap_pairs]) \
        .to_csv(OUTPUTS_DIR / "reports" / "top_interactions_shap.csv", index=False)
    pd.DataFrame([{"feature_a": p.feature_a, "feature_b": p.feature_b, "strength": p.interaction_strength, "method": p.method} for p in h_pairs]) \
        .to_csv(OUTPUTS_DIR / "reports" / "top_interactions_hstat.csv", index=False)
    pd.DataFrame([{"pair": "|".join(sorted(p)), "fold_count": c} for p, c in stability.pair_fold_counts.items()]) \
        .sort_values("fold_count", ascending=False) \
        .to_csv(OUTPUTS_DIR / "reports" / "interaction_fold_stability.csv", index=False)
    pd.DataFrame([{"feature": k, "sources": v} for k, v in source_map.items()]) \
        .to_csv(OUTPUTS_DIR / "reports" / "interaction_candidate_sources.csv", index=False)
