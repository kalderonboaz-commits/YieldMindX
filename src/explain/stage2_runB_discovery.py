"""Stage 2 Run B: single-parameter relationship discovery + interaction
discovery, LightGBM only, blind to the ground-truth document.

Pipeline:
  1. Fit the locked Run B model (train portion) + 5 GroupKFold fold models
     (full 1,425-feature set) -- reused for single-parameter fold-stability.
  2. Build a "screening set" (top 150 by combined importance) and classify
     each feature's PDP shape (direct / threshold / U-shaped / complex / flat).
  3. Build the INTERACTION CANDIDATE POOL via four explicit channels (not
     top-marginal-importance only):
       A. top individual SHAP importance
       B. nonlinear-shape-flagged features (from step 2)
       C. "unstable/conditional SHAP" features -- high within-decile SHAP
          dispersion relative to overall SHAP spread, a proxy for SHAP
          contribution depending on OTHER features rather than the
          feature's own value alone
       D. tree co-occurrence with the A/B/C anchor set (structural, from
          the fitted booster, no retraining needed)
  4. Fit 5 GroupKFold COMPACT fold models on the candidate pool -- used for
     SHAP-interaction fold-stability (out-of-fold, matching the pattern
     already validated in Stage 1).
  5. Rank SHAP interactions + H-statistic on a compact model (train
     portion), evaluated on the held-out test set.
  6. Save all outputs.

Never reads documents/corelation_data_table.docx.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import shap

from src.data.splits import make_group_kfold, make_holdout_split
from src.explain.interactions import (
    classify_pdp_shape,
    compute_h_statistic,
    compute_tree_cooccurrence,
    one_d_partial_dependence,
    rank_shap_interactions,
)
from src.features.stage2_matrix import load_stage2_dataset
from src.models.trees import make_lightgbm

OUTPUTS_DIR = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports"
RANDOM_STATE = 42


class _NameMapShim:
    def __init__(self, inverse_name_map: dict[str, str]):
        self._inv = inverse_name_map

    def to_original(self, c: str) -> str:
        return self._inv.get(c, c)


def compute_conditional_dispersion(shap_values: np.ndarray, X: pd.DataFrame, n_bins: int = 8) -> pd.Series:
    """Proxy for 'unstable / conditional SHAP behavior': for each feature,
    bin its values into deciles and compute the within-bin std of its SHAP
    contribution, normalized by the feature's overall SHAP std. A HIGH
    ratio means the SHAP contribution varies a lot even among rows with
    similar feature values -- i.e. the contribution depends on something
    ELSE (another feature), which is exactly the structural signature of
    an interaction-driven (rather than purely marginal) effect."""
    scores = {}
    for i, col in enumerate(X.columns):
        vals = X[col].to_numpy()
        sv = shap_values[:, i]
        overall_std = sv.std()
        if overall_std < 1e-12 or len(np.unique(vals)) < n_bins:
            scores[col] = 0.0
            continue
        try:
            bins = pd.qcut(vals, n_bins, duplicates="drop")
        except ValueError:
            scores[col] = 0.0
            continue
        within_bin_std = pd.Series(sv).groupby(bins, observed=True).std().mean()
        scores[col] = float(within_bin_std / overall_std) if overall_std > 0 else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def main():
    ds = load_stage2_dataset(include_review_columns=False)
    shim = _NameMapShim(ds.inverse_name_map)
    holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=RANDOM_STATE)
    X_train = ds.X.iloc[holdout.train_idx].reset_index(drop=True)
    X_test = ds.X.iloc[holdout.test_idx].reset_index(drop=True)
    y_train = ds.y.iloc[holdout.train_idx].reset_index(drop=True)
    y_test = ds.y.iloc[holdout.test_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)

    print("=== Fitting locked Run B final model (train portion) ===")
    final_model = make_lightgbm(random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)

    print("=== Computing full SHAP value matrix on holdout (for conditional-dispersion proxy) ===")
    explainer = shap.TreeExplainer(final_model)
    shap_values_test = explainer.shap_values(X_test)

    print("=== Fitting 5 GroupKFold full-feature fold models (single-parameter stability) ===")
    _, folds = make_group_kfold(groups_train, n_splits=5)
    fold_models = []
    for fold_i, (tr, va) in enumerate(folds):
        m = make_lightgbm(random_state=RANDOM_STATE)
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        fold_models.append((m, X_train.iloc[va].reset_index(drop=True)))
        print(f"  fold {fold_i} fit on {len(tr)} rows", flush=True)

    # ------------------------------------------------------------------
    # Screening set: top 150 by combined importance (union of SHAP/gain/perm top-150)
    # ------------------------------------------------------------------
    combined = pd.read_csv(f"{OUTPUTS_DIR}/stage2_runB_global_importance.csv")
    screen_names = set(combined.sort_values("shap_rank").head(150)["raw_csv_parameter_name"]) \
        | set(combined.sort_values("gain_rank").head(80)["raw_csv_parameter_name"]) \
        | set(combined.sort_values("permutation_rank").head(80)["raw_csv_parameter_name"])
    screen_names = [f for f in screen_names if f in ds.name_map]
    print(f"\n=== Screening set for single-parameter shape classification: {len(screen_names)} features ===")

    print("=== Classifying PDP shape for screening set (this can take a few minutes) ===")
    X_bg = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
    shape_rows = []
    for f in screen_names:
        mcol = ds.name_map[f]
        shape, direction = classify_pdp_shape(final_model, X_bg, mcol, grid_resolution=12)
        shape_rows.append({"feature": f, "shape": shape, "direction": direction})
    shape_df = pd.DataFrame(shape_rows).set_index("feature")
    print(shape_df["shape"].value_counts().to_string())

    nonlinear_flagged = shape_df[shape_df["shape"].isin(
        ["threshold", "U_shaped_process_window", "nonlinear_complex"]
    )].index.tolist()
    print(f"\nNonlinear-flagged features (channel B): {len(nonlinear_flagged)}")

    # ------------------------------------------------------------------
    # Conditional-dispersion proxy (channel C) -- computed for screening set
    # ------------------------------------------------------------------
    print("\n=== Computing conditional-dispersion (unstable/conditional SHAP) proxy ===")
    screen_model_cols = [ds.name_map[f] for f in screen_names]
    screen_idx = [list(X_test.columns).index(c) for c in screen_model_cols]
    disp = compute_conditional_dispersion(shap_values_test[:, screen_idx], X_test[screen_model_cols])
    disp.index = [ds.inverse_name_map[c] for c in disp.index]
    conditional_flagged = disp.head(20).index.tolist()
    print(f"Top 20 conditional-dispersion features (channel C): {conditional_flagged}")

    # ------------------------------------------------------------------
    # Build candidate pool: A (top-30 SHAP) + B (nonlinear) + C (top-20 conditional) + D (co-occurrence)
    # ------------------------------------------------------------------
    top_individual = combined.sort_values("shap_rank").head(20)["raw_csv_parameter_name"].tolist()
    anchors = list(dict.fromkeys(top_individual + nonlinear_flagged[:12] + conditional_flagged[:12]))
    anchor_model_cols = [ds.name_map[f] for f in anchors if f in ds.name_map]
    print(f"\n=== Channel D: tree co-occurrence with {len(anchor_model_cols)} anchors ===")
    coocc = compute_tree_cooccurrence(final_model, anchor_model_cols)
    coocc_original = pd.Series(coocc.values, index=[ds.inverse_name_map.get(c, c) for c in coocc.index])
    coocc_new = [f for f in coocc_original.index if f not in anchors][:12]
    print(f"Weak-marginal / interaction-potential candidates (channel D): {coocc_new}")

    candidate_pool = list(dict.fromkeys(anchors + coocc_new))
    print(f"\n=== FINAL INTERACTION CANDIDATE POOL: {len(candidate_pool)} features ===")
    print(f"  A (top individual SHAP): {len(top_individual)}")
    print(f"  B (nonlinear-flagged, capped 20): {min(20, len(nonlinear_flagged))}")
    print(f"  C (conditional-dispersion, top 20): {len(conditional_flagged)}")
    print(f"  D (tree co-occurrence, new only): {len(coocc_new)}")
    candidate_model_cols = [ds.name_map[f] for f in candidate_pool if f in ds.name_map]

    # save intermediate artifacts before the expensive interaction step
    shape_df.reset_index().to_csv(f"{OUTPUTS_DIR}/_stage2_runB_pdp_shapes_screening.csv", index=False)
    pd.Series(candidate_pool, name="feature").to_csv(f"{OUTPUTS_DIR}/_stage2_runB_interaction_candidate_pool.csv", index=False)

    # ------------------------------------------------------------------
    # Compact model on candidate pool (train) -- for SHAP-interaction + H-statistic
    # ------------------------------------------------------------------
    print("\n=== Fitting compact LightGBM on candidate pool (train portion) ===")
    compact_model = make_lightgbm(random_state=RANDOM_STATE)
    compact_model.fit(X_train[candidate_model_cols], y_train)

    print("=== Ranking SHAP interactions (holdout set) ===")
    shap_pairs = rank_shap_interactions(compact_model, X_test[candidate_model_cols], shim,
                                         sample_size=min(300, len(X_test)), top_k=40)

    print("=== Ranking H-statistic interactions (holdout set, independent cross-check) ===")
    h_pairs = rank_h_statistic_interactions_local(compact_model, X_test[candidate_model_cols], candidate_model_cols, shim)

    print("\n=== Fitting 5 GroupKFold COMPACT fold models (interaction fold-stability, out-of-fold) ===")
    compact_fold_pairs = []
    for fold_i, (tr, va) in enumerate(folds):
        m = make_lightgbm(random_state=RANDOM_STATE)
        m.fit(X_train.iloc[tr][candidate_model_cols], y_train.iloc[tr])
        pairs = rank_shap_interactions(m, X_train.iloc[va][candidate_model_cols].reset_index(drop=True), shim,
                                        sample_size=min(300, len(va)), top_k=40)
        pair_set = {frozenset([p.feature_a, p.feature_b]) for p in pairs}
        compact_fold_pairs.append(pair_set)
        print(f"  interaction-stability fold {fold_i}: top pair = "
              f"{pairs[0].feature_a} x {pairs[0].feature_b}" if pairs else "  (no pairs)", flush=True)

    all_pairs = set().union(*compact_fold_pairs)
    pair_fold_counts = pd.Series({p: sum(p in s for s in compact_fold_pairs) for p in all_pairs})

    # persist everything needed by the downstream synthesis script
    import pickle
    with open(f"{OUTPUTS_DIR}/_stage2_runB_discovery_state.pkl", "wb") as f:
        pickle.dump({
            "shape_df": shape_df, "disp": disp, "candidate_pool": candidate_pool,
            "shap_pairs": shap_pairs, "h_pairs": h_pairs, "pair_fold_counts": pair_fold_counts,
            "n_folds": len(folds),
        }, f)

    print("\n=== Stage 2 Run B discovery pipeline stage 1 complete. State saved for synthesis. ===")
    print(f"Top 15 SHAP interactions:")
    for p in shap_pairs[:15]:
        print(f"  {p.feature_a} x {p.feature_b}  strength={p.interaction_strength:.4f}  "
              f"fold_count={int(pair_fold_counts.get(frozenset([p.feature_a,p.feature_b]), 0))}/{len(folds)}")
    print(f"\nTop 15 H-statistic interactions:")
    for p in h_pairs[:15]:
        print(f"  {p.feature_a} x {p.feature_b}  strength={p.interaction_strength:.4f}")


def rank_h_statistic_interactions_local(model, X, candidate_model_cols, shim, grid_resolution=6, top_k=40, background_size=60):
    from itertools import combinations
    X_bg = X.sample(min(background_size, len(X)), random_state=RANDOM_STATE).astype(np.float64)
    from src.explain.interactions import TopInteraction
    results = []
    pairs = list(combinations(candidate_model_cols, 2))
    for a, b in pairs:
        h = compute_h_statistic(model, X_bg, a, b, grid_resolution=grid_resolution)
        if np.isnan(h):
            continue
        results.append(TopInteraction(feature_a=shim.to_original(a), feature_b=shim.to_original(b),
                                       interaction_strength=h, method="h_statistic"))
    results.sort(key=lambda r: -r.interaction_strength)
    return results[:top_k]


if __name__ == "__main__":
    main()
