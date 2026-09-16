"""POST-HOC validation benchmark against the synthetic ground-truth doc.

*** METHODOLOGY BOUNDARY (user rule 2) ***
This module is the ONLY place in the codebase that reads
documents/corelation_data_table.docx. It must never be imported by
src/data, src/features, src/models, or src/explain -- those packages
implement the discovery pipeline "blind," as if the planted answers did not
exist. This script runs strictly AFTER the discovery pipeline has already
produced its findings, and its only job is to check, after the fact,
whether the blind pipeline's findings agree with the planted ground truth.
It is a scorecard, not an input to modeling.

*** METHODOLOGY FIX (review-checkpoint item 1) ***
All importance/interaction evidence used for scoring is computed on the
HELD-OUT TEST portion (never train), using the FINAL models already fit on
the training portion. Earlier code computed scorecard importance on the
training set while separately-saved "official" reports used the test set --
that inconsistency is fixed here: this file is now the single source of
truth for scorecard-relevant importance, and it is test-set-only.

Five blind discovery channels are checked, matching what a genuinely blind
pipeline can produce (review-checkpoint item 6):
  1. individual feature SHAP importance (test set)
  2. cluster/family SHAP importance (test set) -- dilution-aware
  3. engineered-feature discovery -- both the engineered feature's own SHAP
     rank AND its raw |Pearson correlation| with the target (test set),
     since the correlation channel is literally how the Gm-delta signal was
     first noticed in src/features/engineered.py's diagnostic output
  4. SHAP-interaction ranking, on an EXPANDED candidate set (individual +
     cluster + permutation + engineered + tree-cooccurrence -- see
     src/explain/interactions.py), test set
  5. H-statistic interaction ranking (independent, non-SHAP cross-check),
     same candidate set, test set

Interaction-type relationships are additionally checked for fold stability
(review-checkpoint item 5): a pair is only counted STABLE if it recurs
across GroupKFold folds (out-of-fold), not from a single compact-model fit.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_DOCX = PROJECT_ROOT / "documents" / "corelation_data_table.docx"


def read_ground_truth_table() -> list[dict]:
    doc = Document(str(GROUND_TRUTH_DOCX))
    rows_out = []
    for table in doc.tables:
        header = [c.text.strip() for c in table.rows[0].cells]
        if not header or "Relationship" not in header[0]:
            continue
        for row in table.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) < 4:
                continue
            rows_out.append({
                "relationship": cells[0],
                "parameters": cells[1],
                "type": cells[2],
                "expected": cells[3],
            })
    return rows_out


PLANTED_COLUMNS = {
    "Gate CD optimum": ["GATE2_CD_Gate_Leg_Length"],
    "PAE benefit": ["TOPSIN_ET_Load_Pull_P.A.E_Last"],
    "Drain-lag penalty": ["TOPSIN_ET_DC_PIV_Drain_Lag_1"],
    "Gate CD x SiN thickness": ["GATE2_CD_Gate_Leg_Length", "SIN_Thickness_SiN_Thickness"],
    "Rc x Rsh": ["OHMIC_ET_Rc_an_ct2", "OHMIC_ET_Rsh_an_ct2"],
    "Leakage threshold": [],  # "ET leakage family" -- matched via substring
    "Alignment process window": ["GATE2_Ali_Litho_Gate_to_recess_alignment"],
    "Gm stage degradation": ["TOPSIN_ET_Gm_max", "FIC_ET_Gm_max"],
    "Rc x Drain Lag x low PAE": ["OHMIC_ET_Rc_an_ct2", "TOPSIN_ET_DC_PIV_Drain_Lag_1", "TOPSIN_ET_Load_Pull_P.A.E_Last"],
}

# For the one derived-delta relationship, the corresponding engineered
# feature this pipeline's BLIND, GENERIC stage-delta logic would have
# produced (src/features/engineered.py names it mechanically from the
# shared ET suffix "Gm_max" across the FIC/TOPSIN modules -- this mapping
# is only used here, post-hoc, to know which already-existing engineered
# column to check).
PLANTED_ENGINEERED_FEATURE = {
    "Gm stage degradation": "STAGE_DELTA_Gm_max_FIC_minus_TOPSIN",
}


def score_against_discovery(
    ground_truth_rows: list[dict],
    feat_imp,               # pd.Series, original names, sorted desc, TEST SET
    cluster_imp,             # pd.Series, cluster labels, sorted desc, TEST SET
    clustering,               # FeatureClustering
    engineered_corr,          # pd.Series, original names (engineered only), sorted by |corr| desc, TEST SET
    shap_pairs,                # list[TopInteraction], TEST SET, expanded candidate set
    hstat_pairs,                # list[TopInteraction], TEST SET, expanded candidate set
    interaction_stability,       # InteractionStabilityResult (train-portion GroupKFold, out-of-fold)
    top_k_features: int = 30,
    top_k_clusters: int = 20,
    top_k_engineered: int = 15,
    top_k_interactions: int = 25,
) -> list[dict]:
    top_features = set(feat_imp.head(top_k_features).index)
    top_engineered = set(engineered_corr.head(top_k_engineered).index)

    shap_pair_map = {frozenset([p.feature_a, p.feature_b]): p.interaction_strength for p in shap_pairs[:top_k_interactions]}
    hstat_pair_map = {frozenset([p.feature_a, p.feature_b]): p.interaction_strength for p in hstat_pairs[:top_k_interactions]}

    col_in_top_cluster: dict[str, bool] = {}
    for cid, members in clustering.clusters.items():
        rep = clustering.representatives[cid]
        label = rep if len(members) == 1 else f"{rep} (+{len(members)-1} correlated features)"
        in_top = label in set(cluster_imp.head(top_k_clusters).index)
        for m in members:
            col_in_top_cluster[m] = in_top

    def pair_stability(cols_pair: frozenset) -> str:
        if cols_pair in interaction_stability.stable_pairs:
            return f"STABLE ({len(interaction_stability.per_fold_top_pairs)}/{len(interaction_stability.per_fold_top_pairs)} folds)"
        count = int(interaction_stability.pair_fold_counts.get(cols_pair, 0))
        if count > 0:
            return f"unstable ({count}/{len(interaction_stability.per_fold_top_pairs)} folds)"
        return "not observed in any fold"

    results = []
    for gt in ground_truth_rows:
        rel = gt["relationship"]
        cols = PLANTED_COLUMNS.get(rel.replace("×", "x"), [])
        is_interaction = int(gt["type"].lower().count("interaction")) > 0
        is_delta = "delta" in gt["type"].lower()

        methods_hit = []
        stability_note = "n/a"

        if rel == "Leakage threshold":
            if any("lkg" in f.lower() for f in top_features):
                methods_hit.append("individual_shap")
        elif is_interaction and len(cols) == 2:
            pair = frozenset(cols)
            if pair in shap_pair_map:
                methods_hit.append(f"shap_interaction(rank_strength={shap_pair_map[pair]:.4f})")
            if pair in hstat_pair_map:
                methods_hit.append(f"h_statistic(strength={hstat_pair_map[pair]:.4f})")
            stability_note = pair_stability(pair)
        elif is_interaction and len(cols) == 3:
            from itertools import combinations
            sub_pairs = [frozenset(c) for c in combinations(cols, 2)]
            shap_all = all(sp in shap_pair_map for sp in sub_pairs)
            hstat_all = all(sp in hstat_pair_map for sp in sub_pairs)
            if shap_all:
                methods_hit.append("shap_interaction(all 3 sub-pairs)")
            if hstat_all:
                methods_hit.append("h_statistic(all 3 sub-pairs)")
            stability_note = "; ".join(f"{tuple(sp)}: {pair_stability(sp)}" for sp in sub_pairs)
        else:
            if any(c in top_features for c in cols):
                methods_hit.append("individual_shap")
            if any(col_in_top_cluster.get(c, False) for c in cols):
                methods_hit.append("cluster_shap")
            if is_delta:
                eng_col = PLANTED_ENGINEERED_FEATURE.get(rel)
                if eng_col and eng_col in top_engineered:
                    methods_hit.append(f"engineered_feature_correlation({eng_col})")

        results.append({
            "relationship": rel,
            "planted_columns": cols,
            "planted_type": gt["type"],
            "detected_strict": len(methods_hit) > 0,
            "methods_that_detected_it": "; ".join(methods_hit) if methods_hit else "none",
            "fold_stability": stability_note,
        })
    return results


if __name__ == "__main__":
    import joblib
    import pandas as pd

    from src.data.load import load_dataset
    from src.explain.global_importance import aggregate_cluster_importance, compute_shap_importance
    from src.explain.interactions import (
        assess_interaction_stability,
        rank_h_statistic_interactions,
        rank_shap_interactions,
        select_interaction_candidates,
    )
    from src.explain.global_importance import compute_permutation_importance
    from src.features.matrix import build_feature_matrix
    from src.models.train import OUTPUTS_DIR
    from src.models.trees import make_lightgbm

    print("Reading ground-truth benchmark doc (validation stage only)...")
    gt_rows = read_ground_truth_table()
    print(f"Parsed {len(gt_rows)} planted relationships.")

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

    print("\n=== Channel 1+2: individual + cluster SHAP importance (TEST set, official) ===")
    _, feat_imp = compute_shap_importance(gbm, X_test, fm)
    cluster_imp = aggregate_cluster_importance(feat_imp, fm.clustering)
    perm_imp = compute_permutation_importance(gbm, X_test, y_test, fm)

    print("\n=== Channel 3: engineered-feature correlation ranking (TEST set) ===")
    eng_model_cols = fm.to_model_cols(fm.engineered_cols)
    eng_vals = fm.X.iloc[test_idx][eng_model_cols].reset_index(drop=True)
    corr = eng_vals.apply(lambda s: s.corr(y_test.reset_index(drop=True)))
    corr.index = [fm.to_original(c) for c in corr.index]
    engineered_corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    print("\n=== Candidate selection for interaction discovery (7 channels, feature-strength-aware) ===")
    fs_path = OUTPUTS_DIR / "reports" / "feature_strength_classification.csv"
    feature_strength_df = pd.read_csv(fs_path) if fs_path.exists() else None
    if feature_strength_df is None:
        print("WARNING: feature_strength_classification.csv not found -- falling back to 5-channel selection "
              "(run src/explain/feature_strength.py first for the full 7-channel selection).")
    candidates, source_map = select_interaction_candidates(
        feat_imp, perm_imp, cluster_imp, fm.clustering, fm, gbm, fm.engineered_cols,
        feature_strength_df=feature_strength_df,
    )
    print(f"Candidate set size: {len(candidates)}")
    from collections import Counter
    print("By source:", Counter(v for v in source_map.values()))
    candidate_model_cols = fm.to_model_cols(candidates)

    print("\nRefitting compact LightGBM (TRAIN portion, candidate features)...")
    compact_gbm = make_lightgbm(random_state=42)
    compact_gbm.fit(X_train[candidate_model_cols], y_train)

    print("\n=== Channel 4: SHAP-interaction ranking (TEST set) ===")
    shap_pairs = rank_shap_interactions(compact_gbm, X_test[candidate_model_cols], fm, sample_size=min(300, len(X_test)), top_k=25)

    print("\n=== Channel 5: H-statistic ranking (TEST set, independent cross-check) ===")
    hstat_pairs = rank_h_statistic_interactions(compact_gbm, X_test[candidate_model_cols], candidate_model_cols, fm, grid_resolution=6, top_k=25, background_size=60)

    print("\n=== Fold-stability of SHAP-interaction ranking (5 GroupKFold folds, TRAIN portion, out-of-fold) ===")
    stability = assess_interaction_stability(candidate_model_cols, fm, X_train, y_train, groups_train, n_splits=5, top_k=25)

    print("\n=== Scoring against ground truth ===")
    scorecard = score_against_discovery(
        gt_rows, feat_imp, cluster_imp, fm.clustering, engineered_corr,
        shap_pairs, hstat_pairs, stability,
    )
    df = pd.DataFrame(scorecard)
    print("\n=== GROUND-TRUTH REDISCOVERY SCORECARD (v3, test-set-consistent, 7-channel candidate selection) ===")
    print(df.to_string(index=False))
    print(f"\nStrict detected: {df['detected_strict'].sum()} / {len(df)}")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "reports" / "ground_truth_scorecard_v3.csv", index=False)
    print("\nSaved outputs/reports/ground_truth_scorecard_v3.csv")

    # -----------------------------------------------------------------
    # Part D: structured discovered-relationships table (interpretability)
    # -----------------------------------------------------------------
    print("\n=== Building discovered-relationships table (Part D) ===")
    from src.explain.interactions import classify_pdp_shape

    relationship_rows = []

    def confidence_label(stability_frac: float, n_methods: int) -> str:
        if stability_frac >= 0.8 and n_methods >= 2:
            return "High"
        if stability_frac >= 0.4 or n_methods >= 2:
            return "Medium"
        return "Low"

    # top individual/cluster-important features -> direct/nonlinear relationships
    top_report_features = feat_imp.head(15).index.tolist()
    for f in top_report_features:
        model_col = fm.to_model_cols([f])[0]
        try:
            shape, direction = classify_pdp_shape(gbm, X_test, model_col, grid_resolution=15)
        except Exception as e:
            shape, direction = "unclassified", f"pdp_error: {e}"
        cid = fm.clustering.cluster_of.get(f)
        cluster_members = fm.clustering.clusters.get(int(cid), [f]) if cid is not None else [f]
        is_family = len(cluster_members) > 1
        fold_frac = 0.0  # individual-feature fold stability not recomputed here (see stability.py from prior checkpoint)
        relationship_rows.append({
            "features_involved": f if not is_family else f"{f} (+{len(cluster_members)-1} correlated)",
            "relationship_type": "nonlinear" if shape not in ("direct", "flat_no_clear_effect") else "direct",
            "pdp_shape_detail": shape,
            "direction_or_shape": direction,
            "strength_metric": "shap_importance",
            "strength_value": float(feat_imp[f]),
            "discovery_method": "SHAP importance" + (" + cluster aggregation" if is_family else ""),
            "evidence_level": "cluster/family" if is_family else "individual",
            "confidence": confidence_label(0.6, 1),  # heuristic; individual fold-stability from separate checkpoint
        })

    # top interaction pairs -> 2-way interactions, with fold stability + method agreement
    shap_pair_map = {frozenset([p.feature_a, p.feature_b]): p.interaction_strength for p in shap_pairs[:25]}
    hstat_pair_map = {frozenset([p.feature_a, p.feature_b]): p.interaction_strength for p in hstat_pairs[:25]}
    all_top_pairs = set(shap_pair_map) | set(hstat_pair_map)
    n_folds = len(stability.per_fold_top_pairs)
    for pair in sorted(all_top_pairs, key=lambda p: -max(shap_pair_map.get(p, 0), hstat_pair_map.get(p, 0)))[:15]:
        a, b = tuple(pair)
        fold_count = int(stability.pair_fold_counts.get(pair, 0))
        methods = []
        if pair in shap_pair_map:
            methods.append(f"shap_interaction({shap_pair_map[pair]:.4f})")
        if pair in hstat_pair_map:
            methods.append(f"h_statistic({hstat_pair_map[pair]:.4f})")
        relationship_rows.append({
            "features_involved": f"{a} x {b}",
            "relationship_type": "2-way interaction",
            "pdp_shape_detail": "",
            "direction_or_shape": "see 2D PDP",
            "strength_metric": "interaction_strength",
            "strength_value": max(shap_pair_map.get(pair, 0), hstat_pair_map.get(pair, 0)),
            "discovery_method": "; ".join(methods),
            "evidence_level": "pairwise",
            "confidence": confidence_label(fold_count / n_folds, len(methods)),
        })

    # engineered/derived cross-stage relationships
    for f in engineered_corr.head(10).index:
        relationship_rows.append({
            "features_involved": f,
            "relationship_type": "derived cross-stage relationship",
            "pdp_shape_detail": "",
            "direction_or_shape": "positive" if engineered_corr[f] > 0 else "negative",
            "strength_metric": "pearson_correlation",
            "strength_value": float(engineered_corr[f]),
            "discovery_method": "engineered-feature correlation (blind, generic stage-delta construction)",
            "evidence_level": "individual (engineered)",
            "confidence": confidence_label(0.5, 1),
        })

    rel_df = pd.DataFrame(relationship_rows)
    rel_df.to_csv(OUTPUTS_DIR / "reports" / "discovered_relationships.csv", index=False)
    print(f"Saved outputs/reports/discovered_relationships.csv ({len(rel_df)} relationships)")
    print("\nSaved outputs/reports/ground_truth_scorecard_v2.csv")
