"""Stage 2 Run B: synthesis stage.

Consumes the discovery-pipeline state (src/explain/stage2_runB_discovery.py)
and produces all required labeled deliverables:
  - stage2_runB_single_parameter_relationships.csv
  - stage2_runB_thresholds_process_windows.csv
  - stage2_runB_interactions.csv
  - stage2_runB_candidate_golden_routes.csv
  - stage2_runB_candidate_worsen_routes.csv
  - stage2_runB_relationship_summary.txt

Adds fold-stability assessment for single-parameter relationships (refits
5 GroupKFold full-feature models and re-classifies PDP shape/direction per
fold for the "most promising" screening set) -- interaction fold-stability
was already computed in the discovery stage and is reused here.

Every finding gets an explicit STRONG/MODERATE/WEAK/INCONCLUSIVE evidence
label with a stated reason. Never reads the ground-truth document.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold, make_holdout_split
from src.explain.interactions import classify_pdp_shape, one_d_partial_dependence
from src.features.stage2_matrix import load_stage2_dataset
from src.models.trees import make_lightgbm

OUTPUTS_DIR = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports"
RANDOM_STATE = 42


def label_single_param(shape: str, direction: str, shape_consistency: int, direction_consistency: int, n_folds: int) -> tuple[str, str]:
    if shape == "flat_no_clear_effect":
        return "INCONCLUSIVE", "Main model shows no clear PDP effect (range below flat threshold)."
    if shape == "unclassified":
        return "INCONCLUSIVE", "PDP computation failed or was not classifiable."

    if shape_consistency >= 4 and direction_consistency >= 4:
        return "STRONG", f"Shape and direction consistent in {shape_consistency}/{n_folds} and {direction_consistency}/{n_folds} folds."
    if shape_consistency >= 3 or direction_consistency >= 3:
        return "MODERATE", f"Shape consistent in {shape_consistency}/{n_folds} folds, direction in {direction_consistency}/{n_folds} -- majority but not unanimous."
    if shape_consistency <= 1 and direction_consistency <= 1:
        return "INCONCLUSIVE", f"Shape/direction agree in only {shape_consistency}/{n_folds} and {direction_consistency}/{n_folds} folds -- likely fold-specific artifact."
    return "WEAK", f"Shape consistent in {shape_consistency}/{n_folds} folds, direction in {direction_consistency}/{n_folds} -- inconsistent across folds."


def label_interaction(fold_count: int, n_folds: int, agreement_methods: int) -> tuple[str, str]:
    if fold_count >= 4:
        base = "STRONG"
    elif fold_count >= 2:
        base = "MODERATE"
    elif fold_count == 1:
        base = "WEAK"
    else:
        base = "INCONCLUSIVE"

    reason = f"Appeared in top-40 out-of-fold SHAP-interaction ranking in {fold_count}/{n_folds} folds"
    if agreement_methods >= 2:
        reason += "; corroborated by both SHAP-interaction and H-statistic on the held-out set."
        if base == "MODERATE":
            base = "STRONG" if fold_count >= 3 else base
    else:
        reason += "; found by only one of the two interaction-detection methods on the held-out set."
    return base, reason


def main():
    with open(f"{OUTPUTS_DIR}/_stage2_runB_discovery_state.pkl", "rb") as f:
        state = pickle.load(f)
    shape_df: pd.DataFrame = state["shape_df"]
    candidate_pool: list = state["candidate_pool"]
    shap_pairs = state["shap_pairs"]
    h_pairs = state["h_pairs"]
    pair_fold_counts: pd.Series = state["pair_fold_counts"]
    n_folds = state["n_folds"]

    combined = pd.read_csv(f"{OUTPUTS_DIR}/stage2_runB_global_importance.csv")

    ds = load_stage2_dataset(include_review_columns=False)
    holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=RANDOM_STATE)
    X_train = ds.X.iloc[holdout.train_idx].reset_index(drop=True)
    y_train = ds.y.iloc[holdout.train_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)
    y_full = ds.y  # for route statistics, computed on TRAIN rows only (see below)

    print("Refitting the locked Run B final model (needed for real PDP range values)...")
    final_model_for_pdp = make_lightgbm(random_state=RANDOM_STATE)
    final_model_for_pdp.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Single-parameter fold-stability: refit 5 full-feature fold models,
    # re-classify PDP shape/direction for the screening set.
    # ------------------------------------------------------------------
    screening_features = shape_df.index.tolist()
    print(f"Re-assessing fold stability for {len(screening_features)} screened single-parameter relationships...")
    _, folds = make_group_kfold(groups_train, n_splits=5)

    per_fold_shapes = {f: [] for f in screening_features}
    per_fold_directions = {f: [] for f in screening_features}
    for fold_i, (tr, va) in enumerate(folds):
        m = make_lightgbm(random_state=RANDOM_STATE)
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        X_bg = X_train.iloc[va].reset_index(drop=True).sample(min(150, len(va)), random_state=RANDOM_STATE)
        for f in screening_features:
            mcol = ds.name_map[f]
            shp, drc = classify_pdp_shape(m, X_bg, mcol, grid_resolution=10)
            per_fold_shapes[f].append(shp)
            per_fold_directions[f].append(drc)
        print(f"  fold {fold_i} PDP-shape reclassification done", flush=True)

    rows = []
    for f in screening_features:
        main_shape = shape_df.loc[f, "shape"]
        main_dir = shape_df.loc[f, "direction"]
        shape_consistency = sum(1 for s in per_fold_shapes[f] if s == main_shape)
        direction_consistency = sum(1 for d in per_fold_directions[f] if d == main_dir)
        label, reason = label_single_param(main_shape, main_dir, shape_consistency, direction_consistency, n_folds=5)

        shap_row = combined[combined["raw_csv_parameter_name"] == f]
        shap_rank = int(shap_row["shap_rank"].values[0]) if len(shap_row) else None

        rows.append({
            "raw_csv_parameter_name": f,
            "relationship_type": main_shape,
            "direction": main_dir,
            "shap_rank": shap_rank,
            "shape_fold_consistency": f"{shape_consistency}/5",
            "direction_fold_consistency": f"{direction_consistency}/5",
            "evidence_label": label,
            "reason": reason,
        })

    single_param_df = pd.DataFrame(rows).sort_values("shap_rank", na_position="last")
    single_param_df.to_csv(f"{OUTPUTS_DIR}/stage2_runB_single_parameter_relationships.csv", index=False)
    print(f"\nSaved stage2_runB_single_parameter_relationships.csv ({len(single_param_df)} rows)")
    print(single_param_df["evidence_label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Thresholds / process windows (subset with shape in threshold/U-shaped)
    # ------------------------------------------------------------------
    threshold_rows = []
    X_bg_pdp = X_train.sample(min(300, len(X_train)), random_state=RANDOM_STATE)
    for _, row in single_param_df[single_param_df["relationship_type"].isin(
        ["threshold", "U_shaped_process_window"]
    )].iterrows():
        f = row["raw_csv_parameter_name"]
        mcol = ds.name_map[f]
        grid, values = one_d_partial_dependence(final_model_for_pdp, X_bg_pdp, mcol, grid_resolution=15)
        values = np.asarray(values)
        if row["relationship_type"] == "threshold":
            # locate the grid point where the single largest step in
            # predicted yield occurs -- the practical "elbow" of the curve
            diffs = np.abs(np.diff(values))
            elbow_idx = int(np.argmax(diffs))
            approx_threshold = round(float((grid[elbow_idx] + grid[elbow_idx + 1]) / 2), 4)
            window_low, window_high = None, None
        else:  # U_shaped_process_window
            best_idx = int(np.argmax(values)) if row["direction"] != "negative" else int(np.argmin(values))
            approx_threshold = None
            # "preferred window" = grid points within 10% of the best PDP value's range from the optimum
            span = values.max() - values.min()
            near_best = np.abs(values - values[best_idx]) <= 0.1 * span if span > 0 else np.ones_like(values, dtype=bool)
            window_low = round(float(grid[near_best].min()), 4)
            window_high = round(float(grid[near_best].max()), 4)
        threshold_rows.append({
            "raw_csv_parameter_name": f,
            "relationship_type": row["relationship_type"],
            "evidence_label": row["evidence_label"],
            "observed_range_low": round(float(grid.min()), 4),
            "observed_range_high": round(float(grid.max()), 4),
            "approx_threshold": approx_threshold,
            "preferred_window_low": window_low,
            "preferred_window_high": window_high,
            "shap_rank": row["shap_rank"],
        })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(f"{OUTPUTS_DIR}/stage2_runB_thresholds_process_windows.csv", index=False)
    print(f"\nSaved stage2_runB_thresholds_process_windows.csv ({len(threshold_df)} rows)")

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------
    shap_pair_map = {frozenset([p.feature_a, p.feature_b]): p.interaction_strength for p in shap_pairs}
    h_pair_map = {frozenset([p.feature_a, p.feature_b]): p.interaction_strength for p in h_pairs}
    all_pairs = set(shap_pair_map) | set(h_pair_map)

    interaction_rows = []
    for pair in all_pairs:
        a, b = tuple(pair)
        n_methods = int(pair in shap_pair_map) + int(pair in h_pair_map)
        fold_count = int(pair_fold_counts.get(pair, 0))
        label, reason = label_interaction(fold_count, n_folds, n_methods)
        interaction_rows.append({
            "parameter_A": a, "parameter_B": b,
            "shap_interaction_strength": shap_pair_map.get(pair, np.nan),
            "h_statistic_strength": h_pair_map.get(pair, np.nan),
            "n_methods_agreeing": n_methods,
            "fold_reproducibility": f"{fold_count}/{n_folds}",
            "evidence_label": label,
            "reason": reason,
        })
    interaction_df = pd.DataFrame(interaction_rows).sort_values(
        by=["evidence_label", "shap_interaction_strength"], ascending=[True, False],
        key=lambda s: s if s.name != "evidence_label" else s.map({"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3})
    )
    interaction_df.to_csv(f"{OUTPUTS_DIR}/stage2_runB_interactions.csv", index=False)
    print(f"\nSaved stage2_runB_interactions.csv ({len(interaction_df)} rows)")
    print(interaction_df["evidence_label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Candidate Golden / Worsen Routes -- built from STRONG single-parameter
    # findings and STRONG interactions, evaluated on the TRAIN portion only
    # ------------------------------------------------------------------
    strong_single = single_param_df[single_param_df["evidence_label"] == "STRONG"]
    strong_interactions = interaction_df[interaction_df["evidence_label"] == "STRONG"]

    y_train_full = y_train
    baseline_mean = y_train_full.mean()

    golden_rows, worsen_rows = [], []

    def route_stats(mask: pd.Series, description: str, rows_out: list, kind: str):
        n = int(mask.sum())
        if n < 20:  # too small a subgroup to report a stable estimate
            return
        sub_mean = y_train_full[mask].mean()
        effect = sub_mean - baseline_mean
        if kind == "golden" and effect <= 0:
            return
        if kind == "worsen" and effect >= 0:
            return
        rows_out.append({
            "condition": description,
            "n_wafers_matching": n,
            "pct_of_train_set": round(100 * n / len(y_train_full), 1),
            "mean_yield_in_condition": round(float(sub_mean), 3),
            "baseline_mean_yield": round(float(baseline_mean), 3),
            "estimated_yield_effect": round(float(effect), 3),
            "based_on": "STRONG single-parameter/interaction findings, TRAIN portion only (not holdout-validated)",
        })

    # single-parameter-based routes: top/bottom quartile of each STRONG monotonic feature
    for _, row in strong_single.iterrows():
        f = row["raw_csv_parameter_name"]
        if row["relationship_type"] != "direct":
            continue
        mcol = ds.name_map[f]
        vals = X_train[mcol]
        q75, q25 = vals.quantile(0.75), vals.quantile(0.25)
        if row["direction"] == "positive":
            route_stats(vals >= q75, f"{f} >= {q75:.4g} (top quartile)", golden_rows, "golden")
            route_stats(vals <= q25, f"{f} <= {q25:.4g} (bottom quartile)", worsen_rows, "worsen")
        elif row["direction"] == "negative":
            route_stats(vals <= q25, f"{f} <= {q25:.4g} (bottom quartile)", golden_rows, "golden")
            route_stats(vals >= q75, f"{f} >= {q75:.4g} (top quartile)", worsen_rows, "worsen")

    # interaction-based 2-condition routes for STRONG interactions
    for _, row in strong_interactions.head(10).iterrows():
        a, b = row["parameter_A"], row["parameter_B"]
        if a not in ds.name_map or b not in ds.name_map:
            continue
        va, vb = X_train[ds.name_map[a]], X_train[ds.name_map[b]]
        a_hi, a_lo = va.quantile(0.75), va.quantile(0.25)
        b_hi, b_lo = vb.quantile(0.75), vb.quantile(0.25)
        route_stats((va >= a_hi) & (vb >= b_hi), f"{a} >= {a_hi:.4g} (top quartile) AND {b} >= {b_hi:.4g} (top quartile)", golden_rows, "golden")
        route_stats((va >= a_hi) & (vb >= b_hi), f"{a} >= {a_hi:.4g} (top quartile) AND {b} >= {b_hi:.4g} (top quartile)", worsen_rows, "worsen")
        route_stats((va <= a_lo) & (vb <= b_lo), f"{a} <= {a_lo:.4g} (bottom quartile) AND {b} <= {b_lo:.4g} (bottom quartile)", golden_rows, "golden")
        route_stats((va <= a_lo) & (vb <= b_lo), f"{a} <= {a_lo:.4g} (bottom quartile) AND {b} <= {b_lo:.4g} (bottom quartile)", worsen_rows, "worsen")

    golden_df = pd.DataFrame(golden_rows).sort_values("estimated_yield_effect", ascending=False) if golden_rows else pd.DataFrame(
        columns=["condition", "n_wafers_matching", "pct_of_train_set", "mean_yield_in_condition", "baseline_mean_yield", "estimated_yield_effect", "based_on"])
    worsen_df = pd.DataFrame(worsen_rows).sort_values("estimated_yield_effect", ascending=True) if worsen_rows else pd.DataFrame(
        columns=["condition", "n_wafers_matching", "pct_of_train_set", "mean_yield_in_condition", "baseline_mean_yield", "estimated_yield_effect", "based_on"])
    golden_df.to_csv(f"{OUTPUTS_DIR}/stage2_runB_candidate_golden_routes.csv", index=False)
    worsen_df.to_csv(f"{OUTPUTS_DIR}/stage2_runB_candidate_worsen_routes.csv", index=False)
    print(f"\nSaved stage2_runB_candidate_golden_routes.csv ({len(golden_df)} rows)")
    print(f"Saved stage2_runB_candidate_worsen_routes.csv ({len(worsen_df)} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    write_summary(ds, single_param_df, interaction_df, golden_df, worsen_df, combined)


def write_summary(ds, single_param_df, interaction_df, golden_df, worsen_df, combined):
    lines = []
    lines.append("STAGE 2 -- RUN B RELATIONSHIP DISCOVERY SUMMARY (LightGBM only, blind to ground truth)")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"1. Total Run B predictor count: {ds.X.shape[1]} (1,405 numeric process/ET + 20 categorical dummies)")
    lines.append("2. Zero engineered predictors used (hard-asserted; verified in src/models/stage2_runB_lock.py)")
    lines.append("")

    sp_counts = single_param_df["evidence_label"].value_counts().to_dict()
    lines.append("3. Single-parameter relationships by evidence label (screening set):")
    for lbl in ["STRONG", "MODERATE", "WEAK", "INCONCLUSIVE"]:
        lines.append(f"   {lbl}: {sp_counts.get(lbl, 0)}")
    lines.append("")

    int_counts = interaction_df["evidence_label"].value_counts().to_dict()
    lines.append("4. 2-way interactions by evidence label (candidate pool):")
    for lbl in ["STRONG", "MODERATE", "WEAK", "INCONCLUSIVE"]:
        lines.append(f"   {lbl}: {int_counts.get(lbl, 0)}")
    lines.append("")

    positive = single_param_df[(single_param_df["direction"] == "positive") & (single_param_df["evidence_label"].isin(["STRONG", "MODERATE"]))]
    negative = single_param_df[(single_param_df["direction"] == "negative") & (single_param_df["evidence_label"].isin(["STRONG", "MODERATE"]))]
    lines.append("5. Strongest positive Yield drivers (direct positive relationship, STRONG/MODERATE):")
    for _, r in positive.sort_values("shap_rank").head(10).iterrows():
        lines.append(f"   - {r['raw_csv_parameter_name']} (rank {r['shap_rank']}, {r['evidence_label']})")
    lines.append("")
    lines.append("6. Strongest negative Yield drivers (direct negative relationship, STRONG/MODERATE):")
    for _, r in negative.sort_values("shap_rank").head(10).iterrows():
        lines.append(f"   - {r['raw_csv_parameter_name']} (rank {r['shap_rank']}, {r['evidence_label']})")
    lines.append("")

    thresholds = single_param_df[single_param_df["relationship_type"].isin(["threshold", "U_shaped_process_window"])]
    lines.append("7. Strongest thresholds/process windows (STRONG/MODERATE):")
    for _, r in thresholds[thresholds["evidence_label"].isin(["STRONG", "MODERATE"])].sort_values("shap_rank").head(10).iterrows():
        lines.append(f"   - {r['raw_csv_parameter_name']}: {r['relationship_type']} ({r['evidence_label']})")
    lines.append("")

    lines.append("8. Strongest interaction patterns (STRONG):")
    for _, r in interaction_df[interaction_df["evidence_label"] == "STRONG"].head(10).iterrows():
        lines.append(f"   - {r['parameter_A']} x {r['parameter_B']} (fold reproducibility {r['fold_reproducibility']}, "
                      f"{r['n_methods_agreeing']}/2 methods agree)")
    lines.append("")

    lines.append("9. Candidate Golden Routes (NOT validated production rules):")
    for _, r in golden_df.head(8).iterrows():
        lines.append(f"   - IF {r['condition']} THEN mean yield ~{r['mean_yield_in_condition']:.2f}% "
                      f"(baseline {r['baseline_mean_yield']:.2f}%, effect {r['estimated_yield_effect']:+.2f}, "
                      f"n={r['n_wafers_matching']})")
    lines.append("")
    lines.append("10. Candidate Worsen Routes (NOT validated production rules):")
    for _, r in worsen_df.head(8).iterrows():
        lines.append(f"   - IF {r['condition']} THEN mean yield ~{r['mean_yield_in_condition']:.2f}% "
                      f"(baseline {r['baseline_mean_yield']:.2f}%, effect {r['estimated_yield_effect']:+.2f}, "
                      f"n={r['n_wafers_matching']})")
    lines.append("")

    lines.append("11. Important limitations:")
    lines.append("   - Single model family (LightGBM) only -- no cross-model-class corroboration this checkpoint.")
    lines.append("   - Golden/Worsen routes are computed on the TRAIN portion only and are NOT holdout-validated or")
    lines.append("     significance-tested -- they are shallow, interpretable CANDIDATES for engineering review, not")
    lines.append("     validated production rules.")
    lines.append("   - Interaction candidate pool (~50-60 features) cannot cover all C(1425,2) possible pairs --")
    lines.append("     absence of a pair from this report is not proof of no interaction.")
    lines.append("   - PDP-shape classification uses a simple sign-change heuristic (see src/explain/interactions.py),")
    lines.append("     not a rigorous curve-fitting method -- shape labels are a practical first pass.")
    lines.append("   - No multiple-testing correction applied across the many single-parameter and interaction checks.")
    lines.append("   - Ground-truth document was not read at any point in this discovery pipeline (by design).")
    lines.append("")
    lines.append("12. Relationship types LightGBM appears unable to detect reliably here:")
    inconclusive_shapes = single_param_df[single_param_df["evidence_label"] == "INCONCLUSIVE"]["relationship_type"].value_counts()
    lines.append(f"    Of {len(single_param_df[single_param_df['evidence_label']=='INCONCLUSIVE'])} INCONCLUSIVE single-parameter findings, "
                  f"shape breakdown: {inconclusive_shapes.to_dict()}")
    lines.append("    Interactions with 0/5 fold reproducibility but present in the single official run are the most likely")
    lines.append("    candidates for 'found once, not reliably reproducible' -- see WEAK/INCONCLUSIVE rows in the interactions CSV.")

    text = "\n".join(lines)
    with open(f"{OUTPUTS_DIR}/stage2_runB_relationship_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
