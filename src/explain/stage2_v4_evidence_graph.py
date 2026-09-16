"""Stage 2 Run B v4 -- feature-evidence graph.

Generic, per-feature aggregation of evidence from every discovery layer
already frozen in this project (spline/GAM, v2 model-free univariate,
LightGBM SHAP/gain/permutation importance, v3 broad non-additivity
screening, v3 confirmed interactions, v3 tree-path co-occurrence, v2
cross-stage analysis). Nothing here reads the ground-truth document or
references a specific parameter name -- the evidence sources and vote
thresholds below are generic rules applicable to any dataset with this
same discovery-pipeline shape.

"Conditional-dispersion" evidence (used internally during v1 discovery's
candidate selection) is not available as an independent frozen CSV -- only
as an intermediate value inside a pickled internal state file -- so it is
not re-derived here; its downstream effect is already transitively present
via v1's STRONG/MODERATE interaction labels, which feed the "confirmed
interaction neighborhood" vote below (through v3's channel B/C, which
reused v1's frozen interaction table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split

TOP_PCTILE = 90.0  # "strong marginal importance" = top decile, a generic, non-tuned cut


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original
    n_total = len(numeric_features)

    spline = pd.read_csv(f"{R}/stage2_runB_spline_single_parameter.csv").set_index("raw_csv_parameter_name")
    v2u = pd.read_csv(f"{R}/stage2_runB_v2_univariate_all.csv").set_index("raw_csv_parameter_name")
    gi = pd.read_csv(f"{R}/stage2_runB_global_importance.csv").set_index("raw_csv_parameter_name")
    ps = pd.read_csv(f"{R}/stage2_runB_v3_pair_screen.csv")
    cand3 = pd.read_csv(f"{R}/stage2_runB_v3_pair_candidates.csv")
    conf3 = pd.read_csv(f"{R}/stage2_runB_v3_confirmed_2way_interactions.csv")
    cs = pd.read_csv(f"{R}/stage2_runB_v2_cross_stage_relationships.csv")

    # per-feature aggregates from pair-level frozen files
    def best_rank(feat):
        rows = ps[(ps["parameter_A"] == feat) | (ps["parameter_B"] == feat)]
        if len(rows) == 0:
            return np.nan
        return int(min(rows["rank_score_1"].min(), rows["rank_score_2"].min()))

    def tree_cooccurrence_flag(feat):
        rows = cand3[(cand3["parameter_A"] == feat) | (cand3["parameter_B"] == feat)]
        return bool(rows["channels"].str.contains("D_tree_path_cooccurrence").any())

    def cross_stage_label(feat):
        rows = cs[(cs["parameter_A"] == feat) | (cs["parameter_B"] == feat)]
        if len(rows) == 0:
            return "no_evidence"
        order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
        rows = rows.assign(_o=rows["evidence_label"].map(order)).sort_values("_o")
        return rows.iloc[0]["evidence_label"]

    rows_out = []
    for f in numeric_features:
        spline_label = spline.loc[f, "evidence_label"] if f in spline.index else "not_tested"
        spline_nonlin = bool(spline.loc[f, "nonlinear_advantage"]) if f in spline.index else False
        spline_class = spline.loc[f, "relationship_class"] if f in spline.index else None

        v2_label = v2u.loc[f, "evidence_label"] if f in v2u.index else "not_tested"
        v2_shape = v2u.loc[f, "shape"] if f in v2u.index else None

        shap_pct = float(gi.loc[f, "shap_percentile"]) if f in gi.index else np.nan
        gain_pct = float(gi.loc[f, "gain_percentile"]) if f in gi.index else np.nan
        perm_pct = float(gi.loc[f, "permutation_percentile"]) if f in gi.index else np.nan
        method_agree = gi.loc[f, "method_agreement"] if f in gi.index else None

        rank = best_rank(f)
        in_broadscreen_top300 = bool(not np.isnan(rank))

        conf_rows = conf3[(conf3["parameter_A"] == f) | (conf3["parameter_B"] == f)]
        n_confirmed_strong = int((conf_rows["evidence_label"] == "STRONG").sum())
        n_confirmed_moderate = int((conf_rows["evidence_label"] == "MODERATE").sum())
        best_confirmed_r2 = float(conf_rows["interaction_incremental_r2"].max()) if len(conf_rows) else np.nan

        tree_flag = tree_cooccurrence_flag(f)
        cross_stage_ev = cross_stage_label(f)

        # --- votes (each a simple boolean, generic, no ground-truth involvement) ---
        vote_spline = spline_label in ("STRONG", "MODERATE")
        vote_v2univariate = v2_label in ("STRONG", "MODERATE")
        vote_shap_top = shap_pct >= TOP_PCTILE if not np.isnan(shap_pct) else False
        vote_perm_top = perm_pct >= TOP_PCTILE if not np.isnan(perm_pct) else False
        vote_confirmed_interaction = (n_confirmed_strong + n_confirmed_moderate) > 0
        vote_broadscreen = in_broadscreen_top300
        vote_tree_cooc = tree_flag
        vote_cross_stage = cross_stage_ev in ("STRONG", "MODERATE")

        votes = [vote_spline, vote_v2univariate, vote_shap_top, vote_perm_top,
                 vote_confirmed_interaction, vote_broadscreen, vote_tree_cooc, vote_cross_stage]
        n_methods = int(sum(votes))

        is_marginal = vote_spline or vote_v2univariate or vote_shap_top or vote_perm_top
        is_nonlinear = spline_nonlin or (v2_shape not in (None, "monotonic", "flat", "insufficient_data"))
        is_conditional = vote_confirmed_interaction
        is_interaction_related = vote_broadscreen or vote_confirmed_interaction or vote_tree_cooc
        is_cross_stage = vote_cross_stage

        if n_confirmed_strong > 0 or n_methods >= 3:
            tier = "HIGH_EVIDENCE"
        elif n_methods == 2 or (n_confirmed_moderate > 0) or (vote_broadscreen and is_marginal):
            tier = "MEDIUM_EVIDENCE"
        elif n_methods == 1:
            tier = "LOW_EVIDENCE"
        else:
            tier = "UNSUPPORTED"

        rows_out.append({
            "raw_csv_parameter_name": f,
            "spline_evidence_label": spline_label, "spline_relationship_class": spline_class,
            "spline_nonlinear_advantage": spline_nonlin,
            "v2_univariate_evidence_label": v2_label, "v2_univariate_shape": v2_shape,
            "shap_percentile": round(shap_pct, 2) if not np.isnan(shap_pct) else None,
            "gain_percentile": round(gain_pct, 2) if not np.isnan(gain_pct) else None,
            "permutation_percentile": round(perm_pct, 2) if not np.isnan(perm_pct) else None,
            "method_agreement": method_agree,
            "v3_broadscreen_best_rank": rank if in_broadscreen_top300 else None,
            "v3_confirmed_strong_count": n_confirmed_strong, "v3_confirmed_moderate_count": n_confirmed_moderate,
            "v3_confirmed_best_incremental_r2": round(best_confirmed_r2, 4) if not np.isnan(best_confirmed_r2) else None,
            "v3_tree_path_cooccurrence": tree_flag,
            "cross_stage_best_evidence_label": cross_stage_ev,
            "n_independent_methods_supporting": n_methods,
            "is_marginal_evidence": is_marginal, "is_nonlinear_evidence": is_nonlinear,
            "is_conditional_evidence": is_conditional, "is_interaction_related_evidence": is_interaction_related,
            "is_cross_stage_evidence": is_cross_stage,
            "evidence_tier": tier,
        })

    df = pd.DataFrame(rows_out)
    out_path = f"{R}/stage2_runB_v4_feature_evidence_graph.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} / {n_total} numeric predictors, 100% coverage)")
    print("\nEvidence tier counts:")
    print(df["evidence_tier"].value_counts().to_string())
    print("\nEvidence-character counts:")
    for col in ["is_marginal_evidence", "is_nonlinear_evidence", "is_conditional_evidence",
                "is_interaction_related_evidence", "is_cross_stage_evidence"]:
        print(f"  {col}: {int(df[col].sum())}")


if __name__ == "__main__":
    main()
