"""Stage 2 Run B v5 -- fixes bottleneck #1 identified by the v4 ground-truth
validation: MEDIUM_EVIDENCE x MEDIUM_EVIDENCE pairs had no generous
candidate-generation channel in v4 (channels A/B/E all require >=1
HIGH_EVIDENCE member). This script EXTENDS v4's pair pool (reused in full,
unchanged) with a new MEDIUM x MEDIUM admission path, generic and
evidence-driven -- never a blind full C(387,2) inclusion.

Six independent admission rules (a MEDIUM x MEDIUM pair is admitted if it
satisfies AT LEAST ONE):
  M1. non-additivity (score_2) outlier WITHIN the MEDIUM x MEDIUM subset
      (top 3,000 of its own ~74.7k-pair population -- not competed
      globally against HIGH-tier pairs, which is what excluded these pairs
      from v4's channel C)
  M2. dual-broad-score corroboration: pair ranks in the top 5,000 of BOTH
      score_1 and score_2 WITHIN the subset
  M3. fresh tree-path co-occurrence, mined from a compact model fit on
      just the MEDIUM_EVIDENCE feature set
  M4. shared interaction neighborhood: A and B each have an independently
      CONFIRMED (v4, STRONG/MODERATE) interaction with some common third
      feature X (a generic "two hops via confirmed evidence" rule)
  M5. shared process-family structure: A and B belong to the same
      stage-comparable suffix group (src/features/engineered.py::
      find_stage_groups -- the SAME generic column-naming matcher used,
      unmodified, by v2's cross-stage discovery; different modules
      measuring the same underlying quantity)
  M6. both nonlinear-evidenced: both features show spline/GAM nonlinear
      advantage (v4 evidence graph), ranked by score_2 outlier status
      (top 2,000) to keep this rule bounded

Every admitted pair records exactly which rule(s) admitted it.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v2_pair_screen import compute_pairwise_screen_scores
from src.explain.stage2_v3_common import compute_nonadditivity_screen
from src.explain.stage2_v3_candidates import compute_tree_pair_cooccurrence
from src.features.engineered import find_stage_groups
from src.models.trees import make_lightgbm

M1_TOP_K = 3000
M2_TOP_K = 5000
M3_TOP_K = 2000
M6_TOP_K = 2000


def main():
    split = load_locked_split()
    ds = split.ds
    eg = pd.read_csv(f"{R}/stage2_runB_v4_feature_evidence_graph.csv").set_index("raw_csv_parameter_name")
    numeric_features = ds.numeric_cols_original
    n = len(numeric_features)

    medium = sorted(eg[eg["evidence_tier"] == "MEDIUM_EVIDENCE"].index)
    print(f"MEDIUM_EVIDENCE features: {len(medium)}")
    n_possible_mm = len(medium) * (len(medium) - 1) // 2
    print(f"Theoretically possible MEDIUM x MEDIUM pairs: {n_possible_mm}")

    med_idx_global = {f: numeric_features.index(f) for f in medium}
    med_pos = {f: i for i, f in enumerate(medium)}

    # ---- full-universe broad scores (reused formulas, recomputed once) ----
    model_cols_all = [ds.name_map[f] for f in numeric_features]
    X_all = split.X_train[model_cols_all].to_numpy(dtype=np.float64)
    y_arr = split.y_train.to_numpy()
    X_num_df = split.X_train[model_cols_all].copy()
    X_num_df.columns = numeric_features
    print("Recomputing full-universe broad scores (score_1, score_2)...")
    score1_full = compute_pairwise_screen_scores(X_num_df, y_arr, numeric_features)
    score2_full = compute_nonadditivity_screen(X_all, y_arr, n_bins=4)

    med_global_idx = np.array([numeric_features.index(f) for f in medium])
    s1_mm = score1_full[np.ix_(med_global_idx, med_global_idx)]
    s2_mm = score2_full[np.ix_(med_global_idx, med_global_idx)]
    iu = np.triu_indices(len(medium), k=1)
    s1_flat, s2_flat = s1_mm[iu], s2_mm[iu]

    def pair_at(idx):
        return medium[iu[0][idx]], medium[iu[1][idx]]

    admitted: dict[frozenset, set] = {}

    def admit(pairs, label):
        for a, b in pairs:
            if a == b:
                continue
            admitted.setdefault(frozenset([a, b]), set()).add(label)

    # --- M1: non-additivity outlier within the MEDIUM x MEDIUM subset ---
    m1_idx = np.argsort(-s2_flat)[:M1_TOP_K]
    admit([pair_at(i) for i in m1_idx], "M1_nonadditivity_outlier_within_medium")
    print(f"M1 (non-additivity outlier, top {M1_TOP_K} of {n_possible_mm}): {len(m1_idx)} pairs")

    # --- M2: dual-score corroboration within the subset ---
    top1_idx = set(np.argsort(-s1_flat)[:M2_TOP_K])
    top2_idx = set(np.argsort(-s2_flat)[:M2_TOP_K])
    m2_idx = sorted(top1_idx & top2_idx)
    admit([pair_at(i) for i in m2_idx], "M2_dual_broadscore_corroboration")
    print(f"M2 (top {M2_TOP_K} of BOTH scores, intersection): {len(m2_idx)} pairs")

    # --- M3: fresh tree-path co-occurrence among MEDIUM features only ---
    med_model_cols = [ds.name_map[f] for f in medium]
    print(f"Fitting compact model on {len(medium)} MEDIUM_EVIDENCE features for tree co-occurrence...")
    tree_model = make_lightgbm(random_state=42)
    tree_model.fit(split.X_train[med_model_cols], split.y_train)
    pair_counts_model = compute_tree_pair_cooccurrence(tree_model, med_model_cols)
    pair_counts = {frozenset([ds.inverse_name_map[a], ds.inverse_name_map[b]]): c for (a, b), c in pair_counts_model.items()}
    top_m3 = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:M3_TOP_K]
    admit([tuple(p) for p, _ in top_m3], "M3_tree_path_cooccurrence_medium_only")
    print(f"M3 (tree co-occurrence among MEDIUM features, top {M3_TOP_K} of {len(pair_counts)}): {len(top_m3)} pairs")

    # --- M4: shared interaction neighborhood via v4 confirmed edges ---
    conf4 = pd.read_csv(f"{R}/stage2_runB_v4_confirmed_2way_interactions.csv")
    strong_mod4 = conf4[conf4["evidence_label"].isin(["STRONG", "MODERATE"])]
    adjacency: dict[str, set] = {}
    for a, b in zip(strong_mod4["parameter_A"], strong_mod4["parameter_B"]):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    medium_set = set(medium)
    m4_pairs = []
    for i in range(len(medium)):
        a = medium[i]
        neigh_a = adjacency.get(a, set())
        if not neigh_a:
            continue
        for j in range(i + 1, len(medium)):
            b = medium[j]
            neigh_b = adjacency.get(b, set())
            if neigh_a & neigh_b:
                m4_pairs.append((a, b))
    admit(m4_pairs, "M4_shared_interaction_neighborhood")
    print(f"M4 (shared confirmed-interaction neighborhood): {len(m4_pairs)} pairs")

    # --- M5: shared process-family structure (find_stage_groups, reused unmodified) ---
    stage_groups = find_stage_groups(numeric_features)
    m5_pairs = []
    for suffix, module_map in stage_groups.groups.items():
        cols = [c for c in module_map.values() if c in medium_set]
        for a, b in combinations(sorted(cols), 2):
            m5_pairs.append((a, b))
    admit(m5_pairs, "M5_shared_process_family")
    print(f"M5 (shared process-family / stage-group structure): {len(m5_pairs)} pairs")

    # --- M6: both nonlinear-evidenced, ranked by non-additivity outlier ---
    nonlin_set = set(eg[eg["spline_nonlinear_advantage"] == True].index) & medium_set
    nonlin_pair_idx = [i for i in range(len(iu[0])) if medium[iu[0][i]] in nonlin_set and medium[iu[1][i]] in nonlin_set]
    nonlin_pair_idx_sorted = sorted(nonlin_pair_idx, key=lambda i: -s2_flat[i])[:M6_TOP_K]
    admit([pair_at(i) for i in nonlin_pair_idx_sorted], "M6_both_nonlinear_evidenced")
    print(f"M6 (both nonlinear-evidenced, top {M6_TOP_K} of {len(nonlin_pair_idx)} eligible by non-additivity): {len(nonlin_pair_idx_sorted)} pairs")

    n_admitted = len(admitted)
    print(f"\nTotal MEDIUM x MEDIUM pairs admitted to V5 (union of M1-M6): {n_admitted} "
          f"({100*n_admitted/n_possible_mm:.2f}% of {n_possible_mm} possible)")

    # --- union with v4's pool (reused in full, unchanged) ---
    v4_pool = pd.read_csv(f"{R}/stage2_runB_v4_pair_candidates.csv")
    rows = []
    for _, r in v4_pool.iterrows():
        rows.append({"parameter_A": r["parameter_A"], "parameter_B": r["parameter_B"],
                     "channels": r["channels"], "n_channels": r["n_channels"],
                     "source": "v4_reused"})
    for pair, labels in admitted.items():
        a, b = tuple(pair)
        rows.append({"parameter_A": a, "parameter_B": b, "channels": "+".join(sorted(labels)),
                     "n_channels": len(labels), "source": "v5_medium_x_medium"})

    df = pd.DataFrame(rows)
    # dedup: a pair might already be in v4 (e.g. via D reuse) AND newly admitted here -- merge channel labels
    def merge_group(g):
        all_channels = set()
        for ch in g["channels"]:
            all_channels.update(ch.split("+"))
        sources = set(g["source"])
        return pd.Series({
            "channels": "+".join(sorted(all_channels)),
            "n_channels": len(all_channels),
            "source": "+".join(sorted(sources)),
        })
    df["pair_key"] = df.apply(lambda r: frozenset([r["parameter_A"], r["parameter_B"]]), axis=1)
    dedup_rows = []
    for key, g in df.groupby("pair_key"):
        a, b = tuple(key)
        merged = merge_group(g)
        dedup_rows.append({"parameter_A": a, "parameter_B": b, **merged.to_dict()})
    df_final = pd.DataFrame(dedup_rows).sort_values("n_channels", ascending=False)

    eg_lookup = eg["evidence_tier"]
    df_final["evidence_tier_A"] = df_final["parameter_A"].map(eg_lookup)
    df_final["evidence_tier_B"] = df_final["parameter_B"].map(eg_lookup)

    out_path = f"{R}/stage2_runB_v5_pair_candidates.csv"
    df_final.to_csv(out_path, index=False)

    union_features = sorted(set(df_final["parameter_A"]) | set(df_final["parameter_B"]))
    v4_features = sorted(set(v4_pool["parameter_A"]) | set(v4_pool["parameter_B"]))
    print(f"\nSaved {out_path}")
    print(f"V4 pool: {len(v4_pool)} pairs, {len(v4_features)} features ({100*len(v4_features)/n:.1f}%)")
    print(f"V5 pool: {len(df_final)} pairs, {len(union_features)} features ({100*len(union_features)/n:.1f}%)")
    print(f"New pairs added beyond v4: {len(df_final) - len(v4_pool)}")
    mm_new = df_final[df_final["source"].str.contains("v5_medium_x_medium")]
    print(f"MEDIUM x MEDIUM pairs in final v5 pool: {len(mm_new)}")


if __name__ == "__main__":
    main()
