"""Stage 2 Run B v3 -- multi-channel candidate-pool construction for
expensive 2-way interaction confirmation. UNION of six generic channels
(never gated by a single top-N ranking alone, and never seeded with any
known planted parameter name):

  A. highest full-universe non-additivity scores (both v3 broad scores)
  B. strongest EXISTING LightGBM SHAP-interaction evidence (v1 + v2
     confirmed STRONG/MODERATE pairs, already-frozen, reused not recomputed)
  C. strongest EXISTING H-statistic/PDP interaction evidence (same source,
     ranked by h_statistic_strength instead of shap_interaction_strength)
  D. LightGBM tree-path co-occurrence, mined fresh from a compact model
     fit on a broad feature union (global-importance top features + all
     channel A/B/C features)
  E. pairs with strong non-additivity (score_2) where at least one member
     has WEAK standalone importance (bottom-70th-percentile SHAP
     importance) -- explicitly targets "weak marginal, strong conditional"
  F. pairs where the two broad screen scores strongly DISAGREE in rank
     (one score ranks it as notable, the other does not) -- surfaces cases
     the single-score v2 method could not have flagged as ambiguous
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import compute_nonadditivity_screen
from src.explain.stage2_v2_pair_screen import compute_pairwise_screen_scores
from src.models.trees import make_lightgbm

CHANNEL_CAPS = dict(A_score1=50, A_score2=50, B=50, C=50, D=50, E=30, F=20)


def compute_tree_pair_cooccurrence(model, feature_names: list[str]) -> dict:
    trees_df = model.booster_.trees_to_dataframe().dropna(subset=["split_feature"])
    counts: dict[tuple, int] = {}
    fset = set(feature_names)
    for _, group in trees_df.groupby("tree_index"):
        feats = sorted(set(group["split_feature"].unique()) & fset)
        if len(feats) < 2:
            continue
        for pair in combinations(feats, 2):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def main():
    split = load_locked_split()
    ds = split.ds
    numeric_features = ds.numeric_cols_original
    model_cols = [ds.name_map[f] for f in numeric_features]
    n = len(numeric_features)

    y_arr = split.y_train.to_numpy()
    X_num = split.X_train[model_cols]
    X_num.columns = numeric_features
    X_arr = X_num.to_numpy(dtype=np.float64)

    print("Recomputing full-universe broad scores (same formulas as stage2_v3_pair_screen)...")
    score1 = compute_pairwise_screen_scores(X_num, y_arr, numeric_features)
    score2 = compute_nonadditivity_screen(X_arr, y_arr, n_bins=4)
    iu = np.triu_indices(n, k=1)
    s1_flat, s2_flat = score1[iu], score2[iu]
    rank1 = pd.Series(-s1_flat).rank(method="min").astype(int).to_numpy()
    rank2 = pd.Series(-s2_flat).rank(method="min").astype(int).to_numpy()

    def pair_at(idx):
        i, j = iu[0][idx], iu[1][idx]
        return numeric_features[i], numeric_features[j]

    sources: dict[frozenset, set] = {}

    def add(pairs, label):
        for a, b in pairs:
            sources.setdefault(frozenset([a, b]), set()).add(label)

    # --- Channel A: top of each broad score ---
    top_s1 = [pair_at(idx) for idx in np.argsort(-s1_flat)[:CHANNEL_CAPS["A_score1"]]]
    top_s2 = [pair_at(idx) for idx in np.argsort(-s2_flat)[:CHANNEL_CAPS["A_score2"]]]
    add(top_s1, "A_top_score1_nonadditivity")
    add(top_s2, "A_top_score2_nonadditivity")

    # --- Channel B & C: existing frozen SHAP-interaction / H-statistic evidence ---
    v1i = pd.read_csv(f"{R}/stage2_runB_interactions.csv")
    v2i = pd.read_csv(f"{R}/stage2_runB_v2_confirmed_interactions.csv")
    existing = pd.concat([
        v1i[["parameter_A", "parameter_B", "shap_interaction_strength", "h_statistic_strength", "evidence_label"]],
        v2i[["parameter_A", "parameter_B", "shap_interaction_strength", "h_statistic_strength", "evidence_label"]],
    ], ignore_index=True)
    existing = existing[existing["evidence_label"].isin(["STRONG", "MODERATE"])]
    existing = existing[existing["parameter_A"].isin(numeric_features) & existing["parameter_B"].isin(numeric_features)]

    top_b = existing.sort_values("shap_interaction_strength", ascending=False).head(CHANNEL_CAPS["B"])
    add(list(zip(top_b["parameter_A"], top_b["parameter_B"])), "B_existing_shap_interaction")

    top_c = existing.dropna(subset=["h_statistic_strength"]).sort_values("h_statistic_strength", ascending=False).head(CHANNEL_CAPS["C"])
    add(list(zip(top_c["parameter_A"], top_c["parameter_B"])), "C_existing_h_statistic")

    # --- Channel D: fresh tree-path co-occurrence on a broad feature union ---
    gi = pd.read_csv(f"{R}/stage2_runB_global_importance.csv")
    top_importance_feats = gi.sort_values("shap_importance", ascending=False).head(150)["raw_csv_parameter_name"].tolist()
    channel_features_so_far = sorted(set(f for p in sources for f in p))
    broad_union = sorted(set(top_importance_feats) | set(channel_features_so_far))
    broad_union = [f for f in broad_union if f in ds.name_map]
    broad_union_cols = [ds.name_map[f] for f in broad_union]
    print(f"\nFitting compact model for tree-path co-occurrence mining on {len(broad_union)} broad-union features...")
    tree_model = make_lightgbm(random_state=42)
    tree_model.fit(split.X_train[broad_union_cols], split.y_train)
    pair_counts_model_cols = compute_tree_pair_cooccurrence(tree_model, broad_union_cols)
    pair_counts = {frozenset([ds.inverse_name_map[a], ds.inverse_name_map[b]]): c for (a, b), c in pair_counts_model_cols.items()}
    top_d = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:CHANNEL_CAPS["D"]]
    add([tuple(p) for p, _ in top_d], "D_tree_path_cooccurrence")

    # --- Channel E: strong non-additivity + weak marginal importance on >=1 side ---
    imp_pctile = gi.set_index("raw_csv_parameter_name")["shap_percentile"]
    weak_thresh = 70.0  # bottom 70% of ranked importance = "weak" marginal signal
    e_candidates = []
    for idx in np.argsort(-s2_flat):
        a, b = pair_at(idx)
        pa, pb = imp_pctile.get(a, 0.0), imp_pctile.get(b, 0.0)
        if min(pa, pb) < (100 - weak_thresh) or max(pa, pb) < weak_thresh:
            # at least one member sits in the bottom ~weak_thresh percentile of importance
            if pa < weak_thresh or pb < weak_thresh:
                e_candidates.append((a, b))
        if len(e_candidates) >= CHANNEL_CAPS["E"]:
            break
    add(e_candidates, "E_weak_marginal_strong_conditional")

    # --- Channel F: strong disagreement between the two broad scores ---
    notable = np.where((rank1 <= 3000) | (rank2 <= 3000))[0]
    disagreement = np.abs(rank1[notable].astype(np.int64) - rank2[notable].astype(np.int64))
    order = notable[np.argsort(-disagreement)][:CHANNEL_CAPS["F"]]
    add([pair_at(idx) for idx in order], "F_score_disagreement")

    # --- assemble final candidate pool ---
    rows = []
    for pair, labels in sources.items():
        a, b = tuple(pair)
        idx_lookup = None
        rows.append({
            "parameter_A": a, "parameter_B": b,
            "channels": "+".join(sorted(labels)),
            "n_channels": len(labels),
        })
    df = pd.DataFrame(rows).sort_values("n_channels", ascending=False)
    out_path = f"{R}/stage2_runB_v3_pair_candidates.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {out_path} ({len(df)} unique candidate pairs)")
    print("\nPer-channel unique-pair contribution:")
    for label in ["A_top_score1_nonadditivity", "A_top_score2_nonadditivity", "B_existing_shap_interaction",
                  "C_existing_h_statistic", "D_tree_path_cooccurrence", "E_weak_marginal_strong_conditional",
                  "F_score_disagreement"]:
        cnt = sum(1 for labels in sources.values() if label in labels)
        print(f"  {label}: {cnt}")
    print(f"\nFinal union size: {len(df)} unique pairs")
    print(f"Pairs found by >=2 channels (cross-channel corroboration): {int((df['n_channels'] >= 2).sum())}")


if __name__ == "__main__":
    main()
