"""Stage 2 Run B v4 -- evidence-graph-driven pair candidate generation.

Replaces v3's fixed "top-N per channel" gate with membership rules over the
feature-evidence graph (stage2_runB_v4_feature_evidence_graph.csv). Channels
(per project instruction, all generic -- no ground-truth involvement):

  A. HIGH_EVIDENCE x HIGH_EVIDENCE (all pairs)
  B. HIGH_EVIDENCE x (conditional OR interaction-related OR cross-stage
     evidence, tier below HIGH)
  C. full-universe broad-screen (non-additivity) outliers among pairs where
     NEITHER feature is individually HIGH_EVIDENCE -- "the pair matters more
     than either predictor alone"
  D. pairs already repeatedly surfaced by >=2 independent v3 candidate
     channels (reused directly from the frozen v3 candidate pool)
  E. features with STRONG spline/GAM single-parameter evidence but weak v3
     interaction-candidate representation, paired against every
     HIGH_EVIDENCE feature (guarantees a single-parameter-confirmed-real
     feature is not silently excluded from interaction testing just because
     it never surfaced in v3's indirect ranking channels)

The additive-vs-joint core confirmation test costs ~2.5ms/pair (measured),
so even tens of thousands of candidates remain confirmable in minutes --
channels A (HIGH x HIGH) and D (reused v3 multi-channel pairs) are
therefore used in FULL, uncapped. Channels B and E are cross-products that
would otherwise reach tens/hundreds of thousands of pairs from a
combinatorial blow-up rather than genuine evidentiary density; each is
explicitly capped, by the SAME full-universe non-additivity score used
elsewhere in this pipeline (not an arbitrary index cut), at a disclosed
top-K. Channel C's own definition is inherently a top-K ("outlier") rule.
All caps are stated as constants at the top of this file and reported in
the build log and summary -- no silent truncation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import compute_nonadditivity_screen

C_OUTLIER_TOP_K = 1000  # explicit, disclosed bound for channel C: the top 1000
                         # non-additivity scores among ALL pairs where neither
                         # member is individually HIGH_EVIDENCE -- i.e. the top
                         # ~0.1% of the ~880k such pairs, a generic percentile
                         # cut (not tuned to any known answer)

# Channels B and E are cross-products (HIGH x special-evidence, and
# underrepresented-spline x HIGH) and can combinatorially explode well past
# what is useful to report even though the confirmation test itself is cheap
# enough to run all of it. Rather than leave that implicit, each is
# explicitly capped at its own top-K by the SAME full-universe non-
# additivity score used elsewhere in this pipeline (evidence-based
# prioritization, not an arbitrary index cut) -- exactly the "strong broad-
# screen outliers next" ordering the checkpoint spec itself suggests.
B_TOP_K = 5000
E_TOP_K = 3000


def main():
    split = load_locked_split()
    ds = split.ds
    eg = pd.read_csv(f"{R}/stage2_runB_v4_feature_evidence_graph.csv").set_index("raw_csv_parameter_name")
    numeric_features = ds.numeric_cols_original
    n = len(numeric_features)

    high = set(eg[eg["evidence_tier"] == "HIGH_EVIDENCE"].index)
    special_evidence = set(eg[(eg["evidence_tier"] != "HIGH_EVIDENCE") &
                               (eg["is_conditional_evidence"] | eg["is_interaction_related_evidence"] | eg["is_cross_stage_evidence"])].index)
    print(f"HIGH_EVIDENCE features: {len(high)}")
    print(f"MEDIUM/LOW features with conditional/interaction-related/cross-stage evidence: {len(special_evidence)}")

    sources: dict[frozenset, set] = {}

    def add(pairs, label):
        for a, b in pairs:
            if a == b:
                continue
            sources.setdefault(frozenset([a, b]), set()).add(label)

    def topk_by_score(pairs, score_lookup, k):
        if len(pairs) <= k:
            return pairs
        scored = [(score_lookup(a, b), a, b) for a, b in pairs]
        scored.sort(key=lambda t: -t[0])
        return [(a, b) for _, a, b in scored[:k]]

    # --- full-universe non-additivity score, computed once, reused to
    # --- prioritize (not gate) every oversized channel below ---
    model_cols = [ds.name_map[f] for f in numeric_features]
    X_arr = split.X_train[model_cols].to_numpy(dtype=np.float64)
    y_arr = split.y_train.to_numpy()
    print("Recomputing full-universe non-additivity score (shared by channels B, C, E)...")
    score2 = compute_nonadditivity_screen(X_arr, y_arr, n_bins=4)
    feat_idx = {f: i for i, f in enumerate(numeric_features)}

    def s2(a, b):
        return float(score2[feat_idx[a], feat_idx[b]])

    # --- Channel A: HIGH x HIGH, full (highest priority, per spec, kept uncapped) ---
    high_list = sorted(high)
    from itertools import combinations
    chan_a = list(combinations(high_list, 2))
    add(chan_a, "A_high_x_high")
    print(f"Channel A (HIGH x HIGH, full, no cap): {len(chan_a)} pairs")

    # --- Channel B: HIGH x special-evidence (non-HIGH), capped by non-additivity score ---
    chan_b_full = [(h, s) for h in high_list for s in special_evidence]
    chan_b = topk_by_score(chan_b_full, s2, B_TOP_K)
    add(chan_b, "B_high_x_special_evidence")
    print(f"Channel B (HIGH x special-evidence, top {B_TOP_K} by non-additivity score of "
          f"{len(chan_b_full)} eligible pairs): {len(chan_b)} pairs")

    # --- Channel C: broad-screen outliers among non-HIGH x non-HIGH pairs ---
    iu = np.triu_indices(n, k=1)
    is_high_mask = np.array([f in high for f in numeric_features])
    neither_high = ~(is_high_mask[iu[0]] | is_high_mask[iu[1]])
    s2_flat = score2[iu]
    eligible_idx = np.where(neither_high)[0]
    eligible_scores = s2_flat[eligible_idx]
    top_order = eligible_idx[np.argsort(-eligible_scores)[:C_OUTLIER_TOP_K]]
    chan_c = [(numeric_features[iu[0][idx]], numeric_features[iu[1][idx]]) for idx in top_order]
    add(chan_c, "C_broadscreen_outlier_neither_high")
    print(f"Channel C (broad-screen outliers, neither HIGH, top {C_OUTLIER_TOP_K} of "
          f"{int(neither_high.sum())} eligible pairs): {len(chan_c)} pairs")

    # --- Channel D: reused v3 multi-channel-corroborated pairs ---
    v3cand = pd.read_csv(f"{R}/stage2_runB_v3_pair_candidates.csv")
    multi = v3cand[v3cand["n_channels"] >= 2]
    chan_d = list(zip(multi["parameter_A"], multi["parameter_B"]))
    add(chan_d, "D_v3_multichannel_corroborated")
    print(f"Channel D (v3 pairs surfaced by >=2 independent v3 channels, reused, no cap): {len(chan_d)} pairs")

    # --- Channel E: spline-STRONG but v3-underrepresented, x every HIGH feature, capped ---
    underrep = eg[(eg["spline_evidence_label"] == "STRONG") &
                  (eg["v3_confirmed_strong_count"] == 0) &
                  (eg["v3_confirmed_moderate_count"] == 0) &
                  (eg["v3_broadscreen_best_rank"].isna())].index.tolist()
    chan_e_full = [(u, h) for u in underrep for h in high_list if u != h]
    chan_e = topk_by_score(chan_e_full, s2, E_TOP_K)
    add(chan_e, "E_spline_strong_v3_underrepresented")
    print(f"Channel E ({len(underrep)} spline-STRONG/v3-invisible features x {len(high_list)} HIGH features, "
          f"top {E_TOP_K} by non-additivity score of {len(chan_e_full)} eligible pairs): {len(chan_e)} pairs")

    rows = []
    for pair, labels in sources.items():
        a, b = tuple(pair)
        rows.append({"parameter_A": a, "parameter_B": b, "channels": "+".join(sorted(labels)), "n_channels": len(labels)})
    df = pd.DataFrame(rows).sort_values("n_channels", ascending=False)
    out_path = f"{R}/stage2_runB_v4_pair_candidates.csv"
    df.to_csv(out_path, index=False)

    union_features = sorted(set(df["parameter_A"]) | set(df["parameter_B"]))
    print(f"\nSaved {out_path} ({len(df)} unique candidate pairs, {len(union_features)} unique features, "
          f"{100*len(union_features)/n:.1f}% of the {n}-feature numeric universe)")
    print("\nPer-channel unique-pair contribution:")
    for label in ["A_high_x_high", "B_high_x_special_evidence", "C_broadscreen_outlier_neither_high",
                  "D_v3_multichannel_corroborated", "E_spline_strong_v3_underrepresented"]:
        cnt = sum(1 for labs in sources.values() if label in labs)
        print(f"  {label}: {cnt}")
    print(f"\nPairs found by >=2 channels: {int((df['n_channels'] >= 2).sum())}")
    print("Truncation disclosure: channels A and D used in FULL (no cap). Channel C capped at "
          f"top-{C_OUTLIER_TOP_K} of its own eligible universe. Channels B and E are cross-products that "
          f"would otherwise combinatorially explode ({len(chan_b_full)} and {len(chan_e_full)} raw pairs "
          f"respectively) -- each is explicitly capped (B: top {B_TOP_K}, E: top {E_TOP_K}) by the SAME "
          "full-universe non-additivity score used elsewhere in this pipeline, not an arbitrary index cut.")

    with open(f"{R}/_stage2_v4_pair_candidate_build_log.txt", "w") as f:
        f.write(f"high_evidence_features={len(high)}\n")
        f.write(f"special_evidence_features={len(special_evidence)}\n")
        f.write(f"channel_A_pairs={len(chan_a)} (uncapped)\n")
        f.write(f"channel_B_pairs_raw={len(chan_b_full)} channel_B_pairs_used={len(chan_b)} (capped at top {B_TOP_K} by non-additivity score)\n")
        f.write(f"channel_C_pairs={len(chan_c)} (capped at top {C_OUTLIER_TOP_K} by non-additivity score)\n")
        f.write(f"channel_D_pairs={len(chan_d)} (uncapped)\n")
        f.write(f"channel_E_pairs_raw={len(chan_e_full)} channel_E_pairs_used={len(chan_e)} ({len(underrep)} underrepresented features, capped at top {E_TOP_K} by non-additivity score)\n")
        f.write(f"final_union_pairs={len(df)}\n")
        f.write(f"unique_features_represented={len(union_features)}\n")
        f.write(f"pct_of_numeric_universe={100*len(union_features)/n:.2f}\n")


if __name__ == "__main__":
    main()
