"""Stage 2 Run B EBM -- section 9: pairwise interaction discovery.

Uses EBM's own automatic interaction-selection mechanism (FAST/greedy
detection, generic, no hard-coded pairs) -- the final model's up-to-15
selected pairwise terms are the candidate pool, exactly as EBM chose them.

Fold reproducibility is measured two ways, both from the 5 cached grouped-
CV fold models (each of which independently ran EBM's OWN interaction
detection on that fold's training data):
  1. SELECTION reproducibility: was this exact pair ALSO selected as one
     of that fold's own interaction terms? (a pair EBM would not even
     consider "significant enough to include" in a different lot split is
     inherently less trustworthy than one selected consistently)
  2. Among folds where it WAS selected, is its relative importance rank
     and 2D pattern consistent?

This is a stronger reproducibility bar than just recomputing a score for
a fixed pair -- it tests whether the pair reliably passes threshold, not
just whether the number itself is somewhat stable.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.explain.stage2_ebm_common import real_bin_layout
from src.explain.stage2_ebm_train_eval import CACHE_PATH
from src.explain.stage2_v2_common import R, load_locked_split


def pair_bin_layout(ebm, term_idx):
    feat_a, feat_b = ebm.term_features_[term_idx]
    edges_a = np.asarray(ebm.bins_[feat_a][0], dtype=np.float64)
    edges_b = np.asarray(ebm.bins_[feat_b][0], dtype=np.float64)
    scores = np.asarray(ebm.term_scores_[term_idx], dtype=np.float64)
    # interaction term scores: shape (n_bins_a, n_bins_b) incl. a leading missing row/col
    n_real_a = min(len(edges_a) + 1, scores.shape[0] - 1)
    n_real_b = min(len(edges_b) + 1, scores.shape[1] - 1)
    real_scores = scores[1:1 + n_real_a, 1:1 + n_real_b]
    return edges_a, edges_b, real_scores


def bin_repr(edges, i, n_real):
    if len(edges) == 0:
        return 0.0
    if i == 0:
        return edges[0] - (edges[1] - edges[0]) / 2 if len(edges) > 1 else edges[0] - 1.0
    if i == n_real - 1:
        return edges[-1] + (edges[-1] - edges[-2]) / 2 if len(edges) > 1 else edges[-1] + 1.0
    return (edges[i - 1] + edges[i]) / 2


def main():
    split = load_locked_split()
    cache = joblib.load(CACHE_PATH)
    final_ebm = cache["final_ebm"]
    fold_models = cache["fold_models"]

    pair_terms = [(i, t) for i, t in enumerate(final_ebm.term_features_) if len(t) == 2]
    importances = final_ebm.term_importances()
    print(f"EBM selected {len(pair_terms)} pairwise interaction terms (automatic selection, cap=15)")

    feature_names = list(final_ebm.feature_names_in_)

    # build fold-model pair sets keyed by feature-name-pair for selection reproducibility
    fold_pair_sets = []
    for fm in fold_models:
        ebm_f = fm["ebm"]
        names_f = list(ebm_f.feature_names_in_)
        pairs_f = set()
        for t in ebm_f.term_features_:
            if len(t) == 2:
                pairs_f.add(frozenset([names_f[t[0]], names_f[t[1]]]))
        fold_pair_sets.append(pairs_f)

    rows = []
    for rank, (term_idx, (fa, fb)) in enumerate(sorted(pair_terms, key=lambda kv: -importances[kv[0]]), start=1):
        name_a, name_b = feature_names[fa], feature_names[fb]
        strength = float(importances[term_idx])
        edges_a, edges_b, surface = pair_bin_layout(final_ebm, term_idx)
        n_a, n_b = surface.shape
        best = np.unravel_index(np.argmax(surface), surface.shape)
        worst = np.unravel_index(np.argmin(surface), surface.shape)
        high_region = f"{name_a}~{bin_repr(edges_a, best[0], n_a):.4g}, {name_b}~{bin_repr(edges_b, best[1], n_b):.4g}"
        low_region = f"{name_a}~{bin_repr(edges_a, worst[0], n_a):.4g}, {name_b}~{bin_repr(edges_b, worst[1], n_b):.4g}"

        # conditional pattern: effect of A (min row -> max row) at low/mid/high B tercile
        b_tercile_idx = [0, n_b // 2, n_b - 1]
        b_labels = ["low", "mid", "high"]
        effect_desc_parts = []
        for bi, blab in zip(b_tercile_idx, b_labels):
            col = surface[:, bi]
            effect_desc_parts.append(f"{blab} {name_b}: {name_a} range effect={col.max() - col.min():.3f}")
        conditional_pattern = "; ".join(effect_desc_parts)

        pair_key = frozenset([name_a, name_b])
        n_folds_selected = sum(1 for s in fold_pair_sets if pair_key in s)
        if n_folds_selected >= 4:
            confidence = "STRONG"
        elif n_folds_selected == 3:
            confidence = "MODERATE"
        elif n_folds_selected <= 1:
            confidence = "INCONCLUSIVE"
        else:
            confidence = "WEAK"

        rows.append({
            "parameter_A": name_a, "parameter_B": name_b,
            "interaction_strength": round(strength, 6), "rank": rank,
            "high_yield_region": high_region, "low_yield_region": low_region,
            "conditional_pattern": conditional_pattern,
            "fold_selection_reproducibility": f"{n_folds_selected}/5",
            "confidence": confidence,
        })

    df = pd.DataFrame(rows)
    out_path = f"{R}/stage2_runB_ebm_2way_interactions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} EBM-selected pairwise interactions)")
    print(df["confidence"].value_counts().to_string())
    print("\nAll pairs (ranked):")
    print(df[["parameter_A", "parameter_B", "interaction_strength", "rank", "fold_selection_reproducibility", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
