"""Stage 2 Run B v2, item 6: generic 3-way interaction discovery.

Candidate triples are generated systematically (never from ground truth):
  (a) chaining: for every STRONG/MODERATE confirmed 2-way pair (A,B), any
      third feature C that also appears in another STRONG/MODERATE
      confirmed pair with A or with B becomes a candidate triple (A,B,C)
  (b) tree-path co-occurrence: features that appear TOGETHER as split
      features in the same LightGBM tree, 3 or more at once, tallied
      across all trees of a compact model fit on the Stage-B union
      features -- purely structural, no target-shuffle or ground-truth
      involvement

Each candidate triple is tested with a classic 2^3 factorial interaction
contrast (median-split each variable into low/high, compute the 8 cell
means of yield, then the Yates three-way contrast = the part of the joint
pattern not explained by any main effect or 2-way effect). This is a
standard, generic statistic for isolating true 3-way effects, entirely
independent of any ground-truth knowledge.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_v2_common import R, load_locked_split
from src.models.trees import make_lightgbm

RANDOM_STATE = 42
MIN_CELL_SIZE = 15


def three_way_contrast(a: np.ndarray, b: np.ndarray, c: np.ndarray, y: np.ndarray) -> dict | None:
    a_bin = (a > np.median(a)).astype(int)
    b_bin = (b > np.median(b)).astype(int)
    c_bin = (c > np.median(c)).astype(int)

    cell_means = {}
    cell_counts = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                mask = (a_bin == i) & (b_bin == j) & (c_bin == k)
                n = int(mask.sum())
                cell_counts[(i, j, k)] = n
                cell_means[(i, j, k)] = float(y[mask].mean()) if n >= MIN_CELL_SIZE else np.nan

    if any(np.isnan(v) for v in cell_means.values()):
        return None

    contrast = 0.0
    for (i, j, k), m in cell_means.items():
        sign = (2 * i - 1) * (2 * j - 1) * (2 * k - 1)
        contrast += sign * m
    contrast /= 8.0

    # 2-way contrasts within the same cube (collapsing the third variable) -- for comparison
    def two_way(collapse_dim):
        # average over the collapsed dimension, then compute the 2x2 contrast
        dims = [0, 1, 2]
        dims.remove(collapse_dim)
        sub = {}
        for i in (0, 1):
            for j in (0, 1):
                vals = []
                for k in (0, 1):
                    key = [None, None, None]
                    key[collapse_dim] = k
                    key[dims[0]] = i
                    key[dims[1]] = j
                    vals.append(cell_means[tuple(key)])
                sub[(i, j)] = np.mean(vals)
        return 0.25 * (sub[(1, 1)] - sub[(1, 0)] - sub[(0, 1)] + sub[(0, 0)])

    ab = two_way(2)  # collapse C
    ac = two_way(1)  # collapse B
    bc = two_way(0)  # collapse A
    max_2way = max(abs(ab), abs(ac), abs(bc))

    best_cell = max(cell_means, key=cell_means.get)
    worst_cell = min(cell_means, key=cell_means.get)

    return {
        "contrast_3way": contrast,
        "max_2way_contrast": max_2way,
        "adds_beyond_lower_order": abs(contrast) > max_2way,
        "high_yield_cell": best_cell, "high_yield_mean": cell_means[best_cell],
        "low_yield_cell": worst_cell, "low_yield_mean": cell_means[worst_cell],
        "min_cell_count": min(cell_counts.values()),
    }


def generate_candidates(confirmed: pd.DataFrame, tree_triplet_counts: dict, top_n_chain: int = 15, top_n_tree: int = 20) -> list[tuple]:
    strong_mod = confirmed[confirmed["evidence_label"].isin(["STRONG", "MODERATE"])]
    triples = set()

    top_pairs = list(zip(strong_mod["parameter_A"], strong_mod["parameter_B"]))[:top_n_chain]
    member_to_partners = {}
    for a, b in zip(strong_mod["parameter_A"], strong_mod["parameter_B"]):
        member_to_partners.setdefault(a, set()).add(b)
        member_to_partners.setdefault(b, set()).add(a)

    for a, b in top_pairs:
        for c in member_to_partners.get(a, set()) | member_to_partners.get(b, set()):
            if c not in (a, b):
                triples.add(tuple(sorted([a, b, c])))

    ranked_tree_triplets = sorted(tree_triplet_counts.items(), key=lambda kv: -kv[1])[:top_n_tree]
    for (a, b, c), _ in ranked_tree_triplets:
        triples.add(tuple(sorted([a, b, c])))

    return sorted(triples)


def compute_tree_triplet_counts(model, feature_names: list[str]) -> dict:
    trees_df = model.booster_.trees_to_dataframe().dropna(subset=["split_feature"])
    counts = {}
    for tree_idx, group in trees_df.groupby("tree_index"):
        feats = set(group["split_feature"].unique()) & set(feature_names)
        if len(feats) < 3:
            continue
        for trip in combinations(sorted(feats), 3):
            counts[trip] = counts.get(trip, 0) + 1
    return counts


def main():
    split = load_locked_split()
    ds = split.ds

    confirmed = pd.read_csv(f"{R}/stage2_runB_v2_confirmed_interactions.csv")
    union_features = sorted(set(confirmed["parameter_A"]) | set(confirmed["parameter_B"]))
    model_cols = [ds.name_map[f] for f in union_features]

    print(f"Refitting compact model on {len(union_features)} confirmed-interaction-pool features for tree-path mining...")
    model = make_lightgbm(random_state=RANDOM_STATE)
    model.fit(split.X_train[model_cols], split.y_train)

    print("Mining tree paths for 3-feature co-occurrence...")
    model_col_triplet_counts = compute_tree_triplet_counts(model, model_cols)
    orig_triplet_counts = {
        tuple(sorted(ds.inverse_name_map.get(c, c) for c in trip)): cnt
        for trip, cnt in model_col_triplet_counts.items()
    }
    print(f"  {len(orig_triplet_counts)} distinct 3-feature tree co-occurrence combinations found")

    candidates = generate_candidates(confirmed, orig_triplet_counts)
    print(f"\nTotal 3-way candidate triples generated: {len(candidates)}")

    y_train_arr = split.y_train.to_numpy()
    _, folds = make_group_kfold(split.groups_train, n_splits=5)

    rows = []
    for a, b, c in candidates:
        if a not in ds.name_map or b not in ds.name_map or c not in ds.name_map:
            continue
        av = split.X_train[ds.name_map[a]].to_numpy()
        bv = split.X_train[ds.name_map[b]].to_numpy()
        cv = split.X_train[ds.name_map[c]].to_numpy()
        result = three_way_contrast(av, bv, cv, y_train_arr)
        if result is None:
            rows.append({"parameter_A": a, "parameter_B": b, "parameter_C": c,
                          "contrast_3way": None, "evidence_label": "INCONCLUSIVE",
                          "reason": "insufficient cell size for main split"})
            continue

        fold_signs = []
        for tr, va in folds:
            av_f = split.X_train.iloc[va][ds.name_map[a]].to_numpy()
            bv_f = split.X_train.iloc[va][ds.name_map[b]].to_numpy()
            cv_f = split.X_train.iloc[va][ds.name_map[c]].to_numpy()
            y_f = split.y_train.iloc[va].to_numpy()
            r_f = three_way_contrast(av_f, bv_f, cv_f, y_f)
            if r_f is not None:
                fold_signs.append(np.sign(r_f["contrast_3way"]))

        main_sign = np.sign(result["contrast_3way"])
        sign_consistency = sum(1 for s in fold_signs if s == main_sign)
        n_valid_folds = len(fold_signs)

        if n_valid_folds >= 4 and sign_consistency >= 4 and result["adds_beyond_lower_order"]:
            label = "STRONG"
        elif n_valid_folds >= 3 and sign_consistency >= 3:
            label = "MODERATE"
        elif sign_consistency >= 1:
            label = "WEAK"
        else:
            label = "INCONCLUSIVE"

        rows.append({
            "parameter_A": a, "parameter_B": b, "parameter_C": c,
            "contrast_3way": round(result["contrast_3way"], 4),
            "max_2way_contrast_in_cube": round(result["max_2way_contrast"], 4),
            "adds_beyond_lower_order": result["adds_beyond_lower_order"],
            "high_yield_region_(A,B,C)_bins": result["high_yield_cell"], "high_yield_mean": round(result["high_yield_mean"], 3),
            "low_yield_region_(A,B,C)_bins": result["low_yield_cell"], "low_yield_mean": round(result["low_yield_mean"], 3),
            "min_cell_count": result["min_cell_count"],
            "fold_sign_reproducibility": f"{sign_consistency}/{n_valid_folds}" if n_valid_folds else "0/0",
            "evidence_label": label,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/stage2_runB_v2_three_way_interactions.csv", index=False)
    print(f"\nSaved {R}/stage2_runB_v2_three_way_interactions.csv ({len(df)} candidate triples tested)")
    print(df["evidence_label"].value_counts().to_string())
    print("\nSTRONG 3-way findings:")
    strong = df[df["evidence_label"] == "STRONG"]
    print(strong.to_string(index=False) if len(strong) else "  (none)")


if __name__ == "__main__":
    main()
