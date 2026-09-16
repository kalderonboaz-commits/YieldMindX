"""Relationship Validation Engine -- broad hierarchical 3-way candidate
screen (COVERAGE, not confirmation).

Generates candidate triples generically from multiple, independently-
capped channels (no channel individually gates promotion; no channel
references a known planted parameter name):
  1. triangles within THIS engine's own confirmed STRONG/MODERATE 2-way
     edges (stage2_relval_2way.py output)
  2. shared-node chaining among the same edges -- the THIRD pairwise edge
     is explicitly NOT required to exist (this is the generic fix for the
     v3-identified "real triple blocked because one leg was never
     confirmed" bottleneck, reused as a design principle here, not as
     code, since this is a fresh engine)
  3. fresh LightGBM tree-path 3-feature co-occurrence, mined from a
     compact model fit on the confirmed-edge node union (a lightweight
     SCREENING tool, consistent with its use throughout every prior Stage
     2 checkpoint -- not a new predictive model added to the pipeline)
  4. residual-conditional third-variable evidence: for the top confirmed
     STRONG edges, correlate the pair's own joint-fit residual against
     every other numeric feature
  5. corroboration from the frozen v5 3-way graph/interactions (edges and
     STRONG triples v5 already found become additional candidate seeds --
     reused as external evidence, not re-derived)

Triple-count is bounded by a disclosed cap on the confirmed-edge subgraph
feeding channels 1/2 (ranked by this engine's own interaction_strength_
delta_r2, not by an external file), analogous to v5's evidence-adaptive
(not magnitude-only) edge admission, redesigned fresh for this engine.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.explain.stage2_relval_common import R, load_locked_split
from src.explain.stage2_v3_common import fit_additive_joint, predict_joint
from src.models.trees import make_lightgbm

EDGE_TOPK_PER_NODE = 8   # per-node guaranteed local top-k edges (replaces the old global top-2500
                         # cutoff -- see candidate-generation coverage audit). A node with a real,
                         # FDR-confirmed edge whose raw magnitude ranks outside any GLOBAL cutoff
                         # still gets its own strongest edges into the subgraph.
TRIANGLE_CAP = 2000
CHAIN_CAP = 4000
TREE_UNION_TOP_N = 250
TREE_TRIPLET_TOP_N = 150
RESIDUAL_TOP_PAIRS = 15
RESIDUAL_THIRDVARS_PER_PAIR = 3


def compute_tree_triplet_counts(model, feature_names):
    trees_df = model.booster_.trees_to_dataframe().dropna(subset=["split_feature"])
    counts = {}
    fset = set(feature_names)
    for _, group in trees_df.groupby("tree_index"):
        feats = sorted(set(group["split_feature"].unique()) & fset)
        if len(feats) < 3:
            continue
        for trip in combinations(feats, 3):
            counts[trip] = counts.get(trip, 0) + 1
    return counts


def main():
    split = load_locked_split()
    ds = split.ds
    conf2 = pd.read_csv(f"{R}/stage2_runB_relationship_validation_2way.csv")
    strong_mod = conf2[conf2["confidence"].isin(["STRONG", "MODERATE"])].copy()
    strong = conf2[conf2["confidence"] == "STRONG"].sort_values("interaction_strength_delta_r2", ascending=False)
    print(f"This engine's own confirmed 2-way edges: {len(strong_mod)} STRONG+MODERATE ({len(strong)} STRONG) "
          f"over {len(set(strong_mod.parameter_A) | set(strong_mod.parameter_B))} nodes")

    # Per-node guaranteed local top-K edge selection (replaces the old global
    # top-2500-by-magnitude cutoff): for every node that appears in the
    # confirmed edge set, keep ITS OWN strongest EDGE_TOPK_PER_NODE edges,
    # then take the union across all nodes. This guarantees a node's best
    # edges enter the subgraph even if their raw magnitude would lose a
    # GLOBAL ranking race against unrelated, larger-effect-size edges.
    edge_score = {}
    for a, b, s in zip(strong_mod["parameter_A"], strong_mod["parameter_B"], strong_mod["interaction_strength_delta_r2"]):
        edge_score[frozenset([a, b])] = s
    node_edges: dict[str, list] = {}
    for a, b in zip(strong_mod["parameter_A"], strong_mod["parameter_B"]):
        node_edges.setdefault(a, []).append(b)
        node_edges.setdefault(b, []).append(a)
    subgraph_edge_set = set()
    for node, partners in node_edges.items():
        ranked = sorted(partners, key=lambda p: -edge_score[frozenset([node, p])])[:EDGE_TOPK_PER_NODE]
        for p in ranked:
            subgraph_edge_set.add(frozenset([node, p]))
    print(f"Confirmed-edge subgraph for triangle/chaining channels: {len(subgraph_edge_set)} edges "
          f"(per-node top-{EDGE_TOPK_PER_NODE} union over {len(node_edges)} nodes, of {len(strong_mod)} total confirmed edges)")

    adjacency: dict[str, set] = {}
    for edge in subgraph_edge_set:
        a, b = tuple(edge)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    triples: set[tuple] = set()
    channel_of: dict[tuple, set] = {}

    def add_triple(t, label):
        t = tuple(sorted(t))
        triples.add(t)
        channel_of.setdefault(t, set()).add(label)

    # --- Channel 1: triangles ---
    n_triangles = 0
    nodes = sorted(adjacency.keys())
    for a in nodes:
        neighbors_a = adjacency[a]
        for b in neighbors_a:
            if b <= a:
                continue
            common = neighbors_a & adjacency.get(b, set())
            for c in common:
                if c > b:
                    add_triple((a, b, c), "1_triangle")
                    n_triangles += 1
    print(f"Channel 1 (triangles): {n_triangles}")

    # --- Channel 2: shared-node chaining (third edge NOT required), capped ---
    # Overflow fix (per the candidate-generation coverage audit): the
    # original implementation took a RANDOM subsample when raw_chains
    # exceeded CHAIN_CAP -- a non-evidence-based information loss. Now
    # ranked by summed constituent-edge score (the same two edges already
    # used to form the chain), so the retained CHAIN_CAP chains are the
    # ones with the strongest underlying pairwise evidence, not an
    # arbitrary random draw.
    raw_chains = []
    for node, neighbors in adjacency.items():
        neighbors_l = sorted(neighbors)
        for i in range(len(neighbors_l)):
            for j in range(i + 1, len(neighbors_l)):
                b, c = neighbors_l[i], neighbors_l[j]
                chain_score = edge_score[frozenset([node, b])] + edge_score[frozenset([node, c])]
                raw_chains.append((chain_score, node, b, c))
    raw_chains.sort(key=lambda t: -t[0])
    kept_chains = raw_chains[:CHAIN_CAP]
    for _, node, b, c in kept_chains:
        add_triple((node, b, c), "2_shared_node_chaining")
    print(f"Channel 2 (shared-node chaining, top {CHAIN_CAP} of {len(raw_chains)} by summed edge score, "
          f"evidence-ranked not random): {len(kept_chains)} (cumulative unique: {len(triples)})")

    # --- Channel 3: fresh tree-path co-occurrence ---
    node_union = sorted(adjacency.keys())
    top_nodes = node_union[:TREE_UNION_TOP_N] if len(node_union) > TREE_UNION_TOP_N else node_union
    model_cols = [ds.name_map[f] for f in top_nodes if f in ds.name_map]
    print(f"Fitting compact LightGBM (screening tool only) on {len(model_cols)} confirmed-edge-node features for tree-triplet mining...")
    tree_model = make_lightgbm(random_state=42)
    tree_model.fit(split.X_train[model_cols], split.y_train)
    triplet_counts_model = compute_tree_triplet_counts(tree_model, model_cols)
    triplet_counts = {tuple(sorted(ds.inverse_name_map[c] for c in trip)): cnt for trip, cnt in triplet_counts_model.items()}
    ranked = sorted(triplet_counts.items(), key=lambda kv: -kv[1])[:TREE_TRIPLET_TOP_N]
    for trip, _ in ranked:
        add_triple(trip, "3_tree_path_cooccurrence")
    print(f"Channel 3 (tree-path co-occurrence, top {TREE_TRIPLET_TOP_N}): {len(ranked)}")

    # --- Channel 4: residual-conditional third-variable evidence ---
    y_train_arr = split.y_train.to_numpy()
    numeric_features = ds.numeric_cols_original
    X_all = split.X_train[[ds.name_map[f] for f in numeric_features]].to_numpy(dtype=np.float64)
    top_resid_pairs = strong.head(RESIDUAL_TOP_PAIRS)
    n_ch4 = 0
    for _, row in top_resid_pairs.iterrows():
        a, b = row["parameter_A"], row["parameter_B"]
        xa = split.X_train[ds.name_map[a]].to_numpy(dtype=np.float64)
        xb = split.X_train[ds.name_map[b]].to_numpy(dtype=np.float64)
        fit_ab = fit_additive_joint(xa, xb, y_train_arr, n_bins=3)
        pred = predict_joint(fit_ab, xa, xb)
        residual = y_train_arr - pred
        resid_c = residual - residual.mean()
        resid_std = resid_c.std()
        if resid_std < 1e-9:
            continue
        X_std = (X_all - X_all.mean(axis=0)) / np.where(X_all.std(axis=0) < 1e-9, 1.0, X_all.std(axis=0))
        corr = (X_std * resid_c[:, None]).mean(axis=0) / resid_std
        order = np.argsort(-np.abs(corr))
        added = 0
        for idx in order:
            c = numeric_features[idx]
            if c in (a, b):
                continue
            add_triple((a, b, c), "4_residual_conditional_structure")
            added += 1
            n_ch4 += 1
            if added >= RESIDUAL_THIRDVARS_PER_PAIR:
                break
    print(f"Channel 4 (residual-conditional, top {RESIDUAL_TOP_PAIRS} STRONG pairs x {RESIDUAL_THIRDVARS_PER_PAIR}): {n_ch4}")

    # --- Channel 5: corroboration from frozen v5 3-way results (external evidence, reused) ---
    v5three = pd.read_csv(f"{R}/stage2_runB_v5_three_way_interactions.csv")
    v5_strong = v5three[v5three["evidence_label"] == "STRONG"]
    n_ch5 = 0
    for _, row in v5_strong.iterrows():
        add_triple((row["parameter_A"], row["parameter_B"], row["parameter_C"]), "5_v5_frozen_strong_corroboration")
        n_ch5 += 1
    print(f"Channel 5 (frozen v5 STRONG 3-way triples, reused as external evidence): {n_ch5}")

    rows = []
    for t, labs in channel_of.items():
        a, b, c = t
        if a not in ds.name_map or b not in ds.name_map or c not in ds.name_map:
            continue
        rows.append({"parameter_A": a, "parameter_B": b, "parameter_C": c,
                     "generation_channels": "+".join(sorted(labs)), "n_channels": len(labs)})
    df = pd.DataFrame(rows).sort_values("n_channels", ascending=False)
    out_path = f"{R}/stage2_runB_relationship_validation_triple_screen.csv"
    df.to_csv(out_path, index=False)

    union_feats = sorted(set(df["parameter_A"]) | set(df["parameter_B"]) | set(df["parameter_C"]))
    n_numeric = len(ds.numeric_cols_original)
    n_possible_triples = n_numeric * (n_numeric - 1) * (n_numeric - 2) // 6
    print(f"\nSaved {out_path} ({len(df)} unique candidate triples, {len(union_feats)} unique features, "
          f"{100*len(union_feats)/n_numeric:.1f}% of the {n_numeric}-feature numeric universe)")
    print("\nPer-channel unique-triple contribution:")
    for label in ["1_triangle", "2_shared_node_chaining", "3_tree_path_cooccurrence",
                  "4_residual_conditional_structure", "5_v5_frozen_strong_corroboration"]:
        cnt = sum(1 for labs in channel_of.values() if label in labs)
        print(f"  {label}: {cnt}")
    print(f"Triples found by >=2 channels: {int((df['n_channels'] >= 2).sum())}")

    with open(f"{R}/relval_cache/triple_screen_coverage.txt", "w") as f:
        f.write(f"eligible_numeric_predictors={n_numeric}\n")
        f.write(f"total_theoretical_triples={n_possible_triples}\n")
        f.write(f"triples_cheaply_generated={len(df)}\n")
        f.write(f"unique_features_in_generated_triples={len(union_feats)}\n")
        f.write(f"pct_numeric_universe_covered={100*len(union_feats)/n_numeric:.4f}\n")
        f.write(f"estimated_search_coverage_pct_of_theoretical={100*len(df)/n_possible_triples:.10f}\n")


if __name__ == "__main__":
    main()
