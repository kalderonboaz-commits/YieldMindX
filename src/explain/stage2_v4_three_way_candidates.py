"""Stage 2 Run B v4 -- interaction-graph-driven 3-way candidate generation.

Nodes: original Run B numeric predictors. Edges: v4 confirmed STRONG/
MODERATE 2-way interactions (stage2_runB_v4_confirmed_2way_interactions.csv).

The v4 confirmed-edge set (47,871 STRONG + 20,443 MODERATE over ~1,000
nodes) is far denser than v3's (176 STRONG over 177 nodes) -- enumerating
triangles or shared-node motifs over the FULL edge set would itself be
combinatorially intractable (a near-complete graph on ~1,000 nodes has up
to ~1.6e8 possible triangles). Motif search is therefore run on an
explicit, disclosed, evidence-ranked SUBSET of the edges (the strongest
edges by incremental R^2) -- the same "prioritize by evidence, disclose the
cap" principle used throughout the v4 pair-candidate script, applied here
because the graph itself, not the confirmation cost, is the bottleneck.

Channels:
  1. triangles within the top-EDGE_CAP-strongest-edges subgraph
  2. two edges (within the same subgraph) sharing a node -- chaining,
     explicitly NOT requiring the third pairwise edge to exist (this is
     the direct fix for v3's failure mode: a real 3-way interaction whose
     third pairwise component was never itself confirmed can still be
     generated here from any two of its three edges)
  3. one strong edge (top-N by incremental R^2) + a third HIGH_EVIDENCE
     feature that independently carries conditional-interaction evidence
     (evidence_tier=HIGH_EVIDENCE and is_conditional_evidence=True)
  4. LightGBM tree-path 3-feature co-occurrence, mined fresh from a
     compact model fit on the top-evidence feature union
  5. residual-conditional third-variable evidence (reused from v3's
     design): for the top STRONG v4 pairs, correlate the pair's own
     joint-fit residual against every other numeric feature
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import fit_additive_joint, predict_joint
from src.models.trees import make_lightgbm

EDGE_CAP = 3000        # explicit, disclosed: top 3000 STRONG edges by incremental R^2,
                        # used to build a tractable subgraph for triangle/chaining motifs
                        # (out of 47,871 STRONG edges -- top ~6.3%)
EDGE_PLUS_HIGH_CAP = 40   # top-N strongest edges paired with HIGH-evidence-conditional features
TREE_UNION_TOP_N = 300    # feature union size for tree-path mining (evidence-ranked)
TREE_TRIPLET_TOP_N = 150
RESIDUAL_TOP_PAIRS = 15
RESIDUAL_THIRDVARS_PER_PAIR = 3


def compute_tree_triplet_counts(model, feature_names):
    from itertools import combinations
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
    conf = pd.read_csv(f"{R}/stage2_runB_v4_confirmed_2way_interactions.csv")
    eg = pd.read_csv(f"{R}/stage2_runB_v4_feature_evidence_graph.csv").set_index("raw_csv_parameter_name")
    strong_mod = conf[conf["evidence_label"].isin(["STRONG", "MODERATE"])]
    strong = conf[conf["evidence_label"] == "STRONG"].sort_values("interaction_incremental_r2", ascending=False)
    print(f"v4 confirmed edges available: {len(strong_mod)} STRONG+MODERATE ({len(strong)} STRONG) "
          f"over {len(set(strong_mod.parameter_A)|set(strong_mod.parameter_B))} nodes")

    subgraph_edges = strong.head(EDGE_CAP)
    print(f"Motif subgraph: top {len(subgraph_edges)} STRONG edges by incremental R^2 (of {len(strong)} STRONG total)")

    adjacency: dict[str, set] = {}
    edge_set = set()
    for a, b in zip(subgraph_edges["parameter_A"], subgraph_edges["parameter_B"]):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        edge_set.add(frozenset([a, b]))

    triples: set[tuple] = set()
    channel_of: dict[tuple, set] = {}

    def add_triple(t, label):
        t = tuple(sorted(t))
        triples.add(t)
        channel_of.setdefault(t, set()).add(label)

    # --- Channel 1: triangles in the subgraph ---
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
    print(f"Channel 1 (triangles in top-{EDGE_CAP}-edge subgraph): {n_triangles} triangles")

    # --- Channel 2: two edges sharing a node (chaining, no triangle required) ---
    n_chain = 0
    for node, neighbors in adjacency.items():
        neighbors_l = sorted(neighbors)
        for i in range(len(neighbors_l)):
            for j in range(i + 1, len(neighbors_l)):
                b, c = neighbors_l[i], neighbors_l[j]
                add_triple((node, b, c), "2_shared_node_chaining")
                n_chain += 1
    print(f"Channel 2 (shared-node chaining, third edge not required): {n_chain} triples generated "
          f"(cumulative unique so far: {len(triples)})")

    # --- Channel 3: top edge + third HIGH_EVIDENCE+conditional feature ---
    high_cond = eg[(eg["evidence_tier"] == "HIGH_EVIDENCE") & (eg["is_conditional_evidence"])].index.tolist()
    top_edges_for_ch3 = strong.head(EDGE_PLUS_HIGH_CAP)
    n_ch3 = 0
    for a, b in zip(top_edges_for_ch3["parameter_A"], top_edges_for_ch3["parameter_B"]):
        for h in high_cond:
            if h not in (a, b):
                add_triple((a, b, h), "3_edge_plus_high_conditional")
                n_ch3 += 1
    print(f"Channel 3 (top {EDGE_PLUS_HIGH_CAP} edges x {len(high_cond)} HIGH+conditional features): {n_ch3} triples")

    # --- Channel 4: fresh tree-path 3-way co-occurrence on top-evidence feature union ---
    order_cols = ["v3_confirmed_best_incremental_r2"]
    eg_sorted = eg.copy()
    eg_sorted["_score"] = eg_sorted["n_independent_methods_supporting"].fillna(0)
    top_evidence_feats = eg_sorted.sort_values("_score", ascending=False).head(TREE_UNION_TOP_N).index.tolist()
    top_evidence_feats = [f for f in top_evidence_feats if f in ds.name_map]
    model_cols = [ds.name_map[f] for f in top_evidence_feats]
    print(f"Fitting compact model on top {len(top_evidence_feats)} evidence features for tree-triplet mining...")
    tree_model = make_lightgbm(random_state=42)
    tree_model.fit(split.X_train[model_cols], split.y_train)
    triplet_counts_model = compute_tree_triplet_counts(tree_model, model_cols)
    triplet_counts = {tuple(sorted(ds.inverse_name_map[c] for c in trip)): cnt for trip, cnt in triplet_counts_model.items()}
    ranked = sorted(triplet_counts.items(), key=lambda kv: -kv[1])[:TREE_TRIPLET_TOP_N]
    for trip, _ in ranked:
        add_triple(trip, "4_tree_path_cooccurrence")
    print(f"Channel 4 (tree-path co-occurrence, top {TREE_TRIPLET_TOP_N}): {len(ranked)} triples")

    # --- Channel 5: residual-conditional third-variable evidence ---
    y_train_arr = split.y_train.to_numpy()
    numeric_features = ds.numeric_cols_original
    X_all = split.X_train[[ds.name_map[f] for f in numeric_features]].to_numpy(dtype=np.float64)
    top_resid_pairs = strong.head(RESIDUAL_TOP_PAIRS)
    n_ch5 = 0
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
            add_triple((a, b, c), "5_residual_conditional_structure")
            added += 1
            n_ch5 += 1
            if added >= RESIDUAL_THIRDVARS_PER_PAIR:
                break
    print(f"Channel 5 (residual-conditional, top {RESIDUAL_TOP_PAIRS} pairs x {RESIDUAL_THIRDVARS_PER_PAIR} third-vars): {n_ch5} triples")

    rows = []
    for t, labs in channel_of.items():
        a, b, c = t
        rows.append({"parameter_A": a, "parameter_B": b, "parameter_C": c,
                     "channels": "+".join(sorted(labs)), "n_channels": len(labs)})
    df = pd.DataFrame(rows).sort_values("n_channels", ascending=False)
    out_path = f"{R}/stage2_runB_v4_three_way_candidates.csv"
    df.to_csv(out_path, index=False)

    union_feats = sorted(set(df["parameter_A"]) | set(df["parameter_B"]) | set(df["parameter_C"]))
    print(f"\nSaved {out_path} ({len(df)} unique candidate triples, {len(union_feats)} unique features)")
    print("\nPer-channel unique-triple contribution:")
    for label in ["1_triangle", "2_shared_node_chaining", "3_edge_plus_high_conditional",
                  "4_tree_path_cooccurrence", "5_residual_conditional_structure"]:
        cnt = sum(1 for labs in channel_of.values() if label in labs)
        print(f"  {label}: {cnt}")
    print(f"Triples found by >=2 channels: {int((df['n_channels'] >= 2).sum())}")


if __name__ == "__main__":
    main()
