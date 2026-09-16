"""Stage 2 Run B v5 -- 3-way candidate generation from the evidence-adaptive
graph (stage2_runB_v5_three_way_graph.csv, built by stage2_v5_three_way_
graph.py). Six channels, none requiring all 3 pairwise edges to exist and
none requiring an edge to be globally top-ranked by magnitude (only to be
IN the evidence-adaptive graph, which already applied quality/hub/
multi-channel admission instead of a magnitude cutoff):

  1. triangles in the v5 graph
  2. shared-node chaining (two edges from the graph sharing a node; the
     third pairwise edge is NOT required to exist)
  3. one graph edge + a third feature that is HIGH or MEDIUM evidence AND
     carries independent conditional-interaction evidence
  4. fresh LightGBM tree-path 3-feature co-occurrence, mined from a
     compact model on the graph's own node set
  5. residual-conditional third-variable evidence (top STRONG v5 edges)
  6. hub-triple interaction-neighborhood motifs: triples of hub features
     (top-decile confirmed-STRONG degree) with at least one pairwise edge
     among them already in the graph
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import fit_additive_joint, predict_joint
from src.models.trees import make_lightgbm

TREE_UNION_TOP_N = 350
TREE_TRIPLET_TOP_N = 200
RESIDUAL_TOP_PAIRS = 20
RESIDUAL_THIRDVARS_PER_PAIR = 3
EDGE_PLUS_COND_CAP = 60  # top-60 graph edges (by quality_score) x HIGH/MEDIUM+conditional features

# The 41,515-edge evidence-adaptive graph is dense enough that UNRESTRICTED
# triangle/chaining/hub-triple enumeration produces ~6.5 MILLION candidate
# triples -- confirmable in principle (each triple costs ~2.85ms) but not
# in practice within one checkpoint (~5 hours). Rather than re-introduce a
# PAIRWISE-EDGE rank cutoff (which is exactly what bottleneck #2 removed),
# each of these three motif channels is bounded by ranking the GENERATED
# TRIPLES themselves by the SUM of their constituent edges' quality_score
# (the same composite, non-magnitude evidence measure used to build the
# graph) -- a triple-level, not edge-level, evidence-adaptive cap.
TRIANGLE_CAP = 5000
CHAIN_CAP = 20000
HUB_TRIPLE_CAP = 3000


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
    graph = pd.read_csv(f"{R}/stage2_runB_v5_three_way_graph.csv")
    eg = pd.read_csv(f"{R}/stage2_runB_v4_feature_evidence_graph.csv").set_index("raw_csv_parameter_name")

    adjacency: dict[str, set] = {}
    edge_quality: dict[frozenset, float] = {}
    for a, b, q in zip(graph["parameter_A"], graph["parameter_B"], graph["quality_score"]):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        edge_quality[frozenset([a, b])] = q
    print(f"v5 3-way graph: {len(graph)} edges, {len(adjacency)} nodes")

    def eq(a, b):
        return edge_quality.get(frozenset([a, b]), 0.0)

    triples: set[tuple] = set()
    channel_of: dict[tuple, set] = {}

    def add_triple(t, label):
        t = tuple(sorted(t))
        triples.add(t)
        channel_of.setdefault(t, set()).add(label)

    # --- Channel 1: triangles, capped by summed edge quality (triple-level, not edge-level) ---
    raw_triangles = []
    nodes = sorted(adjacency.keys())
    for a in nodes:
        neighbors_a = adjacency[a]
        for b in neighbors_a:
            if b <= a:
                continue
            common = neighbors_a & adjacency.get(b, set())
            for c in common:
                if c > b:
                    score = eq(a, b) + eq(a, c) + eq(b, c)
                    raw_triangles.append((score, a, b, c))
    raw_triangles.sort(key=lambda t: -t[0])
    for score, a, b, c in raw_triangles[:TRIANGLE_CAP]:
        add_triple((a, b, c), "1_triangle")
    print(f"Channel 1 (triangles, top {TRIANGLE_CAP} of {len(raw_triangles)} by summed edge quality): "
          f"{min(TRIANGLE_CAP, len(raw_triangles))}")

    # --- Channel 2: shared-node chaining, capped by summed edge quality ---
    raw_chains = []
    for node, neighbors in adjacency.items():
        neighbors_l = sorted(neighbors)
        for i in range(len(neighbors_l)):
            for j in range(i + 1, len(neighbors_l)):
                b, c = neighbors_l[i], neighbors_l[j]
                score = eq(node, b) + eq(node, c)
                raw_chains.append((score, node, b, c))
    raw_chains.sort(key=lambda t: -t[0])
    for score, node, b, c in raw_chains[:CHAIN_CAP]:
        add_triple((node, b, c), "2_shared_node_chaining")
    print(f"Channel 2 (shared-node chaining, top {CHAIN_CAP} of {len(raw_chains)} by summed edge quality): "
          f"{min(CHAIN_CAP, len(raw_chains))} (cumulative unique: {len(triples)})")

    # --- Channel 3: top graph edge + HIGH/MEDIUM+conditional third feature ---
    high_med_cond = eg[(eg["evidence_tier"].isin(["HIGH_EVIDENCE", "MEDIUM_EVIDENCE"])) &
                        (eg["is_conditional_evidence"])].index.tolist()
    top_edges_ch3 = graph.sort_values("quality_score", ascending=False).head(EDGE_PLUS_COND_CAP)
    n_ch3 = 0
    for a, b in zip(top_edges_ch3["parameter_A"], top_edges_ch3["parameter_B"]):
        for h in high_med_cond:
            if h not in (a, b):
                add_triple((a, b, h), "3_edge_plus_high_or_medium_conditional")
                n_ch3 += 1
    print(f"Channel 3 (top {EDGE_PLUS_COND_CAP} edges x {len(high_med_cond)} HIGH/MEDIUM+conditional): {n_ch3}")

    # --- Channel 4: fresh tree-path co-occurrence on the graph's node union ---
    eg_sorted = eg.copy()
    eg_sorted["_score"] = eg_sorted["n_independent_methods_supporting"].fillna(0)
    node_set = set(adjacency.keys())
    top_evidence_feats = eg_sorted[eg_sorted.index.isin(node_set)].sort_values("_score", ascending=False).head(TREE_UNION_TOP_N).index.tolist()
    model_cols = [ds.name_map[f] for f in top_evidence_feats if f in ds.name_map]
    print(f"Fitting compact model on {len(model_cols)} graph-node features for tree-triplet mining...")
    tree_model = make_lightgbm(random_state=42)
    tree_model.fit(split.X_train[model_cols], split.y_train)
    triplet_counts_model = compute_tree_triplet_counts(tree_model, model_cols)
    triplet_counts = {tuple(sorted(ds.inverse_name_map[c] for c in trip)): cnt for trip, cnt in triplet_counts_model.items()}
    ranked = sorted(triplet_counts.items(), key=lambda kv: -kv[1])[:TREE_TRIPLET_TOP_N]
    for trip, _ in ranked:
        add_triple(trip, "4_tree_path_cooccurrence")
    print(f"Channel 4 (tree-path co-occurrence, top {TREE_TRIPLET_TOP_N}): {len(ranked)}")

    # --- Channel 5: residual-conditional third-variable evidence ---
    y_train_arr = split.y_train.to_numpy()
    numeric_features = ds.numeric_cols_original
    X_all = split.X_train[[ds.name_map[f] for f in numeric_features]].to_numpy(dtype=np.float64)
    strong_edges = graph[graph["evidence_label"] == "STRONG"].sort_values("quality_score", ascending=False)
    top_resid_pairs = strong_edges.head(RESIDUAL_TOP_PAIRS)
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
    print(f"Channel 5 (residual-conditional, top {RESIDUAL_TOP_PAIRS} STRONG edges x {RESIDUAL_THIRDVARS_PER_PAIR}): {n_ch5}")

    # --- Channel 6: hub-triple interaction-neighborhood motifs ---
    strong_only = pd.read_csv(f"{R}/stage2_runB_v5_confirmed_2way_interactions.csv")
    strong_only = strong_only[strong_only["evidence_label"] == "STRONG"]
    degree = pd.concat([strong_only["parameter_A"], strong_only["parameter_B"]]).value_counts()
    hub_cutoff = np.percentile(degree.values, 90.0)
    hubs = sorted(degree[degree >= hub_cutoff].index)
    from itertools import combinations
    raw_hub_triples = []
    for h1, h2, h3 in combinations(hubs, 3):
        score = eq(h1, h2) + eq(h1, h3) + eq(h2, h3)  # 0 for any pair with no graph edge
        edges_present = sum([
            h2 in adjacency.get(h1, set()), h3 in adjacency.get(h1, set()), h3 in adjacency.get(h2, set())
        ])
        if edges_present >= 1:
            raw_hub_triples.append((score, h1, h2, h3))
    raw_hub_triples.sort(key=lambda t: -t[0])
    for score, h1, h2, h3 in raw_hub_triples[:HUB_TRIPLE_CAP]:
        add_triple((h1, h2, h3), "6_hub_triple_interaction_neighborhood")
    print(f"Channel 6 (hub-triple motifs, {len(hubs)} hubs, >=1 pairwise edge present, "
          f"top {HUB_TRIPLE_CAP} of {len(raw_hub_triples)} by summed edge quality): "
          f"{min(HUB_TRIPLE_CAP, len(raw_hub_triples))}")

    rows = []
    for t, labs in channel_of.items():
        a, b, c = t
        rows.append({"parameter_A": a, "parameter_B": b, "parameter_C": c,
                     "channels": "+".join(sorted(labs)), "n_channels": len(labs)})
    df = pd.DataFrame(rows).sort_values("n_channels", ascending=False)
    out_path = f"{R}/stage2_runB_v5_three_way_candidates.csv"
    df.to_csv(out_path, index=False)

    union_feats = sorted(set(df["parameter_A"]) | set(df["parameter_B"]) | set(df["parameter_C"]))
    print(f"\nSaved {out_path} ({len(df)} unique candidate triples, {len(union_feats)} unique features)")
    print("Per-channel unique-triple contribution:")
    for label in ["1_triangle", "2_shared_node_chaining", "3_edge_plus_high_or_medium_conditional",
                  "4_tree_path_cooccurrence", "5_residual_conditional_structure", "6_hub_triple_interaction_neighborhood"]:
        cnt = sum(1 for labs in channel_of.values() if label in labs)
        print(f"  {label}: {cnt}")
    print(f"Triples found by >=2 channels: {int((df['n_channels'] >= 2).sum())}")


if __name__ == "__main__":
    main()
