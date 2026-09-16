"""Stage 2 Run B v5 -- fixes bottleneck #2 identified by the v4 ground-
truth validation: 3-way motif generation was built from only the top 3,000
CONFIRMED edges by raw incremental-R^2 MAGNITUDE, which excluded genuinely
real, STRONG, unanimously-fold-reproducible edges (e.g. Rc x Drain_Lag_1,
STRONG/5-5, r^2=0.0464) purely because their effect size ranked outside
the top 3,000 among ~60,000+ STRONG edges -- a magnitude-rank blind spot,
not a quality problem.

Replaces the single magnitude-rank cutoff with a QUALITY-based composite
admission rule plus two supplementary evidence-driven rules that do not
depend on magnitude rank at all:

  G1. composite quality score (evidence_label + fold reproducibility +
      candidate-channel corroboration + endpoint evidence-tier strength),
      top-K by this composite -- NOT by raw incremental R^2 -- so a
      unanimous-fold, multi-channel-corroborated, HIGH-endpoint edge with a
      modest effect size ranks ABOVE a single-channel, barely-4/5-fold
      edge with a larger but less-reproducible effect size
  G2. ALL STRONG/MODERATE edges touching a "hub" feature (top decile of
      features by confirmed-STRONG degree) -- guarantees a major
      interaction hub's real edges are never excluded purely by magnitude,
      directly targeting the v4 failure mode
  G3. ALL edges supported by >=2 independent pair candidate-generation
      channels (multi-channel corroboration), regardless of magnitude rank

This uses v5's CONFIRMED 2-way results (which already include v4's
preserved edges plus the newly-confirmed MEDIUM x MEDIUM edges from
bottleneck-#1's fix) -- unchanged confirmation logic, only the GRAPH built
from it is redesigned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import R

G1_TOP_K = 6000
HUB_PERCENTILE = 90.0  # top decile of features by confirmed-STRONG degree = "hub"


def main():
    conf = pd.read_csv(f"{R}/stage2_runB_v5_confirmed_2way_interactions.csv")
    strong_mod = conf[conf["evidence_label"].isin(["STRONG", "MODERATE"])].copy()
    print(f"v5 confirmed STRONG+MODERATE edges available: {len(strong_mod)} "
          f"over {len(set(strong_mod.parameter_A)|set(strong_mod.parameter_B))} nodes")

    # --- composite quality score (NOT magnitude-based) ---
    tier_pts = {"HIGH_EVIDENCE": 1.0, "MEDIUM_EVIDENCE": 0.5, "LOW_EVIDENCE": 0.2, "UNSUPPORTED": 0.0}
    def fold_frac(s):
        try:
            num, den = s.split("/")
            return int(num) / max(int(den), 1)
        except Exception:
            return 0.0
    def n_channels(s):
        return len(str(s).split("+"))

    strong_mod["_conf_pts"] = strong_mod["evidence_label"].map({"STRONG": 3.0, "MODERATE": 2.0})
    strong_mod["_fold_pts"] = strong_mod["fold_reproducibility"].apply(fold_frac)
    strong_mod["_channel_pts"] = strong_mod["channels"].apply(n_channels).clip(upper=3) * 0.5
    strong_mod["_endpoint_pts"] = (strong_mod["evidence_tier_A"].map(tier_pts).fillna(0.3) +
                                    strong_mod["evidence_tier_B"].map(tier_pts).fillna(0.3))
    strong_mod["quality_score"] = (strong_mod["_conf_pts"] + strong_mod["_fold_pts"] +
                                    strong_mod["_channel_pts"] + strong_mod["_endpoint_pts"])

    # --- G1: top-K by composite quality (not magnitude) ---
    g1 = strong_mod.sort_values("quality_score", ascending=False).head(G1_TOP_K)
    g1_edges = set(zip(g1["parameter_A"], g1["parameter_B"]))
    print(f"G1 (top {G1_TOP_K} by composite quality score, of {len(strong_mod)}): {len(g1_edges)} edges")

    # --- G2: all STRONG/MODERATE edges touching a hub feature ---
    strong_only = conf[conf["evidence_label"] == "STRONG"]
    degree = pd.concat([strong_only["parameter_A"], strong_only["parameter_B"]]).value_counts()
    hub_cutoff = np.percentile(degree.values, HUB_PERCENTILE)
    hubs = set(degree[degree >= hub_cutoff].index)
    print(f"Hub features (top decile by confirmed-STRONG degree, degree>={hub_cutoff:.0f}): {len(hubs)}")
    g2_mask = strong_mod["parameter_A"].isin(hubs) | strong_mod["parameter_B"].isin(hubs)
    g2 = strong_mod[g2_mask]
    g2_edges = set(zip(g2["parameter_A"], g2["parameter_B"]))
    print(f"G2 (all STRONG/MODERATE edges touching a hub feature): {len(g2_edges)} edges")

    # --- G3: all edges with >=2 independent candidate-generation channels ---
    g3 = strong_mod[strong_mod["channels"].apply(n_channels) >= 2]
    g3_edges = set(zip(g3["parameter_A"], g3["parameter_B"]))
    print(f"G3 (multi-channel-corroborated, >=2 channels): {len(g3_edges)} edges")

    union_edges = g1_edges | g2_edges | g3_edges
    print(f"\nFinal v5 3-way graph: {len(union_edges)} edges (union of G1/G2/G3), "
          f"vs. v4's fixed top-3,000-by-magnitude ({len(union_edges)/3000:.2f}x)")

    rows = []
    edge_rule = {}
    for a, b in g1_edges:
        edge_rule.setdefault((a, b), set()).add("G1_composite_quality_topK")
    for a, b in g2_edges:
        edge_rule.setdefault((a, b), set()).add("G2_hub_feature_edge")
    for a, b in g3_edges:
        edge_rule.setdefault((a, b), set()).add("G3_multichannel_corroborated")

    strong_mod_idx = strong_mod.set_index(["parameter_A", "parameter_B"])
    for (a, b), rules in edge_rule.items():
        try:
            row = strong_mod_idx.loc[(a, b)]
        except KeyError:
            continue
        rows.append({
            "parameter_A": a, "parameter_B": b,
            "evidence_label": row["evidence_label"], "interaction_incremental_r2": row["interaction_incremental_r2"],
            "fold_reproducibility": row["fold_reproducibility"], "quality_score": round(row["quality_score"], 3),
            "evidence_tier_A": row["evidence_tier_A"], "evidence_tier_B": row["evidence_tier_B"],
            "admission_rule": "+".join(sorted(rules)),
        })
    df = pd.DataFrame(rows)
    out_path = f"{R}/stage2_runB_v5_three_way_graph.csv"
    df.to_csv(out_path, index=False)

    nodes = set(df["parameter_A"]) | set(df["parameter_B"])
    max_possible_edges = len(nodes) * (len(nodes) - 1) / 2
    density = len(df) / max_possible_edges if max_possible_edges else 0
    print(f"\nSaved {out_path} ({len(df)} edges, {len(nodes)} nodes, density={density:.4f})")
    print("\nAdmission-rule breakdown:")
    print(df["admission_rule"].apply(lambda s: s).value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
