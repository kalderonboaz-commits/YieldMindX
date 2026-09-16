"""Stage 2 Run B MLP -- section 11: comparison against existing Stage 2
methods (LightGBM v1, spline/GAM, v2 univariate/cross-stage, v3/v4/v5
interaction discovery). Comparison only -- reads already-frozen outputs,
no retraining.
"""
from __future__ import annotations

import pandas as pd

from src.explain.stage2_v2_common import R

FOUND = {"STRONG", "MODERATE"}


def main():
    rows = []

    # ---- single-parameter comparison ----
    mlp_sp = pd.read_csv(f"{R}/stage2_runB_mlp_single_parameter_relationships.csv")
    spline = pd.read_csv(f"{R}/stage2_runB_spline_single_parameter.csv").set_index("raw_csv_parameter_name")
    v2u = pd.read_csv(f"{R}/stage2_runB_v2_univariate_all.csv").set_index("raw_csv_parameter_name")

    for _, r in mlp_sp.iterrows():
        f = r["raw_csv_parameter_name"]
        mlp_found = r["evidence_label"] in FOUND
        spline_found = f in spline.index and spline.loc[f, "evidence_label"] in FOUND
        v2_found = f in v2u.index and v2u.loc[f, "evidence_label"] in FOUND
        existing_found = spline_found or v2_found
        if mlp_found and existing_found:
            status = "SUPPORTED_BY_BOTH"
        elif mlp_found and not existing_found:
            status = "MLP_ONLY"
        elif not mlp_found and existing_found:
            status = "EXISTING_ONLY"
        else:
            status = "INCONCLUSIVE_BOTH"
        rows.append({
            "relationship_type": "single_parameter", "parameter_A": f, "parameter_B": None, "parameter_C": None,
            "mlp_evidence_label": r["evidence_label"], "mlp_shape": r["relationship_class"],
            "existing_spline_label": spline.loc[f, "evidence_label"] if f in spline.index else "not_tested",
            "existing_v2_label": v2u.loc[f, "evidence_label"] if f in v2u.index else "not_tested",
            "status": status,
        })

    # ---- 2-way comparison ----
    mlp_2way = pd.read_csv(f"{R}/stage2_runB_mlp_2way_interactions.csv")
    v5_2way = pd.read_csv(f"{R}/stage2_runB_v5_confirmed_2way_interactions.csv")
    v5_pairs_found = set()
    for _, r in v5_2way[v5_2way["evidence_label"].isin(FOUND)].iterrows():
        v5_pairs_found.add(frozenset([r["parameter_A"], r["parameter_B"]]))

    for _, r in mlp_2way.iterrows():
        pair = frozenset([r["parameter_A"], r["parameter_B"]])
        mlp_found = r["confidence"] in FOUND
        existing_found = pair in v5_pairs_found
        if mlp_found and existing_found:
            status = "SUPPORTED_BY_BOTH"
        elif mlp_found and not existing_found:
            status = "MLP_ONLY"
        elif not mlp_found and existing_found:
            status = "EXISTING_ONLY"
        else:
            status = "INCONCLUSIVE_BOTH"
        rows.append({
            "relationship_type": "2way", "parameter_A": r["parameter_A"], "parameter_B": r["parameter_B"], "parameter_C": None,
            "mlp_evidence_label": r["confidence"], "mlp_shape": None,
            "existing_spline_label": None,
            "existing_v2_label": "v5_confirmed" if existing_found else "not_confirmed_in_v5",
            "status": status,
        })

    # ---- 3-way comparison ----
    mlp_3way = pd.read_csv(f"{R}/stage2_runB_mlp_3way_interactions.csv")
    v5_3way = pd.read_csv(f"{R}/stage2_runB_v5_three_way_interactions.csv")
    v5_triples_found = set()
    for _, r in v5_3way[v5_3way["evidence_label"].isin(FOUND)].iterrows():
        v5_triples_found.add(frozenset([r["parameter_A"], r["parameter_B"], r["parameter_C"]]))

    for _, r in mlp_3way.iterrows():
        trip = frozenset([r["parameter_A"], r["parameter_B"], r["parameter_C"]])
        mlp_found = r["confidence"] in FOUND
        existing_found = trip in v5_triples_found
        if mlp_found and existing_found:
            status = "SUPPORTED_BY_BOTH"
        elif mlp_found and not existing_found:
            status = "MLP_ONLY"
        elif not mlp_found and existing_found:
            status = "EXISTING_ONLY"
        else:
            status = "INCONCLUSIVE_BOTH"
        rows.append({
            "relationship_type": "3way", "parameter_A": r["parameter_A"], "parameter_B": r["parameter_B"], "parameter_C": r["parameter_C"],
            "mlp_evidence_label": r["confidence"], "mlp_shape": None,
            "existing_spline_label": None,
            "existing_v2_label": "v5_confirmed" if existing_found else "not_confirmed_in_v5",
            "status": status,
        })

    df = pd.DataFrame(rows)
    out_path = f"{R}/stage2_runB_mlp_vs_existing_methods.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")
    print("\nBy relationship type and status:")
    print(df.groupby(["relationship_type", "status"]).size().to_string())


if __name__ == "__main__":
    main()
