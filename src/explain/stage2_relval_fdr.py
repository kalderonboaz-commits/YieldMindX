"""Relationship Validation Engine -- consolidated FDR accounting.

Reads the already-frozen per-family result files (each of which computed
its own empirical p-value and Benjamini-Hochberg q-value internally,
scoped to that family's own hypothesis set) and assembles one long-format
CSV with [relationship_type, identifier(s), raw_statistic, empirical_p_value,
bh_adjusted_q_value, survives_fdr, family_size] for every hypothesis
tested across all three families, plus prints the required aggregate
counts. No recomputation -- purely a consolidation pass over already-frozen
outputs.
"""
from __future__ import annotations

import pandas as pd

from src.explain.stage2_relval_common import R


def main():
    rows = []

    single = pd.read_csv(f"{R}/stage2_runB_relationship_validation_single.csv")
    for _, r in single.iterrows():
        rows.append({
            "relationship_type": "single_parameter", "parameter_A": r["raw_csv_parameter_name"],
            "parameter_B": None, "parameter_C": None,
            "raw_statistic_delta_r2": r["raw_statistic_delta_r2"],
            "empirical_p_value": r["empirical_p_value"], "bh_adjusted_q_value": r["bh_adjusted_q_value"],
            "survives_fdr": r["survives_fdr"], "family_size": len(single),
        })

    pair = pd.read_csv(f"{R}/stage2_runB_relationship_validation_2way.csv")
    for _, r in pair.iterrows():
        rows.append({
            "relationship_type": "2way", "parameter_A": r["parameter_A"], "parameter_B": r["parameter_B"],
            "parameter_C": None,
            "raw_statistic_delta_r2": r["interaction_strength_delta_r2"],
            "empirical_p_value": r["empirical_p_value"], "bh_adjusted_q_value": r["bh_adjusted_q_value"],
            "survives_fdr": r["survives_fdr"], "family_size": len(pair),
        })

    triple = pd.read_csv(f"{R}/stage2_runB_relationship_validation_3way.csv")
    for _, r in triple.iterrows():
        rows.append({
            "relationship_type": "3way", "parameter_A": r["parameter_A"], "parameter_B": r["parameter_B"],
            "parameter_C": r["parameter_C"],
            "raw_statistic_delta_r2": r["delta_r2"],
            "empirical_p_value": r["empirical_p_value"], "bh_adjusted_q_value": r["bh_adjusted_q_value"],
            "survives_fdr": r["survives_fdr"], "family_size": len(triple),
        })

    df = pd.DataFrame(rows)
    out_path = f"{R}/stage2_runB_relationship_validation_fdr.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} total hypotheses across 3 families)")

    print("\nPer-family FDR accounting (Benjamini-Hochberg, alpha=0.05, computed SEPARATELY per family):")
    for fam in ["single_parameter", "2way", "3way"]:
        sub = df[df["relationship_type"] == fam]
        n_tested = len(sub)
        n_survive = int(sub["survives_fdr"].sum())
        print(f"  {fam}: {n_tested} hypotheses tested, {n_survive} survive FDR ({100*n_survive/n_tested:.1f}%)")

    print(f"\nTOTAL: {len(df)} hypotheses tested across all 3 families, {int(df['survives_fdr'].sum())} survive FDR")


if __name__ == "__main__":
    main()
