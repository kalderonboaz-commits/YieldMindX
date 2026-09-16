"""Stage 2 Run B -- spline/GAM method vs. existing frozen v1 (LightGBM
SHAP/PDP-based) and v2 (model-free quantile-binning) single-parameter
results. Comparison only -- reads the already-frozen output CSVs, does not
retrain or modify anything. Ground-truth document is never read here; the
purpose is method complementarity, not scoring against known answers.

v1 only screened 154/1,405 features (the ones with meaningful LightGBM
split importance) -- features outside that set are marked
"not_screened_by_v1", which is itself informative (a coverage gap, not a
"not found" verdict). v2 and the spline method both cover all 1,405.

Shape-family mapping (coarse, for cross-method agreement only):
  spline relationship_class -> family
    MONOTONIC_POSITIVE/NEGATIVE          -> monotonic
    THRESHOLD, SATURATION                -> threshold
    U_SHAPED                             -> u_or_window
    INVERTED_U, PROCESS_WINDOW           -> u_or_window
    COMPLEX_NONLINEAR                    -> complex
    FLAT                                 -> flat
    INCONCLUSIVE                         -> inconclusive
  v2 shape -> family
    monotonic -> monotonic; threshold -> threshold; U_shape/inverted_U -> u_or_window;
    complex_nonmonotonic -> complex; flat -> flat; insufficient_data -> inconclusive
  v1 relationship_type -> family
    direct -> monotonic; threshold -> threshold; U_shaped_process_window -> u_or_window;
    nonlinear_complex -> complex; flat_no_clear_effect -> flat
"""
from __future__ import annotations

import pandas as pd

from src.explain.stage2_v2_common import R

FOUND_LABELS = {"STRONG", "MODERATE"}

SPLINE_FAMILY = {
    "MONOTONIC_POSITIVE": "monotonic", "MONOTONIC_NEGATIVE": "monotonic",
    "THRESHOLD": "threshold", "SATURATION": "threshold",
    "U_SHAPED": "u_or_window", "INVERTED_U": "u_or_window", "PROCESS_WINDOW": "u_or_window",
    "COMPLEX_NONLINEAR": "complex", "FLAT": "flat", "INCONCLUSIVE": "inconclusive",
}
V2_FAMILY = {
    "monotonic": "monotonic", "threshold": "threshold", "U_shape": "u_or_window",
    "inverted_U": "u_or_window", "complex_nonmonotonic": "complex", "flat": "flat",
    "insufficient_data": "inconclusive",
}
V1_FAMILY = {
    "direct": "monotonic", "threshold": "threshold", "U_shaped_process_window": "u_or_window",
    "nonlinear_complex": "complex", "flat_no_clear_effect": "flat",
}


def classify_pair(spline_found: bool, other_found: bool, spline_family: str | None, other_family: str | None,
                   other_screened: bool) -> str:
    if not other_screened:
        return "SPLINE_ONLY_NOT_SCREENED_BY_OTHER" if spline_found else "NEITHER_OR_NOT_SCREENED"
    if spline_found and other_found:
        if spline_family is not None and other_family is not None and spline_family == other_family:
            return "BOTH_AGREE_SHAPE"
        return "BOTH_FOUND_DIFFERENT_SHAPE"
    if spline_found and not other_found:
        return "SPLINE_ONLY"
    if other_found and not spline_found:
        return "OTHER_METHOD_ONLY"
    return "NEITHER"


def main():
    spline = pd.read_csv(f"{R}/stage2_runB_spline_single_parameter.csv")
    v1 = pd.read_csv(f"{R}/stage2_runB_single_parameter_relationships.csv")
    v2 = pd.read_csv(f"{R}/stage2_runB_v2_univariate_all.csv")

    v1_idx = v1.set_index("raw_csv_parameter_name")
    v2_idx = v2.set_index("raw_csv_parameter_name")

    rows = []
    for _, r in spline.iterrows():
        f = r["raw_csv_parameter_name"]
        spline_label = r["evidence_label"]
        spline_found = spline_label in FOUND_LABELS
        spline_family = SPLINE_FAMILY.get(r["relationship_class"])

        v1_screened = f in v1_idx.index
        v1_label = v1_idx.loc[f, "evidence_label"] if v1_screened else "not_screened_by_v1"
        v1_type = v1_idx.loc[f, "relationship_type"] if v1_screened else None
        v1_found = v1_screened and v1_label in FOUND_LABELS
        v1_family = V1_FAMILY.get(v1_type) if v1_screened else None

        v2_label = v2_idx.loc[f, "evidence_label"] if f in v2_idx.index else "not_screened_by_v2"
        v2_shape = v2_idx.loc[f, "shape"] if f in v2_idx.index else None
        v2_found = v2_label in FOUND_LABELS
        v2_family = V2_FAMILY.get(v2_shape)

        rows.append({
            "raw_csv_parameter_name": f,
            "spline_relationship_class": r["relationship_class"],
            "spline_evidence_label": spline_label,
            "spline_nonlinear_advantage": r["nonlinear_advantage"],
            "v1_relationship_type": v1_type,
            "v1_evidence_label": v1_label,
            "v2_shape": v2_shape,
            "v2_evidence_label": v2_label,
            "vs_v1_lightgbm": classify_pair(spline_found, v1_found, spline_family, v1_family, v1_screened),
            "vs_v2_binned": classify_pair(spline_found, v2_found, spline_family, v2_family, True),
        })

    df = pd.DataFrame(rows)
    out_path = f"{R}/stage2_runB_spline_vs_lightgbm.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")

    print("\n--- vs. v1 (LightGBM SHAP/PDP, 154/1405 screened) ---")
    print(df["vs_v1_lightgbm"].value_counts().to_string())
    print("\n--- vs. v2 (model-free quantile-binning, 1405/1405 screened) ---")
    print(df["vs_v2_binned"].value_counts().to_string())

    print("\nFeatures found ONLY by spline (both comparisons), sample:")
    only_spline = df[(df["vs_v1_lightgbm"].isin(["SPLINE_ONLY", "SPLINE_ONLY_NOT_SCREENED_BY_OTHER"])) &
                      (df["vs_v2_binned"] == "SPLINE_ONLY")]
    print(f"  count: {len(only_spline)}")
    print(only_spline.head(15)[["raw_csv_parameter_name", "spline_relationship_class", "spline_evidence_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
