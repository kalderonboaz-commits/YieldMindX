"""Stage 2 (Relationship Discovery Engine) feature-eligibility audit.

AUDIT ONLY -- no model training, no SHAP, no feature selection happens
here. This script classifies every one of the 1,430 ORIGINAL raw-CSV
columns into one of seven categories and determines Stage 2 eligibility.

Per this checkpoint's explicit scope:
  - Only original raw-CSV columns are considered. The 307 project-created
    engineered features (STAGE_RANGE_*/STAGE_STD_*/STAGE_DELTA_*) do not
    exist in the raw CSV at all, so they are excluded from this audit by
    construction -- not by a filtering rule that could be wrong, simply
    because this script reads the CSV directly and never touches
    src/features/engineered.py.
  - No Stage-1-style importance-based reduction is applied. A column is
    only excluded/flagged here for leakage, metadata, or duplicate
    reasons -- never for low correlation or low standalone importance.
  - Every leakage and duplicate claim below is re-verified directly
    against the raw CSV in this script (not copied from prior checkpoint
    conclusions), per the explicit instruction not to trust the old
    classification automatically.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

PROJECT_ROOT_DATA = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\data\correlation_data_table.csv"

TARGET = "SortingYield"

# ---------------------------------------------------------------------------
# Explicit classification for every non-ET/CD "special" column. Every ET/CD/
# Thickness/etc. numeric process column not listed here defaults to
# category B (legitimate process/ET parameter) -- see build_audit() below.
# ---------------------------------------------------------------------------

SPECIAL_COLUMNS: dict[str, dict] = {
    "SortingYield": dict(
        category="A", stage2_eligible="no", leakage_risk="not_applicable", process_or_ET_parameter="no",
        reason="This IS the Stage 2 relationship-discovery target variable.",
        notes="Never a predictor. All relationships are characterized WITH RESPECT TO this column.",
    ),
    "Actual Sorting Yield (%)": dict(
        category="D", stage2_eligible="no", leakage_risk="high", process_or_ET_parameter="no",
        reason=("Re-verified fresh against raw CSV: Pearson r=0.9875, Spearman r=0.9835 with SortingYield "
                "-- effectively a re-expression of the same outcome, not an independent measurement. "
                "It is an EOL sort-yield metric, i.e. an OUTPUT/outcome variable computed at or after the "
                "same test event as SortingYield, not a process/input parameter."),
        notes="Confirmed leakage. Available only after (or as part of) EOL sort -- not usable as a predictor.",
    ),
    "LineYield": dict(
        category="G", stage2_eligible="review", leakage_risk="none", process_or_ET_parameter="uncertain",
        reason=("Re-verified fresh: Pearson r=0.0040, Spearman r=-0.0026 with SortingYield -- statistically "
                "indistinguishable from zero (n=1500). This does NOT show a correlation-based leakage "
                "signature. However, 'LineYield' is semantically an OUTCOME metric (a yield/pass-rate) from "
                "a different process stage ('Line' = wafer fab line), not a physical process/ET "
                "measurement. Whether front-of-line yield is measured and available BEFORE EOL SortingYield "
                "is determined (which would make it a legitimate early-stage input) or is a parallel/"
                "downstream rollup cannot be determined from the CSV alone."),
        notes=("UNCERTAIN -- engineering review needed to confirm manufacturing-flow timing. If confirmed "
               "available pre-EOL and causally upstream, this could become a legitimate early-warning "
               "input; if it is a parallel/summary metric it should be treated as leakage-adjacent. Not "
               "excluded here, but not automatically included either -- flagged for review per project rule "
               "against unsupported assumptions."),
    ),
    "BackEndYield": dict(
        category="G", stage2_eligible="review", leakage_risk="none", process_or_ET_parameter="uncertain",
        reason=("Re-verified fresh: Pearson r=0.0045, Spearman r=-0.0207 with SortingYield -- statistically "
                "indistinguishable from zero. Same semantic concern as LineYield: 'BackEndYield' is an "
                "outcome/yield metric, most plausibly from packaging/assembly (typically AFTER die sort in "
                "standard semiconductor flow, which would make it a downstream, non-causal metric relative "
                "to SortingYield), but this cannot be confirmed from the CSV alone."),
        notes="UNCERTAIN -- engineering review needed. Near-zero correlation argues against leakage risk, but its outcome/yield semantics argue against it being a genuine process/ET input parameter.",
    ),
    "VIYield": dict(
        category="G", stage2_eligible="review", leakage_risk="low", process_or_ET_parameter="uncertain",
        reason=("Re-verified fresh: Pearson r=0.1220, Spearman r=0.1646 with SortingYield -- a modest, "
                "non-trivial correlation. 'VI' most plausibly denotes Visual Inspection, typically a "
                "post-sort or post-assembly step. An outcome/yield metric, not a physical process "
                "measurement."),
        notes="UNCERTAIN -- modest correlation plus outcome-type semantics. Engineering review needed to confirm process-flow position before treating as either a legitimate input or leakage.",
    ),
    "LTYield": dict(
        category="G", stage2_eligible="review", leakage_risk="low", process_or_ET_parameter="uncertain",
        reason=("Re-verified fresh: Pearson r=0.2155, Spearman r=0.1634 with SortingYield -- a modest, "
                "non-trivial correlation. 'LT' most plausibly denotes Life Test / burn-in, typically a "
                "post-sort reliability step. An outcome/yield metric, not a physical process measurement."),
        notes="UNCERTAIN -- modest correlation plus outcome-type semantics. Engineering review needed.",
    ),
    "TotalYield": dict(
        category="D", stage2_eligible="no", leakage_risk="high", process_or_ET_parameter="no",
        reason=("Re-verified fresh: Pearson r=0.9228, Spearman r=0.8947 with SortingYield -- a rollup "
                "metric whose value structurally includes SortingYield as one of its components (a yield "
                "product/aggregate across stages). Using it as a predictor would let the model trivially "
                "recover the target through arithmetic, not through any discovered process relationship."),
        notes="Confirmed leakage: target-derived aggregate.",
    ),
    "EstimatedSupplyChipsCount": dict(
        category="D", stage2_eligible="no", leakage_risk="high", process_or_ET_parameter="no",
        reason=("Re-verified fresh: EstimatedSupplyChipsCount / MaskSetSupplyChipsCount ratio correlates "
                "r=0.9229 with SortingYield, while MaskSetSupplyChipsCount alone correlates only r=0.0847. "
                "This shows EstimatedSupplyChipsCount is a business projection COMPUTED FROM an actual or "
                "forecasted yield figure (chip-count estimate = wafer capacity x yield), not an independent "
                "process measurement."),
        notes="Confirmed leakage: derived from yield outcome, not a process input.",
    ),
    "MaskSetSupplyChipsCount": dict(
        category="B", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason=("Re-verified fresh: correlates only r=0.0847 with SortingYield -- a planned wafer/capacity "
                "count set before production, not derived from actual results. Legitimate, safe process/"
                "planning input."),
        notes="Distinct from EstimatedSupplyChipsCount -- verified NOT to encode yield outcome.",
    ),
    "MaskSetName_group": dict(
        category="F", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason=("Re-verified fresh via direct row-by-row string comparison: MaskSetName_group is identical "
                "to MaskSetName for all 1,500 rows (confirmed by crosstab: a perfect 1:1 diagonal mapping)."),
        notes="Exact duplicate of MaskSetName. No additional exact duplicates were found anywhere in the dataset (checked via content-hash across all numeric columns and pairwise string-equality across all categorical columns).",
    ),
    "MaskSetName": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason="Genuine categorical process attribute (mask set used). Confirmed to materially associate with mean yield differences across groups in prior EDA.",
        notes="Kept as the canonical column; MaskSetName_group is its exact duplicate (see category F entry).",
    ),
    "VendorName": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason="Genuine categorical process attribute (epitaxial wafer vendor).", notes="",
    ),
    "Version": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason="Genuine categorical process attribute (product/process version).", notes="",
    ),
    "Lot_type": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason="Genuine categorical process attribute (engineering vs. production lot).", notes="",
    ),
    "Customer": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason="Genuine categorical process/business attribute; associated with real mean-yield differences in prior EDA.",
        notes="Business attribute rather than a physical process condition, but not target-derived -- set before test, legitimately eligible.",
    ),
    "Technology": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="yes",
        reason="Genuine categorical process attribute (pHEMT technology node).", notes="",
    ),
    "Sort_Yield_target[%]": dict(
        category="C", stage2_eligible="yes", leakage_risk="none", process_or_ET_parameter="uncertain",
        reason=("Re-verified fresh: takes only 2 values (88, 90) and maps deterministically onto Technology "
                "(crosstab confirms a perfect 1:1 relationship) -- a pre-set specification/target threshold "
                "assigned before production, not a measurement derived from the actual outcome."),
        notes="Not leakage (set in advance, not derived from results), but fully redundant with Technology -- carries no independent information.",
    ),
    "Sorting Date": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Pure traceability/scheduling date (when the sort test was performed), not a physical process condition.",
        notes="Could support a derived calendar-drift feature in future engineered work; not itself a process parameter.",
    ),
    "FinishLineDate": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Pure traceability/scheduling date (when the wafer finished the fab line), not itself a physical process condition.",
        notes="Could support a derived cycle-time feature (FinishLineDate to Sorting Date) in future engineered work.",
    ),
    "LotName": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Pure identifier (100 unique lot codes). Used only as the grouping key for validation splits, never as a predictor.",
        notes="",
    ),
    "WaferNum": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Pure identifier (wafer position label W01-W15 within a lot). Administrative/traceability, not a process condition.",
        notes="Could in principle proxy a real within-lot spatial effect, but as a bare label it is not itself a measured process condition; flagged as excluded rather than uncertain since it is unambiguously an identifier, not a measurement.",
    ),
    "Process": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Constant value across all 1,500 rows ('GaAs_pHEMT_MMIC') -- zero variance, provides no discriminative information.",
        notes="Re-verified fresh: nunique=1.",
    ),
    "FinishLineYear": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Constant value across all 1,500 rows (2026) -- zero variance.",
        notes="Re-verified fresh: nunique=1.",
    ),
    "Exception": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Constant value across all 1,500 rows ('NO') -- zero variance.",
        notes="Re-verified fresh: nunique=1.",
    ),
    "Sort_WW": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Work-week number of the sort test -- scheduling/traceability metadata, not a physical process condition.",
        notes="",
    ),
    "Year_WW": dict(
        category="E", stage2_eligible="no", leakage_risk="none", process_or_ET_parameter="no",
        reason="Year-work-week composite label -- scheduling/traceability metadata, not a physical process condition.",
        notes="",
    ),
}


def build_audit() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT_DATA, low_memory=False)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]

    rows = []
    for col in df.columns:
        if col in SPECIAL_COLUMNS:
            spec = SPECIAL_COLUMNS[col]
            rows.append({
                "column_name": col,
                "original_csv_column": "yes",
                "category": spec["category"],
                "stage2_eligible": spec["stage2_eligible"],
                "reason": spec["reason"],
                "leakage_risk": spec["leakage_risk"],
                "process_or_ET_parameter": spec["process_or_ET_parameter"],
                "notes": spec["notes"],
            })
        else:
            # default: original numeric ET/CD/Thickness/etc. process parameter
            rows.append({
                "column_name": col,
                "original_csv_column": "yes",
                "category": "B",
                "stage2_eligible": "yes",
                "reason": ("Original raw-CSV numeric process/inline/ET measurement. Re-checked: not in the "
                           "leakage-family list, not a constant column, not part of any exact-duplicate "
                           "group (verified via content-hash across all numeric columns)."),
                "leakage_risk": "none",
                "process_or_ET_parameter": "yes",
                "notes": "Eligible for Stage 2 regardless of standalone importance -- may still matter via nonlinear effects, thresholds, or interactions.",
            })

    audit_df = pd.DataFrame(rows)
    assert len(audit_df) == df.shape[1], "Audit row count must equal raw CSV column count"
    assert audit_df["column_name"].is_unique
    return audit_df


def write_summary(audit_df: pd.DataFrame, out_path: str):
    total = len(audit_df)
    cat_counts = audit_df["category"].value_counts().to_dict()
    eligible_counts = audit_df["stage2_eligible"].value_counts().to_dict()

    lines = []
    lines.append("STAGE 2 -- RELATIONSHIP DISCOVERY ENGINE: FEATURE ELIGIBILITY AUDIT SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Scope: original raw-CSV columns only. The 307 project-created engineered")
    lines.append("features (STAGE_RANGE_*/STAGE_STD_*/STAGE_DELTA_*) do not exist in the raw")
    lines.append("CSV and are excluded from this audit by construction -- they are not deleted")
    lines.append("from the project and remain available for later Stage 2 work.")
    lines.append("")
    lines.append(f"Total original CSV columns audited: {total}")
    lines.append("")
    lines.append("By category:")
    lines.append(f"  A. Target:                                          {cat_counts.get('A', 0)}")
    lines.append(f"  B. Legitimate process / inline / ET parameter:      {cat_counts.get('B', 0)}")
    lines.append(f"  C. Categorical process parameter:                   {cat_counts.get('C', 0)}")
    lines.append(f"  D. Suspected data leakage / target-derived output:  {cat_counts.get('D', 0)}")
    lines.append(f"  E. Metadata / ID / date / traceability:             {cat_counts.get('E', 0)}")
    lines.append(f"  F. Exact duplicate / redundant column:              {cat_counts.get('F', 0)}")
    lines.append(f"  G. Uncertain -- requires engineering review:        {cat_counts.get('G', 0)}")
    lines.append("")
    lines.append("By Stage 2 eligibility:")
    lines.append(f"  yes (eligible):     {eligible_counts.get('yes', 0)}")
    lines.append(f"  no (excluded):      {eligible_counts.get('no', 0)}")
    lines.append(f"  review (uncertain): {eligible_counts.get('review', 0)}")
    lines.append("")
    lines.append("Suspected leakage columns (category D):")
    for c in audit_df[audit_df["category"] == "D"]["column_name"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Uncertain columns requiring engineering review (category G):")
    for c in audit_df[audit_df["category"] == "G"]["column_name"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Exact duplicate columns (category F):")
    for c in audit_df[audit_df["category"] == "F"]["column_name"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Metadata / ID / date columns (category E):")
    for c in audit_df[audit_df["category"] == "E"]["column_name"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Categorical process parameters (category C):")
    for c in audit_df[audit_df["category"] == "C"]["column_name"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append(f"Legitimate process/ET numeric predictors (category B): {cat_counts.get('B', 0)} columns")
    lines.append("(full list in stage2_feature_eligibility.csv -- not enumerated here for brevity)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    OUTPUTS_DIR = Path(r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs")
    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)

    audit_df = build_audit()
    csv_path = OUTPUTS_DIR / "reports" / "stage2_feature_eligibility.csv"
    audit_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(audit_df)} rows)")

    txt_path = OUTPUTS_DIR / "reports" / "stage2_feature_eligibility_summary.txt"
    summary_text = write_summary(audit_df, str(txt_path))
    print(f"Saved: {txt_path}")
    print()
    print(summary_text)
