"""Stage 2 Run B -- Ground Truth Validation checkpoint.

Compares the ALREADY-FROZEN, blindly-produced Run B discovery outputs
against the synthetic ground-truth relationships. Reads
documents/corelation_data_table.docx ONLY here, in this dedicated
post-hoc validation script -- never during discovery. Does not retrain,
does not modify the candidate pool, does not touch any discovery output
file.
"""
from __future__ import annotations

import pandas as pd

R = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports"

ROWS = [
    dict(
        id=1, relationship="Gate CD optimum",
        parameters="GATE2_CD_Gate_Leg_Length", expected_type="Nonlinear U-shaped / process window",
        exact_param_in_outputs="no", relationship_type_recovered="no", direction_correct="n/a",
        threshold_or_window_correct="n/a", interaction_found="n/a",
        stability_confidence="n/a -- feature has ZERO SHAP/gain/permutation importance (tied with 146 other unused features, rank 1279/1425)",
        evidence_file="stage2_runB_global_importance.csv",
        classification="MISSED",
        explanation=("The exact named parameter was never used by the LightGBM model at all -- shap_importance="
                     "gain_importance=permutation_importance=0.0 exactly. It never entered the 154-feature "
                     "screening set, the single-parameter table, or the 50-feature interaction candidate pool. "
                     "This is a genuine model-capability gap (a symmetric U-shape gives no single-split gain "
                     "advantage against hundreds of stronger near-linear predictors), not merely a screening-"
                     "threshold artifact -- the FULL 1,425-feature model already found zero use for it."),
    ),
    dict(
        id=2, relationship="PAE benefit",
        parameters="TOPSIN_ET_Load_Pull_P.A.E_Last", expected_type="Direct positive",
        exact_param_in_outputs="yes", relationship_type_recovered="no", direction_correct="not determinable (flat PDP)",
        threshold_or_window_correct="n/a", interaction_found="n/a",
        stability_confidence="INCONCLUSIVE (flat_no_clear_effect shape, but 5/5 fold-consistent as flat -- i.e. consistently NOT captured as monotonic)",
        evidence_file="stage2_runB_global_importance.csv; stage2_runB_single_parameter_relationships.csv",
        classification="PARTIALLY FOUND",
        explanation=("The exact parameter IS present with real, high-agreement individual importance (SHAP rank 54, "
                     "gain rank 113, permutation rank 77 out of 1425 -- a genuine top-8% signal, not noise). This "
                     "confirms the model detects SOME real relationship. However, the PDP-shape classifier labels "
                     "its curve 'flat_no_clear_effect' consistently across all 5 folds -- the expected direct-"
                     "positive SHAPE was NOT recovered by this analysis, despite the feature's importance being "
                     "real and stable. Partial credit: existence confirmed, shape/direction not confirmed."),
    ),
    dict(
        id=3, relationship="Drain-lag penalty",
        parameters="TOPSIN_ET_DC_PIV_Drain_Lag_1", expected_type="Threshold / negative",
        exact_param_in_outputs="no (exact column rank 837/1425, never entered screening or candidate pool)",
        relationship_type_recovered="yes -- but on a different, closely-related column (TOPSIN_ET_DC_PIV_Drain_Lag_0)",
        direction_correct="yes (negative) for the proxy column",
        threshold_or_window_correct="approximately -- proxy column's discovered threshold ~8.89 vs. planted ~7 for the exact column (same order of magnitude, same units, same test family)",
        interaction_found="n/a (this entry is single-parameter)",
        stability_confidence="STRONG for the proxy column (shape 5/5, direction 5/5 folds)",
        evidence_file="stage2_runB_global_importance.csv; stage2_runB_single_parameter_relationships.csv; stage2_runB_thresholds_process_windows.csv",
        classification="APPROXIMATELY FOUND",
        explanation=("The EXACT named column (_1) was essentially unused by the model (rank 837/1425). Its close "
                     "sibling, TOPSIN_ET_DC_PIV_Drain_Lag_0 (same test family, consecutive naming, almost "
                     "certainly measuring the same physical drain-lag phenomenon under a related condition), "
                     "ranks #5/1425 and is confidently, stably (5/5 folds) characterized as a NEGATIVE THRESHOLD "
                     "relationship with a discovered threshold (~8.89) in the same order of magnitude as the "
                     "planted ~7. Correct phenomenon and correct relationship type recovered via a proxy column, "
                     "not the literal named column -- approximately, not exactly, found."),
    ),
    dict(
        id=4, relationship="Gate CD x SiN thickness",
        parameters="GATE2_CD_Gate_Leg_Length + SIN_Thickness_SiN_Thickness", expected_type="2-way interaction",
        exact_param_in_outputs="no (neither parameter entered the 50-feature interaction candidate pool)",
        relationship_type_recovered="no", direction_correct="n/a", threshold_or_window_correct="n/a",
        interaction_found="no -- pair never tested",
        stability_confidence="n/a -- not searched",
        evidence_file="_stage2_runB_interaction_candidate_pool.csv; stage2_runB_interactions.csv (pair absent)",
        classification="NOT TESTABLE UNDER CURRENT METHOD",
        explanation=("Neither parameter entered the interaction candidate pool (GATE2_CD_Gate_Leg_Length has zero "
                     "model importance, as in #1; SIN_Thickness_SiN_Thickness was not in the 154-feature "
                     "screening set at all). Since this project's interaction search only tests pairs drawn from "
                     "the candidate pool, this specific pair was never evaluated -- this is a coverage gap, not a "
                     "negative search result, so 'MISSED' would overstate what was actually tested."),
    ),
    dict(
        id=5, relationship="Rc x Rsh",
        parameters="OHMIC_ET_Rc_an_ct2 + OHMIC_ET_Rsh_an_ct2", expected_type="2-way interaction",
        exact_param_in_outputs="partial -- Rsh yes (rank 15, in pool), Rc no (zero importance, rank 1279, not in pool)",
        relationship_type_recovered="n/a for the interaction itself",
        direction_correct="n/a",
        threshold_or_window_correct="n/a for the interaction; Rsh alone: STRONG threshold/negative single-parameter finding",
        interaction_found="no -- pair never tested (Rc absent from candidate pool)",
        stability_confidence="Rsh single-parameter finding is STRONG (5/5, 5/5); the interaction itself has no evidence at all",
        evidence_file="stage2_runB_single_parameter_relationships.csv (Rsh); _stage2_runB_interaction_candidate_pool.csv; stage2_runB_interactions.csv (pair absent)",
        classification="NOT TESTABLE UNDER CURRENT METHOD",
        explanation=("OHMIC_ET_Rsh_an_ct2 alone is well-characterized (STRONG threshold/negative, rank 15/1425) "
                     "and matches the ground truth's description of Rsh's role. But OHMIC_ET_Rc_an_ct2 has "
                     "exactly zero model importance (same pattern as #1), so it never entered the candidate pool "
                     "and the Rc x Rsh PAIR was never tested. Per the strict grading rule, a strong single-"
                     "parameter finding for one half of a planted interaction does not constitute finding the "
                     "interaction -- this is graded as not testable, not as a partial interaction discovery."),
    ),
    dict(
        id=6, relationship="Leakage threshold",
        parameters="ET leakage family (lkg_*)", expected_type="Threshold",
        exact_param_in_outputs="yes -- family unambiguously present and dominant",
        relationship_type_recovered="yes", direction_correct="yes (negative)",
        threshold_or_window_correct="qualitatively yes (threshold-type curves found); exact numeric thresholds (0.20/0.35 in the ground truth) not individually re-verified column-by-column against different lkg columns' own scales",
        interaction_found="n/a (single-parameter/family relationship)",
        stability_confidence="STRONG for at least one representative (TOPSIN_ET_lkg_5: shape 4/5, direction 4/5); MODERATE for the single most-important overall predictor (MEASURE_ET_lkg_50_at20V: shape 3/5, direction 4/5)",
        evidence_file="stage2_runB_global_importance.csv; stage2_runB_single_parameter_relationships.csv",
        classification="EXACTLY FOUND",
        explanation=("MEASURE_ET_lkg_50_at20V is the single strongest predictor in the ENTIRE 1,425-feature set "
                     "by all three importance methods (SHAP, gain, permutation rank #1 each), and is classified "
                     "threshold-shaped. TOPSIN_ET_lkg_5 (rank 20) is independently classified STRONG "
                     "threshold/negative with full 4/5-fold stability. The leakage family's identity, dominance, "
                     "relationship type (threshold), and direction (negative) are all unambiguously and robustly "
                     "recovered blind -- the clearest, most confident finding in this validation."),
    ),
    dict(
        id=7, relationship="Alignment process window",
        parameters="GATE2_Ali_Litho_Gate_to_recess_alignment", expected_type="Absolute-value threshold / process window",
        exact_param_in_outputs="no (present in the model with modest importance, but never entered the 154-feature screening set)",
        relationship_type_recovered="no", direction_correct="n/a", threshold_or_window_correct="n/a",
        interaction_found="n/a",
        stability_confidence="n/a -- not analyzed beyond raw importance ranking",
        evidence_file="stage2_runB_global_importance.csv",
        classification="MISSED",
        explanation=("The parameter has non-trivial but modest importance (SHAP rank 354, gain rank 279, "
                     "permutation rank 340 out of 1425 -- roughly top-20-25%), with high cross-method agreement. "
                     "This is NOT a zero-importance feature like #1/#5/#8, but it fell outside the ~150-feature "
                     "screening threshold used for deeper PDP-shape/threshold analysis, so no shape, direction, "
                     "or window was ever characterized for it. This is a genuine screening-coverage miss, not a "
                     "model-capability miss."),
    ),
    dict(
        id=8, relationship="Gm stage degradation",
        parameters="TOPSIN_ET_Gm_max, FIC_ET_Gm_max (planted as a derived delta, TOPSIN minus FIC)", expected_type="Stage-to-stage degradation (derived delta)",
        exact_param_in_outputs="partial -- FIC_ET_Gm_max yes (rank 45, in candidate pool); TOPSIN_ET_Gm_max no (zero importance, rank 1279)",
        relationship_type_recovered="no", direction_correct="n/a", threshold_or_window_correct="n/a",
        interaction_found="no -- 3 candidate FIC_ET_Gm_max interaction pairs found only by H-statistic (not SHAP-interaction), 0/5 fold reproducibility, none involve TOPSIN_ET_Gm_max or Drain Lag/PAE",
        stability_confidence="INCONCLUSIVE for all FIC_ET_Gm_max interaction candidates",
        evidence_file="stage2_runB_global_importance.csv; stage2_runB_interactions.csv",
        classification="NOT TESTABLE UNDER CURRENT METHOD",
        explanation=("Stage 2 deliberately excludes all engineered/derived features, so the planted TOPSIN-minus-"
                     "FIC delta has no direct representation. Reconstruction via the two RAW columns would "
                     "require the model to have learned an implicit interaction between them -- but "
                     "TOPSIN_ET_Gm_max has exactly zero importance and never entered any candidate pool, so no "
                     "interaction test involving it was ever possible. FIC_ET_Gm_max alone shows real importance "
                     "but a flat marginal shape and only weak, unreproduced (0/5 folds) interaction candidates "
                     "with unrelated features. Honest conclusion: this pipeline, as built, CANNOT currently "
                     "reconstruct a cross-stage delta relationship when one of the two raw components is "
                     "functionally invisible to the model -- this is a compounding effect of the engineered-"
                     "feature exclusion (by design) and a model-capability gap for TOPSIN_ET_Gm_max specifically."),
    ),
    dict(
        id=9, relationship="Rc x Drain Lag x low PAE",
        parameters="OHMIC_ET_Rc_an_ct2, TOPSIN_ET_DC_PIV_Drain_Lag_1 (or proxy _0), TOPSIN_ET_Load_Pull_P.A.E_Last",
        expected_type="3-way interaction",
        exact_param_in_outputs="partial -- only the Drain-Lag proxy (_0) entered the candidate pool; Rc and PAE did not",
        relationship_type_recovered="no", direction_correct="n/a", threshold_or_window_correct="n/a",
        interaction_found="no -- none of the 3 required pairwise legs (Rc x DrainLag, Rc x PAE, DrainLag x PAE) appear anywhere in the interactions table",
        stability_confidence="n/a -- not searched",
        evidence_file="stage2_runB_interactions.csv (all three required pairs absent); _stage2_runB_interaction_candidate_pool.csv",
        classification="NOT TESTABLE UNDER CURRENT METHOD",
        explanation=("Two of the three required parameters (Rc, PAE) never entered the candidate pool, so none "
                     "of the three pairwise sub-interactions needed even as a partial signal were tested -- "
                     "checked directly against the full interactions table. Beyond the coverage gap, this "
                     "pipeline's interaction search is PAIRWISE ONLY by construction; it has no mechanism to "
                     "directly test a genuine 3-way interaction even when all three legs are available. Per the "
                     "explicit grading instruction, finding only separate 2-way pieces would not earn full "
                     "credit for a 3-way relationship regardless -- and here, not even the 2-way pieces were "
                     "found."),
    ),
]


def main():
    df = pd.DataFrame(ROWS)
    df.to_csv(f"{R}/stage2_runB_ground_truth_validation.csv", index=False)
    print(f"Saved {R}/stage2_runB_ground_truth_validation.csv ({len(df)} rows)")

    counts = df["classification"].value_counts()
    exact = counts.get("EXACTLY FOUND", 0)
    approx = counts.get("APPROXIMATELY FOUND", 0)
    partial = counts.get("PARTIALLY FOUND", 0)
    missed = counts.get("MISSED", 0)
    not_testable = counts.get("NOT TESTABLE UNDER CURRENT METHOD", 0)
    n = len(df)

    strict_rate = exact / n
    practical_rate = (exact + approx) / n

    print(f"\nExact: {exact}/{n} | Approx: {approx}/{n} | Partial: {partial}/{n} | Missed: {missed}/{n} | Not testable: {not_testable}/{n}")
    print(f"Strict recovery rate: {strict_rate:.1%}")
    print(f"Practical recovery rate (exact+approx): {practical_rate:.1%}")

    return df, dict(exact=exact, approx=approx, partial=partial, missed=missed, not_testable=not_testable, n=n,
                     strict_rate=strict_rate, practical_rate=practical_rate)


if __name__ == "__main__":
    main()
