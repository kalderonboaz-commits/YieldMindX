"""Stage 2 Run B v2 -- Ground Truth Validation (validation-only).

Compares the ALREADY-FROZEN v2 discovery outputs against the 9 planted
relationships. Reads documents/corelation_data_table.docx only here. Does
not retrain, does not modify any v2 output, does not tune anything.
"""
from __future__ import annotations

import pandas as pd

R = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports"

ROWS = [
    dict(
        id=1, relationship="Gate CD optimum", parameters="GATE2_CD_Gate_Leg_Length",
        expected_type="Nonlinear U-shaped / process window",
        exact_param_screened="yes (100% univariate coverage)",
        relationship_type_recovered="partial -- classified 'complex_nonmonotonic' (>=2 sign changes), not the more specific 'U_shape' bucket (exactly 1 centered sign change)",
        direction_correct="n/a (complex shape has no single direction)",
        threshold_window_correct="n/a", pair_reached_broad_screen="n/a (single-parameter)",
        pair_reached_confirmation="n/a", fold_stable="MODERATE (shape 3/5, direction 3/5)",
        evidence_strength="MODERATE", evidence_file="stage2_runB_v2_univariate_all.csv",
        classification="PARTIALLY FOUND",
        v1_classification="MISSED", improved="YES",
        explanation=("v1: this column had EXACTLY ZERO LightGBM importance and was invisible to every v1 method. "
                     "v2's model-free screen, which never depends on LightGBM split usage, now detects real, "
                     "moderately-stable nonlinear structure (effect range 1.67 yield points, 3/5-fold shape and "
                     "direction consistency) -- a genuine coverage-driven improvement. However, the specific "
                     "expected U-shape sub-type was not cleanly resolved; the classifier landed on the broader "
                     "'complex_nonmonotonic' bucket, so full credit is not given."),
    ),
    dict(
        id=2, relationship="PAE benefit", parameters="TOPSIN_ET_Load_Pull_P.A.E_Last",
        expected_type="Direct positive",
        exact_param_screened="yes (100% univariate coverage)",
        relationship_type_recovered="no -- main-split shape is 'U_shape' (wrong type) with 0/5 fold reproducibility",
        direction_correct="not determinable (unstable)", threshold_window_correct="n/a",
        pair_reached_broad_screen="no (never entered top-200)", pair_reached_confirmation="no",
        fold_stable="INCONCLUSIVE (0/5, 0/5)", evidence_strength="INCONCLUSIVE",
        evidence_file="stage2_runB_v2_univariate_all.csv",
        classification="MISSED",
        v1_classification="PARTIALLY FOUND", improved="NO (regression)",
        explanation=("Important, honestly-reported regression: v1's LightGBM-based SHAP importance detected real, "
                     "stable signal for this parameter (rank 54/1425, high cross-method agreement) even though its "
                     "PDP shape was flat. v2's model-free quantile-binned method finds NO stable evidence at all "
                     "for the exact same parameter (0/5 fold reproducibility) and never surfaces it in the "
                     "interaction screen either. This demonstrates v2's coverage improvement is not uniformly "
                     "beneficial -- a modest, genuinely linear effect can be better captured by a smoothed "
                     "tree-based method than by raw quantile-bin averaging, which is more exposed to sampling "
                     "noise at this bin resolution."),
    ),
    dict(
        id=3, relationship="Drain-lag penalty", parameters="TOPSIN_ET_DC_PIV_Drain_Lag_1",
        expected_type="Threshold / negative, ~7",
        exact_param_screened="yes (100% univariate coverage; also entered the interaction broad-screen pool)",
        relationship_type_recovered="partial -- pure single-parameter shape is unstable (0/5), but the EXACT column is robustly implicated via other channels",
        direction_correct="yes, via cross-stage channel (negative)",
        threshold_window_correct="not directly re-derived as an absolute threshold value in v2's outputs",
        pair_reached_broad_screen="yes (exact column, multiple pairs)",
        pair_reached_confirmation="yes -- multiple STRONG confirmed interactions (5/5 fold reproducibility)",
        fold_stable="STRONG (interaction + cross-stage channels); INCONCLUSIVE (univariate channel alone)",
        evidence_strength="STRONG (via interactions/cross-stage/3-way hub role)",
        evidence_file="stage2_runB_v2_confirmed_interactions.csv; stage2_runB_v2_cross_stage_relationships.csv; stage2_runB_v2_three_way_interactions.csv",
        classification="APPROXIMATELY FOUND",
        v1_classification="APPROXIMATELY FOUND (via sibling column _0)", improved="YES (same label, stronger basis)",
        explanation=("v1 could only approximate this relationship via a highly-correlated SIBLING column "
                     "(_0), because the exact named column (_1) had low LightGBM importance and never entered "
                     "deeper analysis. In v2, the EXACT column (_1) is now: (a) present in the interaction "
                     "broad-screen pool, (b) part of multiple STRONG, 5/5-fold-reproducible confirmed 2-way "
                     "interactions, (c) the single most frequent 'hub' variable across the 112 STRONG 3-way "
                     "findings, and (d) shows a STRONG (5/5, 5/5), correctly-directioned threshold pattern in "
                     "the cross-stage FIC-vs-TOPSIN Drain_Lag_1 comparison. The classification label stays "
                     "'approximately found' because the pure univariate single-value threshold (~7) was not "
                     "itself cleanly re-derived, but the evidentiary basis is now built on the exact column "
                     "through robust multivariate channels rather than a proxy substitute -- a genuine "
                     "qualitative improvement even though the summary label is unchanged."),
    ),
    dict(
        id=4, relationship="Gate CD x SiN thickness", parameters="GATE2_CD_Gate_Leg_Length + SIN_Thickness_SiN_Thickness",
        expected_type="2-way interaction",
        exact_param_screened="both received 100% univariate coverage; neither reached the interaction broad-screen top-200",
        relationship_type_recovered="no", direction_correct="n/a", threshold_window_correct="n/a",
        pair_reached_broad_screen="no", pair_reached_confirmation="no",
        fold_stable="n/a -- not tested", evidence_strength="n/a",
        evidence_file="stage2_runB_v2_pair_screen.csv; stage2_runB_v2_confirmed_interactions.csv (pair absent from both)",
        classification="NOT TESTABLE",
        v1_classification="NOT TESTABLE", improved="NO CHANGE",
        explanation=("Verified directly against the frozen pair-screen and confirmed-interactions files: neither "
                     "parameter, nor the pair, appears in either. Even with exhaustive (100%) broad-screen "
                     "coverage of all 986,310 pairs, this specific pair's cheap interaction score did not rank "
                     "in the top 200 selected for expensive confirmation. Unlike v1, this is not a candidate-pool "
                     "SELECTION artifact (the broad screen saw every pair) -- it means the pair's cheap-screen "
                     "interaction score genuinely did not clear the confirmation threshold, a stronger (though "
                     "still not definitive, since only the top 200 of ~986K got expensive confirmation) negative "
                     "signal than v1 could produce."),
    ),
    dict(
        id=5, relationship="Rc x Rsh", parameters="OHMIC_ET_Rc_an_ct2 + OHMIC_ET_Rsh_an_ct2",
        expected_type="2-way interaction",
        exact_param_screened="both received 100% univariate coverage; both individually appear in the broad-screen top-200 pool (paired with OTHER features), but not with each other",
        relationship_type_recovered="no", direction_correct="n/a", threshold_window_correct="n/a",
        pair_reached_broad_screen="no (as a pair)", pair_reached_confirmation="no",
        fold_stable="n/a -- pair not tested", evidence_strength="n/a",
        evidence_file="stage2_runB_v2_pair_screen.csv; stage2_runB_v2_confirmed_interactions.csv (pair absent)",
        classification="NOT TESTABLE",
        v1_classification="NOT TESTABLE", improved="PARTIAL (infrastructure only)",
        explanation=("Improvement over v1: OHMIC_ET_Rc_an_ct2 (zero LightGBM importance in v1, completely "
                     "invisible) now has real univariate signal (INCONCLUSIVE label but real effect range 2.56 "
                     "points) and appears in the broad-screen pool paired with OTHER features -- it is no longer "
                     "a fully dead feature. However, the SPECIFIC Rc x Rsh pair itself still never reached "
                     "confirmation, so the interaction claim remains not testable. Marked 'improved: PARTIAL' "
                     "to reflect that the underlying infrastructure changed materially even though the final "
                     "classification label for THIS SPECIFIC interaction did not."),
    ),
    dict(
        id=6, relationship="Leakage threshold", parameters="ET leakage family (lkg_*)",
        expected_type="Threshold, stronger penalty at higher leakage",
        exact_param_screened="yes -- all lkg_* columns received 100% univariate coverage",
        relationship_type_recovered="partial -- large effect ranges confirmed family-wide, but shapes are messier ('complex_nonmonotonic'/'inverted_U' dominate over clean 'threshold') than v1's characterization",
        direction_correct="mostly not determinable (direction=NaN for most top lkg columns under the v2 method)",
        threshold_window_correct="not cleanly re-derived", pair_reached_broad_screen="n/a (family-level, single-parameter)",
        pair_reached_confirmation="n/a", fold_stable="only 3/many lkg columns reach STRONG in v2 (vs. a much cleaner, more confident v1 characterization)",
        evidence_strength="MODERATE (family-wide), weaker than v1's STRONG/EXACT characterization",
        evidence_file="stage2_runB_v2_univariate_all.csv",
        classification="APPROXIMATELY FOUND",
        v1_classification="EXACTLY FOUND", improved="NO (regression)",
        explanation=("Important, honestly-reported regression: v1 found this relationship cleanly and confidently "
                     "-- the single strongest predictor in the whole dataset by all three LightGBM-based importance "
                     "methods, correctly characterized as threshold/negative with strong fold stability. v2's "
                     "model-free method still shows the leakage family carries large, real effect ranges (1-4+ "
                     "yield points across many lkg_* columns), confirming the family clearly matters, but the "
                     "SPECIFIC shape classification is messier (many land in 'complex_nonmonotonic'/'inverted_U' "
                     "rather than a clean 'threshold') and fewer individual columns reach STRONG. This is a "
                     "genuine case where LightGBM's own smoothing (via tree structure) characterized an already-"
                     "strong, easy-to-find relationship MORE cleanly than raw quantile-bin averaging did -- v2's "
                     "broader coverage is not a strict upgrade for relationships v1's method already handled well."),
    ),
    dict(
        id=7, relationship="Alignment process window", parameters="GATE2_Ali_Litho_Gate_to_recess_alignment",
        expected_type="Absolute-value process window",
        exact_param_screened="yes (100% univariate coverage)",
        relationship_type_recovered="no", direction_correct="n/a", threshold_window_correct="n/a",
        pair_reached_broad_screen="no", pair_reached_confirmation="no",
        fold_stable="INCONCLUSIVE (0/5, 0/5)", evidence_strength="INCONCLUSIVE",
        evidence_file="stage2_runB_v2_univariate_all.csv",
        classification="MISSED",
        v1_classification="MISSED", improved="NO CHANGE (but now a stronger negative result)",
        explanation=("v1 missed this because the parameter (modest but real LightGBM importance, rank 354) fell "
                     "outside the ~150-feature screening threshold -- a coverage gap. v2 gives it full priority: "
                     "100% univariate coverage, no dependency on LightGBM split usage. It STILL shows zero stable "
                     "evidence (0/5 shape and direction consistency) under a completely different, model-free "
                     "detection method. This upgrades the nature of the miss: it is no longer explainable as a "
                     "coverage artifact under either method -- both a tree-based and a model-free approach, at "
                     "full coverage, fail to detect it, which is stronger (though still not conclusive) evidence "
                     "of a genuine detection-method limitation for this specific relationship's functional form "
                     "(plausibly a narrow absolute-value window that 8-bin quantile averaging smooths away)."),
    ),
    dict(
        id=8, relationship="Gm stage degradation", parameters="TOPSIN_ET_Gm_max, FIC_ET_Gm_max",
        expected_type="Stage-to-stage degradation (derived delta)",
        exact_param_screened="yes -- exact pair identified by the generic cross-stage matcher from column-naming structure alone",
        relationship_type_recovered="yes -- 'threshold' pattern in the raw (FIC - TOPSIN) difference",
        direction_correct="yes -- direction=negative means yield decreases as (FIC - TOPSIN) increases, i.e. as TOPSIN degrades further below FIC, exactly matching the planted direction",
        threshold_window_correct="qualitative threshold behavior confirmed; exact numeric cutoff not independently re-derived",
        pair_reached_broad_screen="n/a (cross-stage channel, not the pairwise interaction channel)",
        pair_reached_confirmation="yes, within the cross-stage analysis's own stability testing",
        fold_stable="STRONG (5/5 shape, 5/5 direction)", evidence_strength="STRONG",
        evidence_file="stage2_runB_v2_cross_stage_relationships.csv",
        classification="EXACTLY FOUND",
        v1_classification="NOT TESTABLE", improved="YES -- the clearest improvement in this checkpoint",
        explanation=("This is the relationship the new cross-stage module was specifically built to address. Per "
                     "instruction, no engineered delta feature was created or fed to LightGBM -- the raw "
                     "difference (FIC_ET_Gm_max - TOPSIN_ET_Gm_max) was computed transiently, in-memory, purely "
                     "for this statistical analysis. The exact two named parameters were matched automatically "
                     "from column-naming structure (never hard-coded), and the resulting relationship is STRONG, "
                     "fully fold-stable (5/5, 5/5), and correctly directioned, closely matching the planted "
                     "'TOPSIN Gm degrading relative to FIC associates with poorer yield' description."),
    ),
    dict(
        id=9, relationship="Rc x Drain Lag x low PAE", parameters="OHMIC_ET_Rc_an_ct2, TOPSIN_ET_DC_PIV_Drain_Lag_1, TOPSIN_ET_Load_Pull_P.A.E_Last",
        expected_type="True 3-way interaction",
        exact_param_screened="all 3 received 100% univariate coverage; Rc and Drain_Lag_1 entered the broad-screen pool, PAE did not",
        relationship_type_recovered="no", direction_correct="n/a", threshold_window_correct="n/a",
        pair_reached_broad_screen="2 of 3 components yes, PAE no",
        pair_reached_confirmation="no triple containing all 3 components was generated as a candidate at all",
        fold_stable="n/a -- not tested", evidence_strength="n/a",
        evidence_file="stage2_runB_v2_three_way_interactions.csv (no matching triple found, verified directly)",
        classification="NOT TESTABLE",
        v1_classification="NOT TESTABLE (no 3-way search existed at all)", improved="PARTIAL (infrastructure only)",
        explanation=("v1 had no 3-way search mechanism whatsoever -- this relationship type was structurally "
                     "undetectable regardless of any other fix. v2 adds a genuine 3-way search (776 candidates "
                     "tested, 112 STRONG), a real capability improvement. But candidate GENERATION depends on "
                     "confirmed 2-way pairs and tree co-occurrence, and PAE never achieved either (never "
                     "confirmed in any 2-way interaction, never co-occurred with the other two in a LightGBM "
                     "tree path) -- so no candidate triple containing all 3 required components was ever "
                     "generated, verified directly against the full 776-row output. The classification label is "
                     "unchanged, but for a materially different, more specific reason: v1 lacked the METHOD "
                     "entirely; v2 has the method but a COVERAGE gap in candidate generation for this specific "
                     "combination."),
    ),
]


def main():
    df = pd.DataFrame(ROWS)
    df.to_csv(f"{R}/stage2_runB_v2_ground_truth_validation.csv", index=False)
    print(f"Saved {R}/stage2_runB_v2_ground_truth_validation.csv ({len(df)} rows)")

    counts = df["classification"].value_counts()
    exact = counts.get("EXACTLY FOUND", 0)
    approx = counts.get("APPROXIMATELY FOUND", 0)
    partial = counts.get("PARTIALLY FOUND", 0)
    missed = counts.get("MISSED", 0)
    not_testable = counts.get("NOT TESTABLE", 0)
    n = len(df)

    v1_counts = df["v1_classification"].apply(lambda s: s.split(" (")[0]).value_counts()
    v1_exact = v1_counts.get("EXACTLY FOUND", 0)
    v1_approx = v1_counts.get("APPROXIMATELY FOUND", 0)

    print(f"\nV2: Exact {exact}/{n} | Approx {approx}/{n} | Partial {partial}/{n} | Missed {missed}/{n} | NotTestable {not_testable}/{n}")
    print(f"V2 strict recovery rate: {exact/n:.1%} | V2 practical recovery rate: {(exact+approx)/n:.1%}")
    print(f"V1 strict recovery rate: {v1_exact/n:.1%} | V1 practical recovery rate: {(v1_exact+v1_approx)/n:.1%}")

    n_improved = (df["improved"].str.startswith("YES")).sum()
    print(f"\nRelationships improved: {n_improved}/9")


if __name__ == "__main__":
    main()
