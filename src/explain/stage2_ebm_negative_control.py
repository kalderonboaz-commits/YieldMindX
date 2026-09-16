"""Stage 2 Run B EBM -- section 7: relationship-evidence-level negative
control (predictive-level negative control already run and frozen in
stage2_ebm_train_eval.py -- reproduced here for one consolidated file).

For each of the 5 shuffled-target fold models (already fit and cached by
stage2_ebm_train_eval.py), counts how many main-effect terms are "non-flat"
by the SAME relative-to-median-term-range criterion used for the real
analysis (classify_ebm_main_shape's flat gate, recomputed per shuffled
model using ITS OWN median term range as the reference), and compares
typical pairwise interaction-strength magnitudes.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.explain.stage2_ebm_common import classify_ebm_main_shape, real_bin_layout
from src.explain.stage2_ebm_train_eval import CACHE_PATH
from src.explain.stage2_v2_common import R

RANDOM_STATE = 42


def main():
    cache = joblib.load(CACHE_PATH)
    shuf_fold_models = cache["shuf_fold_models"]

    metrics = pd.read_csv(f"{R}/stage2_runB_ebm_model_metrics.csv").iloc[0]
    lines = ["STAGE 2 RUN B EBM -- NEGATIVE CONTROL (shuffled-target, one permutation)",
             "=" * 74, ""]

    lines += [
        "PART 1: PREDICTIVE PERFORMANCE (grouped 5-fold CV)",
        "-" * 74,
        f"Real-target CV R^2 (mean): {metrics['cv_val_r2_mean']:.4f}",
        f"Shuffled-target CV R^2 (mean): {metrics['shuffled_target_cv_r2_mean']:.4f}",
        f"-> {'CLEAN COLLAPSE' if metrics['shuffled_target_cv_r2_mean'] < 0.05 else 'WARNING -- residual predictive power under shuffle'}",
        "",
    ]

    print("Part 2: counting apparently-strong main effects under shuffle (5 shuffled fold models)...")
    real_main = pd.read_csv(f"{R}/stage2_runB_ebm_main_effects.csv")
    real_nonflat = int((real_main["relationship_class"] != "FLAT").sum())
    real_strong = int((real_main["evidence_label"] == "STRONG").sum())

    shuf_nonflat_counts = []
    for fm in shuf_fold_models:
        ebm_s = fm["ebm"]
        ranges = []
        for t_idx, t in enumerate(ebm_s.term_features_):
            if len(t) != 1:
                continue
            _, s, _ = real_bin_layout(ebm_s, t_idx)
            ranges.append(float(s.max() - s.min()))
        flat_ref = float(np.median(ranges))
        nonflat = 0
        for t_idx, t in enumerate(ebm_s.term_features_):
            if len(t) != 1:
                continue
            c, s, sd = real_bin_layout(ebm_s, t_idx)
            r = classify_ebm_main_shape(c, s, sd, flat_ref)
            if r["shape"] not in ("FLAT", None):
                nonflat += 1
        shuf_nonflat_counts.append(nonflat)
    shuf_nonflat_mean = float(np.mean(shuf_nonflat_counts))
    print(f"Shuffled non-flat main-effect counts per fold: {shuf_nonflat_counts} (mean={shuf_nonflat_mean:.1f})")

    lines += [
        "PART 2: MAIN-EFFECT 'NON-FLAT' COUNT UNDER SHUFFLE (relative-to-own-median criterion)",
        "-" * 74,
        f"Real-target: {real_nonflat}/1425 non-flat terms ({real_strong} STRONG by 5-fold shape consistency)",
        f"Shuffled-target (5 fold models, own median-range reference each): {shuf_nonflat_counts}, mean={shuf_nonflat_mean:.1f}/1425",
    ]
    if shuf_nonflat_mean < 0.5 * real_nonflat:
        lines.append("-> CLEAN-ISH SEPARATION: shuffled non-flat count is well below the real count, "
                      "though note this criterion is RELATIVE (each model's own median range), so some "
                      "non-flat terms are structurally expected under shuffle too -- the STRONG label (which "
                      "additionally requires 4/5 shape-consistency across independently-fit fold models) is "
                      "the more reliable real-target signal.")
    else:
        lines.append("-> WARNING: shuffled non-flat count is not clearly separated from the real count.")
    lines.append("")
    print(lines[-2])

    print("Part 3: comparing pairwise interaction-strength magnitude under shuffle...")
    real_2way = pd.read_csv(f"{R}/stage2_runB_ebm_2way_interactions.csv")
    shuf_int_strengths = []
    for fm in shuf_fold_models:
        ebm_s = fm["ebm"]
        imp = ebm_s.term_importances()
        for t_idx, t in enumerate(ebm_s.term_features_):
            if len(t) == 2:
                shuf_int_strengths.append(float(imp[t_idx]))
    shuf_int_strengths = np.array(shuf_int_strengths)

    lines += [
        "PART 3: PAIRWISE INTERACTION-STRENGTH MAGNITUDE UNDER SHUFFLE",
        "-" * 74,
        f"Real-target (15 EBM-selected pairs): strength range {real_2way['interaction_strength'].min():.6f} - "
        f"{real_2way['interaction_strength'].max():.6f}, mean={real_2way['interaction_strength'].mean():.6f}",
        f"Shuffled-target (interaction terms selected by the 5 shuffled fold models, "
        f"n={len(shuf_int_strengths)}): "
        + (f"range {shuf_int_strengths.min():.6f} - {shuf_int_strengths.max():.6f}, mean={shuf_int_strengths.mean():.6f}"
           if len(shuf_int_strengths) else "0 pairwise terms were selected at all under any shuffled fold"),
    ]
    if len(shuf_int_strengths) == 0 or shuf_int_strengths.mean() < real_2way["interaction_strength"].mean():
        lines.append("-> Consistent with genuine (if extremely weak) real-target signal: shuffled interaction "
                      "strengths are not larger than real-target strengths. IMPORTANT CONTEXT: the real-target "
                      "interactions ALSO showed 0/5 fold-selection reproducibility (see "
                      "stage2_runB_ebm_2way_interactions.csv) -- so even the 'real' pairwise interactions are "
                      "not being reported as confirmed findings regardless of this shuffle comparison.")
    else:
        lines.append("-> WARNING: shuffled interaction strengths are comparable to or exceed real-target "
                      "strengths.")
    lines.append("")
    print(lines[-2])

    out_path = f"{R}/stage2_runB_ebm_negative_control.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
