"""Stage 2 Run B spline/GAM method -- shuffle-based negative control.

Reruns the EXACT same spline-vs-linear pipeline (src/explain/
stage2_spline_single_parameter.py::run_all) on all 1,405 numeric Run B
predictors, but with the training target randomly permuted ONE time (same
convention as the v2 negative control: one full-coverage shuffle on the
cheap/complete stage, since this pipeline is fast enough to run the full
1,405 features either way). Any relationship found here is by construction
pure noise -- the resulting STRONG/MODERATE counts serve as an estimated
chance floor for the real-target results.
"""
from __future__ import annotations

import numpy as np

from src.explain.stage2_spline_single_parameter import run_all
from src.explain.stage2_v2_common import R, load_locked_split

RANDOM_STATE = 42


def main():
    split = load_locked_split()
    y_real = split.y_train.to_numpy()
    rng = np.random.RandomState(RANDOM_STATE)
    y_shuffled = y_real.copy()
    rng.shuffle(y_shuffled)

    print("Negative control: full 1,405-feature spline/GAM screen re-run with a single "
          "shuffled-target permutation (target values randomly reassigned across rows, "
          "breaking any true x-y relationship by construction).")
    df = run_all(y_override=y_shuffled)

    counts = df["evidence_label"].value_counts()
    strong = int(counts.get("STRONG", 0))
    moderate = int(counts.get("MODERATE", 0))
    weak = int(counts.get("WEAK", 0))
    inconclusive = int(counts.get("INCONCLUSIVE", 0))
    total = len(df)

    lines = [
        "STAGE 2 RUN B -- SPLINE/GAM METHOD NEGATIVE CONTROL (shuffled-target)",
        "=" * 70,
        "",
        f"Predictors re-screened under one shuffled-target permutation: {total} / 1405 (100%)",
        "",
        "Evidence label counts under shuffled target:",
        f"  STRONG:       {strong} ({100*strong/total:.1f}%)",
        f"  MODERATE:     {moderate} ({100*moderate/total:.1f}%)",
        f"  WEAK:         {weak} ({100*weak/total:.1f}%)",
        f"  INCONCLUSIVE: {inconclusive} ({100*inconclusive/total:.1f}%)",
        "",
        "Nonlinear-advantage (spline reproducibly beat linear null) under shuffled target: "
        f"{int(df['nonlinear_advantage'].sum())} / {total} ({100*df['nonlinear_advantage'].sum()/total:.1f}%)",
        "",
        "Interpretation: these are the pipeline's own false-discovery rates under pure",
        "noise, for both the shape/fold-stability gate (STRONG/MODERATE counts) and the",
        "linear-vs-spline improvement gate (nonlinear-advantage count). Compare directly",
        "against the real-target rates reported in stage2_runB_spline_summary.txt. A real",
        "rate not meaningfully above this shuffled floor means the method's thresholds are",
        "not trustworthy and must be recalibrated before further use.",
    ]
    out_path = f"{R}/stage2_runB_spline_negative_control.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
