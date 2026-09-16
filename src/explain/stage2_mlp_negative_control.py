"""Stage 2 Run B MLP -- section 6 negative control, extended.

Part 1 (already run in stage2_mlp_train_eval.py): grouped-CV predictive
R^2 under a shuffled target -- already collapsed cleanly (real=0.289,
shuffled=-1.162). Reproduced here from the frozen metrics file.

Part 2 (NEW, this script): the feature-evidence layer (section 7) showed a
suspiciously high 93% STRONG rate, and the single-parameter layer showed
91% STRONG -- both far higher than any tree-based/spline method in this
project ever produced on the full universe. Given the documented
SUBSTANTIAL train-val gap, this is plausibly an overfitting artifact
(an overfit model's predictions can be destabilized by almost ANY
perturbation, inflating apparent "importance" broadly) rather than
genuinely widespread real structure. This script retrains fold models on
a SHUFFLED target and reruns the SAME permutation-importance procedure on
a disclosed random subset of features (200 of 1,425 -- computationally
reasonable, consistent with this project's established negative-control
scoping convention) to check whether the high STRONG rate itself survives
under shuffling.

Part 3: reruns the 2-way and 3-way interaction tests on their own
already-tested candidate pools with a shuffled target, using the SAME
shuffled fold models from Part 2.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explain.stage2_mlp_2way import GRID_RES_A, BACKGROUND_SIZE_A, interaction_scores
from src.explain.stage2_mlp_3way import three_way_contrast, BACKGROUND_SIZE as BG3
from src.explain.stage2_mlp_common import fit_mlp_grouped_early_stopping, predict_scaled, rmse
from src.explain.stage2_v2_common import R, load_locked_split

RANDOM_STATE = 42
N_FEATURES_SAMPLE = 200
N_PERMUTATION_REPEATS = 2


def main():
    split = load_locked_split()
    ds = split.ds
    X_train = split.X_train.to_numpy(dtype=np.float64)
    y_train = split.y_train.to_numpy(dtype=np.float64)
    groups_train = split.groups_train.to_numpy()
    feature_cols = list(split.X_train.columns)
    train_means = X_train.mean(axis=0)

    rng = np.random.RandomState(RANDOM_STATE)
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)

    lines = ["STAGE 2 RUN B MLP -- NEGATIVE CONTROL (shuffled-target, one permutation)",
             "=" * 74, ""]

    # ---- Part 1: predictive performance (reproduced from frozen metrics) ----
    metrics = pd.read_csv(f"{R}/stage2_runB_mlp_model_metrics.csv").iloc[0]
    lines += [
        "PART 1: PREDICTIVE PERFORMANCE (grouped 5-fold CV)",
        "-" * 74,
        f"Real-target CV R^2 (mean): {metrics['cv_val_r2_mean']:.4f}",
        f"Shuffled-target CV R^2 (mean): {metrics['shuffled_target_cv_r2_mean']:.4f}",
        f"-> {'CLEAN COLLAPSE' if metrics['shuffled_target_cv_r2_mean'] < 0.05 else 'WARNING -- residual predictive power under shuffle'}",
        "",
    ]

    # ---- Part 2: feature-evidence STRONG rate under shuffle ----
    print("Part 2: retraining 5 fold models on shuffled target, testing feature-evidence rate on 200-feature sample...")
    t0 = time.time()
    shuf_fold_models = []
    for fold_i, (tr, va) in enumerate(split.folds):
        fit = fit_mlp_grouped_early_stopping(X_train[tr], y_shuf[tr], groups_train[tr])
        shuf_fold_models.append({"tr": tr, "va": va, "fit": fit})
        print(f"  shuffled fold {fold_i} fitted (best_epoch={fit.best_epoch})", flush=True)

    sample_idx = rng.choice(len(feature_cols), size=N_FEATURES_SAMPLE, replace=False)
    baseline_rmse = []
    for fm in shuf_fold_models:
        p = predict_scaled(fm["fit"], X_train[fm["va"]])
        baseline_rmse.append(rmse(y_shuf[fm["va"]], p))

    consistency = np.zeros(N_FEATURES_SAMPLE)
    mean_effect = np.zeros(N_FEATURES_SAMPLE)
    for fold_i, fm in enumerate(shuf_fold_models):
        va = fm["va"]
        X_va = X_train[va]
        y_va = y_shuf[va]
        for k, j in enumerate(sample_idx):
            deltas = []
            for r in range(N_PERMUTATION_REPEATS):
                X_perm = X_va.copy()
                perm_idx = rng.permutation(len(va))
                X_perm[:, j] = X_perm[perm_idx, j]
                p_perm = predict_scaled(fm["fit"], X_perm)
                deltas.append(rmse(y_va, p_perm) - baseline_rmse[fold_i])
            eff = np.mean(deltas)
            mean_effect[k] += eff / 5
            if eff > 0:
                consistency[k] += 1
    strong_count = int(np.sum((consistency >= 4) & (mean_effect > 0)))
    shuf_strong_rate = 100 * strong_count / N_FEATURES_SAMPLE
    print(f"Part 2 done in {time.time()-t0:.1f}s -- shuffled STRONG rate: {shuf_strong_rate:.1f}% "
          f"({strong_count}/{N_FEATURES_SAMPLE})")

    real_fe = pd.read_csv(f"{R}/stage2_runB_mlp_feature_evidence.csv")
    real_strong_rate = 100 * (real_fe["evidence_label"] == "STRONG").sum() / len(real_fe)
    lines += [
        "PART 2: FEATURE-EVIDENCE STRONG RATE (permutation importance, 200-feature random sample)",
        "-" * 74,
        f"Real-target STRONG rate (full 1,425-feature run): {real_strong_rate:.1f}%",
        f"Shuffled-target STRONG rate (this 200-feature sample, same procedure): {shuf_strong_rate:.1f}%",
    ]
    if shuf_strong_rate >= 30:
        lines.append("-> WARNING: a substantial fraction of features still register STRONG under a shuffled "
                      "target. This confirms the near-universal real-target STRONG rate (93%) is inflated by "
                      "the model's documented overfitting rather than reflecting genuinely widespread real "
                      "structure -- MLP single-feature 'STRONG' labels should be read as heavily diluted by "
                      "noise and are NOT comparable in reliability to the tree-based/spline STRONG rates "
                      "elsewhere in this project.")
    else:
        lines.append("-> Shuffled rate is a minority -- some inflation relative to a theoretical zero-noise "
                      "floor is still plausible given the small sample size, but not a wholesale collapse of "
                      "the evidence layer's meaning.")
    lines.append("")
    print(lines[-2])

    # ---- Part 3: 2-way and 3-way interaction confidence rate under shuffle ----
    print("\nPart 3: re-testing the ALREADY-CONFIRMED 2-way and 3-way candidate pools under the shuffled-target models...")
    shuf_pipes = [Pipeline([("scaler", fm["fit"].scaler), ("mlp", fm["fit"].model)]) for fm in shuf_fold_models]

    twoway = pd.read_csv(f"{R}/stage2_runB_mlp_2way_interactions.csv")
    X_bg = split.X_train.sample(min(BACKGROUND_SIZE_A, len(split.X_train)), random_state=RANDOM_STATE).astype(np.float64)
    name_to_col = {orig: col for orig, col in zip(
        [ds.inverse_name_map.get(c, c) for c in feature_cols], feature_cols)}
    shuf_h_vals = []
    for _, row in twoway.iterrows():
        ca, cb = name_to_col.get(row["parameter_A"]), name_to_col.get(row["parameter_B"])
        if ca is None or cb is None:
            continue
        res = interaction_scores(shuf_pipes[0], X_bg, ca, cb, GRID_RES_A)  # one shuffled model, cost-bounded
        if res is not None:
            shuf_h_vals.append(res[0])
    shuf_h_vals = np.array(shuf_h_vals)

    lines += [
        "PART 3: 2-WAY H-STATISTIC MAGNITUDE UNDER SHUFFLE (same 100-pair pool, 1 shuffled-target fold model)",
        "-" * 74,
        f"Real-target H-statistic: mean={twoway['mlp_h_statistic'].mean():.4f}, "
        f"max={twoway['mlp_h_statistic'].max():.4f}, STRONG count={int((twoway['confidence']=='STRONG').sum())}/100",
        f"Shuffled-target H-statistic (same pairs): mean={shuf_h_vals.mean():.4f}, max={shuf_h_vals.max():.4f}",
    ]
    if shuf_h_vals.mean() < 0.5 * twoway['mlp_h_statistic'].mean():
        lines.append("-> Shuffled-target H-statistic magnitude is meaningfully lower than real-target -- "
                      "consistent with genuine (not purely noise-driven) interaction signal in the real pairs, "
                      "though single-fold-model comparison is a lighter check than the full 5-fold stability "
                      "test used for the real-target confidence labels.")
    else:
        lines.append("-> WARNING: shuffled-target H-statistic magnitude is comparable to real-target -- the "
                      "2-way interaction score itself may not be well-calibrated for this severely "
                      "overparameterized model; reported as-is, not recalibrated against ground truth.")
    lines.append("")
    print(lines[-2])

    # ---- Part 3b: 3-way contrast magnitude under shuffle (top 30 triples, cost-bounded) ----
    threeway = pd.read_csv(f"{R}/stage2_runB_mlp_3way_interactions.csv")
    top30 = threeway.reindex(threeway["contrast_3way"].abs().sort_values(ascending=False).index).head(30)
    X_bg3 = split.X_train.sample(min(BG3, len(split.X_train)), random_state=RANDOM_STATE)
    shuf_contrast_vals = []
    for _, row in top30.iterrows():
        ca, cb, cc = name_to_col.get(row["parameter_A"]), name_to_col.get(row["parameter_B"]), name_to_col.get(row["parameter_C"])
        if not all([ca, cb, cc]):
            continue
        contrast, _, _, _ = three_way_contrast(shuf_pipes[0], X_bg3, ca, cb, cc)
        shuf_contrast_vals.append(abs(contrast))
    shuf_contrast_vals = np.array(shuf_contrast_vals)
    real_top30_mean_abs = top30["contrast_3way"].abs().mean()
    lines += [
        "PART 3b: 3-WAY CONTRAST MAGNITUDE UNDER SHUFFLE (top 30 |contrast| triples, 1 shuffled-target fold model)",
        "-" * 74,
        f"Real-target top-30 mean |contrast_3way|: {real_top30_mean_abs:.5f}",
        f"Shuffled-target mean |contrast_3way| (same 30 triples): {shuf_contrast_vals.mean():.5f}",
    ]
    if shuf_contrast_vals.mean() < real_top30_mean_abs:
        lines.append("-> Shuffled-target contrast magnitude is lower than real-target, consistent with genuine "
                      "signal, but ALL contrast magnitudes in this analysis (real included) are extremely "
                      "small in absolute yield-point terms (see stage2_runB_mlp_summary.txt) -- practical "
                      "significance of even the 'real' 3-way findings is doubtful regardless of this check.")
    else:
        lines.append("-> WARNING: shuffled-target contrast magnitude is not clearly smaller than real-target.")
    lines.append("")
    print(lines[-2])

    out_path = f"{R}/stage2_runB_mlp_negative_control.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
