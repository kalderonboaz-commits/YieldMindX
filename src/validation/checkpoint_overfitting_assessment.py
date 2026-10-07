"""Overfitting assessment for the four LOCKED checkpoint models
(OHMIC/FIC/SIN/THIN), synthetic run of 2026-10-05.

Read-only with respect to models and selection: loads the already-saved
pipelines and metadata, reconstructs the exact original training partition
(same deterministic split code/seed/data already verified in
outputs/checkpoint_training/provenance_retrospective/), and computes
Train/CV/Test metrics for comparison. No model is trained, tuned, or
reselected here, and nothing under outputs/checkpoint_training/ is modified
-- all assessment output goes to a separate directory.

Three numbers per checkpoint, each measuring something different:
  - Train: the locked pipeline (fit on ALL 80 lots / 1200 rows at lock time)
    scored on those SAME rows -- in-sample by construction, expected to look
    better than any out-of-sample number regardless of overfitting.
  - CV: the mean (+/- fold std) of the WINNING config's 5 outer-fold scores,
    each from a pipeline fit on ~4/5 of the train partition (a SMALLER
    training set than the locked pipeline saw) and scored on the held-out
    1/5 -- out-of-sample, but on less data per fit than the locked model.
  - Test: the locked pipeline scored once on the untouched 20-lot holdout --
    out-of-sample, same training-set size as Train, never used for
    selection or re-tuning.

ALL RESULTS ARE SYNTHETIC-DATA RESULTS.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.checkpoint_contract import CheckpointContract
from src.data.checkpoint_splits import build_shared_split, verify_no_lot_crosses_partitions
from src.data.checkpoint_validation import filter_process, load_raw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "outputs" / "checkpoint_training"
OUT_DIR = PROJECT_ROOT / "outputs" / "checkpoint_overfitting_assessment"
CHECKPOINT_ORDER = ("OHMIC", "FIC", "SIN", "THIN")


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _reconstruct_train_partition():
    """Exact same deterministic split/filter code used at training time --
    read-only, no model touched. Already cross-checked against the saved
    artifacts in provenance_retrospective/reconstruction_verification_RETROSPECTIVE.json."""
    contract = CheckpointContract.load()
    assert contract.mode == "synthetic", (
        f"This assessment targets the completed synthetic run; contract.mode={contract.mode!r}. "
        f"Point it back at the synthetic contract before running this script."
    )
    df = load_raw(contract)
    filtered, _ = filter_process(df, contract)
    shared = build_shared_split(filtered, contract.group_column, contract.chronological_candidates,
                                 contract.holdout_test_size, contract.random_state, contract.n_cv_splits)
    assert verify_no_lot_crosses_partitions(filtered, contract.group_column, shared.holdout)
    train_df = filtered.iloc[shared.holdout.train_idx].reset_index(drop=True)
    return contract, train_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, train_df = _reconstruct_train_partition()
    print(f"Reconstructed training partition: {len(train_df)} rows / "
          f"{train_df[contract.group_column].nunique()} lots (read-only, matches the locked-run partition).")

    fold_csv = pd.read_csv(TRAIN_DIR / "cv_fold_results_all_24_configs.csv")

    rows = []
    for cp_name in CHECKPOINT_ORDER:
        with open(TRAIN_DIR / "models" / f"{cp_name}_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["label"] == "SYNTHETIC DATA RESULT", f"{cp_name}: unexpected metadata label {meta['label']!r}"

        model_name, feature_mode = meta["winning_model"], meta["feature_mode"]
        cols = meta["candidate_feature_names"]

        pipeline = joblib.load(TRAIN_DIR / "models" / f"{cp_name}_final_pipeline.joblib")
        y_true_train = train_df[contract.target]
        y_pred_train = pipeline.predict(train_df[cols])
        train_m = _metrics(y_true_train, y_pred_train)

        winner_folds = fold_csv[(fold_csv.checkpoint == cp_name) & (fold_csv.model == model_name) & (fold_csv["mode"] == feature_mode)]
        assert len(winner_folds) == meta["split_provenance"]["n_cv_splits"], (
            f"{cp_name}: expected {meta['split_provenance']['n_cv_splits']} winner fold rows, found {len(winner_folds)}"
        )
        cv_mean = {m: float(winner_folds[m].mean()) for m in ("mae", "rmse", "r2")}
        cv_std = {m: float(winner_folds[m].std(ddof=1)) for m in ("mae", "rmse", "r2")}

        test_m = meta["test_metrics"]

        for split, m, sd in (
            ("Train", train_m, None),
            ("CV", cv_mean, cv_std),
            ("Test", test_m, None),
        ):
            rows.append({
                "checkpoint": cp_name, "split": split,
                "model": model_name, "feature_mode": feature_mode,
                "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                "rmse_std": (sd["rmse"] if sd else None),
                "mae_std": (sd["mae"] if sd else None),
                "r2_std": (sd["r2"] if sd else None),
                "n_rows_or_folds": (len(train_df) if split == "Train" else
                                    meta["split_provenance"]["n_cv_splits"] if split == "CV" else
                                    meta["split_provenance"]["test_rows"]),
            })
        print(f"{cp_name} ({model_name}/{feature_mode}): "
              f"Train RMSE={train_m['rmse']:.4f}  CV RMSE={cv_mean['rmse']:.4f}+/-{cv_std['rmse']:.4f}  "
              f"Test RMSE={test_m['rmse']:.4f}")

    table = pd.DataFrame(rows)
    table_path = OUT_DIR / "train_cv_test_comparison_SYNTHETIC.csv"
    table.to_csv(table_path, index=False)

    # ---- four-panel RMSE figure, consistent scale, CV fold-std error bars ----
    all_rmse_vals = table["rmse"].tolist() + (table["rmse"] + table["rmse_std"].fillna(0)).tolist()
    y_max = max(all_rmse_vals) * 1.12
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    split_order = ["Train", "CV", "Test"]
    colors = {"Train": "#7a9fc2", "CV": "#d97b4b", "Test": "#4c8c5a"}
    for ax, cp_name in zip(axes, CHECKPOINT_ORDER):
        sub = table[table.checkpoint == cp_name].set_index("split").loc[split_order]
        x = np.arange(len(split_order))
        yerr = [sub.loc[s, "rmse_std"] if s == "CV" and pd.notna(sub.loc[s, "rmse_std"]) else 0 for s in split_order]
        bars = ax.bar(x, sub["rmse"].values, yerr=yerr, capsize=5,
                       color=[colors[s] for s in split_order], edgecolor="#2a2a2a", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(split_order)
        ax.set_title(f"{cp_name}\n({sub['model'].iloc[0]}/{sub['feature_mode'].iloc[0]})", fontsize=10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for xi, v in zip(x, sub["rmse"].values):
            ax.text(xi, v + y_max * 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("RMSE (SortingYield, synthetic)")
    fig.suptitle("Train vs. grouped-CV vs. Test RMSE per checkpoint -- SYNTHETIC DATA\n"
                 "CV error bars = fold-to-fold standard deviation (n=5 folds) -- not a confidence interval",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig_path = OUT_DIR / "train_cv_test_rmse_four_panel_SYNTHETIC.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- interpretation note ----
    gap_lines = []
    for cp_name in CHECKPOINT_ORDER:
        sub = table[table.checkpoint == cp_name].set_index("split")
        train_rmse, cv_rmse, test_rmse = sub.loc["Train", "rmse"], sub.loc["CV", "rmse"], sub.loc["Test", "rmse"]
        gap_lines.append(
            f"- {cp_name}: Train={train_rmse:.4f}, CV={cv_rmse:.4f} (+/-{sub.loc['CV','rmse_std']:.4f}), "
            f"Test={test_rmse:.4f}. Train-vs-CV gap={cv_rmse - train_rmse:.4f}; "
            f"Train-vs-Test gap={test_rmse - train_rmse:.4f}."
        )

    note = f"""OVERFITTING ASSESSMENT -- SYNTHETIC DATA RESULTS ONLY
========================================================
Checkpoints assessed: {", ".join(CHECKPOINT_ORDER)}. No model was trained, tuned, or
reselected to produce this assessment; it only loads the pipelines already
locked in outputs/checkpoint_training/ and scores them against the exact
original training partition ({len(train_df)} rows), the already-saved
grouped-CV fold results, and the already-saved test-set metrics.

Per-checkpoint RMSE gaps:
{chr(10).join(gap_lines)}

What these numbers do and do not mean
--------------------------------------
1. Train scores are in-sample: the locked pipeline was fit on these exact
   rows, so its Train score reflects memorization capacity as well as true
   signal. A Train score better than CV/Test is EXPECTED for any model with
   nonzero capacity and is not, by itself, evidence of problematic
   overfitting -- it is evidence the model fit the training rows, which is
   what fitting means.
2. CV models were fit on fewer rows than the locked model. Each of the 5
   outer-CV folds trained on roughly 4/5 of the 80-lot train partition,
   not all of it. A gap between CV and Test is therefore not fully
   comparable to a Train-vs-Test gap: part of any CV-vs-Test difference can
   come from training-set size alone, not only from how the final,
   full-data-trained model generalizes.
3. CV fold-to-fold standard deviation (shown as the CV error bar) describes
   the spread actually observed across 5 folds. It is NOT a confidence
   interval, is not based on a distributional assumption, and should not be
   read as "the true RMSE lies within +/- 1 std with 95% probability" --
   with only 5 folds the standard deviation itself is a noisy estimate.
4. These plots and this table do not, on their own, prove the absence of
   overfitting. They describe the gaps that exist in this one synthetic run
   and offer context for interpreting them; they cannot rule out that the
   locked models would perform differently on data drawn from a materially
   different distribution (in particular: real production data).
5. No model selection, hyperparameter, or pipeline choice was changed in
   response to any number in this assessment, including the Test numbers
   above, which were already fixed before this assessment began and remain
   fixed after it.

All results above are SYNTHETIC DATA RESULTS (Process == '{contract.process_value}').
No production-accuracy or real-data-compatibility claim is made or implied.
"""
    note_path = OUT_DIR / "OVERFITTING_ASSESSMENT_SYNTHETIC.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    print(f"\nSaved:\n  {table_path}\n  {fig_path}\n  {note_path}")
    print(f"Locked artifacts under {TRAIN_DIR} were not modified (read-only pipeline loads + CSV reads only).")


if __name__ == "__main__":
    main()
