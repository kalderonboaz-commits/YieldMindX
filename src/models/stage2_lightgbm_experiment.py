"""Stage 2 -- first LightGBM relationship-discovery experiment.

Two otherwise-identical runs on the APPROVED Stage 2 original-predictor
universe (outputs/reports/stage2_feature_eligibility.csv), differing only
in whether the four category-G "review" columns (LineYield, BackEndYield,
VIYield, LTYield) are included:

  Run A: category B (1,405) + category C one-hot dummies (20) + 4 review columns = 1,429 columns
  Run B: category B (1,405) + category C one-hot dummies (20)                    = 1,425 columns

Same rows, same LotName-grouped holdout split, same GroupKFold(5), same
LightGBM hyperparameters/seed, same SHAP/permutation-importance
methodology, for both runs. No engineered features anywhere (hard-asserted
in src/features/stage2_matrix.py). No new model type. No Golden/Worsen
Route finalization -- this is a single-model comparison experiment only.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

from src.data.splits import make_group_kfold, make_holdout_split
from src.explain.interactions import rank_shap_interactions
from src.features.stage2_matrix import REVIEW_COLUMNS, Stage2Dataset, load_stage2_dataset
from src.models.trees import make_lightgbm

OUTPUTS_DIR = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs"


class _NameMapShim:
    """Minimal fm-like shim so the existing SHAP/permutation/interaction
    helper functions (written against FeatureMatrix.to_original) can be
    reused unmodified against a Stage2Dataset."""
    def __init__(self, inverse_name_map: dict[str, str]):
        self._inv = inverse_name_map

    def to_original(self, c: str) -> str:
        return self._inv.get(c, c)


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _compute_shap_importance(model, X: pd.DataFrame, shim: _NameMapShim):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns)
    importance.index = [shim.to_original(c) for c in importance.index]
    importance = importance.groupby(level=0).sum()
    return shap_values, importance.sort_values(ascending=False)


def _compute_permutation_importance(model, X, y, shim: _NameMapShim, n_repeats=10, random_state=42):
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1)
    importance = pd.Series(result.importances_mean, index=X.columns)
    importance.index = [shim.to_original(c) for c in importance.index]
    importance = importance.groupby(level=0).sum()
    return importance.sort_values(ascending=False)


def _compute_gain_importance(model, shim: _NameMapShim):
    gains = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()
    importance = pd.Series(gains, index=names)
    importance.index = [shim.to_original(c) for c in importance.index]
    importance = importance.groupby(level=0).sum()
    return importance.sort_values(ascending=False)


@dataclasses.dataclass
class RunResult:
    label: str
    ds: Stage2Dataset
    cv_df: pd.DataFrame
    holdout_metrics: pd.DataFrame
    shap_importance: pd.Series
    gain_importance: pd.Series
    perm_importance: pd.Series
    top_interactions: list
    model: object
    X_test: pd.DataFrame
    y_test: pd.Series
    test_row_index: np.ndarray


def run_experiment(include_review: bool, label: str, random_state: int = 42) -> RunResult:
    ds = load_stage2_dataset(include_review_columns=include_review)
    shim = _NameMapShim(ds.inverse_name_map)

    holdout = make_holdout_split(ds.groups, test_size=0.2, random_state=random_state)
    X_train, X_test = ds.X.iloc[holdout.train_idx].reset_index(drop=True), ds.X.iloc[holdout.test_idx].reset_index(drop=True)
    y_train, y_test = ds.y.iloc[holdout.train_idx].reset_index(drop=True), ds.y.iloc[holdout.test_idx].reset_index(drop=True)
    groups_train = ds.groups.iloc[holdout.train_idx].reset_index(drop=True)

    print(f"\n=== {label}: {ds.X.shape[1]} predictors, {X_train.shape[0]} train rows / {X_test.shape[0]} holdout rows ===")

    _, folds = make_group_kfold(groups_train, n_splits=5)
    cv_records = []
    for fold_i, (tr, va) in enumerate(folds):
        m = make_lightgbm(random_state=random_state)
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = m.predict(X_train.iloc[va])
        met = _metrics(y_train.iloc[va], pred)
        cv_records.append({"fold": fold_i, **met})
        print(f"  [fold {fold_i}] rmse={met['rmse']:.4f} mae={met['mae']:.4f} r2={met['r2']:.4f}")
    cv_df = pd.DataFrame(cv_records)
    print(f"  CV mean: rmse={cv_df['rmse'].mean():.4f}+-{cv_df['rmse'].std():.4f} "
          f"mae={cv_df['mae'].mean():.4f} r2={cv_df['r2'].mean():.4f}")

    final_model = make_lightgbm(random_state=random_state)
    final_model.fit(X_train, y_train)
    pred_test = final_model.predict(X_test)
    holdout_metrics = pd.DataFrame([{"label": label, **_metrics(y_test, pred_test)}])
    print(f"  Holdout: {holdout_metrics.iloc[0].to_dict()}")

    print("  Computing SHAP importance (holdout set)...")
    _, shap_imp = _compute_shap_importance(final_model, X_test, shim)
    print("  Computing gain importance...")
    gain_imp = _compute_gain_importance(final_model, shim)
    print("  Computing permutation importance (holdout set)...")
    perm_imp = _compute_permutation_importance(final_model, X_test, y_test, shim)

    print("  Refitting compact LightGBM on top-40 candidates for tractable, correctly-scoped SHAP interaction search...")
    top40_model_cols = [ds.name_map[f] for f in shap_imp.head(40).index if f in ds.name_map]
    # IMPORTANT: shap's TreeExplainer.shap_interaction_values requires X's
    # column count to match what the model was actually trained on --
    # passing a column SUBSET of a model trained on the full 1,425/1,429
    # features segfaults the C extension. A dedicated compact model, fit
    # only on the candidate columns, avoids this (same pattern already
    # proven safe in Stage 1's src/explain/interactions.py).
    compact_model = make_lightgbm(random_state=random_state)
    compact_model.fit(X_train[top40_model_cols], y_train)
    print("  Ranking top-40-candidate SHAP interactions...")
    top_interactions = rank_shap_interactions(compact_model, X_test[top40_model_cols], shim, sample_size=min(300, len(X_test)), top_k=20)

    return RunResult(
        label=label, ds=ds, cv_df=cv_df, holdout_metrics=holdout_metrics,
        shap_importance=shap_imp, gain_importance=gain_imp, perm_importance=perm_imp,
        top_interactions=top_interactions, model=final_model,
        X_test=X_test, y_test=y_test, test_row_index=holdout.test_idx,
    )


def explain_wafer(model, X: pd.DataFrame, shim: _NameMapShim, row_pos: int, top_n: int = 3) -> str:
    explainer = shap.TreeExplainer(model)
    row = X.iloc[[row_pos]]
    sv = explainer.shap_values(row)[0]
    contrib = pd.Series(sv, index=X.columns)
    contrib.index = [shim.to_original(c) for c in contrib.index]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return "; ".join(f"{f}={v:+.3f}" for f, v in top.items())


def main():
    run_b = run_experiment(include_review=False, label="Run B (exclude 4 review columns)")
    run_a = run_experiment(include_review=True, label="Run A (include 4 review columns)")

    print("\n\n" + "=" * 90)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 90)

    print(f"\nFeature counts: Run A = {run_a.ds.X.shape[1]} | Run B = {run_b.ds.X.shape[1]}")

    print("\n--- CV metrics (mean +/- std) ---")
    for r in (run_a, run_b):
        print(f"  {r.label}: rmse={r.cv_df['rmse'].mean():.4f}+-{r.cv_df['rmse'].std():.4f} "
              f"mae={r.cv_df['mae'].mean():.4f}+-{r.cv_df['mae'].std():.4f} "
              f"r2={r.cv_df['r2'].mean():.4f}+-{r.cv_df['r2'].std():.4f}")

    print("\n--- Holdout metrics ---")
    for r in (run_a, run_b):
        print(f"  {r.label}: {r.holdout_metrics.iloc[0][['rmse','mae','r2']].to_dict()}")

    print("\n--- Top 20 SHAP importance: Run A vs Run B ---")
    top20_a = run_a.shap_importance.head(20)
    top20_b = run_b.shap_importance.head(20)
    print("Run A top 20:")
    print(top20_a.round(4).to_string())
    print("\nRun B top 20:")
    print(top20_b.round(4).to_string())

    print("\n--- Review column standing in Run A ---")
    shap_rank_a = run_a.shap_importance.rank(ascending=False, method="min")
    gain_rank_a = run_a.gain_importance.rank(ascending=False, method="min")
    perm_rank_a = run_a.perm_importance.rank(ascending=False, method="min")
    n_features_a = len(run_a.shap_importance)
    review_summary_rows = []
    for c in REVIEW_COLUMNS:
        row = {
            "column": c,
            "shap_importance": run_a.shap_importance.get(c, 0.0),
            "shap_rank": int(shap_rank_a.get(c, n_features_a)),
            "gain_importance": run_a.gain_importance.get(c, 0.0),
            "gain_rank": int(gain_rank_a.get(c, n_features_a)),
            "permutation_importance": run_a.perm_importance.get(c, 0.0),
            "perm_rank": int(perm_rank_a.get(c, n_features_a)),
            "n_features_total": n_features_a,
            "pearson_corr_with_target": float(run_a.ds.X[run_a.ds.name_map[c]].corr(run_a.ds.y))
                if c in run_a.ds.name_map else None,
        }
        review_summary_rows.append(row)
    review_df = pd.DataFrame(review_summary_rows)
    print(review_df.round(4).to_string(index=False))

    print("\n--- Rank correlation of shared (non-review) features between Run A and Run B ---")
    from scipy.stats import spearmanr
    shared_features = [f for f in run_b.shap_importance.index if f in run_a.shap_importance.index]
    a_ranks = run_a.shap_importance.reindex(shared_features).rank(ascending=False)
    b_ranks = run_b.shap_importance.reindex(shared_features).rank(ascending=False)
    rho, _ = spearmanr(a_ranks, b_ranks)
    print(f"  Spearman rank correlation (SHAP importance) across {len(shared_features)} shared features: {rho:.4f}")

    top20_a_names = set(top20_a.index)
    top20_b_names = set(top20_b.index)
    print(f"  Top-20 overlap (excluding review columns from Run A's list): "
          f"{len(top20_a_names - set(REVIEW_COLUMNS) & top20_b_names)} / "
          f"{len(top20_b_names)} of Run B's top 20 also appear in Run A's top 20")

    print("\n--- Top interactions: Run A vs Run B ---")
    print("Run A top interactions:")
    for p in run_a.top_interactions[:15]:
        print(f"  {p.feature_a} x {p.feature_b}  strength={p.interaction_strength:.4f}")
    print("\nRun B top interactions:")
    for p in run_b.top_interactions[:15]:
        print(f"  {p.feature_a} x {p.feature_b}  strength={p.interaction_strength:.4f}")

    a_pairs = {frozenset([p.feature_a, p.feature_b]) for p in run_a.top_interactions[:15]}
    b_pairs = {frozenset([p.feature_a, p.feature_b]) for p in run_b.top_interactions[:15]}
    print(f"\n  Overlap of top-15 interaction pairs: {len(a_pairs & b_pairs)} / {len(b_pairs)}")
    review_in_interactions = [p for p in run_a.top_interactions if p.feature_a in REVIEW_COLUMNS or p.feature_b in REVIEW_COLUMNS]
    print(f"  Interactions in Run A involving a review column: {len(review_in_interactions)}")
    for p in review_in_interactions:
        print(f"    {p.feature_a} x {p.feature_b}  strength={p.interaction_strength:.4f}")

    print("\n--- High-yield / low-yield wafer driver check ---")
    shim_a = _NameMapShim(run_a.ds.inverse_name_map)
    shim_b = _NameMapShim(run_b.ds.inverse_name_map)
    y_test_a = run_a.y_test
    best_pos_a, worst_pos_a = int(y_test_a.idxmax()), int(y_test_a.idxmin())
    y_test_b = run_b.y_test
    best_pos_b, worst_pos_b = int(y_test_b.idxmax()), int(y_test_b.idxmin())
    print(f"  Run A -- highest-yield wafer ({y_test_a.iloc[best_pos_a]:.2f}%) top drivers: "
          f"{explain_wafer(run_a.model, run_a.X_test, shim_a, best_pos_a)}")
    print(f"  Run B -- highest-yield wafer ({y_test_b.iloc[best_pos_b]:.2f}%) top drivers: "
          f"{explain_wafer(run_b.model, run_b.X_test, shim_b, best_pos_b)}")
    print(f"  Run A -- lowest-yield wafer ({y_test_a.iloc[worst_pos_a]:.2f}%) top drivers: "
          f"{explain_wafer(run_a.model, run_a.X_test, shim_a, worst_pos_a)}")
    print(f"  Run B -- lowest-yield wafer ({y_test_b.iloc[worst_pos_b]:.2f}%) top drivers: "
          f"{explain_wafer(run_b.model, run_b.X_test, shim_b, worst_pos_b)}")

    # ---- save artifacts ----
    import os
    os.makedirs(f"{OUTPUTS_DIR}/reports", exist_ok=True)
    run_a.shap_importance.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runA_shap_importance.csv")
    run_b.shap_importance.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runB_shap_importance.csv")
    run_a.gain_importance.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runA_gain_importance.csv")
    run_b.gain_importance.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runB_gain_importance.csv")
    run_a.perm_importance.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runA_permutation_importance.csv")
    run_b.perm_importance.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runB_permutation_importance.csv")
    review_df.to_csv(f"{OUTPUTS_DIR}/reports/stage2_review_columns_standing.csv", index=False)
    run_a.cv_df.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runA_cv_results.csv", index=False)
    run_b.cv_df.to_csv(f"{OUTPUTS_DIR}/reports/stage2_runB_cv_results.csv", index=False)
    pd.concat([run_a.holdout_metrics, run_b.holdout_metrics]).to_csv(f"{OUTPUTS_DIR}/reports/stage2_holdout_comparison.csv", index=False)
    pd.DataFrame([{"feature_a": p.feature_a, "feature_b": p.feature_b, "strength": p.interaction_strength} for p in run_a.top_interactions]) \
        .to_csv(f"{OUTPUTS_DIR}/reports/stage2_runA_top_interactions.csv", index=False)
    pd.DataFrame([{"feature_a": p.feature_a, "feature_b": p.feature_b, "strength": p.interaction_strength} for p in run_b.top_interactions]) \
        .to_csv(f"{OUTPUTS_DIR}/reports/stage2_runB_top_interactions.csv", index=False)
    print("\nSaved all Stage 2 Run A / Run B comparison artifacts to outputs/reports/")


if __name__ == "__main__":
    main()
