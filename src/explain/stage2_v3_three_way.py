"""Stage 2 Run B v3 -- hierarchical 3-way interaction discovery.

Candidate triples generated from three generic channels (none seeded by
ground truth):
  (a) chaining -- for each of the top confirmed STRONG v3 2-way pairs
      (A,B), any third feature C that also appears in another STRONG/
      MODERATE confirmed pair with A or B
  (b) LightGBM tree-path co-occurrence -- features appearing together as
      split features (3+ at once) in the same tree of a compact model
      fit on the v3 confirmed-pool union features
  (c) residual conditional structure -- NEW channel: for the top STRONG
      2-way pairs, compute the residual of y after that pair's own fitted
      JOINT surface, then find third features whose value correlates with
      THAT RESIDUAL (i.e. explain what the confirmed pair does not) --
      directly implements "residual conditional structure not explained by
      confirmed 2-way interactions."

For each candidate triple: LOWER-ORDER model (main effects + all 3
pairwise joint surfaces) vs. FULL 3-way model (adds a genuine 3-way binned
term, shrunk toward the lower-order prediction) -- see
src/explain/stage2_v3_common.py::fit_threeway for the exact construction.
The 3-way score is the incremental improvement of FULL over LOWER-ORDER,
under grouped 5-fold CV -- not "are all three variables individually
important."
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import fit_additive_joint, lower_vs_full_cv, predict_joint
from src.models.trees import make_lightgbm

RANDOM_STATE = 42


def compute_tree_triplet_counts(model, feature_names: list[str]) -> dict:
    trees_df = model.booster_.trees_to_dataframe().dropna(subset=["split_feature"])
    counts: dict[tuple, int] = {}
    fset = set(feature_names)
    for _, group in trees_df.groupby("tree_index"):
        feats = sorted(set(group["split_feature"].unique()) & fset)
        if len(feats) < 3:
            continue
        for trip in combinations(feats, 3):
            counts[trip] = counts.get(trip, 0) + 1
    return counts


def label_row(cv: dict) -> tuple[str, str]:
    fc, n = cv["folds_full_better"], cv["n_folds_used"]
    inc_r2 = cv["incremental_r2"]
    if cv["full_beats_lower"] and fc == n:
        return "STRONG", f"Full 3-way model beat lower-order in {fc}/{n} folds (unanimous), incremental R^2={inc_r2:.4f}."
    if cv["full_beats_lower"] and fc >= n - 1:
        return "MODERATE", f"Full 3-way model beat lower-order in {fc}/{n} folds, incremental R^2={inc_r2:.4f}."
    if fc >= max(3, int(round(0.6 * n))) and inc_r2 > 0.01:
        return "MODERATE", f"Full 3-way model beat lower-order in {fc}/{n} folds with incremental R^2={inc_r2:.4f} -- majority but not unanimous."
    if inc_r2 > 0 or fc >= max(2, int(round(0.4 * n))):
        return "WEAK", f"Full 3-way model beat lower-order in only {fc}/{n} folds, incremental R^2={inc_r2:.4f} -- inconsistent."
    return "INCONCLUSIVE", f"Full 3-way model did not reproducibly beat lower-order ({fc}/{n} folds, incremental R^2={inc_r2:.4f})."


def main():
    split = load_locked_split()
    ds = split.ds
    confirmed = pd.read_csv(f"{R}/stage2_runB_v3_confirmed_2way_interactions.csv")
    strong = confirmed[confirmed["evidence_label"] == "STRONG"].sort_values("interaction_incremental_r2", ascending=False)
    strong_mod = confirmed[confirmed["evidence_label"].isin(["STRONG", "MODERATE"])]
    union_features = sorted(set(confirmed["parameter_A"]) | set(confirmed["parameter_B"]))
    model_cols = [ds.name_map[f] for f in union_features]
    y_train_arr = split.y_train.to_numpy()

    triples: set[tuple] = set()
    channel_of: dict[tuple, set] = {}

    def add_triple(t, label):
        t = tuple(sorted(t))
        triples.add(t)
        channel_of.setdefault(t, set()).add(label)

    # --- channel (a): chaining from top STRONG confirmed pairs ---
    top_chain_pairs = list(zip(strong["parameter_A"], strong["parameter_B"]))[:30]
    member_to_partners: dict[str, set] = {}
    for a, b in zip(strong_mod["parameter_A"], strong_mod["parameter_B"]):
        member_to_partners.setdefault(a, set()).add(b)
        member_to_partners.setdefault(b, set()).add(a)
    for a, b in top_chain_pairs:
        for c in member_to_partners.get(a, set()) | member_to_partners.get(b, set()):
            if c not in (a, b):
                add_triple((a, b, c), "a_chaining_confirmed_pairs")

    # --- channel (b): tree-path 3-way co-occurrence ---
    print(f"Fitting compact model on {len(union_features)} v3-confirmed-pool features for tree-triplet mining...")
    tree_model = make_lightgbm(random_state=RANDOM_STATE)
    tree_model.fit(split.X_train[model_cols], split.y_train)
    triplet_counts_model = compute_tree_triplet_counts(tree_model, model_cols)
    triplet_counts = {tuple(sorted(ds.inverse_name_map[c] for c in trip)): cnt for trip, cnt in triplet_counts_model.items()}
    ranked_triplets = sorted(triplet_counts.items(), key=lambda kv: -kv[1])[:40]
    for trip, _ in ranked_triplets:
        add_triple(trip, "b_tree_path_cooccurrence")

    # --- channel (c): residual conditional structure ---
    top10_strong = strong.head(10)
    numeric_features = ds.numeric_cols_original
    X_all = split.X_train[[ds.name_map[f] for f in numeric_features]].to_numpy(dtype=np.float64)
    for _, row in top10_strong.iterrows():
        a, b = row["parameter_A"], row["parameter_B"]
        xa = split.X_train[ds.name_map[a]].to_numpy(dtype=np.float64)
        xb = split.X_train[ds.name_map[b]].to_numpy(dtype=np.float64)
        fit_ab = fit_additive_joint(xa, xb, y_train_arr, n_bins=3)
        pred = predict_joint(fit_ab, xa, xb)
        residual = y_train_arr - pred
        resid_c = residual - residual.mean()
        resid_std = resid_c.std()
        if resid_std < 1e-9:
            continue
        X_c = (X_all - X_all.mean(axis=0)) / np.where(X_all.std(axis=0) < 1e-9, 1.0, X_all.std(axis=0))
        corr = (X_c * resid_c[:, None]).mean(axis=0) / (resid_std if resid_std > 1e-9 else 1.0)
        order = np.argsort(-np.abs(corr))
        added = 0
        for idx in order:
            c = numeric_features[idx]
            if c in (a, b):
                continue
            add_triple((a, b, c), "c_residual_conditional_structure")
            added += 1
            if added >= 3:
                break

    print(f"\nTotal 3-way candidate triples generated: {len(triples)}")
    for label in ["a_chaining_confirmed_pairs", "b_tree_path_cooccurrence", "c_residual_conditional_structure"]:
        cnt = sum(1 for t, labs in channel_of.items() if label in labs)
        print(f"  {label}: {cnt}")

    _, folds = make_group_kfold(split.groups_train, n_splits=5)
    rows = []
    tested = 0
    for a, b, c in sorted(triples):
        if a not in ds.name_map or b not in ds.name_map or c not in ds.name_map:
            continue
        xa = split.X_train[ds.name_map[a]].to_numpy(dtype=np.float64)
        xb = split.X_train[ds.name_map[b]].to_numpy(dtype=np.float64)
        xc = split.X_train[ds.name_map[c]].to_numpy(dtype=np.float64)
        cv = lower_vs_full_cv(xa, xb, xc, y_train_arr, folds, n_bins=2)
        tested += 1
        if cv is None:
            rows.append({"parameter_A": a, "parameter_B": b, "parameter_C": c,
                         "channels": "+".join(sorted(channel_of[(a, b, c)])),
                         "evidence_label": "INCONCLUSIVE", "reason": "insufficient fold data"})
            continue
        label, reason = label_row(cv)

        from src.explain.stage2_v3_common import fit_threeway
        full_fit = fit_threeway(xa, xb, xc, y_train_arr, n_bins=2)
        fg = full_fit["full_grid"]
        best = np.unravel_index(np.argmax(fg), fg.shape)
        worst = np.unravel_index(np.argmin(fg), fg.shape)
        bin_label = lambda i: "high" if i == 1 else "low"

        rows.append({
            "parameter_A": a, "parameter_B": b, "parameter_C": c,
            "channels": "+".join(sorted(channel_of[(a, b, c)])),
            "lower_order_cv_rmse": round(cv["lower_cv_rmse"], 4),
            "full_3way_cv_rmse": round(cv["full_cv_rmse"], 4),
            "incremental_rmse_improvement_pct": cv["rmse_improvement_pct"],
            "lower_order_cv_r2": round(cv["lower_cv_r2"], 4),
            "full_3way_cv_r2": round(cv["full_cv_r2"], 4),
            "incremental_r2": cv["incremental_r2"],
            "fold_reproducibility": f"{cv['folds_full_better']}/{cv['n_folds_used']}",
            "high_yield_joint_condition": f"{a}={bin_label(best[0])}, {b}={bin_label(best[1])}, {c}={bin_label(best[2])}",
            "low_yield_joint_condition": f"{a}={bin_label(worst[0])}, {b}={bin_label(worst[1])}, {c}={bin_label(worst[2])}",
            "evidence_label": label,
            "reason": reason,
        })

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label"], key=lambda s: s.map(order) if s.name == "evidence_label" else s)
    out_path = f"{R}/stage2_runB_v3_three_way_interactions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} triples tested out of {len(triples)} generated)")
    print(df["evidence_label"].value_counts().to_string())
    print("\nTop STRONG 3-way findings:")
    print(df[df["evidence_label"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "parameter_C", "incremental_r2", "fold_reproducibility"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
