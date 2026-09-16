"""Stage 2 Run B MLP -- section 10: 3-way interaction evidence.

Candidate triples generated generically from the STRONG/MODERATE MLP 2-way
pairs just confirmed (shared-node chaining -- third pairwise edge not
required, same principle as the v4/v5 interaction-graph fix), capped by
summed edge H-statistic (evidence-based, disclosed) to stay manageable.

Test: a PDP-style Yates 2^3 factorial contrast evaluated through the
network's own background-averaged predictions (same background-sample
averaging convention as the 2-way H-statistic test) -- NOT from raw
observed-data cell means (that is what v2/v3/v4/v5 already do with
LightGBM/binned data; this is the model-prediction analogue). For each of
the 8 low/high combinations of (A,B,C), the three columns are overridden
to representative low/high values across the WHOLE background sample and
the network's mean prediction is taken as that cell's value; the Yates
3-way contrast is the part of the resulting 2x2x2 pattern not explained by
any main effect or 2-way effect, by construction of the factorial
decomposition (the same logic already used by v2/v3's three_way_contrast,
applied here to model predictions instead of raw-data cell means).
"""
from __future__ import annotations

import time
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explain.stage2_mlp_feature_evidence import CACHE_PATH
from src.explain.stage2_v2_common import R, load_locked_split

CHAIN_EDGE_CAP = 150   # cap on generated candidate triples, by summed constituent-edge H-statistic
BACKGROUND_SIZE = 40
LO_PCT, HI_PCT = 20, 80  # representative "low"/"high" percentiles for the 2x2x2 grid


def predict_cell_mean(pipe, X_bg: pd.DataFrame, overrides: dict) -> float:
    X_mod = X_bg.copy()
    for col, val in overrides.items():
        X_mod[col] = val
    return float(pipe.predict(X_mod.astype(np.float64)).mean())


def three_way_contrast(pipe, X_bg: pd.DataFrame, col_a: str, col_b: str, col_c: str):
    los = {}
    his = {}
    for col in (col_a, col_b, col_c):
        col_vals = X_bg[col].to_numpy(dtype=np.float64)
        los[col] = float(np.percentile(col_vals, LO_PCT))
        his[col] = float(np.percentile(col_vals, HI_PCT))

    cell_means = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                overrides = {
                    col_a: his[col_a] if i else los[col_a],
                    col_b: his[col_b] if j else los[col_b],
                    col_c: his[col_c] if k else los[col_c],
                }
                cell_means[(i, j, k)] = predict_cell_mean(pipe, X_bg, overrides)

    contrast = 0.0
    for (i, j, k), m in cell_means.items():
        sign = (2 * i - 1) * (2 * j - 1) * (2 * k - 1)
        contrast += sign * m
    contrast /= 8.0

    best_cell = max(cell_means, key=cell_means.get)
    worst_cell = min(cell_means, key=cell_means.get)
    return contrast, cell_means, best_cell, worst_cell


def main():
    split = load_locked_split()
    cache = joblib.load(CACHE_PATH)
    fold_models = cache["fold_models"]
    full_fit = cache["full_fit"]
    name_to_col = dict(zip(cache["orig_names"], cache["feature_cols"]))
    full_pipe = Pipeline([("scaler", full_fit.scaler), ("mlp", full_fit.model)])
    fold_pipes = [Pipeline([("scaler", fm["fit"].scaler), ("mlp", fm["fit"].model)]) for fm in fold_models]

    twoway = pd.read_csv(f"{R}/stage2_runB_mlp_2way_interactions.csv")
    strong_mod = twoway[twoway["confidence"].isin(["STRONG", "MODERATE"])]
    print(f"Building candidate triples from {len(strong_mod)} STRONG/MODERATE MLP 2-way pairs (shared-node chaining)...")

    adjacency: dict[str, set] = {}
    edge_h: dict[frozenset, float] = {}
    for _, row in strong_mod.iterrows():
        a, b = row["parameter_A"], row["parameter_B"]
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        edge_h[frozenset([a, b])] = row["mlp_h_statistic"]

    raw_triples = []
    for node, neighbors in adjacency.items():
        neighbors_l = sorted(neighbors)
        for i in range(len(neighbors_l)):
            for j in range(i + 1, len(neighbors_l)):
                b, c = neighbors_l[i], neighbors_l[j]
                score = edge_h.get(frozenset([node, b]), 0) + edge_h.get(frozenset([node, c]), 0)
                raw_triples.append((score, tuple(sorted([node, b, c]))))
    raw_triples = list({t: s for s, t in raw_triples}.items())
    raw_triples.sort(key=lambda kv: -kv[1])
    candidate_triples = [t for t, s in raw_triples[:CHAIN_EDGE_CAP]]
    print(f"Candidate triples (top {CHAIN_EDGE_CAP} of {len(raw_triples)} by summed edge H-statistic): {len(candidate_triples)}")

    X_bg_full = split.X_train.sample(min(BACKGROUND_SIZE, len(split.X_train)), random_state=42)

    print("Testing candidate triples with the full model...")
    t0 = time.time()
    rows = []
    for a, b, c in candidate_triples:
        ca, cb, cc = name_to_col.get(a), name_to_col.get(b), name_to_col.get(c)
        if not all([ca, cb, cc]):
            continue
        contrast, cells, best_cell, worst_cell = three_way_contrast(full_pipe, X_bg_full, ca, cb, cc)

        fold_signs = []
        for fm, pipe in zip(fold_models, fold_pipes):
            X_fold_bg = split.X_train.iloc[fm["tr"]].sample(min(BACKGROUND_SIZE, len(fm["tr"])), random_state=42)
            fc, _, _, _ = three_way_contrast(pipe, X_fold_bg, ca, cb, cc)
            fold_signs.append(np.sign(fc))
        main_sign = np.sign(contrast)
        sign_consistency = sum(1 for s in fold_signs if s == main_sign)

        if sign_consistency >= 4:
            label = "STRONG"
        elif sign_consistency == 3:
            label = "MODERATE"
        elif sign_consistency <= 1:
            label = "INCONCLUSIVE"
        else:
            label = "WEAK"

        bin_lab = lambda i: "high" if i else "low"
        rows.append({
            "parameter_A": a, "parameter_B": b, "parameter_C": c,
            "contrast_3way": round(contrast, 4),
            "fold_sign_reproducibility": f"{sign_consistency}/5",
            "high_yield_joint_condition": f"{a}={bin_lab(best_cell[0])}, {b}={bin_lab(best_cell[1])}, {c}={bin_lab(best_cell[2])}",
            "low_yield_joint_condition": f"{a}={bin_lab(worst_cell[0])}, {b}={bin_lab(worst_cell[1])}, {c}={bin_lab(worst_cell[2])}",
            "confidence": label,
        })
    print(f"Done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["confidence", "contrast_3way"],
                         key=lambda s: (s.map(order) if s.name == "confidence" else s.abs()),
                         ascending=[True, False])
    out_path = f"{R}/stage2_runB_mlp_3way_interactions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} tested triples)")
    print(df["confidence"].value_counts().to_string())
    print("\nTop 10 STRONG by |contrast|:")
    print(df[df["confidence"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "parameter_C", "contrast_3way", "fold_sign_reproducibility"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
