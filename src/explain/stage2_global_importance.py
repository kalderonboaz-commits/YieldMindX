"""Stage 2 Run B: combined global importance table (SHAP + gain +
permutation), with per-predictor ranks and a cross-method agreement label.

Reuses the SHAP/gain/permutation importance already computed for Run B in
the prior checkpoint (outputs/reports/stage2_runB_{shap,gain,permutation}_importance.csv)
-- Run B's model, data, and split are unchanged (locked), so recomputing
would produce bit-identical results (already established determinism);
reusing avoids redundant compute.
"""
from __future__ import annotations

import pandas as pd

OUTPUTS_DIR = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports"


def build_combined_importance_table() -> pd.DataFrame:
    shap_s = pd.read_csv(f"{OUTPUTS_DIR}/stage2_runB_shap_importance.csv", index_col=0).iloc[:, 0]
    gain_s = pd.read_csv(f"{OUTPUTS_DIR}/stage2_runB_gain_importance.csv", index_col=0).iloc[:, 0]
    perm_s = pd.read_csv(f"{OUTPUTS_DIR}/stage2_runB_permutation_importance.csv", index_col=0).iloc[:, 0]

    all_features = sorted(set(shap_s.index) | set(gain_s.index) | set(perm_s.index))
    df = pd.DataFrame(index=all_features)
    df["shap_importance"] = shap_s.reindex(all_features).fillna(0.0)
    df["gain_importance"] = gain_s.reindex(all_features).fillna(0.0)
    df["permutation_importance"] = perm_s.reindex(all_features).fillna(0.0)

    df["shap_rank"] = df["shap_importance"].rank(ascending=False, method="min").astype(int)
    df["gain_rank"] = df["gain_importance"].rank(ascending=False, method="min").astype(int)
    df["permutation_rank"] = df["permutation_importance"].rank(ascending=False, method="min").astype(int)

    n = len(df)
    df["shap_percentile"] = 100 * (1 - (df["shap_rank"] - 1) / n)
    df["gain_percentile"] = 100 * (1 - (df["gain_rank"] - 1) / n)
    df["permutation_percentile"] = 100 * (1 - (df["permutation_rank"] - 1) / n)

    pct_cols = ["shap_percentile", "gain_percentile", "permutation_percentile"]
    df["percentile_spread"] = df[pct_cols].max(axis=1) - df[pct_cols].min(axis=1)

    def agreement_label(spread: float, min_pct: float) -> str:
        if min_pct < 50:
            # if all three methods agree the feature is unimportant (bottom half), that's
            # itself a form of agreement
            return "high_agreement" if spread <= 15 else ("moderate_agreement" if spread <= 35 else "low_agreement")
        return "high_agreement" if spread <= 10 else ("moderate_agreement" if spread <= 25 else "low_agreement")

    df["method_agreement"] = [
        agreement_label(row["percentile_spread"], row[pct_cols].min())
        for _, row in df.iterrows()
    ]

    df = df.sort_values("shap_rank")
    df.index.name = "raw_csv_parameter_name"
    return df.reset_index()


if __name__ == "__main__":
    df = build_combined_importance_table()
    print(f"Combined global importance table: {len(df)} predictors")
    print("\nTop 25 by SHAP importance:")
    print(df.head(25)[["raw_csv_parameter_name", "shap_importance", "shap_rank",
                        "gain_importance", "gain_rank", "permutation_importance", "permutation_rank",
                        "method_agreement"]].to_string(index=False))

    print("\nMethod agreement distribution:")
    print(df["method_agreement"].value_counts().to_string())

    print("\nFeatures with HIGH gain/perm importance but LOW SHAP rank (or vice versa) -- top 10 largest spreads among top-200-by-any-method:")
    top_by_any = df[(df["shap_rank"] <= 200) | (df["gain_rank"] <= 200) | (df["permutation_rank"] <= 200)]
    print(top_by_any.sort_values("percentile_spread", ascending=False).head(10)[
        ["raw_csv_parameter_name", "shap_rank", "gain_rank", "permutation_rank", "percentile_spread"]
    ].to_string(index=False))

    df.to_csv(f"{OUTPUTS_DIR}/stage2_runB_global_importance.csv", index=False)
    print(f"\nSaved {OUTPUTS_DIR}/stage2_runB_global_importance.csv")
