"""Per-wafer explanation: "why did this wafer predict low/high yield?"

Produces a SHAP-based signed contribution breakdown for a single wafer,
plus a short natural-language narrative. Where the top contributors belong
to a known multicollinear cluster, the narrative names the CLUSTER rather
than an arbitrarily-selected single column (user rule 3) -- e.g. "elevated
leakage-current family" rather than picking one of 70 near-identical `lkg_*`
columns to blame.

Explicitly prediction/explanation, not causal attribution (user rule 4):
the narrative describes what drove the MODEL's prediction, not a proven
physical root cause.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import shap

from src.features.clustering import FeatureClustering
from src.features.matrix import FeatureMatrix


@dataclasses.dataclass
class WaferExplanation:
    row_id: int
    predicted_yield: float
    base_value: float
    contributions: pd.DataFrame  # columns: feature, cluster_label, value, shap_contribution
    narrative: str


def _cluster_label_for(feature: str, clustering: FeatureClustering) -> str:
    cid = clustering.cluster_of.get(feature)
    if cid is None:
        return feature
    members = clustering.clusters.get(int(cid), [feature])
    if len(members) <= 1:
        return feature
    rep = clustering.representatives[int(cid)]
    return f"{rep} family ({len(members)} correlated ET columns)"


def explain_wafer(
    model, X: pd.DataFrame, fm: FeatureMatrix, row_pos: int, top_n: int = 8
) -> WaferExplanation:
    explainer = shap.TreeExplainer(model)
    row = X.iloc[[row_pos]]
    sv = explainer.shap_values(row)[0]
    base_value = float(explainer.expected_value)
    pred = base_value + sv.sum()

    contrib = pd.Series(sv, index=X.columns)
    contrib.index = [fm.to_original(c) for c in contrib.index]
    raw_values = row.iloc[0].copy()
    raw_values.index = [fm.to_original(c) for c in raw_values.index]

    ordered = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    top = ordered.head(top_n)

    rows = []
    for feat, val in top.items():
        rows.append({
            "feature": feat,
            "cluster_label": _cluster_label_for(feat, fm.clustering),
            "value": raw_values.get(feat, np.nan),
            "shap_contribution": val,
        })
    contributions = pd.DataFrame(rows)

    direction = "below" if pred < base_value else "above"
    delta = abs(pred - base_value)
    lines = [
        f"Predicted SortingYield = {pred:.2f}% ({delta:.2f} points {direction} the average of {base_value:.2f}%)."
    ]
    pos = contributions[contributions["shap_contribution"] < 0] if pred < base_value else contributions[contributions["shap_contribution"] > 0]
    drivers = pos.head(3)
    if len(drivers):
        driver_desc = "; ".join(
            f"{r.cluster_label} (value={r.value:.4g}, contribution={r.shap_contribution:+.3f} pts)"
            for r in drivers.itertuples()
        )
        lines.append(f"Primary model-attributed drivers: {driver_desc}.")
    lines.append(
        "NOTE: this is the model's learned association for this wafer, not a confirmed physical root cause -- "
        "treat as a prioritized starting point for engineering investigation."
    )
    narrative = " ".join(lines)

    return WaferExplanation(
        row_id=row_pos, predicted_yield=pred, base_value=base_value,
        contributions=contributions, narrative=narrative,
    )


if __name__ == "__main__":
    import joblib

    from src.data.load import load_dataset
    from src.features.matrix import build_feature_matrix
    from src.models.train import OUTPUTS_DIR

    ds = load_dataset()
    fm = build_feature_matrix(ds)
    split = joblib.load(OUTPUTS_DIR / "models" / "holdout_split.joblib")
    test_idx = split["test_idx"]

    gbm = joblib.load(OUTPUTS_DIR / "models" / "lightgbm.joblib")
    X_test = fm.X.iloc[test_idx].reset_index(drop=True)
    y_test = ds.y.iloc[test_idx].reset_index(drop=True)

    worst_pos = y_test.idxmin()
    best_pos = y_test.idxmax()

    for label, pos in [("LOWEST actual yield wafer in test set", worst_pos), ("HIGHEST actual yield wafer in test set", best_pos)]:
        print(f"\n=== {label} (actual={y_test.iloc[pos]:.2f}%) ===")
        expl = explain_wafer(gbm, X_test, fm, pos)
        print(expl.narrative)
        print(expl.contributions.round(4).to_string(index=False))
