"""Tree-ensemble models: Random Forest and LightGBM.

These are the primary predictive models -- they handle nonlinearity,
interactions, and the dataset's heavy multicollinearity natively, without
manual feature selection. Both are trained on the FULL feature matrix
(raw numeric + engineered + one-hot categorical); unlike the linear
baseline, no de-collinearization is needed for these to fit well, though
(per user rule 3) their importance outputs still need cluster-level
aggregation downstream -- trees do not solve the attribution problem, they
just tolerate the redundancy during training.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def make_random_forest(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


def make_lightgbm(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        verbose=-1,
    )


if __name__ == "__main__":
    from src.data.load import load_dataset
    from src.features.matrix import build_feature_matrix

    ds = load_dataset()
    fm = build_feature_matrix(ds)

    rf = make_random_forest()
    rf.fit(fm.X, ds.y)
    print("RF trained. In-sample R2:", rf.score(fm.X, ds.y).__round__(4))

    gbm = make_lightgbm()
    gbm.fit(fm.X, ds.y)
    pred = gbm.predict(fm.X)
    import numpy as np
    r2 = 1 - ((ds.y - pred) ** 2).sum() / ((ds.y - ds.y.mean()) ** 2).sum()
    print("LightGBM trained. In-sample R2:", round(r2, 4))
