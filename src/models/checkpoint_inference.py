"""Inference entry point for the four locked checkpoint pipelines
(OHMIC / FIC / SIN / THIN).

Loads one checkpoint's saved joblib pipeline + metadata and predicts
SortingYield from a measurements table that contains ONLY that
checkpoint's own candidate predictor columns (plus, optionally, the
LotName/WaferNum identifiers, which are passed through untouched and
never used as predictors). The caller never needs to supply
SortingYield or any later-checkpoint's columns -- e.g. the OHMIC
predictor needs only OHMIC_* columns, not FIC_/SIN_/THIN_* or the
target.

This module performs no training and no feature selection: the fitted
pipeline already contains its own imputation/constant-removal/
redundancy-selection/scaling steps, fit once at lock time. Column order
does not matter -- the pipeline is given exactly the candidate columns
by name, in the name order recorded in metadata at training time.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_NAMES = ("OHMIC", "FIC", "SIN", "THIN")

ID_COLUMNS = ("LotName", "WaferNum")


def _models_dir(mode: str = "synthetic") -> Path:
    base = PROJECT_ROOT / "outputs" / ("checkpoint_training" if mode == "synthetic" else f"checkpoint_training_{mode}")
    return base / "models"


class CheckpointPredictor:
    """Loads one checkpoint's locked pipeline + metadata once; call
    .predict(measurements) as many times as needed."""

    def __init__(self, checkpoint_name: str, mode: str = "synthetic", models_dir: Path | None = None):
        if checkpoint_name not in CHECKPOINT_NAMES:
            raise ValueError(f"Unknown checkpoint {checkpoint_name!r}; expected one of {CHECKPOINT_NAMES}")
        self.checkpoint_name = checkpoint_name
        self.mode = mode
        md = models_dir or _models_dir(mode)

        meta_path = md / f"{checkpoint_name}_metadata.json"
        pipeline_path = md / f"{checkpoint_name}_final_pipeline.joblib"
        if not meta_path.exists() or not pipeline_path.exists():
            raise FileNotFoundError(
                f"[{checkpoint_name}] expected both {meta_path} and {pipeline_path} to exist. "
                f"Run src/models/checkpoint_train_run.py for mode={mode!r} first."
            )

        with open(meta_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Two independent layers against loading an artifact trained under a
        # different mode: (1) synthetic/real artifacts live in physically
        # separate directories (_models_dir), so this only fires if a file
        # was manually copied/misplaced across them; (2) cross-check the
        # mode actually recorded in the metadata at training time, in BOTH
        # directions -- not just "reject non-synthetic metadata if synthetic
        # was requested", which would silently accept a real-mode request
        # being served synthetic-trained artifacts.
        recorded_mode = self.metadata.get("split_provenance", {}).get("mode")
        if recorded_mode != mode:
            raise AssertionError(
                f"[{checkpoint_name}] requested mode={mode!r} but the loaded artifact at {meta_path} "
                f"was trained with mode={recorded_mode!r} -- refusing to load a cross-mode artifact."
            )
        expected_label = f"{mode.upper()} DATA RESULT"
        if self.metadata.get("label") != expected_label:
            raise AssertionError(
                f"[{checkpoint_name}] expected label {expected_label!r}, got {self.metadata.get('label')!r}."
            )

        self.required_columns: list[str] = self.metadata["candidate_feature_names"]
        self.pipeline = joblib.load(pipeline_path)

    def predict(self, measurements: pd.DataFrame) -> pd.DataFrame:
        """measurements: a DataFrame containing at least this checkpoint's
        required_columns (any extra columns, including SortingYield or
        later-checkpoint columns, are ignored, never required)."""
        missing = [c for c in self.required_columns if c not in measurements.columns]
        if missing:
            raise ValueError(
                f"[{self.checkpoint_name}] measurements are missing {len(missing)} required column(s): "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )

        X = measurements[self.required_columns]
        preds = self.pipeline.predict(X)

        out = pd.DataFrame(index=measurements.index)
        for id_col in ID_COLUMNS:
            if id_col in measurements.columns:
                out[id_col] = measurements[id_col].values
        out["Predicted Yield"] = preds
        out["checkpoint"] = self.checkpoint_name
        return out


def predict_checkpoint(checkpoint_name: str, measurements: pd.DataFrame, mode: str = "synthetic") -> pd.DataFrame:
    """Convenience one-shot wrapper around CheckpointPredictor for a single call."""
    return CheckpointPredictor(checkpoint_name, mode=mode).predict(measurements)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict SortingYield for one checkpoint from a measurements CSV.")
    parser.add_argument("checkpoint", choices=CHECKPOINT_NAMES)
    parser.add_argument("measurements_csv", help="CSV containing at least this checkpoint's required predictor columns.")
    parser.add_argument("--mode", default="synthetic", choices=("synthetic", "real"))
    parser.add_argument("--out", default=None, help="Output CSV path; defaults to stdout preview.")
    args = parser.parse_args()

    df_in = pd.read_csv(args.measurements_csv, low_memory=False)
    result = predict_checkpoint(args.checkpoint, df_in, mode=args.mode)
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote {len(result)} predictions to {args.out}")
    else:
        print(result.head(10).to_string(index=False))
        print(f"... {len(result)} rows total")
