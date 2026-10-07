"""Local-only backend connecting the four LOCKED checkpoint models
(OHMIC/FIC/SIN/THIN) to the dashboard in dashboard/output/.

Scope: predicted final SortingYield only. No actual end-of-line yield is
connected here (IT will connect that later) -- this backend never writes to
or claims anything about an "actual yield" series.

Reuses src/models/checkpoint_inference.py's CheckpointPredictor AS-IS: no
retraining, no model-selection change, no modification to that file. This
module only adds (1) a thin HTTP/validation layer around it, (2) append-only
local persistence of every prediction made, and (3) static file serving for
the existing dashboard HTML/JS so the browser can call the API same-origin
(avoiding file:// CORS restrictions).

Binds to 127.0.0.1 only. No cloud calls, no telemetry, no global installs
(Flask is already present in the project .venv).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, redirect

from src.models.checkpoint_inference import CheckpointPredictor, CHECKPOINT_NAMES, ID_COLUMNS
from src.models.checkpoint_provenance import sha256_file, utc_now

MODE = "synthetic"  # the only mode with trained artifacts today; see dashboard/backend/README.md
DASHBOARD_OUTPUT_DIR = PROJECT_ROOT / "dashboard" / "output"
MODELS_DIR = PROJECT_ROOT / "outputs" / "checkpoint_training" / "models"
# Optional overrides, for isolated testing only (e.g. the missing-actual-result
# verification): a second instance can point at its own temp DB/port without
# ever touching the real dashboard/backend/predictions.db or its port. Normal
# launches (Launch_Dashboard.bat / no env vars set) are completely unchanged.
DB_PATH = Path(os.environ.get("YMX_DASHBOARD_DB_PATH", str(Path(__file__).resolve().parent / "predictions.db")))
PORT = int(os.environ.get("YMX_DASHBOARD_PORT", "8765"))

CHART_LABEL = {"OHMIC": "OHMIC", "FIC": "BFIC", "SIN": "SIN", "THIN": "THIN"}

app = Flask(__name__, static_folder=str(DASHBOARD_OUTPUT_DIR), static_url_path="/output")


# ----------------------------------------------------------------------
# Load the four locked predictors ONCE at startup. A load failure for one
# checkpoint does not take down the others -- it is reported per-checkpoint
# by /api/health and surfaced as a clear error if that checkpoint is used.
# ----------------------------------------------------------------------
PREDICTORS: dict[str, CheckpointPredictor] = {}
MODEL_INFO: dict[str, dict] = {}
LOAD_ERRORS: dict[str, str] = {}

for _name in CHECKPOINT_NAMES:
    try:
        _predictor = CheckpointPredictor(_name, mode=MODE)
        PREDICTORS[_name] = _predictor
        _pipeline_path = MODELS_DIR / f"{_name}_final_pipeline.joblib"
        MODEL_INFO[_name] = {
            "winning_model": _predictor.metadata["winning_model"],
            "feature_mode": _predictor.metadata["feature_mode"],
            "label": _predictor.metadata["label"],
            "mode": _predictor.metadata["split_provenance"]["mode"],
            "pipeline_sha256": sha256_file(_pipeline_path),
            "required_columns_count": len(_predictor.required_columns),
        }
    except (FileNotFoundError, AssertionError) as exc:
        LOAD_ERRORS[_name] = str(exc)

print(f"[startup] loaded predictors: {sorted(PREDICTORS)}", flush=True)
if LOAD_ERRORS:
    print(f"[startup] FAILED to load: {LOAD_ERRORS}", flush=True)


# ----------------------------------------------------------------------
# Persistence: append-only SQLite. Every prediction is a new row -- repeated
# predictions for the same LotName+WaferNum+checkpoint are never merged or
# overwritten, so history and repeats are both preserved.
# ----------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_name TEXT NOT NULL,
                wafer_num TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                predicted_yield REAL NOT NULL,
                created_at_utc TEXT NOT NULL,
                model_name TEXT NOT NULL,
                feature_mode TEXT NOT NULL,
                mode TEXT NOT NULL,
                label TEXT NOT NULL,
                pipeline_sha256 TEXT NOT NULL,
                source_filename TEXT,
                is_simulation INTEGER NOT NULL DEFAULT 0,
                simulation_date TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_checkpoint ON predictions(checkpoint)")
        # Migration path for a predictions.db created before simulation mode
        # existed: ALTER TABLE ADD COLUMN is a no-op-safe check first, so
        # existing live history (is_simulation defaults to 0, simulation_date
        # stays NULL) is preserved exactly as-is, never deleted or rewritten.
        existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
        if "is_simulation" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN is_simulation INTEGER NOT NULL DEFAULT 0")
        if "simulation_date" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN simulation_date TEXT")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS actual_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_name TEXT NOT NULL,
                wafer_num TEXT NOT NULL,
                sorting_yield REAL NOT NULL,
                result_date TEXT NOT NULL,
                uploaded_at_utc TEXT NOT NULL,
                source_filename TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_actual_results_wafer ON actual_results(lot_name, wafer_num)")


_init_db()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return redirect("/output/YieldMinx_dashboard_rev5.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "mode": MODE,
        "loaded_checkpoints": sorted(PREDICTORS),
        "load_errors": LOAD_ERRORS,
        "model_info": MODEL_INFO,
    })


SIMULATION_DATE_COLUMN = "SimulationDate"


@app.route("/api/predict", methods=["POST"])
def predict():
    checkpoint = request.form.get("checkpoint", "").strip().upper()
    if checkpoint not in CHECKPOINT_NAMES:
        return jsonify({"error": f"Unknown checkpoint {checkpoint!r}; expected one of {CHECKPOINT_NAMES}."}), 400
    if checkpoint not in PREDICTORS:
        return jsonify({"error": f"Checkpoint {checkpoint} failed to load at startup: "
                                  f"{LOAD_ERRORS.get(checkpoint, 'unknown error')}"}), 503

    is_simulation = request.form.get("is_simulation", "false").strip().lower() in ("1", "true", "yes")

    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "No measurements CSV was uploaded."}), 400

    try:
        df = pd.read_csv(upload.stream, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure as a clear 400, not a 500
        return jsonify({"error": f"Could not parse the uploaded file as CSV: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "The uploaded CSV has a header but no data rows."}), 400

    missing_ids = [c for c in ID_COLUMNS if c not in df.columns]
    if missing_ids:
        return jsonify({"error": f"Missing required identifier column(s): {missing_ids}. "
                                  f"Both LotName and WaferNum are required for every row."}), 400

    # Simulation dates are read here, for persistence/chart-positioning only,
    # and then dropped before the dataframe ever reaches the predictor --
    # SimulationDate is never a model input, in either mode.
    simulation_dates: dict[tuple, str] = {}
    predict_df = df
    if is_simulation:
        if SIMULATION_DATE_COLUMN not in df.columns:
            return jsonify({"error": f"Simulation uploads require a '{SIMULATION_DATE_COLUMN}' column."}), 400
        parsed = pd.to_datetime(df[SIMULATION_DATE_COLUMN], errors="coerce")
        bad_rows = df.index[parsed.isna()].tolist()
        if bad_rows:
            return jsonify({"error": f"{SIMULATION_DATE_COLUMN} could not be parsed on row(s) "
                                      f"{[r+2 for r in bad_rows][:10]} (1-based, including header)."}), 400
        for _, r in df.iterrows():
            simulation_dates[(str(r["LotName"]), str(r["WaferNum"]))] = str(r[SIMULATION_DATE_COLUMN])
        predict_df = df.drop(columns=[SIMULATION_DATE_COLUMN])  # <- never passed to the predictor

    predictor = PREDICTORS[checkpoint]
    try:
        result = predictor.predict(predict_df)
    except ValueError as exc:
        # predictor.predict() already lists the exact missing measurement columns
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - never leak a raw 500/traceback to the dashboard
        return jsonify({"error": f"Prediction failed: {exc}"}), 400

    info = MODEL_INFO[checkpoint]
    created_at = utc_now()  # the REAL prediction-creation timestamp, recorded in every mode
    saved = []
    with _db() as conn:
        for _, row in result.iterrows():
            lot_name, wafer_num = str(row.get("LotName", "")), str(row.get("WaferNum", ""))
            sim_date = simulation_dates.get((lot_name, wafer_num)) if is_simulation else None

            if is_simulation:
                # Upsert: a simulation point represents "the" simulated event for
                # this wafer+checkpoint. Re-uploading (identical or changed values)
                # replaces it in place rather than appending a duplicate chart
                # point. Live-mode rows (is_simulation=0) are never touched here.
                conn.execute(
                    "DELETE FROM predictions WHERE lot_name=? AND wafer_num=? AND checkpoint=? AND is_simulation=1",
                    (lot_name, wafer_num, checkpoint),
                )

            cur = conn.execute(
                "INSERT INTO predictions "
                "(lot_name, wafer_num, checkpoint, predicted_yield, created_at_utc, "
                " model_name, feature_mode, mode, label, pipeline_sha256, source_filename, "
                " is_simulation, simulation_date) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (lot_name, wafer_num, checkpoint,
                 float(row["Predicted Yield"]), created_at,
                 info["winning_model"], info["feature_mode"], info["mode"], info["label"],
                 info["pipeline_sha256"], upload.filename, int(is_simulation), sim_date),
            )
            saved.append({
                "id": cur.lastrowid, "lot_name": lot_name, "wafer_num": wafer_num, "checkpoint": checkpoint,
                "predicted_yield": float(row["Predicted Yield"]), "created_at_utc": created_at,
                "model_name": info["winning_model"], "feature_mode": info["feature_mode"],
                "mode": info["mode"], "label": info["label"],
                "is_simulation": is_simulation, "simulation_date": sim_date,
            })

    return jsonify({
        "checkpoint": checkpoint, "dashboard_label": CHART_LABEL[checkpoint],
        "is_simulation": is_simulation, "n_predicted": len(saved), "predictions": saved,
    })


@app.route("/api/predictions/<checkpoint>")
def list_predictions(checkpoint: str):
    checkpoint = checkpoint.strip().upper()
    if checkpoint not in CHECKPOINT_NAMES:
        return jsonify({"error": f"Unknown checkpoint {checkpoint!r}."}), 400
    # Default (no query param, or simulation=false) is EXACTLY the original
    # query -- normal/live-mode behavior is unchanged by simulation mode's
    # existence. simulation=true returns only the isolated simulation rows.
    is_simulation = request.args.get("simulation", "false").strip().lower() in ("1", "true", "yes")
    order_col = "simulation_date" if is_simulation else "created_at_utc"
    with _db() as conn:
        rows = conn.execute(
            f"SELECT * FROM predictions WHERE checkpoint = ? AND is_simulation = ? ORDER BY {order_col} ASC",
            (checkpoint, int(is_simulation)),
        ).fetchall()
    return jsonify([_row_to_dict(r) for r in rows])


@app.route("/api/actual-results", methods=["POST"])
def upload_actual_results():
    """Simulation-only: uploads actual final-yield results, matched to
    simulated predictions by LotName+WaferNum in the dashboard. Never feeds
    into any predictor -- this table is never read by /api/predict."""
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "No actual-results CSV was uploaded."}), 400
    try:
        df = pd.read_csv(upload.stream, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not parse the uploaded file as CSV: {exc}"}), 400
    if df.empty:
        return jsonify({"error": "The uploaded CSV has a header but no data rows."}), 400

    required = ["LotName", "WaferNum", "SortingYield", "ResultDate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required column(s): {missing}. "
                                  f"Expected: {required}."}), 400

    parsed_dates = pd.to_datetime(df["ResultDate"], errors="coerce")
    bad_rows = df.index[parsed_dates.isna()].tolist()
    if bad_rows:
        return jsonify({"error": f"ResultDate could not be parsed on row(s) "
                                  f"{[r+2 for r in bad_rows][:10]} (1-based, including header)."}), 400

    uploaded_at = utc_now()
    summary = {"new": 0, "duplicate": 0, "updated": 0}
    details = []
    with _db() as conn:
        for _, r in df.iterrows():
            lot_name, wafer_num = str(r["LotName"]), str(r["WaferNum"])
            new_yield, new_date = float(r["SortingYield"]), str(r["ResultDate"])
            existing = conn.execute(
                "SELECT sorting_yield, result_date FROM actual_results "
                "WHERE lot_name=? AND wafer_num=? ORDER BY id DESC LIMIT 1",
                (lot_name, wafer_num),
            ).fetchone()
            if existing is None:
                status = "new"
            elif abs(existing["sorting_yield"] - new_yield) < 1e-9 and existing["result_date"] == new_date:
                status = "duplicate"  # identical repeat -- recorded again for audit, but not flagged as a conflict
            else:
                status = "updated"  # conflicting value -- most recent upload wins for display; nothing deleted
            summary[status] += 1
            conn.execute(
                "INSERT INTO actual_results (lot_name, wafer_num, sorting_yield, result_date, "
                "uploaded_at_utc, source_filename) VALUES (?,?,?,?,?,?)",
                (lot_name, wafer_num, new_yield, new_date, uploaded_at, upload.filename),
            )
            details.append({"lot_name": lot_name, "wafer_num": wafer_num, "status": status})

    return jsonify({"n_rows": len(df), "summary": summary, "details": details})


@app.route("/api/actual-results")
def list_actual_results():
    """Returns the LATEST uploaded record per (LotName, WaferNum) -- the
    value the dashboard should display. Pass ?history=true for every row
    ever uploaded (nothing is ever deleted on conflict)."""
    full_history = request.args.get("history", "false").strip().lower() in ("1", "true", "yes")
    with _db() as conn:
        if full_history:
            rows = conn.execute("SELECT * FROM actual_results ORDER BY lot_name, wafer_num, id ASC").fetchall()
        else:
            rows = conn.execute("""
                SELECT ar.* FROM actual_results ar
                INNER JOIN (
                    SELECT lot_name, wafer_num, MAX(id) AS max_id FROM actual_results GROUP BY lot_name, wafer_num
                ) latest ON ar.id = latest.max_id
                ORDER BY ar.lot_name, ar.wafer_num
            """).fetchall()
    return jsonify([_row_to_dict(r) for r in rows])


if __name__ == "__main__":
    print(f"[startup] DB at {DB_PATH}", flush=True)
    print(f"[startup] serving dashboard from {DASHBOARD_OUTPUT_DIR}", flush=True)
    print(f"[startup] port {PORT}", flush=True)
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)
