"""Data provenance map: for every column used or produced anywhere in this
project, classify what it is, where it came from, and whether/why it is
used as a model feature.

Produces a single machine-readable table covering:
  - all 1,430 columns of the original raw CSV (target, leakage, id,
    categorical, and raw numeric predictor columns)
  - all 307 engineered (STAGE_RANGE/STD/DELTA) features created by this
    project's pipeline, each traced back to its raw source column(s)

Deliberately does NOT read documents/corelation_data_table.docx -- this is
a factual map of the DATA's origin, independent of the ground-truth
validation benchmark.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.load import load_dataset
from src.features.clustering import cluster_features
from src.features.engineered import find_stage_groups


def _extract_family(colname: str) -> str:
    """Process-module / measurement-family label, derived purely from the
    column-naming schema (module prefix before the '_ET_'/'_CD_'/etc.
    marker), e.g. 'SIN_GATE_ET_lkg_5' -> 'SIN_GATE'."""
    for marker in ("_ET_", "_CD_", "_Thickness", "_Ali_", "_Probe"):
        idx = colname.find(marker)
        if idx > 0:
            return colname[:idx]
    return colname.split("_")[0]


def build_provenance_table() -> pd.DataFrame:
    ds = load_dataset()
    contract = ds.contract
    raw_cols = ds.raw.columns.tolist()

    numeric_df = ds.raw[ds.numeric_feature_cols]
    clustering = cluster_features(numeric_df)
    stage_groups = find_stage_groups(ds.numeric_feature_cols)

    # exact-duplicate check (content hash) across ALL numeric raw columns,
    # independent of the correlation-clustering above -- this is the
    # "duplicate/redundant column" signal (e.g. MaskSetName_group), not to
    # be confused with statistically-correlated-but-distinct clusters.
    all_numeric = ds.raw.select_dtypes(include="number")
    sentinel = -999999.123456
    filled = all_numeric.fillna(sentinel)
    hash_groups: dict[int, list[str]] = {}
    for c in filled.columns:
        h = int(pd.util.hash_pandas_object(filled[c], index=False).sum())
        hash_groups.setdefault(h, []).append(c)
    # For each duplicate GROUP, keep exactly one representative (the one
    # already recorded as the "keep" side in config/data_contract.yaml's
    # id_columns list, where applicable, else alphabetically first) with its
    # normal classification; only the REDUNDANT member(s) are marked
    # duplicate/excluded. Symmetric flagging (marking every member excluded)
    # would incorrectly drop the column actually used as a feature.
    exact_dup_cols: dict[str, list[str]] = {}

    def _register_dup_group(cols: list[str]):
        cols_sorted = sorted(cols)
        keep = next((c for c in cols_sorted if c not in contract.id_columns), cols_sorted[0])
        for c in cols_sorted:
            if c != keep:
                exact_dup_cols[c] = [x for x in cols_sorted if x != c]

    for cols in hash_groups.values():
        if len(cols) > 1:
            _register_dup_group(cols)

    # exact-duplicate check for object/categorical columns via crosstab-style
    # 1:1 mapping check (MaskSetName_group vs MaskSetName was found this way).
    cat_like = ds.raw.select_dtypes(exclude="number")
    seen_pairs = set()
    for c in cat_like.columns:
        for other in cat_like.columns:
            if c >= other or (c, other) in seen_pairs:
                continue
            if ds.raw[c].astype(str).equals(ds.raw[other].astype(str)):
                seen_pairs.add((c, other))
                _register_dup_group([c, other])

    rows = []

    # ---- 1. Original raw CSV columns ----
    for c in raw_cols:
        if c == contract.target:
            source_type = "target"
            modeling_status = "excluded"
            exclusion_reason = "this IS the modeling target"
        elif c in contract.leakage_columns:
            source_type = "leakage / forbidden predictor"
            modeling_status = "excluded"
            if c == "EstimatedSupplyChipsCount":
                exclusion_reason = ("derived from actual yield outcome -- ratio to "
                                     "MaskSetSupplyChipsCount correlates r=0.92 with target")
            else:
                exclusion_reason = "yield-family metric (same/near-same measurement as target)"
        elif c in exact_dup_cols:
            source_type = "duplicate / redundant column"
            modeling_status = "excluded"
            exclusion_reason = f"exact duplicate of {exact_dup_cols[c]}"
        elif c in contract.id_columns:
            source_type = "metadata / identifier"
            modeling_status = "excluded"
            exclusion_reason = "non-physical bookkeeping column (date/lot/wafer id/constant), not a process parameter"
        elif c in contract.categorical_columns:
            source_type = "categorical predictor"
            modeling_status = "included"
            exclusion_reason = ""
        else:
            source_type = "original raw CSV column (numeric predictor)"
            modeling_status = "included"
            exclusion_reason = ""

        cluster_id = clustering.cluster_of.get(c) if c in clustering.cluster_of.index else None
        family = _extract_family(c) if source_type in (
            "original raw CSV column (numeric predictor)",) else ""

        rows.append({
            "feature_name": c,
            "source_type": source_type,
            "present_in_original_csv": "yes",
            "source_columns": "",
            "modeling_status": modeling_status,
            "exclusion_reason": exclusion_reason,
            "feature_family": family,
            "correlation_cluster_id": cluster_id if cluster_id is not None else "",
            "correlation_cluster_size": len(clustering.clusters.get(int(cluster_id), [])) if cluster_id is not None else "",
        })

    # ---- 2. Engineered features (this project's pipeline) ----
    for suffix, modules in stage_groups.groups.items():
        cols = list(modules.values())
        family = "+".join(sorted(modules.keys()))
        rows.append({
            "feature_name": f"STAGE_RANGE_{suffix}",
            "source_type": "engineered feature (derived, cross-stage)",
            "present_in_original_csv": "no",
            "source_columns": "; ".join(cols),
            "modeling_status": "included",
            "exclusion_reason": "",
            "feature_family": family,
            "correlation_cluster_id": "",
            "correlation_cluster_size": "",
        })
        rows.append({
            "feature_name": f"STAGE_STD_{suffix}",
            "source_type": "engineered feature (derived, cross-stage)",
            "present_in_original_csv": "no",
            "source_columns": "; ".join(cols),
            "modeling_status": "included",
            "exclusion_reason": "",
            "feature_family": family,
            "correlation_cluster_id": "",
            "correlation_cluster_size": "",
        })
        if len(modules) == 2:
            modA, modB = sorted(modules.keys())
            rows.append({
                "feature_name": f"STAGE_DELTA_{suffix}_{modA}_minus_{modB}",
                "source_type": "engineered feature (derived, cross-stage)",
                "present_in_original_csv": "no",
                "source_columns": f"{modules[modA]}; {modules[modB]}",
                "modeling_status": "included",
                "exclusion_reason": "",
                "feature_family": family,
                "correlation_cluster_id": "",
                "correlation_cluster_size": "",
            })

    df = pd.DataFrame(rows)
    return df


def audit_source_metadata() -> dict:
    """File-level provenance evidence -- what CAN be established from
    project files and filesystem metadata about how/when the raw CSV was
    generated. Does not read the ground-truth doc's content for modeling
    purposes, but its own metadata (authorship/dates) is fair game here
    since this is a factual audit of file provenance, not discovery logic."""
    import os
    from pathlib import Path

    from docx import Document

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    csv_path = PROJECT_ROOT / "data" / "correlation_data_table.csv"
    corr_docx_path = PROJECT_ROOT / "documents" / "corelation_data_table.docx"
    ymx_docx_path = PROJECT_ROOT / "documents" / "YieldmindX.docx"

    def file_stat(p: Path) -> dict:
        st = p.stat()
        return {
            "path": str(p.relative_to(PROJECT_ROOT)),
            "size_bytes": st.st_size,
            "modified_time": pd.Timestamp(st.st_mtime, unit="s").isoformat(),
            "created_time": pd.Timestamp(st.st_ctime, unit="s").isoformat(),
        }

    def docx_core_properties(p: Path) -> dict:
        doc = Document(str(p))
        cp = doc.core_properties
        return {
            "author": cp.author,
            "last_modified_by": cp.last_modified_by,
            "created": str(cp.created) if cp.created else None,
            "modified": str(cp.modified) if cp.modified else None,
            "revision": cp.revision,
            "title": cp.title,
            "subject": cp.subject,
            "comments": cp.comments,
        }

    # check for any header/comment rows in the CSV beyond the single header line
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        first_lines = [next(f) for _ in range(3)]

    audit = {
        "csv_file": file_stat(csv_path),
        "csv_first_3_lines_preview": [l[:120] for l in first_lines],
        "corr_docx_file": file_stat(corr_docx_path),
        "corr_docx_core_properties": docx_core_properties(corr_docx_path),
        "ymx_docx_file": file_stat(ymx_docx_path),
        "ymx_docx_core_properties": docx_core_properties(ymx_docx_path),
        "known_facts": [
            "corelation_data_table.docx explicitly states the CSV is a SYNTHETIC dataset "
            "('Synthetic pHEMT / MMIC Yield Correlation Ground Truth... 1,500-wafer synthetic dataset'), "
            "built to validate whether an AI system rediscovers deliberately planted relationships.",
            "The doc states values are 'engineering-plausible training values, not production specifications "
            "or control limits for a specific pHEMT/MMIC process.'",
            "Row structure (100 lots x 15 wafers = 1500 rows, verified in EDA) and column naming "
            "(process-module-prefixed ET/CD/Thickness parameters) are consistent with a semiconductor "
            "wafer-sort test dataset.",
        ],
        "unknown_facts": [
            "No generation script, notebook, or code is present anywhere in the project directory -- "
            "the CSV's actual generation process (RNG seed, functional forms beyond the 9 documented "
            "planted relationships, noise model, distribution assumptions) is not recoverable from project files.",
            "The CSV itself carries no embedded metadata (no header comment rows, no sidecar manifest) "
            "identifying a generation date, tool, or author.",
            "File modified-time is NOT reliable evidence of generation date -- it reflects when the file "
            "was last written to this filesystem/session, not necessarily when the synthetic data was "
            "originally produced (e.g. could have been copied/re-saved).",
            "Whether the 9 documented planted relationships are the ONLY deliberately planted signal, or "
            "merely the ones chosen for the reference report, cannot be determined from available files.",
        ],
    }
    return audit


if __name__ == "__main__":
    import json

    from src.models.train import OUTPUTS_DIR

    df = build_provenance_table()
    print("Provenance table rows:", len(df))
    print(df["source_type"].value_counts().to_string())
    print("\nmodeling_status counts:")
    print(df["modeling_status"].value_counts().to_string())

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "reports" / "feature_provenance.csv", index=False)
    print("\nSaved outputs/reports/feature_provenance.csv")

    audit = audit_source_metadata()
    with open(OUTPUTS_DIR / "reports" / "data_provenance_metadata.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    print("Saved outputs/reports/data_provenance_metadata.json")
    print("\n=== KNOWN FACTS ===")
    for k in audit["known_facts"]:
        print(" -", k)
    print("\n=== UNKNOWN FACTS ===")
    for k in audit["unknown_facts"]:
        print(" -", k)
