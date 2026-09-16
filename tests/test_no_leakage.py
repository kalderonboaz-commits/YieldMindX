"""Guards the leakage contract: target-family and derived-leakage columns
must never end up in the model-facing feature matrix, and the ground-truth
answer key must never be imported by the discovery pipeline.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_dataset
from src.features.matrix import build_feature_matrix


def test_leakage_columns_excluded_from_features():
    ds = load_dataset()
    leaked = set(ds.contract.leakage_columns) & set(ds.feature_cols)
    assert not leaked, f"Leakage columns present in feature set: {leaked}"


def test_target_excluded_from_features():
    ds = load_dataset()
    assert ds.contract.target not in ds.feature_cols


def test_id_columns_excluded_from_features():
    ds = load_dataset()
    leaked = set(ds.contract.id_columns) & set(ds.feature_cols)
    assert not leaked, f"ID columns present in feature set: {leaked}"


def test_feature_matrix_has_no_leakage_columns():
    ds = load_dataset()
    fm = build_feature_matrix(ds)
    excluded = set(ds.contract.leakage_columns) | {ds.contract.target}
    present_original_names = set(fm.numeric_cols) | set(fm.categorical_cols)
    assert not (excluded & present_original_names)


def test_no_missing_values_in_final_matrix():
    ds = load_dataset()
    fm = build_feature_matrix(ds)
    assert fm.X.isna().sum().sum() == 0


def test_discovery_pipeline_does_not_import_ground_truth_docx():
    """The ground-truth doc must only ever be READ (via python-docx) by
    src/validation/benchmark_ground_truth.py -- never by data/feature/model/
    explain code, which must operate blind. A prose mention of the filename
    in a docstring (explaining WHY it's excluded) is fine; an actual
    docx import/open/Document(...) call is not."""
    forbidden_dirs = ["data", "features", "models", "explain"]
    suspicious_patterns = ["import docx", "from docx", "Document(", "GROUND_TRUTH_DOCX"]
    for d in forbidden_dirs:
        for py_file in (PROJECT_ROOT / "src" / d).glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for pattern in suspicious_patterns:
                assert pattern not in text, (
                    f"{py_file} contains '{pattern}' -- discovery code must stay blind to the ground-truth doc"
                )


if __name__ == "__main__":
    test_leakage_columns_excluded_from_features()
    test_target_excluded_from_features()
    test_id_columns_excluded_from_features()
    test_feature_matrix_has_no_leakage_columns()
    test_no_missing_values_in_final_matrix()
    test_discovery_pipeline_does_not_import_ground_truth_docx()
    print("All leakage-guard tests passed.")
