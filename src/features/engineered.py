"""Generic, blind cross-process-stage derived features.

Methodology note (per user rule 2): this feature set is built purely from
the RAW COLUMN NAMING SCHEMA of the dataset -- i.e. the fact that the same
electrical-test suffix (e.g. "Rc_an_ct2", "Gm_max") is measured under
multiple different process-module prefixes (e.g. ANODE_ET_, OHMIC_ET_,
THIN_ET_, TOPSIN_ET_, FIC_ET_...). That module-prefix structure is metadata
about the fab process flow, not information read from
documents/corelation_data_table.docx. No specific pair of stages or specific
suffix was chosen because the ground-truth doc mentioned it; the procedure
below is applied uniformly and automatically to every suffix that happens to
recur across >=2 modules.

The `MEASURE_ET_*` block is deliberately excluded from this pairing logic:
an EDA check (src/eda/profile.py) showed that MEASURE_ET_<suffix> columns
are essentially uncorrelated with module_ET_<suffix> columns sharing the
same suffix (e.g. r=0.16 for C_AN_2), i.e. they are a separate, independent
measurement set despite the coincidental naming overlap -- pairing them as
"stages" would be physically meaningless.

For each suffix shared by >=2 non-MEASURE modules, for a given row:
  - STAGE_RANGE_<suffix>  = max(value across modules) - min(value across modules)
  - STAGE_STD_<suffix>    = std(value across modules)
  - STAGE_DELTA_<suffix>_<modA>_minus_<modB>  (only when exactly 2 modules
    share the suffix, so the signed difference is unambiguous)

"Probe_Number"-type suffixes (test-equipment/site metadata, not a physical
electrical quantity) are excluded from this generation, since a stage-delta
of a probe index number is not physically meaningful.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd

_SUFFIX_PATTERN = re.compile(r"^([A-Za-z0-9]+)_ET_(.+)$")
_EXCLUDED_SUFFIX_SUBSTRINGS = ("Probe_Number",)


@dataclasses.dataclass
class StageGroups:
    groups: dict[str, dict[str, str]]  # suffix -> {module_prefix: column_name}


def find_stage_groups(numeric_feature_cols: list[str]) -> StageGroups:
    groups: dict[str, dict[str, str]] = {}
    for col in numeric_feature_cols:
        m = _SUFFIX_PATTERN.match(col)
        if not m:
            continue
        prefix, suffix = m.group(1), m.group(2)
        if prefix == "MEASURE":
            continue
        if any(bad in suffix for bad in _EXCLUDED_SUFFIX_SUBSTRINGS):
            continue
        groups.setdefault(suffix, {})[prefix] = col

    multi = {suf: mods for suf, mods in groups.items() if len(mods) >= 2}
    return StageGroups(groups=multi)


def build_stage_features(df: pd.DataFrame, stage_groups: StageGroups) -> pd.DataFrame:
    out = {}
    for suffix, modules in stage_groups.groups.items():
        cols = list(modules.values())
        block = df[cols]
        out[f"STAGE_RANGE_{suffix}"] = block.max(axis=1) - block.min(axis=1)
        out[f"STAGE_STD_{suffix}"] = block.std(axis=1)
        if len(modules) == 2:
            (modA, colA), (modB, colB) = sorted(modules.items())
            out[f"STAGE_DELTA_{suffix}_{modA}_minus_{modB}"] = df[colA] - df[colB]
    return pd.DataFrame(out, index=df.index)


if __name__ == "__main__":
    from src.data.load import load_dataset

    ds = load_dataset()
    stage_groups = find_stage_groups(ds.numeric_feature_cols)
    print("Stage-comparable suffix groups found:", len(stage_groups.groups))
    sizes = pd.Series({s: len(m) for s, m in stage_groups.groups.items()})
    print("Module-count distribution across groups:\n", sizes.value_counts().sort_index().to_string())

    feats = build_stage_features(ds.raw, stage_groups)
    print("\nEngineered feature matrix shape:", feats.shape)
    print("Sample engineered columns:", feats.columns[:10].tolist())

    # quick, blind sanity scan: correlate engineered features with target
    # (for our own diagnostic purposes only -- NOT used to pick which
    # features to keep; all engineered features are passed to the models,
    # which perform their own selection/regularization).
    corr = feats.apply(lambda s: s.corr(ds.y))
    print("\nTop 15 engineered features by |corr| with target (diagnostic only):")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).round(4).to_string())
