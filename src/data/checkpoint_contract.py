"""Loader/validator for config/checkpoint_contract.yaml -- the single
source of truth for the four-checkpoint final-yield prediction pipeline
(OHMIC -> +FIC -> +SIN -> +THIN).

Mirrors the same load-a-small-YAML-contract pattern already used by
src/data/load.py::DataContract, kept as a separate file/class because this
contract describes a different, additional concern (checkpoints, process
mode) that the original Stage 1 data contract does not and should not have
to know about.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_CONTRACT_PATH = PROJECT_ROOT / "config" / "checkpoint_contract.yaml"


@dataclasses.dataclass
class CheckpointDef:
    name: str
    order: int
    prefixes: list[str]


@dataclasses.dataclass
class CheckpointContract:
    mode: str
    target: str
    group_column: str
    secondary_id_column: str
    input_csv_path: Path
    process_column: str
    process_value: str
    checkpoints: list[CheckpointDef]
    leakage_columns: list[str]
    chronological_candidates: list[str]
    holdout_test_size: float
    random_state: int
    n_cv_splits: int

    @classmethod
    def load(cls, path: Path = DEFAULT_CHECKPOINT_CONTRACT_PATH) -> "CheckpointContract":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        mode = raw["mode"]
        assert mode in ("synthetic", "real"), f"mode must be 'synthetic' or 'real', got {mode!r}"

        checkpoints = [
            CheckpointDef(name=c["name"], order=c["order"], prefixes=list(c["prefixes"]))
            for c in sorted(raw["checkpoints"], key=lambda c: c["order"])
        ]
        # cumulative-order sanity check: each checkpoint's prefix set must be a
        # strict superset of the previous one's, matching the "cumulative
        # predictor set" requirement -- fail loudly rather than silently
        # running a non-cumulative pipeline.
        for prev, cur in zip(checkpoints, checkpoints[1:]):
            assert set(prev.prefixes) < set(cur.prefixes), (
                f"Checkpoint '{cur.name}' prefixes must be a strict superset of "
                f"'{prev.name}' prefixes to satisfy the cumulative requirement; "
                f"got prev={prev.prefixes} cur={cur.prefixes}"
            )

        input_csv_path = PROJECT_ROOT / raw["input_csv_path"][mode]
        process_value = raw["process_value"][mode]

        # the two modes must never share the same process_value -- this is the
        # explicit "do not relabel synthetic rows as real 0.25 HP data" guard.
        other_mode = "real" if mode == "synthetic" else "synthetic"
        assert process_value != raw["process_value"][other_mode], (
            "process_value must differ between synthetic and real modes "
            f"(both currently set to {process_value!r}) -- refusing to proceed, "
            "this would silently relabel one mode's rows as the other's."
        )

        return cls(
            mode=mode,
            target=raw["target"],
            group_column=raw["group_column"],
            secondary_id_column=raw["secondary_id_column"],
            input_csv_path=input_csv_path,
            process_column=raw["process_column"],
            process_value=process_value,
            checkpoints=checkpoints,
            leakage_columns=list(raw["leakage_columns"]),
            chronological_candidates=list(raw["chronological_candidates"]),
            holdout_test_size=float(raw["holdout_test_size"]),
            random_state=int(raw["random_state"]),
            n_cv_splits=int(raw["n_cv_splits"]),
        )


if __name__ == "__main__":
    contract = CheckpointContract.load()
    print(f"mode={contract.mode}")
    print(f"input_csv_path={contract.input_csv_path}")
    print(f"process: {contract.process_column} == {contract.process_value!r}")
    for c in contract.checkpoints:
        print(f"  checkpoint {c.order} '{c.name}': prefixes={c.prefixes}")
