"""Ground truth: pull the labels out of the raw mirror and keep them apart from telemetry.

A raw RCAEval case carries its answer in three places:

1. the directory name, ``{suite}{system}_{service}_{fault}_{repetition}``;
2. ``cases.parquet``, the case index (``root_cause_service``, ``fault``, ...);
3. ``root_cause.txt``, the root-cause log line (8 RE3-SS cases).

This module is the only code that reads them. It produces one label table per
dataset under ``data/labels/``, outside the blind view, for an evaluator to join
predictions against by ``case_id``. RCA methods never import it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

from aiops.data.catalog import ROOT_CAUSE_FILE, RawCase, normalize_dataset

# Metric family RCAEval scores each fault against at the fine-grained level
# (see RCAEval/benchmark/scoring.py: fault_to_indicator).
_FAULT_INDICATORS = {
    "cpu": "cpu",
    "mem": "mem",
    "socket": "socket",
    "delay": "latency",
    "loss": "latency",
}


def fault_indicator(fault: str, dataset: str) -> str | None:
    """The indicator metric of a fault; None for the RE3 code-level faults (f1-f5)."""
    if fault == "disk":
        return "latency" if normalize_dataset(dataset).startswith("RE1") else "diskio"
    return _FAULT_INDICATORS.get(fault)


@dataclass(frozen=True)
class GroundTruth:
    case_id: str  # anonymised id used in the blind view
    dataset: str
    source_case: str  # raw directory name: label-bearing, never shown to a method
    root_cause_service: str
    fault: str
    fault_description: str | None
    indicator: str | None  # e.g. "cpu", "latency", "diskio"; None for code-level faults
    inject_time: int
    repetition: int
    root_cause_log: str | None  # raw root_cause.txt line, when the case ships one


LABEL_COLUMNS = tuple(f.name for f in fields(GroundTruth))


def extract_ground_truth(
    case: RawCase, case_id: str, index: pd.DataFrame | None = None
) -> GroundTruth:
    """Labels of one raw case. ``index`` (cases.parquet) supplies the fault description."""
    name = case.name
    description = None
    if index is not None and str(name) in index.index:
        description = index.loc[str(name), "fault_description"]
    root_cause_file = case.path / ROOT_CAUSE_FILE
    root_cause_log = root_cause_file.read_text().strip() if root_cause_file.is_file() else None
    return GroundTruth(
        case_id=case_id,
        dataset=name.dataset,
        source_case=str(name),
        root_cause_service=name.service,
        fault=name.fault,
        fault_description=description,
        indicator=fault_indicator(name.fault, name.dataset),
        inject_time=case.inject_time,
        repetition=name.repetition,
        root_cause_log=root_cause_log or None,
    )


def check_against_index(labels: list[GroundTruth], index: pd.DataFrame) -> list[str]:
    """Cross-check labels parsed from directory names against cases.parquet.

    ``index`` must be indexed by case name. Returns one message per disagreement;
    an empty list means the two sources of truth agree.
    """
    problems = []
    for label in labels:
        if label.source_case not in index.index:
            problems.append(f"{label.source_case}: not in the case index")
            continue
        row = index.loc[label.source_case]
        expected = {
            "dataset": label.dataset,
            "root_cause_service": label.root_cause_service,
            "fault": label.fault,
            "repetition": label.repetition,
            "inject_time": label.inject_time,
        }
        for column, value in expected.items():
            if row[column] != value:
                problems.append(f"{label.source_case}: {column} is {value!r} on disk but {row[column]!r} in the index")
    return problems


def labels_path(labels_dir: Path, dataset: str) -> Path:
    return Path(labels_dir) / f"{normalize_dataset(dataset)}.parquet"


def labels_frame(labels: list[GroundTruth]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(label) for label in labels], columns=list(LABEL_COLUMNS))
    for column in LABEL_COLUMNS:
        if column in ("inject_time", "repetition"):
            frame[column] = frame[column].astype("int64")
        else:
            frame[column] = frame[column].astype("string")
    return frame.sort_values("case_id", ignore_index=True)


def load_labels(labels_dir: Path, dataset: str) -> pd.DataFrame:
    """The label table of one dataset, indexed by ``case_id``. For evaluators only."""
    path = labels_path(labels_dir, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"no labels at {path}; run `aiops-data build -d {dataset}` first")
    return pd.read_parquet(path).set_index("case_id")
