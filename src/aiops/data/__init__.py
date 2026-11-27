"""Phase 1 data pipeline: RCAEval download, telemetry readers, ground-truth isolation.

RCA methods use only the label-free side::

    from aiops.data import BlindDataset
    for case in BlindDataset.open("RE2-OB"):
        case.metrics, case.logs, case.traces, case.inject_time

Evaluators load the answers separately::

    from aiops.data.ground_truth import load_labels
"""

from aiops.data.catalog import DATASETS, DataLayout
from aiops.data.dataset import BlindCase, BlindDataset
from aiops.data.readers import metrics_long, read_logs, read_metrics, read_traces

__all__ = [
    "DATASETS",
    "BlindCase",
    "BlindDataset",
    "DataLayout",
    "metrics_long",
    "read_logs",
    "read_metrics",
    "read_traces",
]
