"""Label-free access to the blind view: the data API RCA methods use.

Cases are opened by anonymised id from ``data/blind/<DATASET>/``, which holds
only telemetry and ``case.json``. Nothing reachable from a :class:`BlindCase`
names the root cause.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from aiops.data.catalog import TELEMETRY_FILES, DataLayout, normalize_dataset
from aiops.data.readers import read_logs, read_metrics, read_traces

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
CASE_FILE = "case.json"
CASES_DIR = "cases"


class BlindCase:
    """One anonymised failure case. Telemetry loads lazily, once per object."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.info = json.loads((self.path / CASE_FILE).read_text())

    @property
    def case_id(self) -> str:
        return self.info["case_id"]

    @property
    def dataset(self) -> str:
        return self.info["dataset"]

    @property
    def system(self) -> str:
        return self.info["system"]

    @property
    def inject_time(self) -> int:
        """When the fault was injected (unix s). RCAEval hands this to RCA methods
        as the anomaly-detection time; it is an input, not part of the answer."""
        return self.info["inject_time"]

    @property
    def time_range(self) -> tuple[int, int]:
        return self.info["time_range"]["start"], self.info["time_range"]["end"]

    @property
    def candidate_services(self) -> list[str]:
        return list(self.info["candidate_services"])

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self.info["files"])

    def _file(self, kind: str) -> Path | None:
        name = self.info["files"].get(kind)
        return self.path / name if name else None

    @cached_property
    def metrics(self) -> pd.DataFrame:
        """Wide frame indexed by ``time``; see :func:`aiops.data.readers.read_metrics`."""
        return read_metrics(self._file("metrics"))

    @cached_property
    def logs(self) -> pd.DataFrame | None:
        """None when the system records no logs (RE1, one RE2-TT case)."""
        path = self._file("logs")
        return read_logs(path) if path else None

    @cached_property
    def traces(self) -> pd.DataFrame | None:
        """None when the system records no traces (RE1, Sock Shop)."""
        path = self._file("traces")
        return read_traces(path) if path else None

    def __repr__(self) -> str:
        return f"BlindCase({self.case_id!r}, sources={'+'.join(self.sources)})"


class BlindDataset(Sequence[BlindCase]):
    """The blind view of one dataset, e.g. ``BlindDataset.open("RE2-OB")``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{self.root} is not a blind view (no {MANIFEST_FILE}); run `aiops-data build` first"
            )
        self.manifest = json.loads(manifest_path.read_text())

    @classmethod
    def open(cls, dataset: str, layout: DataLayout | None = None) -> BlindDataset:
        layout = layout or DataLayout.default()
        return cls(layout.blind / normalize_dataset(dataset))

    @property
    def dataset(self) -> str:
        return self.manifest["dataset"]

    @property
    def case_ids(self) -> list[str]:
        return list(self.manifest["case_ids"])

    @property
    def candidate_services(self) -> list[str]:
        return list(self.manifest["candidate_services"])

    def __len__(self) -> int:
        return len(self.manifest["case_ids"])

    def __getitem__(self, key: int | str) -> BlindCase:  # type: ignore[override]
        case_id = self.manifest["case_ids"][key] if isinstance(key, int) else key
        if case_id not in self.manifest["case_ids"]:
            raise KeyError(f"{case_id!r} is not a case of {self.dataset}")
        return BlindCase(self.root / CASES_DIR / case_id)

    def __iter__(self) -> Iterator[BlindCase]:
        for case_id in self.manifest["case_ids"]:
            yield BlindCase(self.root / CASES_DIR / case_id)

    def common_metric_columns(self) -> list[str]:
        """Metric columns present in every case of the dataset, in first-case order.

        RCAEval's metric set varies between cases, and which columns a case has
        partly reveals its root cause (see the leakage audit). Restricting a
        method to these columns removes that shortcut.
        """
        common: list[str] | None = None
        for case_id in self.manifest["case_ids"]:
            path = self.root / CASES_DIR / case_id / TELEMETRY_FILES["metrics"]
            names = [n for n in pq.read_schema(path).names if n != "time"]
            present = set(names)
            common = names if common is None else [n for n in common if n in present]
        return common or []

    def __repr__(self) -> str:
        return f"BlindDataset({self.dataset!r}, {len(self)} cases)"
