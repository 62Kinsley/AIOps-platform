"""Run an RCA method over blind cases and record what it predicts.

This is the method side of :mod:`aiops.evaluation`: nothing here reads labels.
A method sees one :class:`~aiops.data.BlindCase` at a time, in case-id order
(the ids are salted hashes, so the order says nothing about when cases were
injected), and returns a :class:`Prediction`.

The runner appends one JSON line per case to ``predictions.jsonl`` as it goes,
so an interrupted run resumes where it stopped. A case whose method raised is
recorded with the error, scored as a miss, and tried again on the next run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from aiops.data.catalog import DATASETS, DataLayout
from aiops.data.dataset import BlindCase, BlindDataset

PREDICTIONS_FILE = "predictions.jsonl"
RUN_FILE = "run.json"
KEEP = 20  # ranked entries stored per case; scoring looks at the top 5
COLUMNS = ["case_id", "dataset", "services", "indicators", "seconds", "error", "info"]


@dataclass
class Prediction:
    """What a method says about one case, most likely root cause first."""

    services: list[str]
    # Optional metric-level ranking, "<service>_<metric>" (e.g. "cartservice_cpu").
    indicators: list[str] = field(default_factory=list)
    # JSON-serialisable extras worth keeping: evidence, tool calls, token usage, ...
    info: dict[str, Any] = field(default_factory=dict)


class Method(Protocol):
    name: str
    params: dict[str, Any]  # what distinguishes this configuration; recorded with the run

    def rank(self, case: BlindCase) -> Prediction: ...


def run_method(method: Method, cases: Mapping[str, Iterable[str]], layout: DataLayout,
               run_dir: str | Path, *, fresh: bool = False,
               progress: Callable[[str, str | None], None] | None = None) -> pd.DataFrame:
    """Run ``method`` on ``cases`` (dataset -> case ids) and return all of the run's predictions.

    Cases already predicted in ``run_dir`` without an error are skipped, unless
    ``fresh``. A run directory belongs to one method configuration: resuming it
    with other ``params`` raises ``ValueError``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path, pred_path = run_dir / RUN_FILE, run_dir / PREDICTIONS_FILE
    meta = {"method": method.name, "params": method.params}
    if fresh:
        pred_path.unlink(missing_ok=True)
    elif meta_path.is_file():
        previous = json.loads(meta_path.read_text())
        if {key: previous.get(key) for key in meta} != meta:
            raise ValueError(f"{run_dir} holds a run of {previous.get('method')} "
                             f"{previous.get('params')}; pass --fresh to replace it or --name "
                             "to start another run")
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps({**meta, "updated": updated}, indent=2) + "\n")

    done: set[str] = set()
    if pred_path.is_file() and pred_path.stat().st_size:
        existing = load_predictions(run_dir)
        done = set(existing.loc[existing["error"].isna(), "case_id"])
        if not pred_path.read_bytes().endswith(b"\n"):  # a write cut short by an interrupt
            with pred_path.open("a") as out:
                out.write("\n")
    with pred_path.open("a") as out:
        for dataset, case_ids in cases.items():
            blind = BlindDataset.open(dataset, layout)
            for case_id in case_ids:
                if case_id in done:
                    continue
                started = time.perf_counter()
                try:
                    prediction, error = method.rank(blind[case_id]), None
                except Exception as exc:  # one failing case must not end a long run
                    prediction, error = Prediction(services=[]), f"{type(exc).__name__}: {exc}"
                record = {
                    "case_id": case_id,
                    "dataset": blind.dataset,
                    "services": [str(s) for s in prediction.services][:KEEP],
                    "indicators": [str(s) for s in prediction.indicators][:KEEP],
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": error,
                    "info": prediction.info,
                }
                out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                out.flush()
                if progress:
                    progress(case_id, error)
    return load_predictions(run_dir)


def load_predictions(run_dir: str | Path) -> pd.DataFrame:
    """The predictions of a run, one row per case (a retried case keeps its last record)."""
    path = Path(run_dir) / PREDICTIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no predictions in {run_dir}")
    records: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # empty, or cut short by an interrupt: that case reruns
            continue
        records[record["case_id"]] = record
    frame = pd.DataFrame(list(records.values()), columns=COLUMNS)
    order = frame["dataset"].map({d: i for i, d in enumerate(DATASETS)})
    return frame.assign(_order=order).sort_values(["_order", "case_id"]).drop(columns="_order") \
        .reset_index(drop=True)
