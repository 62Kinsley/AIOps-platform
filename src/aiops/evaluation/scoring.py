"""Score RCA predictions against the ground truth, the way RCAEval does.

This is the evaluator side of :mod:`aiops.evaluation` and the only part of it
that reads labels; methods run through :mod:`aiops.evaluation.runner`, which
never does.

Metrics, as in RCAEval (``RCAEval/benchmark/evaluation.py``):

- AC@k: the share of cases whose root cause is among the method's top k;
- Avg@5: the mean of AC@1 ... AC@5.

Service level: a ranked name counts as the service before its first underscore,
with ``-db`` removed (``carts-db`` counts as ``carts``), and only a service's
first appearance counts (RCAEval ``scoring.py``: ``to_service``, ``service_ranks``).

Metric level ("fine"): the answer is the root-cause service plus the indicator of
its fault (cpu, mem, socket, diskio, or latency for delay and loss; latency for
RE1 disk faults). ``latency-90`` counts as ``latency`` and ``latency-50`` is not
scored, as RCAEval's preprocessing does. Code-level faults (RE3 f1-f5) have no
indicator: RCAEval scores them 0 at the metric level, here they are left out of it.

The chance reference ``random AC@1`` is what a uniformly random ranking of the
candidate services scores: 1 / the number of candidates, on average.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from aiops.data.catalog import DataLayout
from aiops.data.dataset import BlindDataset
from aiops.data.ground_truth import load_labels

TOP_K = (1, 3, 5)
AVG_K = 5
SPLITS = ("all", "dev", "test")
SCORES_FILE = "scores.json"
CASES_FILE = "case_scores.csv"


def to_service(name: str) -> str:
    """``carts-db_cpu`` -> ``carts``: RCAEval's service of a ranked name."""
    return name.split("_")[0].replace("-db", "")


def service_ranking(names: Iterable[str]) -> list[str]:
    """Services in order of first appearance."""
    return list(dict.fromkeys(to_service(name) for name in names))


def indicator_ranking(names: Iterable[str]) -> list[tuple[str, str]]:
    """``(service, indicator)`` pairs of a metric-level ranking, as RCAEval scores them."""
    ranking = []
    for name in names:
        service, _, metric = name.partition("_")
        if metric == "latency-50":
            continue
        ranking.append((service, "latency" if metric == "latency-90" else metric or "unknown"))
    return ranking


def rank_of(answer, ranking: Sequence) -> float:
    """1-based position of ``answer`` in ``ranking``; NaN when it is not there."""
    try:
        return float(list(ranking).index(answer) + 1)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- splits

def assign_splits(labels: pd.DataFrame) -> pd.Series:
    """``dev`` or ``test`` for every case of a label table (indexed by ``case_id``).

    In each (root-cause service, fault) group the case with the smallest id goes
    to dev, the others to test. Ids are salted hashes, so the pick is random but
    stable, and dev holds each root cause and fault of the dataset once: a fifth
    of RE1 (5 repetitions), a third of RE2 (3). Tune a method on dev; report test.
    """
    ordered = labels.sort_index()
    dev = ~ordered.duplicated(["root_cause_service", "fault"])
    split = pd.Series(np.where(dev, "dev", "test"), index=ordered.index, name="split")
    return split.reindex(labels.index)


def select_cases(layout: DataLayout, dataset: str, split: str = "all",
                 limit: int | None = None) -> list[str]:
    """Case ids of one dataset's split, in id order; the first ``limit`` of them if given."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {', '.join(SPLITS)}")
    labels = load_labels(layout.labels, dataset)
    ids = labels.index if split == "all" else labels.index[assign_splits(labels) == split]
    ids = sorted(ids)
    return ids[:limit] if limit else ids


# --------------------------------------------------------------------------- scoring

def score_cases(predictions: pd.DataFrame, layout: DataLayout) -> pd.DataFrame:
    """One row per predicted case: its truth, split, and where the method ranked the truth."""
    rows = []
    for dataset, group in predictions.groupby("dataset", sort=False):
        labels = load_labels(layout.labels, dataset)
        splits = assign_splits(labels)
        blind = BlindDataset.open(dataset, layout)
        for record in group.itertuples(index=False):
            truth = labels.loc[record.case_id]
            service = to_service(truth["root_cause_service"])
            candidates = set(service_ranking(blind[record.case_id].candidate_services))
            indicator = None if pd.isna(truth["indicator"]) else truth["indicator"]
            services = service_ranking(record.services)
            rows.append({
                "case_id": record.case_id,
                "dataset": dataset,
                "split": splits[record.case_id],
                "root_cause_service": truth["root_cause_service"],
                "fault": truth["fault"],
                "indicator": indicator,
                "rank": rank_of(service, services),
                "rank_fine": (rank_of((truth["root_cause_service"], indicator),
                                      indicator_ranking(record.indicators))
                              if indicator else np.nan),
                "n_candidates": len(candidates),
                "chance_ac1": 1 / len(candidates) if service in candidates else 0.0,
                "error": record.error,
                "predicted_top3": " > ".join(services[:3]),
            })
    return pd.DataFrame(rows)


def accuracy(ranks: pd.Series, k: int) -> float:
    """AC@k of a series of ranks (NaN rank = a miss)."""
    return float((ranks <= k).mean()) if len(ranks) else np.nan


def average_accuracy(ranks: pd.Series, k: int = AVG_K) -> float:
    """Avg@k = (AC@1 + ... + AC@k) / k."""
    return float(np.mean([accuracy(ranks, j) for j in range(1, k + 1)])) if len(ranks) else np.nan


def summarize(cases: pd.DataFrame, by: Sequence[str] = ("dataset",)) -> pd.DataFrame:
    """AC@1/3/5, Avg@5 (service and metric level) and the chance reference, per group."""
    rows = []
    for key, group in cases.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        fine = group.loc[group["indicator"].notna(), "rank_fine"]
        row = dict(zip(by, key), cases=len(group))
        for k in TOP_K:
            row[f"AC@{k}"] = accuracy(group["rank"], k)
        row["Avg@5"] = average_accuracy(group["rank"])
        row["fine AC@1"] = accuracy(fine, 1)
        row["fine Avg@5"] = average_accuracy(fine)
        row["random AC@1"] = float(group["chance_ac1"].mean())
        row["errors"] = int(group["error"].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def macro_mean(summary: pd.DataFrame) -> dict:
    """Every metric averaged over the rows of a summary (datasets weigh the same)."""
    metrics = [c for c in summary.columns if "@" in c]
    return {"cases": int(summary["cases"].sum()), **summary[metrics].mean().to_dict(),
            "errors": int(summary["errors"].sum())}


def write_scores(run_dir: str | Path, cases: pd.DataFrame) -> dict:
    """Write ``scores.json`` and ``case_scores.csv`` for every scored case of a run.

    ``scores.json`` has a summary per split (all, dev, test): per dataset, the
    mean over datasets, and per (dataset, fault). Both files hold ground truth:
    a scored run directory is label-bearing.
    """
    run_dir = Path(run_dir)
    scores = {}
    for split in SPLITS:
        part = cases if split == "all" else cases[cases["split"] == split]
        if part.empty:
            continue
        summary = summarize(part)
        scores[split] = {
            "datasets": _records(summary),
            "mean": {k: None if pd.isna(v) else v for k, v in macro_mean(summary).items()},
            "by_fault": _records(summarize(part, by=("dataset", "fault"))),
        }
    (run_dir / SCORES_FILE).write_text(json.dumps(scores, indent=2) + "\n")
    cases.to_csv(run_dir / CASES_FILE, index=False)
    return scores


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))  # NaN -> null
