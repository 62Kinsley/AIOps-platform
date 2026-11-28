"""Baselines that need no LLM: reference points for the agent, at no cost.

``anomaly`` ranks the services by their most anomalous metric series. Every
series is scored against its own pre-alert baseline with the robust z-score of
:mod:`aiops.monitoring.anomaly`, the scorer behind the ``detect_anomalies``
tool, and a service ranks where its strongest series ranks. This is close in
spirit to BARO's robust scorer, one of RCAEval's strongest metric-based methods.

``anomaly-common`` does the same on the metric columns that every case of the
dataset has, which removes the column-presence shortcut of the leakage audit.
"""

from __future__ import annotations

from aiops.data.catalog import DataLayout
from aiops.data.dataset import BlindCase, BlindDataset
from aiops.evaluation.runner import Prediction
from aiops.monitoring import anomaly


class AnomalyRanking:
    """Services ranked by the score of their strongest metric series."""

    def __init__(self, layout: DataLayout | None = None, *, common_metrics_only: bool = False):
        self.layout = layout or DataLayout.default()
        self.common_metrics_only = common_metrics_only
        self.name = "anomaly-common" if common_metrics_only else "anomaly"
        self.params = {"common_metrics_only": common_metrics_only}
        self._common: dict[str, set[str]] = {}

    def rank(self, case: BlindCase) -> Prediction:
        metrics = case.metrics
        if self.common_metrics_only:
            common = self._common_columns(case.dataset)
            metrics = metrics[[c for c in metrics.columns if c in common]]
        metrics = metrics.set_axis(metrics.index - case.inject_time, axis=0)
        scores = anomaly.score_series(metrics)  # strongest first
        services = list(dict.fromkeys(scores["service"]))
        # services without a scorable series come last, in candidate order
        services += [s for s in case.candidate_services if s not in services]
        return Prediction(services=services, indicators=scores["series"].tolist())

    def _common_columns(self, dataset: str) -> set[str]:
        if dataset not in self._common:
            blind = BlindDataset.open(dataset, self.layout)
            self._common[dataset] = set(blind.common_metric_columns())
        return self._common[dataset]
