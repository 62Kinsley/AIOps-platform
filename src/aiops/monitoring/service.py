"""The Monitoring Server's tools: one case's telemetry, read from the blind view only.

Each tool takes a case id and answers from that case's metrics, logs and traces.
Nothing here can reach the labels: :class:`MonitoringService` opens cases through
:class:`~aiops.data.BlindDataset`, never lists case ids (a method that could
enumerate cases could read other cases' inject times, which give the labels away
through the injection order), and rejects ids that aren't in the manifest.

Times in every request and result are seconds relative to the alert, the case's
RCAEval inject time: negative before it. The part of a window before the alert
is the baseline that the part after it is compared with. Results are bounded
(top-k lists, downsampled series) so they fit in an LLM's context.
"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from aiops.data.catalog import DataLayout, normalize_dataset
from aiops.data.dataset import BlindCase, BlindDataset
from aiops.data.readers import SERVICE_ALIASES, split_metric_name
from aiops.mcp_support import InvalidRequest, close_matches
from aiops.monitoring import anomaly
from aiops.monitoring import logs as logtools
from aiops.monitoring import traces as tracetools

MAX_SERIES = 30  # query_metrics: series summarised per call
MAX_SERIES_WITH_VALUES = 12  # ... of which this many come with their values
MAX_LOG_SERVICES = 15
EXAMPLE_CHARS = 300

METRIC_KINDS = {
    "cpu": "container CPU usage",
    "mem": "container memory usage, bytes",
    "diskio": "container disk I/O",
    "socket": "open sockets",
    "workload": "request rate",
    "error": "error rate",
    "latency-50": "median response time, seconds",
    "latency-90": "90th-percentile response time, seconds",
}

CONVENTIONS = (
    "All times are seconds relative to the alert (negative = before it). The window before "
    "the alert is each signal's baseline; tools compare the window after the alert with it. "
    "Pass start/end (seconds relative to the alert) to narrow a window."
)


class CaseView:
    """One case with the derived frames the tools share, each built once, on first use."""

    def __init__(self, case: BlindCase, columns: frozenset[str] | None):
        self.case = case
        self.case_id = case.case_id
        self.alert = case.inject_time
        start, end = case.time_range
        self.window = (float(start - self.alert), float(end - self.alert))
        self._columns = columns
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}

    def _get(self, key: str, build: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = build()
            return self._cache[key]

    def _load(self) -> BlindCase:
        # A fresh BlindCase per frame, so its cached raw telemetry is freed after derivation.
        return BlindCase(self.case.path)

    @property
    def metrics(self) -> pd.DataFrame:
        """Wide metrics, index = seconds relative to the alert."""
        def build() -> pd.DataFrame:
            metrics = self._load().metrics
            if self._columns is not None:
                metrics = metrics[[c for c in metrics.columns if c in self._columns]]
            metrics = metrics.set_axis(metrics.index - self.alert, axis=0)
            metrics.index.name = "t"
            return metrics
        return self._get("metrics", build)

    @property
    def metric_services(self) -> list[str]:
        return self._get("metric_services", lambda: sorted(
            {split_metric_name(c)[0] for c in self.metrics.columns}))

    @property
    def logs(self) -> pd.DataFrame | None:
        """Columns t (s, relative), service, message; None when the case has no logs."""
        def build() -> pd.DataFrame | None:
            raw = self._load().logs
            if raw is None:
                return None
            return pd.DataFrame({"t": raw["time"] - self.alert,
                                 "service": raw["service"].astype("category"),
                                 "message": raw["message"]})
        return self._get("logs", build)

    @property
    def log_index(self) -> pd.DataFrame | None:
        """:attr:`logs` plus ``template`` and ``error`` (error-like) columns."""
        def build() -> pd.DataFrame | None:
            logs = self.logs
            if logs is None:
                return None
            return logs.assign(template=logtools.templates(logs["message"]),
                               error=logtools.error_like(logs["message"]))
        return self._get("log_index", build)

    @property
    def spans(self) -> pd.DataFrame | None:
        def build() -> pd.DataFrame | None:
            raw = self._load().traces
            return None if raw is None else tracetools.prepare(raw, self.alert)
        return self._get("spans", build)

    @property
    def calls(self) -> pd.DataFrame | None:
        return self._get("calls", lambda: None if self.spans is None else tracetools.calls(self.spans))


class MonitoringService:
    """Telemetry tools over the blind view of every built dataset."""

    def __init__(self, layout: DataLayout | None = None, *, common_metrics_only: bool = False,
                 cache_size: int = 2):
        self.layout = layout or DataLayout.default()
        self.common_metrics_only = common_metrics_only
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self._datasets: dict[str, tuple[BlindDataset, frozenset[str], frozenset[str] | None]] = {}
        self._views: OrderedDict[str, CaseView] = OrderedDict()

    # ------------------------------------------------------------------ case access

    def view(self, case_id: str) -> CaseView:
        case_id = normalize_case_id(case_id)
        with self._lock:
            view = self._views.get(case_id)
            if view is None:
                dataset, ids, columns = self._dataset(case_id.rsplit("-", 1)[0])
                if case_id not in ids:
                    raise InvalidRequest(f"there is no case {case_id!r} in {dataset.dataset}")
                view = self._views[case_id] = CaseView(dataset[case_id], columns)
                while len(self._views) > self.cache_size:
                    self._views.popitem(last=False)
            self._views.move_to_end(case_id)
            return view

    def _dataset(self, name: str) -> tuple[BlindDataset, frozenset[str], frozenset[str] | None]:
        if name not in self._datasets:
            try:
                dataset = BlindDataset.open(name, self.layout)
            except FileNotFoundError:
                raise InvalidRequest(f"dataset {name} is not available on this server "
                                     f"(no blind view under {self.layout.blind})") from None
            columns = frozenset(dataset.common_metric_columns()) if self.common_metrics_only else None
            self._datasets[name] = (dataset, frozenset(dataset.case_ids), columns)
        return self._datasets[name]

    # ------------------------------------------------------------------ tools

    def incident_overview(self, case_id: str) -> dict:
        view = self.view(case_id)
        metrics, logs, spans = view.metrics, view.logs, view.spans
        kinds: dict[str, list[str]] = {}
        for column in metrics.columns:
            service, kind = split_metric_name(column)
            kinds.setdefault(service, []).append(kind)
        log_counts = logs["service"].value_counts() if logs is not None else pd.Series(dtype=int)
        span_counts = spans["service"].value_counts() if spans is not None else pd.Series(dtype=int)
        names = sorted(set(kinds) | set(log_counts[log_counts > 0].index)
                       | set(span_counts[span_counts > 0].index))
        services = []
        for name in names:
            row: dict[str, Any] = {"service": name, "metrics": ",".join(kinds.get(name, [])) or None}
            if logs is not None:
                row["log_lines"] = int(log_counts.get(name, 0))
            if spans is not None:
                row["spans"] = int(span_counts.get(name, 0))
            services.append(row)
        present = sorted({k for ks in kinds.values() for k in ks})
        interval = pd.Series(metrics.index).diff().median() if len(metrics) > 1 else None
        return {
            "case_id": view.case_id,
            "system": view.case.system,
            "alert_time_utc": _utc(view.alert),
            "window_s": list(view.window),
            "conventions": CONVENTIONS,
            "telemetry": {
                "metrics": {"series": metrics.shape[1], "services": len(kinds),
                            "sample_interval_s": interval,
                            "columns": "common to every case of the dataset"
                                       if self.common_metrics_only else "all recorded"},
                "logs": None if logs is None else {"lines": len(logs),
                                                   "services": int((log_counts > 0).sum())},
                "traces": None if spans is None else {
                    "spans": len(spans), "traces": int(spans["trace_id"].nunique()),
                    "services_with_spans": int((span_counts > 0).sum()),
                    "status_codes": bool(spans["has_status"].any())},
            },
            "metric_kinds": {k: METRIC_KINDS.get(k, "") for k in present},
            "services": services,
            "root_cause_candidates": view.case.candidate_services,
        }

    def detect_anomalies(self, case_id: str, *, service: str | None = None,
                         metric: str | None = None, top_k: int = 15) -> dict:
        view = self.view(case_id)
        columns = self._columns(view, service, metric)
        scores = anomaly.score_series(view.metrics[columns])
        notable = scores[scores["score"] >= anomaly.NOTABLE]
        rows = scores.head(top_k).drop(columns="series")
        summary = anomaly.service_summary(scores)
        summary = summary[summary["anomalous_series"] > 0].head(15)
        return {
            "case_id": view.case_id,
            "method": "robust z-score of each metric series after the alert against its own "
                      "baseline before it (90th percentile of |z|); >=3 notable, >=10 strong; "
                      "onset_s = first of 3 consecutive samples beyond 3",
            "series_scored": len(scores),
            "series_anomalous": len(notable),
            "series_skipped": len(columns) - len(scores),
            "anomalies": _records(rows),
            "services": _records(summary),
        }

    def query_metrics(self, case_id: str, *, service: str | None = None, metric: str | None = None,
                      start: float | None = None, end: float | None = None, points: int = 24) -> dict:
        view = self.view(case_id)
        if not service and not metric:
            raise InvalidRequest("give a service, a metric, or both (e.g. service='cartservice' "
                                 "or metric='cpu')")
        columns = self._columns(view, service, metric)
        lo, hi = self._window(view, start, end)
        metrics = view.metrics
        window = metrics.loc[(metrics.index >= lo) & (metrics.index <= hi), columns]
        scores = anomaly.score_series(window).set_index("series")
        order = list(scores.index) + sorted(c for c in columns if c not in scores.index)
        bucket = max(1, math.ceil((hi - lo) / points))
        n_buckets = max(1, math.ceil((hi - lo) / bucket))
        # bucket i covers [lo + i*bucket, lo + (i+1)*bucket); the sample at t=hi joins the last one
        bins = np.clip(((window.index.to_numpy() - lo) // bucket).astype(int), 0, n_buckets - 1)
        means = window.groupby(bins).mean().reindex(range(n_buckets))
        series = []
        for i, column in enumerate(order[:MAX_SERIES]):
            service_name, kind = split_metric_name(column)
            row: dict[str, Any] = {"service": service_name, "metric": kind}
            if column in scores.index:
                s = scores.loc[column]
                row.update(score=s["score"], direction=s["direction"], baseline=s["baseline"],
                           after_median=s["after_median"], onset_s=s["onset_s"])
            if i < MAX_SERIES_WITH_VALUES:
                row["values"] = [None if pd.isna(v) else float(v) for v in means[column]]
            series.append(row)
        return {
            "case_id": view.case_id,
            "window_s": [lo, hi],
            "bucket_s": bucket,
            "aggregation": "mean per bucket; null = not observed",
            "t": [lo + i * bucket for i in range(len(means))],
            "order": "most anomalous first",
            "series": series,
            "series_without_values": max(0, min(len(order), MAX_SERIES) - MAX_SERIES_WITH_VALUES),
            "series_omitted": max(0, len(order) - MAX_SERIES),
        }

    def search_logs(self, case_id: str, *, service: str | None = None, query: str | None = None,
                    start: float | None = None, end: float | None = None, limit: int = 15) -> dict:
        view = self.view(case_id)
        if view.logs is None:
            return {"case_id": view.case_id, "logs_available": False,
                    "note": "this case records no logs (RE1 cases have metrics only)"}
        lo, hi = self._window(view, start, end)
        name = self._service(service, sorted(view.logs["service"].cat.categories), "logs")
        logs = view.log_index
        if name:
            logs = logs[logs["service"] == name]
        if query:
            logs = logs[logtools.matches(logs["message"], query)]
        result = {"case_id": view.case_id, "window_s": [lo, hi],
                  "filters": {"service": name, "query": query or None}}
        result.update(logtools.summarize(logs, lo, hi, limit=limit))
        if len(result["services"]) > MAX_LOG_SERVICES:
            result["services_omitted"] = len(result["services"]) - MAX_LOG_SERVICES
            result["services"] = result["services"][:MAX_LOG_SERVICES]
        if query:
            in_window = logs[(logs["t"] >= lo) & (logs["t"] <= hi)]
            after = in_window[in_window["t"] >= 0].head(limit)
            before = in_window[in_window["t"] < 0].tail(limit - len(after))
            examples = pd.concat([before, after])
            result["examples"] = [
                {"t": row.t, "service": row.service, "message": row.message[:EXAMPLE_CHARS]}
                for row in examples.itertuples(index=False)]
        return result

    def query_traces(self, case_id: str, *, service: str | None = None,
                     operation: str | None = None, trace_id: str | None = None,
                     start: float | None = None, end: float | None = None, limit: int = 10) -> dict:
        view = self.view(case_id)
        spans = view.spans
        if spans is None:
            return {"case_id": view.case_id, "traces_available": False,
                    "note": "this case records no traces (Sock Shop and RE1 cases have none)"}
        if trace_id:
            return {"case_id": view.case_id, **tracetools.trace_tree(spans, trace_id.strip())}
        lo, hi = self._window(view, start, end)
        known = sorted(set(spans["service"].cat.categories) | set(view.metric_services))
        name = self._service(service, known, "telemetry")
        selected = spans[(spans["t"] >= lo) & (spans["t"] <= hi)]
        notes = []
        if name:
            own = selected["service"] == name
            if not own.any():
                # e.g. Online Boutique's cartservice: only its callers' client spans name it
                # ("hipstershop.CartService/GetCart")
                own = _contains(selected["operation"], name)
                notes.append(f"{name} records no spans of its own here; showing the spans whose "
                             "operation names mention it (its callers' client spans)")
            selected = selected[own]
        if operation:
            selected = selected[_contains(selected["operation"], operation)]
        pairs = view.calls
        pairs = pairs[(pairs["t"] >= lo) & (pairs["t"] <= hi)]
        if name:
            pairs = pairs[(pairs["caller"] == name) | (pairs["callee"] == name)]
        if not spans["has_status"].any():
            notes.append("this system records no span status codes, so error rates are unknown")
        result = {"case_id": view.case_id, "window_s": [lo, hi],
                  "filters": {"service": name, "operation": operation or None},
                  "spans": {"before_alert": int((selected["t"] < 0).sum()),
                            "after_alert": int((selected["t"] >= 0).sum())}}
        result.update(tracetools.operation_report(selected, limit=limit))
        result.update(tracetools.call_report(pairs, limit=limit))
        if notes:
            result["notes"] = notes
        return result

    # ------------------------------------------------------------------ argument checks

    def _window(self, view: CaseView, start: float | None, end: float | None) -> tuple[float, float]:
        first, last = view.window
        lo = first if start is None else max(float(start), first)
        hi = last if end is None else min(float(end), last)
        if not lo < hi:
            raise InvalidRequest(f"empty window start={start}, end={end}: this case covers "
                                 f"{first:g} to {last:g} s around the alert")
        return lo, hi

    @staticmethod
    def _service(service: str | None, known: list[str], source: str) -> str | None:
        if service is None or not service.strip():
            return None
        name = SERVICE_ALIASES.get(service.strip(), service.strip())
        if name not in known:
            raise InvalidRequest(f"no service {service!r} in this case's {source}"
                                 + close_matches(name, known))
        return name

    def _columns(self, view: CaseView, service: str | None, metric: str | None) -> list[str]:
        """Metric columns selected by service and/or metric kind ('latency' = all latencies)."""
        columns = list(view.metrics.columns)
        name = self._service(service, view.metric_services, "metrics")
        if name:
            columns = [c for c in columns if split_metric_name(c)[0] == name]
        if metric and metric.strip():
            wanted = metric.strip()
            if wanted in columns:  # a full series name such as 'cartservice_cpu'
                return [wanted]
            kinds = sorted({split_metric_name(c)[1] for c in columns})
            chosen = {k for k in kinds if k == wanted.lower()} or {
                k for k in kinds if k.startswith(wanted.lower())}
            if not chosen:
                where = f" for {name}" if name else ""
                raise InvalidRequest(f"no metric {metric!r}{where}" + close_matches(wanted, kinds))
            columns = [c for c in columns if split_metric_name(c)[1] in chosen]
        return columns


def normalize_case_id(case_id: str) -> str:
    """``re2-ob-0078B4EE`` -> ``RE2-OB-0078b4ee``; InvalidRequest for anything else."""
    text = str(case_id).strip()
    head, sep, tail = text.rpartition("-")
    try:
        dataset = normalize_dataset(head) if sep else None
    except ValueError:
        dataset = None
    if dataset is None or not re.fullmatch(r"[0-9a-fA-F]{8}", tail):
        raise InvalidRequest(f"{case_id!r} is not a case id; case ids look like RE2-OB-0123abcd")
    return f"{dataset}-{tail.lower()}"


def _contains(values: pd.Series, text: str) -> pd.Series:
    """Case-insensitive substring match on a categorical column, evaluated per category."""
    categories = values.cat.categories
    hits = categories[pd.Series(categories.astype(str)).str.contains(text, case=False, regex=False)]
    return values.isin(hits)


def _records(frame: pd.DataFrame) -> list[dict]:
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    return [dict(row) for row in rows]


def _utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
