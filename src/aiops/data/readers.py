"""Pandas readers for RCAEval telemetry: metrics, logs and traces.

Each reader turns one Parquet file of the Hugging Face layout into a normalised
frame. All three share one clock: ``time`` is int64 unix seconds, the unit of
``metrics.parquet`` and ``inject_time.txt``, so a log line or a span can be
lined up with the metric sample of the same second. Service names are also
unified across sources (see :data:`SERVICE_ALIASES`).

Raw layouts these readers expect (as published on Hugging Face):

- metrics.parquet: ``time`` (int64 s) + one float64 column per ``{service}_{metric}``;
  null where a metric was not observed at that second.
- logs.parquet: ``timestamp`` (int64 s), ``container_name``, ``message``.
- traces.parquet: ``traceID``, ``spanID``, ``parentSpanID``, ``serviceName``,
  ``operationName``, ``methodName``, ``startTime`` (int64 us), ``duration`` (int64 us),
  ``statusCode``, plus the redundant ``time`` ("HH:MM") and ``startTimeMillis``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# RE2-OB traces call the frontend "frontendservice"; its metrics and logs (and
# every RE3-OB source) call it "frontend". Map trace/log names onto metric names.
SERVICE_ALIASES = {"frontendservice": "frontend"}

LOG_COLUMNS = ("time", "service", "message")
TRACE_COLUMNS = (
    "time",
    "start_time_us",
    "duration_us",
    "trace_id",
    "span_id",
    "parent_span_id",
    "service",
    "operation",
    "method",
    "status_code",
)

_RAW_TRACE_COLUMNS = [
    "traceID",
    "spanID",
    "parentSpanID",
    "serviceName",
    "operationName",
    "methodName",
    "startTime",
    "duration",
    "statusCode",
]


def normalize_services(names: pd.Series) -> pd.Series:
    return names.astype("string").replace(SERVICE_ALIASES)


def read_metrics(path: str | Path, *, fill: bool = False) -> pd.DataFrame:
    """Wide metric frame indexed by ``time`` (int64 unix seconds, ascending).

    Columns are ``{service}_{metric}`` (float64), e.g. ``cartservice_cpu`` or
    ``ts-order-service_latency-90``. Nulls are kept by default because they mean
    "not observed at that second"; ``fill=True`` forward-fills, then back-fills,
    then sets never-observed columns to 0.0, the convention RCAEval baselines use.
    """
    df = pd.read_parquet(path)
    if "time" not in df.columns:
        raise ValueError(f"{path}: metrics file has no 'time' column")
    df = df.set_index("time").sort_index()
    df.index = df.index.astype("int64")
    df = df.astype("float64")
    if fill:
        df = df.ffill().bfill().fillna(0.0)
    return df


def split_metric_name(column: str) -> tuple[str, str]:
    """``ts-order-service_latency-90`` -> ``("ts-order-service", "latency-90")``.

    Service names in RCAEval use hyphens, never underscores, so the first
    underscore separates service from metric.
    """
    service, sep, metric = column.partition("_")
    if not (sep and service and metric):
        raise ValueError(f"metric column {column!r} is not '<service>_<metric>'")
    return service, metric


def metric_services(metrics: pd.DataFrame) -> list[str]:
    """Services that have at least one metric column, sorted."""
    return sorted({split_metric_name(c)[0] for c in metrics.columns})


def metrics_long(metrics: pd.DataFrame) -> pd.DataFrame:
    """Long format (``time``, ``service``, ``metric``, ``value``) without the nulls."""
    names = pd.DataFrame(
        [split_metric_name(c) for c in metrics.columns],
        index=metrics.columns,
        columns=["service", "metric"],
    )
    long = (
        metrics.rename_axis(index="time")
        .reset_index()
        .melt(id_vars="time", var_name="column", value_name="value")
        .dropna(subset=["value"])
        .join(names, on="column")
    )
    long["service"] = long["service"].astype("string")
    long["metric"] = long["metric"].astype("string")
    return long[["time", "service", "metric", "value"]].sort_values(
        ["time", "service", "metric"], ignore_index=True
    )


def read_logs(path: str | Path) -> pd.DataFrame:
    """Log lines with columns ``time`` (int64 s), ``service``, ``message``.

    ``service`` is the emitting container, which matches the metric service
    names. A missing message becomes "" so that every row stays a log line
    (log volume is itself a signal).
    """
    raw = pd.read_parquet(path, columns=["timestamp", "container_name", "message"])
    logs = pd.DataFrame(
        {
            "time": raw["timestamp"].astype("int64"),
            "service": normalize_services(raw["container_name"]),
            "message": raw["message"].astype("string").fillna(""),
        }
    )
    return logs.sort_values("time", kind="stable", ignore_index=True)


def read_traces(path: str | Path) -> pd.DataFrame:
    """One row per span, ordered by start time, with the columns in :data:`TRACE_COLUMNS`.

    ``time`` is the span's start second (int64), ``start_time_us`` and
    ``duration_us`` keep microsecond precision, ``parent_span_id`` is null for
    root spans, and ``status_code`` is nullable (Train Ticket records none;
    Online Boutique mixes gRPC codes, 0 = OK, with a few HTTP codes).
    """
    raw = pd.read_parquet(path, columns=_RAW_TRACE_COLUMNS)
    start = raw["startTime"].astype("int64")
    parent = raw["parentSpanID"].astype("string")
    traces = pd.DataFrame(
        {
            "time": start // 1_000_000,
            "start_time_us": start,
            "duration_us": raw["duration"].astype("int64"),
            "trace_id": raw["traceID"].astype("string"),
            "span_id": raw["spanID"].astype("string"),
            "parent_span_id": parent.mask(parent == ""),
            "service": normalize_services(raw["serviceName"]),
            "operation": raw["operationName"].astype("string"),
            "method": raw["methodName"].astype("string"),
            "status_code": raw["statusCode"].astype("Int64"),
        }
    )
    return traces.sort_values("start_time_us", kind="stable", ignore_index=True)
