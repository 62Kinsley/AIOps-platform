import pandas as pd
import pytest

from aiops.data.readers import (
    LOG_COLUMNS,
    TRACE_COLUMNS,
    metric_services,
    metrics_long,
    read_logs,
    read_metrics,
    read_traces,
    split_metric_name,
)
from tests.conftest import DURATION, T0, OB_SERVICES


@pytest.fixture
def case_dir(layout):
    return layout.raw / "re2ob_cartservice_cpu_1"


def test_read_metrics_index_and_dtypes(case_dir):
    metrics = read_metrics(case_dir / "metrics.parquet")
    assert metrics.index.name == "time"
    assert metrics.index.dtype == "int64"
    assert metrics.index.is_monotonic_increasing and metrics.index.is_unique
    assert metrics.index[0] == T0 and len(metrics) == DURATION
    assert set(metrics.dtypes) == {pd.Series(dtype="float64").dtype}
    assert "time" not in metrics.columns
    assert metrics["cartservice_latency-90"].isna().sum() == 5  # nulls kept by default


def test_read_metrics_fill(case_dir, tmp_path):
    raw = pd.read_parquet(case_dir / "metrics.parquet")
    raw["ghost_cpu"] = float("nan")  # never observed
    raw.sample(frac=1, random_state=0).to_parquet(tmp_path / "m.parquet", index=False)
    filled = read_metrics(tmp_path / "m.parquet", fill=True)
    assert not filled.isna().any().any()
    assert (filled["ghost_cpu"] == 0.0).all()
    assert filled.index.is_monotonic_increasing  # shuffled input comes back sorted
    # back-fill: the first observed latency is copied to the unobserved first seconds
    first = raw.sort_values("time")["cartservice_latency-90"].dropna().iloc[0]
    assert filled["cartservice_latency-90"].iloc[0] == first


@pytest.mark.parametrize(
    "column, expected",
    [
        ("cartservice_cpu", ("cartservice", "cpu")),
        ("ts-order-service_latency-90", ("ts-order-service", "latency-90")),
        ("frontend-external_workload", ("frontend-external", "workload")),
    ],
)
def test_split_metric_name(column, expected):
    assert split_metric_name(column) == expected


@pytest.mark.parametrize("bad", ["cpu", "_cpu", "cartservice_"])
def test_split_metric_name_rejects(bad):
    with pytest.raises(ValueError):
        split_metric_name(bad)


def test_metric_services_and_long_format(case_dir):
    metrics = read_metrics(case_dir / "metrics.parquet")
    assert metric_services(metrics) == sorted(OB_SERVICES)
    long = metrics_long(metrics)
    assert list(long.columns) == ["time", "service", "metric", "value"]
    assert len(long) == metrics.notna().sum().sum()  # nulls dropped
    row = long[(long.service == "redis") & (long.metric == "mem")].iloc[0]
    assert row["value"] == metrics.loc[row["time"], "redis_mem"]


def test_read_logs(case_dir):
    logs = read_logs(case_dir / "logs.parquet")
    assert tuple(logs.columns) == LOG_COLUMNS
    assert logs["time"].dtype == "int64"
    assert logs["time"].is_monotonic_increasing
    assert not logs["message"].isna().any()
    assert (logs["message"] == "").sum() == 1  # the null message became ""
    assert set(logs["service"]) <= set(OB_SERVICES)


def test_read_traces(case_dir):
    traces = read_traces(case_dir / "traces.parquet")
    assert tuple(traces.columns) == TRACE_COLUMNS
    assert traces["start_time_us"].is_monotonic_increasing
    assert list(traces["span_id"]) == ["s1", "s3", "s4", "s2"]
    assert (traces["time"] == traces["start_time_us"] // 1_000_000).all()
    assert traces["time"].iloc[0] == T0 + 1
    # RE2-OB's trace-only name is mapped onto the metric service name
    assert "frontendservice" not in set(traces["service"])
    assert set(traces["service"]) <= set(OB_SERVICES)
    # root spans have a null parent, whether stored as null or as ""
    roots = traces[traces["parent_span_id"].isna()]
    assert set(roots["span_id"]) == {"s1", "s3", "s4"}
    assert traces["status_code"].dtype == "Int64"
    assert traces["status_code"].isna().sum() == 1
    assert traces.loc[traces.span_id == "s2", "duration_us"].item() == 900
