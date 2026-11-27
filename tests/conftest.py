"""Synthetic RCAEval mirror in the Hugging Face layout, small enough to build in milliseconds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiops.data.catalog import DataLayout

T0 = 1_705_353_846  # 2024-01-15 21:24:06 UTC
DURATION = 120  # seconds of metrics per case
INJECT_OFFSET = 60

# (case name, has logs, has traces, root_cause.txt line or None)
CASES = [
    ("re2ob_cartservice_cpu_1", True, True, None),
    ("re2ob_cartservice_cpu_2", True, True, None),
    ("re2ob_frontend_delay_1", True, True, None),
    ("re2ob_redis_mem_1", True, True, None),
    ("re3ss_carts_f1_1", True, False,
     '02:51,1732243919404637792,carts,"WARN Request method \'POST\' not supported",carts-7f4f,node-1'),
    ("re3ss_orders_f3_1", True, False, None),
]
OB_SERVICES = ["cartservice", "frontend", "redis", "checkoutservice"]
SS_SERVICES = ["carts", "orders", "front-end", "carts-db"]


def _metrics(services: list[str], start: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"time": np.arange(start, start + DURATION, dtype="int64")}
    for svc in services:
        data[f"{svc}_cpu"] = rng.random(DURATION)
        data[f"{svc}_mem"] = rng.random(DURATION) * 100
        latency = rng.random(DURATION) * 10
        latency[:5] = np.nan  # not observed in the first seconds
        data[f"{svc}_latency-90"] = latency
    return pd.DataFrame(data)


def _logs(services: list[str], start: int) -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append((start + i * 4, services[i % len(services)], f"request {i} ok"))
    frame = pd.DataFrame(rows, columns=["timestamp", "container_name", "message"])
    frame.loc[3, "message"] = None
    return frame


def _traces(start: int) -> pd.DataFrame:
    # Deliberately unsorted, with the RE2-OB "frontendservice" alias and a root span.
    rows = [
        ("t1", "s2", "s1", "cartservice", "GetCart", "grpc.CartService/GetCart", 3_500_000, 900, 0),
        ("t1", "s1", None, "frontendservice", "GetCart", "hipstershop.CartService/GetCart", 1_000_000, 5_000, 0),
        ("t2", "s3", None, "frontendservice", None, "HTTP GET /", 2_250_000, 12_000, None),
        ("t2", "s4", "", "redis", "HGET", "redis.HGET", 2_260_000, 150, 13),
    ]
    frame = pd.DataFrame(rows, columns=["traceID", "spanID", "parentSpanID", "serviceName",
                                        "methodName", "operationName", "startTime", "duration",
                                        "statusCode"])
    frame["startTime"] = frame["startTime"] + start * 1_000_000
    frame["startTimeMillis"] = frame["startTime"] // 1000
    frame["time"] = "21:24"
    frame["statusCode"] = frame["statusCode"].astype("Int64")
    return frame[["time", "traceID", "spanID", "serviceName", "methodName", "operationName",
                  "parentSpanID", "startTimeMillis", "startTime", "duration", "statusCode"]]


def build_raw_mirror(root: Path) -> DataLayout:
    layout = DataLayout(root)
    layout.raw.mkdir(parents=True)
    index_rows = []
    for i, (name, has_logs, has_traces, root_cause) in enumerate(CASES):
        case_dir = layout.raw / name
        case_dir.mkdir()
        start = T0 + i * 3600
        inject = start + INJECT_OFFSET
        services = OB_SERVICES if name.startswith("re2ob") else SS_SERVICES
        _metrics(services, start, seed=i).to_parquet(case_dir / "metrics.parquet", index=False)
        (case_dir / "inject_time.txt").write_text(f"{inject}\n")
        if has_logs:
            _logs(services, start).to_parquet(case_dir / "logs.parquet", index=False)
        if has_traces:
            _traces(start).to_parquet(case_dir / "traces.parquet", index=False)
        if root_cause:
            (case_dir / "root_cause.txt").write_text(root_cause + "\n")
        prefix, service, fault, rep = name.split("_")
        dataset = f"{prefix[:3].upper()}-{prefix[3:].upper()}"
        index_rows.append({
            "case": name, "dataset": dataset, "root_cause_service": service, "fault": fault,
            "fault_description": f"{fault} fault", "repetition": int(rep), "inject_time": inject,
        })
    pd.DataFrame(index_rows).to_parquet(layout.raw / "cases.parquet", index=False)
    return layout


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    return build_raw_mirror(tmp_path / "data")
