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


# --------------------------------------------------------------------------- Phase 2 incident

# The case that gets a planted incident: cartservice slows down 10x after the alert.
PLANTED = "re2ob_cartservice_cpu_1"
PLANTED_ERROR = "ERROR connection refused to cartservice:7070 (attempt {})"


def _incident_traces(start: int, inject: int) -> pd.DataFrame:
    """One trace every 2 s: frontend -> cartservice (traced on both sides) and frontend ->
    checkoutservice (no spans of its own, like Online Boutique's uninstrumented services).
    After the alert the cartservice span takes 10 ms instead of 1 ms, and every third client
    span to it fails with gRPC code 14."""
    rows = []
    for k, t in enumerate(range(start, start + DURATION, 2)):
        after = t >= inject
        server = 10_000 if after else 1_000
        status = 14 if after and k % 3 == 0 else 0
        tid, sid, t_us = f"{k:032x}", f"{k:012x}", t * 1_000_000
        rows += [
            (tid, sid + "0001", None, "frontendservice", None, "frontend", t_us, server + 5_000, None),
            (tid, sid + "0002", sid + "0001", "frontendservice", "GetCart",
             "hipstershop.CartService/GetCart", t_us + 100, server + 500, status),
            (tid, sid + "0003", sid + "0002", "cartservice", "GetCart",
             "/hipstershop.CartService/GetCart", t_us + 300, server, 0),
            (tid, sid + "0004", sid + "0001", "frontendservice", "PlaceOrder",
             "hipstershop.CheckoutService/PlaceOrder", t_us + server + 1_000, 2_000, 0),
        ]
    frame = pd.DataFrame(rows, columns=["traceID", "spanID", "parentSpanID", "serviceName",
                                        "methodName", "operationName", "startTime", "duration",
                                        "statusCode"])
    frame["statusCode"] = frame["statusCode"].astype("Int64")
    frame["startTimeMillis"] = frame["startTime"] // 1000
    frame["time"] = "21:24"
    return frame


def plant_incident(layout: DataLayout) -> None:
    """CPU +20 for cartservice after the alert, a new error log template in frontend, slow
    cartservice spans, and a metric column only this case has (cartservice_diskio)."""
    case_dir = layout.raw / PLANTED
    inject = int((case_dir / "inject_time.txt").read_text())
    metrics = pd.read_parquet(case_dir / "metrics.parquet")
    metrics.loc[metrics["time"] >= inject, "cartservice_cpu"] += 20.0
    metrics["cartservice_diskio"] = 0.0
    metrics.to_parquet(case_dir / "metrics.parquet", index=False)
    logs = pd.read_parquet(case_dir / "logs.parquet")
    extra = pd.DataFrame({"timestamp": [inject + 5 + i for i in range(6)],
                          "container_name": "frontend",
                          "message": [PLANTED_ERROR.format(i) for i in range(6)]})
    pd.concat([logs, extra], ignore_index=True).to_parquet(case_dir / "logs.parquet", index=False)
    _incident_traces(inject - INJECT_OFFSET, inject).to_parquet(case_dir / "traces.parquet",
                                                               index=False)


@pytest.fixture(scope="session")
def incident_layout(tmp_path_factory) -> DataLayout:
    """Blind views of RE2-OB (with the planted incident) and RE3-SS, built once per session."""
    from aiops.data.blind import export_dataset

    layout = build_raw_mirror(tmp_path_factory.mktemp("incident") / "data")
    plant_incident(layout)
    for dataset in ("RE2-OB", "RE3-SS"):
        export_dataset(dataset, layout)
    return layout


@pytest.fixture(scope="session")
def planted_case_id(incident_layout) -> str:
    from aiops.data.ground_truth import load_labels

    labels = load_labels(incident_layout.labels, "RE2-OB")
    return labels.index[labels["source_case"] == PLANTED][0]
