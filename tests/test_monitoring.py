"""Monitoring Server tools on a synthetic blind view with a planted incident (see conftest)."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from aiops.data.dataset import BlindDataset
from aiops.mcp_support import InvalidRequest, to_json
from aiops.monitoring import MonitoringService, normalize_case_id
from aiops.monitoring.anomaly import score_series
from aiops.monitoring.logs import change_score, error_like, matches, templates
from tests.conftest import CASES, INJECT_OFFSET, PLANTED_ERROR


@pytest.fixture(scope="module")
def service(incident_layout):
    return MonitoringService(incident_layout)


@pytest.fixture(scope="module")
def ss_case_id(incident_layout):
    return BlindDataset.open("RE3-SS", incident_layout).case_ids[0]


# --------------------------------------------------------------------------- tools


def test_overview(service, planted_case_id):
    out = service.incident_overview(planted_case_id)
    assert out["case_id"] == planted_case_id
    assert out["system"] == "Online Boutique"
    assert out["window_s"] == [-INJECT_OFFSET, 119 - INJECT_OFFSET]
    services = {row["service"]: row for row in out["services"]}
    assert set(services) == {"cartservice", "checkoutservice", "frontend", "redis"}
    assert services["cartservice"]["metrics"].split(",")[:2] == ["cpu", "mem"]
    assert services["frontend"]["spans"] == 3 * 60 and services["checkoutservice"]["spans"] == 0
    assert out["telemetry"]["traces"]["status_codes"] is True
    assert out["telemetry"]["logs"]["lines"] == 30 + 6
    assert out["metric_kinds"]["cpu"] == "container CPU usage"
    assert "cartservice" in out["root_cause_candidates"]


def test_detect_anomalies_finds_the_planted_cpu_jump(service, planted_case_id):
    out = service.detect_anomalies(planted_case_id)
    top = out["anomalies"][0]
    assert (top["service"], top["metric"], top["direction"]) == ("cartservice", "cpu", "up")
    assert top["score"] > 20 and top["onset_s"] == 0
    assert out["series_anomalous"] == 1  # random noise elsewhere stays below 3
    assert out["services"][0]["service"] == "cartservice"
    assert out["services"][0]["top_metrics"] == ["cpu up"]
    only_cpu = service.detect_anomalies(planted_case_id, metric="cpu", top_k=2)
    assert {a["metric"] for a in only_cpu["anomalies"]} == {"cpu"} and len(only_cpu["anomalies"]) == 2


def test_query_metrics_downsamples(service, planted_case_id):
    out = service.query_metrics(planted_case_id, service="cartservice", metric="cpu", points=4)
    assert out["bucket_s"] == 30 and out["t"] == [-60, -30, 0, 30]
    (series,) = out["series"]
    before, after = series["values"][:2], series["values"][2:]
    assert max(before) < 1 < 20 < min(after)
    latency = service.query_metrics(planted_case_id, metric="latency")
    assert {s["metric"] for s in latency["series"]} == {"latency-90"}
    window = service.query_metrics(planted_case_id, service="redis", start=-10, end=10, points=4)
    assert window["window_s"] == [-10, 10] and window["t"] == [-10, -5, 0, 5]


def test_search_logs_surfaces_the_new_error_template(service, planted_case_id):
    out = service.search_logs(planted_case_id)
    first = out["templates_increased"][0]
    assert first["service"] == "frontend" and first["error_like"] is True
    assert (first["before"], first["after"], first["first_after_s"]) == (0, 6, 5)
    assert "<*>" in first["template"] and first["sample"] == PLANTED_ERROR.format(0)
    frontend = next(s for s in out["services"] if s["service"] == "frontend")
    assert frontend["new_templates_after"] == 1 and frontend["error_like_after"] == 6


def test_search_logs_query(service, planted_case_id):
    out = service.search_logs(planted_case_id, query="refused", limit=3)
    assert out["lines"] == {"before_alert": 0, "after_alert": 6}
    assert [e["message"] for e in out["examples"]] == [PLANTED_ERROR.format(i) for i in range(3)]
    literal = service.search_logs(planted_case_id, query="cartservice:7070 (attempt")  # bad regex
    assert literal["lines"]["after_alert"] == 6


def test_query_traces_compares_operations_and_calls(service, planted_case_id):
    out = service.query_traces(planted_case_id)
    first = out["operations"][0]
    assert (first["service"], first["operation"]) == ("cartservice", "/hipstershop.CartService/GetCart")
    assert first["p90_ratio"] == pytest.approx(10.0)
    client = next(o for o in out["operations"] if o["operation"] == "hipstershop.CartService/GetCart")
    assert client["before"]["error_rate"] == 0 and client["after"]["error_rate"] == pytest.approx(1 / 3, abs=0.05)
    (call,) = out["calls"]
    assert (call["caller"], call["callee"]) == ("frontend", "cartservice")
    assert call["callee_p90_ratio"] == pytest.approx(10.0)
    assert call["caller_p90_ratio"] == pytest.approx(10_500 / 1_500)
    assert out["spans"] == {"before_alert": 4 * 30, "after_alert": 4 * 30}


def test_trace_tree(service, planted_case_id):
    trace_id = service.query_traces(planted_case_id)["operations"][0]["example_trace"]
    tree = service.query_traces(planted_case_id, trace_id=trace_id)
    assert tree["spans"] == 4 and tree["start_s"] >= 0
    lines = tree["tree"]
    assert lines[0].startswith("frontend: frontend | +0.0 ms")
    assert lines[1].startswith("  frontend: hipstershop.CartService/GetCart")
    assert lines[2].startswith("    cartservice: /hipstershop.CartService/GetCart") and "10.00 ms" in lines[2]
    assert lines[3].startswith("  frontend: hipstershop.CheckoutService/PlaceOrder")


def test_uninstrumented_service_is_found_through_its_callers(service, planted_case_id):
    out = service.query_traces(planted_case_id, service="checkoutservice")
    assert [o["operation"] for o in out["operations"]] == ["hipstershop.CheckoutService/PlaceOrder"]
    assert "no spans of its own" in out["notes"][0]


def test_cases_without_traces_or_logs(service, ss_case_id):
    assert service.query_traces(ss_case_id)["traces_available"] is False
    assert service.search_logs(ss_case_id)["lines"]["before_alert"] > 0
    overview = service.incident_overview(ss_case_id)
    assert overview["telemetry"]["traces"] is None and "spans" not in overview["services"][0]


def test_common_metrics_only_hides_columns_some_cases_lack(incident_layout, planted_case_id):
    full = MonitoringService(incident_layout).incident_overview(planted_case_id)
    common = MonitoringService(incident_layout, common_metrics_only=True).incident_overview(planted_case_id)
    cart = lambda out: next(s for s in out["services"] if s["service"] == "cartservice")["metrics"]
    assert "diskio" in cart(full) and "diskio" not in cart(common)
    assert common["telemetry"]["metrics"]["series"] == full["telemetry"]["metrics"]["series"] - 1


# --------------------------------------------------------------------------- argument errors


def test_case_id_checks(service, planted_case_id):
    assert normalize_case_id(" " + planted_case_id.lower().replace("re2-ob", "re2ob") + " ") == planted_case_id
    for bad in ("RE2-OB", "RE2-OB-xyz", "cartservice", "RE9-OB-0123abcd"):
        with pytest.raises(InvalidRequest, match="is not a case id"):
            service.incident_overview(bad)
    with pytest.raises(InvalidRequest, match="no case") as error:
        service.incident_overview("RE2-OB-00000000")
    # the message must not hand out the ids that do exist
    assert not re.search(r"RE2-OB-(?!00000000)[0-9a-f]{8}", str(error.value))
    with pytest.raises(InvalidRequest, match="not available"):
        service.incident_overview("RE1-TT-0123abcd")  # dataset not built


def test_argument_errors(service, planted_case_id):
    with pytest.raises(InvalidRequest, match="did you mean cartservice"):
        service.detect_anomalies(planted_case_id, service="cartsevice")
    with pytest.raises(InvalidRequest, match="no metric 'gpu'"):
        service.query_metrics(planted_case_id, metric="gpu")
    with pytest.raises(InvalidRequest, match="give a service, a metric"):
        service.query_metrics(planted_case_id)
    with pytest.raises(InvalidRequest, match="empty window"):
        service.search_logs(planted_case_id, start=30, end=10)
    with pytest.raises(InvalidRequest, match="not in this case"):
        service.query_traces(planted_case_id, trace_id="f" * 32)


# --------------------------------------------------------------------------- isolation


def test_no_tool_output_names_a_raw_case_or_a_label(service, incident_layout):
    raw_names = [c[0] for c in CASES]
    texts = []
    for dataset in ("RE2-OB", "RE3-SS"):
        for case_id in BlindDataset.open(dataset, incident_layout).case_ids:
            texts += [to_json(service.incident_overview(case_id)),
                      to_json(service.detect_anomalies(case_id)),
                      to_json(service.query_metrics(case_id, metric="cpu")),
                      to_json(service.search_logs(case_id)),
                      to_json(service.query_traces(case_id))]
    blob = "\n".join(texts)
    assert not any(name in blob for name in raw_names)
    assert "root_cause_service" not in blob and "fault_description" not in blob
    assert not re.search(r"\b(cpu|mem|delay|f1|f3) fault\b", blob)  # conftest's fault descriptions


def test_cache_keeps_the_most_recent_cases(incident_layout):
    service = MonitoringService(incident_layout, cache_size=2)
    ids = BlindDataset.open("RE2-OB", incident_layout).case_ids
    for case_id in ids[:3]:
        service.view(case_id)
    assert list(service._views) == ids[1:3]
    assert service.view(ids[1]) is service.view(ids[1])


# --------------------------------------------------------------------------- building blocks


def test_score_series_shift_spike_and_flat_baseline():
    t = np.arange(-120, 120)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a_shift": np.where(t >= 0, 10.0, 0.0) + rng.normal(0, 1, len(t)),
        "b_spike": np.where(t == 30, 50.0, 0.0) + rng.normal(0, 1, len(t)),
        "c_zero": np.where(t >= 60, 5.0, 0.0),
        "d_mixed": np.select([(t >= 0) & (t < 40), t >= 40], [30.0, -30.0], 0.0) + rng.normal(0, 1, len(t)),
    }, index=t)
    scores = score_series(frame).set_index("series")
    assert scores.loc["a_shift", "score"] > 5 and scores.loc["a_shift", "onset_s"] == 0
    assert scores.loc["b_spike", "score"] < 3  # one sample is not a sustained anomaly
    assert scores.loc["c_zero", "flat_baseline"] and scores.loc["c_zero", "onset_s"] == 60
    assert scores.loc["d_mixed", "direction"] == "mixed"
    assert list(scores.index[:1]) == ["c_zero"]  # capped score, ranked first


def test_log_templates_mask_variable_parts():
    messages = pd.Series([
        "2024-02-24 05:08:30.701  INFO 1 --- [io-18898-exec-7] seat.service.SeatServiceImpl : Seat id 42",
        "2024-02-24 05:09:31.002  INFO 8 --- [o-12032-exec-13] seat.service.SeatServiceImpl : Seat id 7",
        "GetCartAsync called with userId=63a9f6d8-d33c-4d1b-95ec-7b681b7e41ec",
        'Posting Customer: {"username":"oFXzDx","password":"qZWQd","email":"a@b.com"}',
        "method=Register username=oFXzDx email=oFXzDx@gmail.com took=2.934901ms",
    ])
    out = templates(messages).astype(str).tolist()
    assert out[0] == out[1] == "INFO seat.service.SeatServiceImpl: Seat id <*>"
    assert out[2] == "GetCartAsync called with userId=<UUID>"
    assert out[3] == 'Posting Customer: {"username":"<*>","password":"<*>","email":"<*>"}'
    assert out[4] == "method=Register username=<*> email=<*> took=<*>"
    assert error_like(pd.Series(["Connection refused", "err=null took=1ms", "err=timeout",
                                 "all good"])).tolist() == [True, False, True, False]
    assert matches(pd.Series(["a (b", "x"]), "a (b").tolist() == [True, False]  # invalid regex


def test_change_score_scales_by_window_length():
    # 10 lines in a 60 s baseline, 20 in a 120 s after-window: same rate, score 0
    assert change_score(np.array([10]), np.array([20]), 60, 120)[0] == pytest.approx(0.0)
    assert change_score(np.array([0]), np.array([7]), 60, 60)[0] == pytest.approx(3.0)


def test_to_json_is_compact_and_finite():
    text = to_json({"a": 1.23456789, "b": float("nan"), "c": np.int64(3), "d": [np.float64(2.0)],
                    "e": pd.NA, "f": np.bool_(True), "g": 123456.7})
    assert json.loads(text) == {"a": 1.235, "b": None, "c": 3, "d": [2], "e": None, "f": True,
                                "g": 123500}
    assert " " not in text
