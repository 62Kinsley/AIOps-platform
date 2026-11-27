import pandas as pd
import pytest

from aiops.data.catalog import RawCase
from aiops.data.ground_truth import (
    check_against_index,
    extract_ground_truth,
    fault_indicator,
    labels_frame,
    load_labels,
)
from tests.conftest import INJECT_OFFSET, T0


@pytest.mark.parametrize(
    "fault, dataset, indicator",
    [
        ("cpu", "RE2-OB", "cpu"),
        ("mem", "RE1-SS", "mem"),
        ("socket", "RE2-TT", "socket"),
        ("delay", "RE2-OB", "latency"),
        ("loss", "RE1-TT", "latency"),
        ("disk", "RE1-OB", "latency"),  # RE1 has no disk-io metric
        ("disk", "RE2-OB", "diskio"),
        ("f3", "RE3-TT", None),  # code-level faults have no metric indicator
    ],
)
def test_fault_indicator(fault, dataset, indicator):
    assert fault_indicator(fault, dataset) == indicator


@pytest.fixture
def index(layout):
    return pd.read_parquet(layout.raw / "cases.parquet").set_index("case")


def test_extract_ground_truth(layout, index):
    label = extract_ground_truth(RawCase(layout.raw / "re2ob_cartservice_cpu_2"), "RE2-OB-x", index)
    assert label.case_id == "RE2-OB-x"
    assert label.dataset == "RE2-OB"
    assert label.source_case == "re2ob_cartservice_cpu_2"
    assert (label.root_cause_service, label.fault, label.repetition) == ("cartservice", "cpu", 2)
    assert label.indicator == "cpu"
    assert label.inject_time == T0 + 3600 + INJECT_OFFSET
    assert label.fault_description == "cpu fault"
    assert label.root_cause_log is None


def test_root_cause_log_is_a_label(layout, index):
    label = extract_ground_truth(RawCase(layout.raw / "re3ss_carts_f1_1"), "RE3-SS-x", index)
    assert "Request method 'POST' not supported" in label.root_cause_log
    assert label.indicator is None


def test_check_against_index(layout, index):
    case = RawCase(layout.raw / "re2ob_redis_mem_1")
    label = extract_ground_truth(case, "id", index)
    assert check_against_index([label], index) == []

    tampered = index.copy()
    tampered.loc["re2ob_redis_mem_1", "root_cause_service"] = "frontend"
    tampered.loc["re2ob_redis_mem_1", "inject_time"] = 0
    problems = check_against_index([label], tampered)
    assert len(problems) == 2
    assert any("root_cause_service" in p for p in problems)

    assert check_against_index([label], index.drop("re2ob_redis_mem_1")) == [
        "re2ob_redis_mem_1: not in the case index"
    ]


def test_labels_frame_types(layout, index):
    labels = [extract_ground_truth(RawCase(layout.raw / n), f"id-{i}", index)
              for i, n in enumerate(["re3ss_orders_f3_1", "re3ss_carts_f1_1"])]
    frame = labels_frame(labels)
    assert list(frame["case_id"]) == ["id-0", "id-1"]
    assert frame["inject_time"].dtype == "int64"
    assert frame["root_cause_log"].isna().sum() == 1


def test_load_labels_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="aiops-data build"):
        load_labels(tmp_path, "RE2-OB")
