import json

import pandas as pd
import pytest

from aiops.data import download
from aiops.data.download import (
    SOURCE_FILE,
    download_cases,
    read_source,
    resolve_revision,
    select_cases,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def index():
    rows = [
        ("re2ob_cartservice_cpu_1", "RE2-OB", "cartservice", "cpu", 1),
        ("re2ob_cartservice_cpu_2", "RE2-OB", "cartservice", "cpu", 2),
        ("re2ob_cartservice_mem_1", "RE2-OB", "cartservice", "mem", 1),
        ("re2ob_frontend_cpu_3", "RE2-OB", "frontend", "cpu", 3),
        ("re2tt_ts-auth-service_cpu_1", "RE2-TT", "ts-auth-service", "cpu", 1),
    ]
    return pd.DataFrame(rows, columns=["case", "dataset", "root_cause_service", "fault", "repetition"])


def test_select_by_dataset_fault_service(index):
    assert list(select_cases(index, datasets=["re2-tt"])["case"]) == ["re2tt_ts-auth-service_cpu_1"]
    assert len(select_cases(index, faults=["cpu"])) == 4
    assert list(select_cases(index, datasets=["RE2-OB"], services=["frontend"])["case"]) == [
        "re2ob_frontend_cpu_3"
    ]
    assert len(select_cases(index)) == 5


def test_select_max_repetitions_keeps_the_lowest(index):
    picked = select_cases(index, max_repetitions=1)
    assert list(picked["case"]) == [
        "re2ob_cartservice_cpu_1",
        "re2ob_cartservice_mem_1",
        "re2ob_frontend_cpu_3",  # its only repetition is #3
        "re2tt_ts-auth-service_cpu_1",
    ]


def test_full_sha_needs_no_network():
    assert resolve_revision(SHA_A) == SHA_A


def test_empty_selection_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "snapshot_download", lambda **kw: pytest.fail("no download"))
    assert download_cases(tmp_path, []) == 0


def test_revision_is_recorded_and_never_mixed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download, "snapshot_download", lambda **kw: calls.append(kw))
    download_cases(tmp_path, ["re2ob_cartservice_cpu_1"], revision=SHA_A)
    assert read_source(tmp_path)["revision"] == SHA_A
    assert calls[0]["allow_patterns"] == ["re2ob_cartservice_cpu_1/*"]
    assert calls[0]["revision"] == SHA_A

    with pytest.raises(RuntimeError, match="refusing to mix"):
        download_cases(tmp_path, ["re2ob_cartservice_cpu_2"], revision=SHA_B)
    assert len(calls) == 1  # refused before fetching anything
    assert json.loads((tmp_path / SOURCE_FILE).read_text())["revision"] == SHA_A
