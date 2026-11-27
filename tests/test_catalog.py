import pytest

from aiops.data.catalog import (
    CaseName,
    dataset_prefix,
    discover_raw_cases,
    normalize_dataset,
    read_inject_time,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("re2ob_checkoutservice_cpu_1", ("re2", "ob", "checkoutservice", "cpu", 1)),
        ("re2tt_ts-auth-service_socket_3", ("re2", "tt", "ts-auth-service", "socket", 3)),
        ("re3ss_front-end_f2_12", ("re3", "ss", "front-end", "f2", 12)),
    ],
)
def test_parse_case_name(name, expected):
    parsed = CaseName.parse(name)
    assert (parsed.suite, parsed.system, parsed.service, parsed.fault, parsed.repetition) == expected
    assert str(parsed) == name


def test_case_name_dataset():
    assert CaseName.parse("re2tt_ts-order-service_cpu_1").dataset == "RE2-TT"


@pytest.mark.parametrize("bad", ["re4ob_x_cpu_1", "re2ob_cpu_1", "RE2-OB-1a2b3c4d", "cases.parquet"])
def test_parse_rejects_non_case_names(bad):
    with pytest.raises(ValueError):
        CaseName.parse(bad)


@pytest.mark.parametrize("given", ["RE2-OB", "re2-ob", "re2ob", " Re2_Ob "])
def test_normalize_dataset(given):
    assert normalize_dataset(given) == "RE2-OB"
    assert dataset_prefix(given) == "re2ob"


def test_normalize_dataset_rejects_unknown():
    with pytest.raises(ValueError, match="unknown dataset"):
        normalize_dataset("RE4-OB")


def test_discover_filters_by_dataset(layout):
    names = [c.path.name for c in discover_raw_cases(layout.raw, ["re3-ss"])]
    assert names == ["re3ss_carts_f1_1", "re3ss_orders_f3_1"]
    assert len(discover_raw_cases(layout.raw)) == 6


def test_discover_flags_incomplete_cases(layout):
    partial = layout.raw / "re2ob_adservice_cpu_1"
    partial.mkdir()
    (partial / "logs.parquet").write_bytes(b"")
    cases = {c.path.name: c for c in discover_raw_cases(layout.raw, ["RE2-OB"])}
    assert not cases["re2ob_adservice_cpu_1"].is_complete
    assert cases["re2ob_redis_mem_1"].is_complete
    assert cases["re2ob_redis_mem_1"].sources == ("metrics", "logs", "traces")
    assert cases["re2ob_adservice_cpu_1"].file("metrics") is None


def test_discover_missing_dir_is_empty(tmp_path):
    assert discover_raw_cases(tmp_path / "nope") == []


def test_read_inject_time(tmp_path):
    path = tmp_path / "inject_time.txt"
    path.write_text("1705354566\n")
    assert read_inject_time(path) == 1705354566
    path.write_text("soon")
    with pytest.raises(ValueError, match="unix timestamp"):
        read_inject_time(path)
