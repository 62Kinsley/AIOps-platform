"""The blind view must hold everything a method needs and nothing that names the answer."""

import hashlib
import json
import re
import shutil

import pandas as pd
import pytest

from aiops.data.blind import (
    assign_case_ids,
    export_dataset,
    schema_shortcut_ac1,
    time_adjacency,
)
from aiops.data.catalog import discover_raw_cases
from aiops.data.dataset import BlindCase, BlindDataset
from aiops.data.ground_truth import GroundTruth, load_labels
from tests.conftest import CASES, INJECT_OFFSET

SOURCES = {"RE2-OB": [c[0] for c in CASES if c[0].startswith("re2ob")],
           "RE3-SS": [c[0] for c in CASES if c[0].startswith("re3ss")]}


@pytest.fixture
def built(layout):
    return {d: export_dataset(d, layout) for d in SOURCES}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_layout_and_ids(built, layout):
    report = built["RE2-OB"]
    assert report.n_cases == 4
    case_dirs = sorted(p.name for p in (layout.blind / "RE2-OB" / "cases").iterdir())
    assert all(re.fullmatch(r"RE2-OB-[0-9a-f]{8}", name) for name in case_dirs)
    assert case_dirs == BlindDataset.open("RE2-OB", layout).case_ids
    assert not list(layout.blind.glob(".*.partial"))
    # labels live outside the blind tree
    assert report.labels_path.parent == layout.labels
    assert layout.blind not in report.labels_path.parents
    assert layout.blind not in report.meta_path.parents


def test_no_answer_reaches_the_blind_view(built, layout):
    labels = {d: load_labels(layout.labels, d) for d in SOURCES}
    for dataset, report in built.items():
        for path in report.blind_dir.rglob("*"):
            rel = str(path.relative_to(report.blind_dir))
            assert not any(source in rel for source in SOURCES[dataset])
            assert path.name not in {"root_cause.txt", "inject_time.txt", "cases.parquet"}
            if path.suffix == ".json":
                text = path.read_text()
                assert not any(source in text for source in SOURCES[dataset])
                assert "root_cause" not in text and '"fault' not in text
        for case_id, row in labels[dataset].iterrows():
            info = json.loads((report.blind_dir / "cases" / case_id / "case.json").read_text())
            assert set(info) == {"schema_version", "case_id", "dataset", "system", "inject_time",
                                 "time_range", "files", "rows", "candidate_services"}
            assert row["root_cause_service"] in info["candidate_services"]
    assert all(c["status"] == "ok" for name, c in built["RE2-OB"].audit.items()
               if name in ("only_telemetry_files", "no_source_names", "file_order",
                           "root_cause_in_candidates"))


def test_labels_join_the_blind_view(built, layout):
    labels = load_labels(layout.labels, "RE3-SS")
    blind = BlindDataset.open("RE3-SS", layout)
    assert sorted(labels.index) == blind.case_ids
    by_source = labels.reset_index().set_index("source_case")
    assert by_source.loc["re3ss_carts_f1_1", "root_cause_service"] == "carts"
    assert "not supported" in by_source.loc["re3ss_carts_f1_1", "root_cause_log"]
    case = blind[by_source.loc["re3ss_carts_f1_1", "case_id"]]
    assert case.inject_time == by_source.loc["re3ss_carts_f1_1", "inject_time"]


def test_telemetry_is_copied_byte_for_byte(built, layout):
    labels = load_labels(layout.labels, "RE2-OB")
    for case_id, row in labels.iterrows():
        for name in ("metrics.parquet", "logs.parquet", "traces.parquet"):
            raw = layout.raw / row["source_case"] / name
            blind = layout.blind / "RE2-OB" / "cases" / case_id / name
            assert _sha(raw) == _sha(blind)
            assert blind.stat().st_nlink == 1
            assert blind.stat().st_ino != raw.stat().st_ino


def test_hardlink_mode_is_flagged(layout):
    report = export_dataset("RE2-OB", layout, hardlink=True)
    check = report.audit["file_order"]
    assert check["status"] == "warn"
    assert check["value"]["hardlinked_files"] == 4 * 3


def test_ids_are_stable_and_salted(built, layout):
    first = BlindDataset.open("RE2-OB", layout).case_ids
    export_dataset("RE2-OB", layout)
    assert BlindDataset.open("RE2-OB", layout).case_ids == first  # salt reused
    (layout.labels / "RE2-OB.json").unlink()
    export_dataset("RE2-OB", layout)
    assert set(BlindDataset.open("RE2-OB", layout).case_ids).isdisjoint(first)  # new salt


def test_assign_case_ids_depends_on_salt():
    a = assign_case_ids("RE2-OB", ["re2ob_x_cpu_1", "re2ob_x_cpu_2"], "salt-a")
    b = assign_case_ids("RE2-OB", ["re2ob_x_cpu_1", "re2ob_x_cpu_2"], "salt-b")
    assert a != b and len(set(a.values())) == 2


def test_refuses_to_replace_a_foreign_directory(layout):
    target = layout.blind / "RE2-OB"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        export_dataset("RE2-OB", layout)
    assert (target / "notes.txt").read_text() == "mine"


def test_incomplete_download_writes_nothing(layout):
    (layout.raw / "re2ob_adservice_cpu_1").mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        export_dataset("RE2-OB", layout)
    assert not layout.blind.exists() and not layout.labels.exists()


def test_index_disagreement_writes_nothing(layout):
    index = pd.read_parquet(layout.raw / "cases.parquet")
    index.loc[index.case == "re2ob_redis_mem_1", "inject_time"] += 1
    index.to_parquet(layout.raw / "cases.parquet", index=False)
    with pytest.raises(ValueError, match="disagree"):
        export_dataset("RE2-OB", layout)
    assert not layout.blind.exists()


def test_missing_dataset(layout):
    with pytest.raises(FileNotFoundError, match="download"):
        export_dataset("RE1-TT", layout)


def test_blind_case_api(built, layout):
    dataset = BlindDataset.open("re2-ob", layout)
    assert len(dataset) == 4 and dataset[0].case_id == dataset.case_ids[0]
    case = dataset[dataset.case_ids[1]]
    assert isinstance(case, BlindCase)
    assert case.sources == ("metrics", "logs", "traces")
    start, end = case.time_range
    assert case.inject_time == start + INJECT_OFFSET and end > case.inject_time
    assert case.metrics.index.min() == start
    assert len(case.logs) == 30 and len(case.traces) == 4
    assert not any(hasattr(case, attr) for attr in ("root_cause_service", "fault", "source_case"))
    with pytest.raises(KeyError):
        dataset["RE2-OB-00000000"]

    ss_case = BlindDataset.open("RE3-SS", layout)[0]
    assert ss_case.traces is None and ss_case.logs is not None


def test_common_metric_columns(built, layout):
    dataset = BlindDataset.open("RE2-OB", layout)
    every = set(dataset[0].metrics.columns)
    assert set(dataset.common_metric_columns()) == every
    # add a column to one case only: it is no longer common
    path = dataset.root / "cases" / dataset.case_ids[0] / "metrics.parquet"
    frame = pd.read_parquet(path)
    frame["cartservice_diskio"] = 0.0
    frame.to_parquet(path, index=False)
    assert "cartservice_diskio" not in dataset.common_metric_columns()


def test_schema_shortcut_ac1():
    column_sets = {
        "a": {"x_cpu", "y_cpu", "x_diskio"},  # only case with x_diskio: votes x
        "b": {"x_cpu", "y_cpu", "y_error"},  # votes y
        "c": {"x_cpu", "y_cpu"},  # no rare columns: no vote
        "d": {"x_cpu", "y_cpu"},
    }
    truth = {"a": "x", "b": "x", "c": "x", "d": "y"}
    assert schema_shortcut_ac1(column_sets, truth) == pytest.approx(1 / 4)


def _label(service, fault, t):
    return GroundTruth("id", "RE2-OB", "src", service, fault, None, None, t, 1, None)


def test_time_adjacency():
    labels = [_label("a", "cpu", 1), _label("a", "cpu", 2), _label("b", "mem", 3), _label("b", "mem", 4)]
    rate, chance = time_adjacency(labels)
    assert rate == pytest.approx(2 / 3)
    assert chance == pytest.approx(4 / 12)
    assert time_adjacency(labels[:1]) == (0.0, 0.0)


def test_rebuild_after_removing_raw_case(built, layout):
    shutil.rmtree(layout.raw / "re2ob_redis_mem_1")
    report = export_dataset("RE2-OB", layout)
    assert report.n_cases == 3
    assert len(load_labels(layout.labels, "RE2-OB")) == 3
    assert len(discover_raw_cases(layout.raw, ["RE2-OB"])) == 3
