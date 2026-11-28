"""Evaluation: RCAEval-style scoring, the dev/test split, the runner and the anomaly baseline."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiops.data import BlindDataset
from aiops.evaluation import cli
from aiops.evaluation.baselines import AnomalyRanking
from aiops.evaluation.runner import PREDICTIONS_FILE, Prediction, load_predictions, run_method
from aiops.evaluation.scoring import (accuracy, assign_splits, average_accuracy, indicator_ranking,
                                      rank_of, score_cases, service_ranking, summarize, to_service)

SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- scoring rules

def test_rcaeval_name_rules():
    assert to_service("carts-db_cpu") == "carts"
    assert to_service("ts-order-service_latency-90") == "ts-order-service"
    assert service_ranking(["carts-db_cpu", "carts_mem", "orders_cpu", "carts"]) == ["carts", "orders"]
    assert indicator_ranking(["a_latency-50", "a_latency-90", "b_cpu", "c"]) == [
        ("a", "latency"), ("b", "cpu"), ("c", "unknown")]
    assert rank_of("b", ["a", "b"]) == 2 and np.isnan(rank_of("z", ["a"]))


def test_accuracy_and_average():
    ranks = pd.Series([1, 2, np.nan, 4, 6], dtype=float)
    assert [accuracy(ranks, k) for k in (1, 3, 5)] == [0.2, 0.4, 0.6]
    assert average_accuracy(ranks) == pytest.approx((0.2 + 0.4 + 0.4 + 0.6 + 0.6) / 5)


def test_summary_leaves_code_faults_out_of_the_metric_level():
    cases = pd.DataFrame({
        "dataset": ["X"] * 4,
        "fault": ["cpu", "cpu", "f1", "f1"],
        "indicator": ["cpu", "cpu", None, None],
        "rank": [1.0, np.nan, 2.0, 1.0],
        "rank_fine": [1.0, 3.0, np.nan, np.nan],
        "chance_ac1": [0.1, 0.1, 0.1, 0.0],
        "error": [None, "RuntimeError: boom", None, None],
    })
    row = summarize(cases).iloc[0]
    assert row["cases"] == 4 and row["AC@1"] == 0.5 and row["AC@3"] == 0.75
    assert row["fine AC@1"] == 0.5 and row["fine Avg@5"] == pytest.approx(0.8)
    assert row["random AC@1"] == pytest.approx(0.075) and row["errors"] == 1
    by_fault = summarize(cases, by=("dataset", "fault")).set_index("fault")
    assert np.isnan(by_fault.loc["f1", "fine Avg@5"]) and by_fault.loc["f1", "AC@1"] == 0.5


def test_split_puts_one_case_per_root_cause_and_fault_in_dev():
    ids = [f"D-{i:08x}" for i in range(12)]
    labels = pd.DataFrame({"root_cause_service": ["a"] * 6 + ["b"] * 6,
                           "fault": (["cpu"] * 3 + ["mem"] * 3) * 2},
                          index=pd.Index(ids[::-1], name="case_id"))
    split = assign_splits(labels)
    assert split.index.equals(labels.index)
    dev = labels[split == "dev"]
    assert len(dev) == 4 and not dev.duplicated(["root_cause_service", "fault"]).any()
    for _, group in labels.groupby(["root_cause_service", "fault"]):
        assert split[min(group.index)] == "dev"  # stable: the smallest (salted) id


# --------------------------------------------------------------------------- baseline

def test_anomaly_baseline_finds_the_planted_incident(incident_layout, planted_case_id):
    case = BlindDataset.open("RE2-OB", incident_layout)[planted_case_id]
    prediction = AnomalyRanking(incident_layout).rank(case)
    assert prediction.services[0] == "cartservice"
    assert prediction.indicators[0] == "cartservice_cpu"
    assert set(prediction.services) == set(case.candidate_services)
    # cartservice_diskio exists only in this case, so the common-columns variant drops it
    assert "cartservice_diskio" in prediction.indicators
    common = AnomalyRanking(incident_layout, common_metrics_only=True).rank(case)
    assert "cartservice_diskio" not in common.indicators and common.services[0] == "cartservice"


# --------------------------------------------------------------------------- runner

class Flaky:
    """A fake method that fails on the given cases and records what it was asked."""

    name = "flaky"

    def __init__(self, fail=(), version=1):
        self.params = {"version": version}
        self.fail = set(fail)
        self.calls = []

    def rank(self, case):
        self.calls.append(case.case_id)
        if case.case_id in self.fail:
            raise RuntimeError("model timed out")
        return Prediction(services=["frontend", "cartservice"], info={"tokens": 7})


def test_runner_records_errors_and_resumes(incident_layout, tmp_path):
    ids = BlindDataset.open("RE2-OB", incident_layout).case_ids
    cases = {"RE2-OB": ids}
    first = Flaky(fail=ids[:1])
    out = run_method(first, cases, incident_layout, tmp_path).set_index("case_id")
    assert first.calls == ids
    assert out.loc[ids[0], "error"] == "RuntimeError: model timed out"
    assert out.loc[ids[1], "info"] == {"tokens": 7}
    scored = score_cases(out.reset_index(), incident_layout).set_index("case_id")
    assert np.isnan(scored.loc[ids[0], "rank"])  # a failed case is a miss

    second = Flaky()
    out = run_method(second, cases, incident_layout, tmp_path)
    assert second.calls == ids[:1]  # only the failed case runs again
    assert len(out) == len(ids) and out["error"].isna().all()

    changed = Flaky(version=2)
    with pytest.raises(ValueError, match="--fresh"):
        run_method(changed, cases, incident_layout, tmp_path)
    run_method(changed, cases, incident_layout, tmp_path, fresh=True)
    assert changed.calls == ids


def test_a_record_cut_short_by_an_interrupt_is_rerun(incident_layout, tmp_path):
    ids = BlindDataset.open("RE2-OB", incident_layout).case_ids
    run_method(Flaky(), {"RE2-OB": ids[:2]}, incident_layout, tmp_path)
    path = tmp_path / PREDICTIONS_FILE
    path.write_text(path.read_text() + '{"case_id": "' + ids[2])
    method = Flaky()
    out = run_method(method, {"RE2-OB": ids[:3]}, incident_layout, tmp_path)
    assert method.calls == [ids[2]] and list(out["case_id"]) == sorted(ids[:3])


def test_the_method_side_never_loads_labels():
    code = ("import sys, aiops.evaluation.runner, aiops.evaluation.baselines; "
            "print([m for m in ('aiops.data.ground_truth', 'aiops.data.blind') if m in sys.modules])")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONPATH": str(SRC)})
    assert result.stdout.strip() == "[]"


# --------------------------------------------------------------------------- command line

def test_run_and_score_from_the_command_line(incident_layout, planted_case_id, tmp_path, capsys):
    base = ["--data-dir", str(incident_layout.root), "--runs-dir", str(tmp_path)]
    assert cli.main(base + ["run", "anomaly", "-d", "RE2-OB", "--by-fault"]) == 0
    out = capsys.readouterr().out
    assert "AC@1" in out and "RE2-OB" in out and "cpu" in out
    run_dir = tmp_path / "anomaly"
    assert len(load_predictions(run_dir)) == 4
    cases = pd.read_csv(run_dir / "case_scores.csv").set_index("case_id")
    assert cases.loc[planted_case_id, "rank"] == 1 and cases.loc[planted_case_id, "rank_fine"] == 1
    scores = json.loads((run_dir / "scores.json").read_text())
    assert scores["all"]["datasets"][0]["dataset"] == "RE2-OB"
    # dev holds one case per (root cause, fault): 3 of this mirror's 4 RE2-OB cases
    assert [scores[s]["datasets"][0]["cases"] for s in ("all", "dev", "test")] == [4, 3, 1]

    # scoring a selection prints it but leaves the run's files covering the whole run
    assert cli.main(base + ["score", "anomaly", "--split", "dev"]) == 0
    assert "RE2-OB       3" in capsys.readouterr().out
    assert json.loads((run_dir / "scores.json").read_text()) == scores
    assert cli.main(base + ["score", "missing-run"]) == 1
    assert "no predictions" in capsys.readouterr().err
