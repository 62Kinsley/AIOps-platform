"""``aiops-eval``: run an RCA method over the blind view and score it the way RCAEval does.

    aiops-eval run anomaly                              # every built dataset, every case
    aiops-eval run anomaly -d RE2-OB --split dev --limit 5
    aiops-eval score anomaly --split test --by-fault    # re-score a finished run

A run lives in <data dir>/runs/<name>/ (name: --name, default the method):
run.json and predictions.jsonl, which are label-free, and after scoring
scores.json and case_scores.csv, which hold the ground truth.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from aiops.data.catalog import DATASETS, DataLayout, normalize_dataset
from aiops.data.dataset import MANIFEST_FILE
from aiops.evaluation.baselines import AnomalyRanking
from aiops.evaluation.runner import Method, load_predictions, run_method
from aiops.evaluation.scoring import (SCORES_FILE, SPLITS, macro_mean, score_cases, select_cases,
                                      summarize, write_scores)

METHODS: dict[str, Callable[[DataLayout], Method]] = {
    "anomaly": lambda layout: AnomalyRanking(layout),
    "anomaly-common": lambda layout: AnomalyRanking(layout, common_metrics_only=True),
}
SUMMARY_COLUMNS = ["cases", "AC@1", "AC@3", "AC@5", "Avg@5", "fine Avg@5", "random AC@1"]
FAULT_COLUMNS = ["cases", "AC@1", "AC@3", "Avg@5", "fine Avg@5"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiops-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="data directory (default: $AIOPS_DATA_DIR or ./data)")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="where runs live (default: <data dir>/runs)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a method on blind cases, then score it")
    run.add_argument("method", choices=sorted(METHODS))
    _selection_args(run)
    run.add_argument("--limit", type=int, help="at most N cases per dataset, in id order")
    run.add_argument("--name", help="run name (default: the method name)")
    run.add_argument("--fresh", action="store_true", help="discard the run's earlier predictions")

    score = sub.add_parser("score", help="score an existing run")
    score.add_argument("run", help="run name, e.g. anomaly")
    _selection_args(score)

    args = parser.parse_args(argv)
    layout = DataLayout(args.data_dir) if args.data_dir else DataLayout.default()
    runs_dir = args.runs_dir or layout.root / "runs"
    try:
        return {"run": _run, "score": _score}[args.command](args, layout, runs_dir)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-d", "--dataset", nargs="+", metavar="DATASET",
                        help="default: every built dataset (run) or every dataset of the run (score)")
    parser.add_argument("--split", choices=SPLITS, default="all",
                        help="dev: one case per (root cause, fault) to tune on; test: the rest")
    parser.add_argument("--by-fault", action="store_true", help="also show scores per fault type")


def _run(args, layout: DataLayout, runs_dir: Path) -> int:
    cases = {d: select_cases(layout, d, args.split, args.limit)
             for d in _datasets(args.dataset, layout)}
    total = sum(len(ids) for ids in cases.values())
    method = METHODS[args.method](layout)
    run_dir = runs_dir / (args.name or method.name)
    print(f"{method.name} on {total} cases ({args.split} split) -> {run_dir}")
    ran = []

    def progress(case_id: str, error: str | None) -> None:
        ran.append(case_id)
        if error:
            print(f"  {case_id}: {error}", file=sys.stderr)

    started = time.perf_counter()
    predictions = run_method(method, cases, layout, run_dir, fresh=args.fresh, progress=progress)
    print(f"ran {len(ran)} cases in {time.perf_counter() - started:.1f} s"
          + (f" ({total - len(ran)} done earlier)" if total > len(ran) else ""))
    wanted = {case_id for ids in cases.values() for case_id in ids}
    return _report(predictions[predictions["case_id"].isin(wanted)], layout, run_dir, args)


def _score(args, layout: DataLayout, runs_dir: Path) -> int:
    run_dir = runs_dir / args.run
    predictions = load_predictions(run_dir)
    if args.dataset:
        wanted = {normalize_dataset(d) for d in args.dataset}
        predictions = predictions[predictions["dataset"].isin(wanted)]
    return _report(predictions, layout, run_dir, args)


def _report(predictions: pd.DataFrame, layout: DataLayout, run_dir: Path, args) -> int:
    """Print the scores of ``predictions`` (the selection asked for); the files in the
    run directory always cover every case the run has predicted."""
    if predictions.empty:
        print("no predictions to score")
        return 1
    scored = score_cases(load_predictions(run_dir), layout)
    write_scores(run_dir, scored)
    cases = scored[scored["case_id"].isin(predictions["case_id"])]
    if args.split != "all":
        cases = cases[cases["split"] == args.split]
    if cases.empty:
        print(f"no predictions in the {args.split} split")
        return 1
    summary = summarize(cases)
    print(_table(summary, ["dataset"], SUMMARY_COLUMNS, footer=macro_mean(summary)))
    if args.by_fault:
        print()
        print(_table(summarize(cases, by=("dataset", "fault")), ["dataset", "fault"], FAULT_COLUMNS))
    errors = int(cases["error"].notna().sum())
    if errors:
        print(f"{errors} cases failed and count as misses; run again to retry them")
    print(f"scores -> {run_dir / SCORES_FILE}")
    return 0


def _datasets(names: list[str] | None, layout: DataLayout) -> list[str]:
    if names:
        return [normalize_dataset(name) for name in names]
    built = [d for d in DATASETS if (layout.blind / d / MANIFEST_FILE).is_file()]
    if not built:
        raise FileNotFoundError(f"no blind view under {layout.blind}; run `aiops-data build` first")
    return built


def _table(frame: pd.DataFrame, keys: list[str], columns: list[str],
           footer: dict | None = None) -> str:
    header = keys + columns
    rows = [[str(row[k]) for k in keys] + [_fmt(row[c]) for c in columns]
            for row in frame.to_dict("records")]
    if footer is not None:
        rows.append(["mean"] + [""] * (len(keys) - 1) + [_fmt(footer[c]) for c in columns])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) if i < len(keys) else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    return "\n".join([line(header)] + [line(row) for row in rows])


def _fmt(value) -> str:
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2f}"


if __name__ == "__main__":
    sys.exit(main())
