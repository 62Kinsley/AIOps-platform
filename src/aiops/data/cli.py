"""``aiops-data``: download RCAEval, build the blind view and labels, inspect cases.

    aiops-data download -d RE2-OB RE2-TT          # or --all; --max-reps 1 for a quick subset
    aiops-data build                              # every dataset with downloaded cases
    aiops-data inspect                            # what is downloaded and built
    aiops-data inspect RE2-OB-1a2b3c4d            # load one blind case's metrics, logs, traces
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from aiops.data.blind import export_dataset
from aiops.data.catalog import DATASETS, HF_REVISION, DataLayout, discover_raw_cases
from aiops.data.dataset import MANIFEST_FILE, BlindCase, BlindDataset
from aiops.data.download import download_cases, fetch_index, select_cases
from aiops.data.readers import metric_services

_MARK = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiops-data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="data directory (default: $AIOPS_DATA_DIR or ./data)")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download cases from Hugging Face")
    which = dl.add_mutually_exclusive_group(required=True)
    which.add_argument("-d", "--dataset", nargs="+", metavar="DATASET",
                       help=f"datasets to fetch: {', '.join(DATASETS)}")
    which.add_argument("--all", action="store_true", help="all 735 cases (3.4 GB)")
    dl.add_argument("--fault", nargs="+", help="only these faults, e.g. cpu mem f1")
    dl.add_argument("--service", nargs="+", help="only these root-cause services")
    dl.add_argument("--max-reps", type=int, help="at most N repetitions per (service, fault)")
    dl.add_argument("--revision", default=HF_REVISION, help="dataset commit (default: pinned)")
    dl.add_argument("--workers", type=int, default=8)

    build = sub.add_parser("build", help="build blind views and ground-truth tables")
    build.add_argument("-d", "--dataset", nargs="+", metavar="DATASET",
                       help="default: every dataset with downloaded cases")
    build.add_argument("--hardlink", action="store_true",
                       help="hardlink telemetry instead of copying it (saves disk, but leaks the "
                            "labels through inode/mtime order)")

    inspect = sub.add_parser("inspect", help="summarise the data, or one blind case")
    inspect.add_argument("case_id", nargs="?", help="e.g. RE2-OB-1a2b3c4d")

    args = parser.parse_args(argv)
    layout = DataLayout(args.data_dir) if args.data_dir else DataLayout.default()
    try:
        return {"download": _download, "build": _build, "inspect": _inspect}[args.command](args, layout)
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _download(args, layout: DataLayout) -> int:
    index = fetch_index(layout.raw, revision=args.revision)
    selected = select_cases(index, datasets=None if args.all else args.dataset,
                            faults=args.fault, services=args.service,
                            max_repetitions=args.max_reps)
    if selected.empty:
        print("no case matches that selection")
        return 1
    counts = selected.groupby("dataset").size()
    print(f"{len(selected)} cases: " + ", ".join(f"{d} {n}" for d, n in counts.items()))
    download_cases(layout.raw, selected["case"], revision=args.revision, max_workers=args.workers)
    print(f"raw mirror (label-bearing): {layout.raw}")
    return 0


def _build(args, layout: DataLayout) -> int:
    datasets = args.dataset or sorted({c.name.dataset for c in discover_raw_cases(layout.raw)},
                                      key=DATASETS.index)
    if not datasets:
        raise FileNotFoundError(f"nothing downloaded under {layout.raw}; run `aiops-data download` first")
    for dataset in datasets:
        report = export_dataset(dataset, layout, hardlink=args.hardlink)
        print(f"{report.dataset}: {report.n_cases} cases -> {report.blind_dir}   "
              f"labels -> {report.labels_path}")
        for name, check in report.audit.items():
            print(f"  {_MARK[check['status']]}  {name:26} {check['detail']}")
    return 0


def _inspect(args, layout: DataLayout) -> int:
    if args.case_id:
        dataset = args.case_id.rsplit("-", 1)[0]
        _describe_case(BlindDataset.open(dataset, layout)[args.case_id])
        return 0
    downloaded: dict[str, int] = {}
    for case in discover_raw_cases(layout.raw):
        downloaded[case.name.dataset] = downloaded.get(case.name.dataset, 0) + case.is_complete
    print(f"{'dataset':8} {'downloaded':>10} {'blind':>6}  example case id")
    for dataset in DATASETS:
        built = layout.blind / dataset / MANIFEST_FILE
        blind = BlindDataset(layout.blind / dataset) if built.is_file() else None
        if not downloaded.get(dataset) and blind is None:
            continue
        print(f"{dataset:8} {downloaded.get(dataset, 0):>10} {len(blind) if blind else '-':>6}  "
              f"{blind.case_ids[0] if blind else ''}")
    return 0


def _describe_case(case: BlindCase) -> None:
    start, end = case.time_range
    print(f"{case.case_id}  ({case.system})")
    print(f"  window   {_utc(start)} -> {_utc(end)} UTC ({(end - start) / 60:.1f} min), "
          f"fault injected at {_utc(case.inject_time)}")
    metrics = case.metrics
    print(f"  metrics  {metrics.shape[0]} samples x {metrics.shape[1]} series from "
          f"{len(metric_services(metrics))} services, {metrics.isna().mean().mean():.1%} null")
    logs = case.logs
    print("  logs     " + ("none recorded" if logs is None else
                          f"{len(logs):,} lines from {logs['service'].nunique()} services"))
    traces = case.traces
    print("  traces   " + ("none recorded" if traces is None else
                          f"{len(traces):,} spans in {traces['trace_id'].nunique():,} traces "
                          f"from {traces['service'].nunique()} services"))
    candidates = case.candidate_services
    shown = ", ".join(candidates[:10]) + (", ..." if len(candidates) > 10 else "")
    print(f"  candidates ({len(candidates)}): {shown}")


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    sys.exit(main())
