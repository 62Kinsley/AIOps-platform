"""Build the blind view: telemetry under anonymised ids, ground truth kept elsewhere.

For dataset D this writes::

    data/blind/D/manifest.json                     case ids, candidate services
    data/blind/D/cases/<case_id>/metrics.parquet   telemetry, byte-identical to the raw files
    data/blind/D/cases/<case_id>/logs.parquet      (when the system records logs)
    data/blind/D/cases/<case_id>/traces.parquet    (when the system records traces)
    data/blind/D/cases/<case_id>/case.json         inject time, time range, files, candidates
    data/labels/D.parquet                          ground truth, keyed by case_id
    data/labels/D.json                             salt, provenance, leakage audit

Left behind in the raw mirror: the answer-bearing directory names,
``cases.parquet``, ``root_cause.txt`` and ``inject_time.txt``. The inject time's
value moves into case.json, because RCAEval hands it to RCA methods as the
anomaly-detection time.

Case ids are prefixes of HMAC-SHA256(salt, source name). The salt is random per
dataset and stored only with the labels: ids stay stable across rebuilds but
cannot be recomputed from the public case names.

Telemetry is copied, in case-id order, by default. A hardlink would share the
raw file's inode number and mtime, and on the Hugging Face mirror both follow
the alphabetical order of the answer-bearing names (rank correlation ~1.0), so
``ls -li`` alone would undo the anonymisation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from aiops.data.catalog import (
    INDEX_FILE,
    SYSTEMS,
    TELEMETRY_FILES,
    DataLayout,
    RawCase,
    discover_raw_cases,
    normalize_dataset,
)
from aiops.data.dataset import CASE_FILE, CASES_DIR, MANIFEST_FILE, SCHEMA_VERSION
from aiops.data.download import read_source
from aiops.data.ground_truth import (
    GroundTruth,
    check_against_index,
    extract_ground_truth,
    labels_path,
    labels_frame,
)
from aiops.data.readers import split_metric_name

ALLOWED_FILES = {MANIFEST_FILE, CASE_FILE, *TELEMETRY_FILES.values()}


@dataclass(frozen=True)
class ExportReport:
    dataset: str
    n_cases: int
    blind_dir: Path
    labels_path: Path
    meta_path: Path
    audit: dict


def assign_case_ids(dataset: str, sources: list[str], salt: str) -> dict[str, str]:
    """source case name -> ``{dataset}-{8 hex}``, keyed by a secret salt."""
    ids = {
        source: f"{dataset}-{hmac.new(salt.encode(), source.encode(), hashlib.sha256).hexdigest()[:8]}"
        for source in sources
    }
    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"case id collision in {dataset}; delete its labels .json to draw a new salt")
    return ids


def export_dataset(dataset: str, layout: DataLayout, *, hardlink: bool = False) -> ExportReport:
    """Build (or rebuild) the blind view and label table of one dataset."""
    dataset = normalize_dataset(dataset)
    raw_cases = discover_raw_cases(layout.raw, [dataset])
    if not raw_cases:
        raise FileNotFoundError(
            f"no {dataset} cases under {layout.raw}; run `aiops-data download -d {dataset}` first"
        )
    incomplete = [case.path.name for case in raw_cases if not case.is_complete]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} incomplete {dataset} case(s), e.g. {incomplete[0]} "
            "(interrupted download?); re-run `aiops-data download` for them"
        )

    out = layout.blind / dataset
    if out.exists() and not (out / MANIFEST_FILE).is_file():
        raise FileExistsError(f"{out} exists but is not a blind view (no {MANIFEST_FILE}); not touching it")

    meta_path = layout.labels / f"{dataset}.json"
    salt = _previous_salt(meta_path) or secrets.token_hex(16)
    ids = assign_case_ids(dataset, [case.path.name for case in raw_cases], salt)

    # Labels first: if the two sources of truth disagree, stop before writing anything.
    index = _load_index(layout.raw)
    labels = [extract_ground_truth(case, ids[case.path.name], index) for case in raw_cases]
    if index is not None:
        problems = check_against_index(labels, index)
        if problems:
            raise ValueError("labels disagree with cases.parquet:\n  " + "\n  ".join(problems[:20]))

    candidates = sorted({svc for case in raw_cases for svc in _metric_services(case)})
    tmp = layout.blind / f".{dataset}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / CASES_DIR).mkdir(parents=True)
    # Create cases in id order, so inode numbers and timestamps follow the ids.
    for case in sorted(raw_cases, key=lambda c: ids[c.path.name]):
        _write_case(case, tmp / CASES_DIR / ids[case.path.name], ids[case.path.name], dataset,
                    candidates, hardlink)
    _write_json(tmp / MANIFEST_FILE, {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "system": SYSTEMS[dataset[-2:].lower()],
        "n_cases": len(ids),
        "case_ids": sorted(ids.values()),
        "candidate_services": candidates,
        "note": "Ground truth is not in this directory; evaluators read it with "
                "aiops.data.ground_truth.load_labels.",
    })

    audit = audit_blind_view(tmp, labels, candidates)
    failed = [name for name, check in audit.items() if check["status"] == "fail"]
    if failed:
        shutil.rmtree(tmp)
        raise RuntimeError(f"leakage audit failed for {dataset}: " + "; ".join(
            f"{name}: {audit[name]['detail']}" for name in failed))

    labels_file = labels_path(layout.labels, dataset)
    layout.labels.mkdir(parents=True, exist_ok=True)
    labels_frame(labels).to_parquet(labels_file.with_suffix(".tmp"), index=False)
    _write_json(meta_path.with_suffix(".json.tmp"), {
        "dataset": dataset,
        "salt": salt,
        "source": read_source(layout.raw),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(labels),
        "telemetry_files": "hardlink" if hardlink else "copy",
        "audit": audit,
    })
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    labels_file.with_suffix(".tmp").replace(labels_file)
    meta_path.with_suffix(".json.tmp").replace(meta_path)
    return ExportReport(dataset, len(labels), out, labels_file, meta_path, audit)


def _write_case(case: RawCase, case_dir: Path, case_id: str, dataset: str,
                candidates: list[str], hardlink: bool) -> None:
    case_dir.mkdir()
    files, rows = {}, {}
    for kind, filename in TELEMETRY_FILES.items():
        src = case.file(kind)
        if src is None:
            continue
        _place(src, case_dir / filename, hardlink)
        files[kind] = filename
        rows[kind] = pq.read_metadata(src).num_rows
    time = pq.read_table(case.file("metrics"), columns=["time"]).column("time").to_pandas()
    _write_json(case_dir / CASE_FILE, {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "dataset": dataset,
        "system": SYSTEMS[case.name.system],
        "inject_time": case.inject_time,
        "time_range": {"start": int(time.min()), "end": int(time.max())},
        "files": files,
        "rows": rows,
        "candidate_services": candidates,
    })


def _place(src: Path, dst: Path, hardlink: bool) -> None:
    """Copy (fresh inode and mtime) or hardlink. Never a symlink: its target is the answer."""
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # other filesystem: fall back to a copy
    shutil.copyfile(src, dst)  # data only; copy2 would carry over the raw mtime


# --------------------------------------------------------------------------- audit


def audit_blind_view(root: Path, labels: list[GroundTruth], candidates: list[str]) -> dict:
    """Look for every way the blind view at ``root`` could still reveal ``labels``.

    Each check is ``{"status": "ok" | "warn" | "fail", "value": ..., "detail": str}``.
    "fail" means the export itself leaks (a bug) and aborts the build; "warn"
    flags a property of the benchmark data that a method could exploit.
    """
    sources = [label.source_case for label in labels]
    truth = {label.case_id: label.root_cause_service for label in labels}
    cases_dir = root / CASES_DIR
    checks: dict[str, dict] = {}

    unexpected = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                        if p.is_file() and p.name not in ALLOWED_FILES)
    checks["only_telemetry_files"] = _check(
        not unexpected, len(unexpected),
        f"unexpected files: {unexpected[:5]}" if unexpected else
        "only metrics/logs/traces.parquet, case.json and manifest.json are present", fail=True)

    hits = _source_name_hits(root, sources)
    checks["no_source_names"] = _check(
        not hits, len(hits),
        f"answer-bearing case names found in: {hits[:5]}" if hits else
        "no raw case name appears in any path, JSON file or Parquet metadata", fail=True)

    missing = [label.case_id for label in labels if label.root_cause_service not in candidates]
    checks["root_cause_in_candidates"] = _check(
        not missing, f"{len(labels) - len(missing)}/{len(labels)}",
        "every root cause is a candidate service" if not missing else
        f"root cause missing from candidates in {missing[:5]}")

    linked = sum(path.stat().st_nlink > 1 for path in cases_dir.rglob("*.parquet"))
    stats = [os.stat(cases_dir / label.case_id / TELEMETRY_FILES["metrics"]) for label in labels]
    order = {
        "inode_rank_corr": _rank_corr([s.st_ino for s in stats], sources),
        "mtime_rank_corr": _rank_corr([s.st_mtime_ns for s in stats], sources),
        # creation time exists on macOS only; an APFS clone would keep the raw one
        "birthtime_rank_corr": _rank_corr([getattr(s, "st_birthtime", None) for s in stats], sources),
    }
    # |rank corr| of a random order has sd ~ 1/sqrt(n-1); only flag clear structure.
    threshold = max(0.5, 3 / math.sqrt(max(len(labels) - 1, 1)))
    ordered = max(abs(corr or 0.0) for corr in order.values()) > threshold
    checks["file_order"] = _check(
        not linked and not ordered, {"hardlinked_files": linked, **order},
        "telemetry files are fresh copies whose inode/time order is unrelated to the raw case names"
        if not linked and not ordered else
        f"{linked} telemetry file(s) are hardlinks to the raw mirror: they share its inode numbers "
        "and mtimes, which follow the answer-bearing names, and `find -samefile` leads straight "
        "to them; rebuild without --hardlink" if linked else
        "file inode/timestamp order follows the raw case names; sorting files by inode or "
        "time recovers the labels")

    column_sets = {label.case_id: {n for n in pq.read_schema(
        cases_dir / label.case_id / TELEMETRY_FILES["metrics"]).names if n != "time"}
        for label in labels}
    shortcut = schema_shortcut_ac1(column_sets, truth)
    random_ac1 = 1 / len(candidates)
    ok = shortcut <= max(2 * random_ac1, random_ac1 + 0.05)
    measured = (f"a cheat that reads only which metric columns a case has, never their values, "
                f"scores AC@1={shortcut:.2f} (random {random_ac1:.2f})")
    checks["metric_schema_shortcut"] = _check(
        ok, {"ac1": round(shortcut, 3), "random_ac1": round(random_ac1, 3)},
        measured if ok else
        f"{measured}; restrict methods to BlindDataset.common_metric_columns() to remove it")

    rate, chance = time_adjacency(labels)
    ok = rate <= max(2 * chance, chance + 0.05)
    measured = (f"{rate:.0%} of cases injected back to back share root cause and fault "
                f"(chance {chance:.0%})")
    checks["time_adjacency"] = _check(
        ok, {"same_label_rate": round(rate, 3), "chance": round(chance, 3)},
        measured if ok else
        f"{measured}; do not let a method see other cases' inject times")
    return checks


def schema_shortcut_ac1(column_sets: dict[str, set[str]], truth: dict[str, str]) -> float:
    """AC@1 of a cheat that ignores all values: per case, predict the service owning
    the most metric columns that fewer than half of the dataset's cases have."""
    n = len(column_sets)
    freq = Counter(column for columns in column_sets.values() for column in columns)
    hits = 0.0
    for case_id, columns in column_sets.items():
        votes = Counter(split_metric_name(c)[0] for c in columns if freq[c] < n / 2)
        if votes:
            top = max(votes.values())
            best = [service for service, count in votes.items() if count == top]
            hits += (truth[case_id] in best) / len(best)  # ties broken at random
    return hits / n


def time_adjacency(labels: list[GroundTruth]) -> tuple[float, float]:
    """Share of consecutive cases (by inject time) with the same (service, fault),
    and the share expected if the injection order were random."""
    ordered = sorted(labels, key=lambda label: label.inject_time)
    keys = [(label.root_cause_service, label.fault) for label in ordered]
    n = len(keys)
    if n < 2:
        return 0.0, 0.0
    rate = sum(a == b for a, b in zip(keys, keys[1:])) / (n - 1)
    chance = sum(k * (k - 1) for k in Counter(keys).values()) / (n * (n - 1))
    return rate, chance


def _source_name_hits(root: Path, sources: list[str]) -> list[str]:
    """Paths whose name, JSON text or Parquet key-value metadata mention a raw case name."""
    pattern = re.compile(b"|".join(re.escape(s.encode()) for s in sources))
    hits = []
    for path in root.rglob("*"):
        rel = str(path.relative_to(root))
        blobs = [rel.encode()]
        if path.suffix == ".json":
            blobs.append(path.read_bytes())
        elif path.suffix == ".parquet":
            for key, value in (pq.read_metadata(path).metadata or {}).items():
                blobs.append(base64.b64decode(value) if key == b"ARROW:schema" else value)
        if any(pattern.search(blob) for blob in blobs):
            hits.append(rel)
    return hits


def _check(ok: bool, value, detail: str, *, fail: bool = False) -> dict:
    return {"status": "ok" if ok else ("fail" if fail else "warn"), "value": value, "detail": detail}


def _rank_corr(values: list, names: list[str]) -> float | None:
    """Spearman correlation of ``values`` with the alphabetical order of ``names``."""
    if len(values) < 3 or any(v is None for v in values):
        return None
    corr = float(pd.Series(values).rank().corr(pd.Series(names).rank()))
    return None if math.isnan(corr) else round(corr, 3)


def _metric_services(case: RawCase) -> set[str]:
    names = pq.read_schema(case.file("metrics")).names
    return {split_metric_name(n)[0] for n in names if n != "time"}


def _load_index(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / INDEX_FILE
    return pd.read_parquet(path).set_index("case") if path.is_file() else None


def _previous_salt(meta_path: Path) -> str | None:
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text()).get("salt")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
