"""Download RCAEval cases from the Hugging Face mirror (Parquet, 3.4 GB for all 735).

The index (``cases.parquet``) comes first; it lists every case with its
dataset, root-cause service, fault and repetition, so a subset can be chosen
before any telemetry is fetched. Downloads are resumable and idempotent:
files already on disk at the same revision are skipped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from aiops.data.catalog import HF_REPO_ID, HF_REVISION, INDEX_FILE, normalize_dataset

SOURCE_FILE = "SOURCE.json"  # which repo and revision the raw mirror came from


def resolve_revision(revision: str) -> str:
    """A branch or tag name -> its commit sha; a full sha is returned as is."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return HfApi().dataset_info(HF_REPO_ID, revision=revision).sha


def fetch_index(raw_dir: Path, *, revision: str = HF_REVISION) -> pd.DataFrame:
    """Download (if needed) and load the case index."""
    raw_dir = Path(raw_dir)
    revision = _claim_revision(raw_dir, revision)
    hf_hub_download(
        HF_REPO_ID, INDEX_FILE, repo_type="dataset", revision=revision, local_dir=raw_dir
    )
    return pd.read_parquet(raw_dir / INDEX_FILE)


def select_cases(
    index: pd.DataFrame,
    *,
    datasets: Iterable[str] | None = None,
    faults: Iterable[str] | None = None,
    services: Iterable[str] | None = None,
    max_repetitions: int | None = None,
) -> pd.DataFrame:
    """Filter the index. ``max_repetitions=1`` keeps one case per (service, fault)."""
    selected = index
    if datasets:
        selected = selected[selected["dataset"].isin({normalize_dataset(d) for d in datasets})]
    if faults:
        selected = selected[selected["fault"].isin(set(faults))]
    if services:
        selected = selected[selected["root_cause_service"].isin(set(services))]
    if max_repetitions is not None:
        rank = selected.groupby(["dataset", "root_cause_service", "fault"])["repetition"].rank(
            method="first"
        )
        selected = selected[rank <= max_repetitions]
    return selected.sort_values("case", ignore_index=True)


def download_cases(
    raw_dir: Path,
    cases: Iterable[str],
    *,
    revision: str = HF_REVISION,
    max_workers: int = 8,
) -> int:
    """Download every file of the named case directories; returns the number of cases."""
    names = sorted(set(cases))
    if not names:
        return 0
    revision = _claim_revision(Path(raw_dir), revision)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=Path(raw_dir),
        allow_patterns=[f"{name}/*" for name in names],
        max_workers=max_workers,
    )
    return len(names)


def read_source(raw_dir: Path) -> dict:
    path = Path(raw_dir) / SOURCE_FILE
    return json.loads(path.read_text()) if path.is_file() else {}


def _claim_revision(raw_dir: Path, revision: str) -> str:
    """Resolve the revision and record it, refusing to mix two revisions in one mirror.

    Runs before any file is fetched, so a refused download leaves the mirror untouched.
    """
    revision = resolve_revision(revision)
    recorded = read_source(raw_dir).get("revision")
    if recorded and recorded != revision:
        raise RuntimeError(
            f"{raw_dir} holds revision {recorded}; refusing to mix in {revision}. "
            "Use a separate --data-dir for another revision."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / SOURCE_FILE).write_text(
        json.dumps({"repo_id": HF_REPO_ID, "revision": revision}, indent=2) + "\n"
    )
    return revision
