"""RCAEval dataset registry and discovery of raw cases on disk.

The raw mirror (``data/raw/rcaeval``) is the Hugging Face copy of RCAEval: one
directory per failure case, named ``{suite}{system}_{service}_{fault}_{repetition}``
(e.g. ``re2tt_ts-order-service_cpu_1``). That name *is* the answer, so this
module is label-bearing: RCA methods must never see a :class:`RawCase`. They get
the anonymised view built by :mod:`aiops.data.blind` instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

HF_REPO_ID = "phamquiluan/RCAEval"
# Pinned dataset commit, so every download sees the same 735 cases.
# Pass revision="main" to follow upstream fixes instead.
HF_REVISION = "8b6c06a7a15bcecdbf70b611a13b789c5925dc17"

SYSTEMS = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket"}
SUITES = ("re1", "re2", "re3")
DATASETS = tuple(f"{suite.upper()}-{system.upper()}" for suite in SUITES for system in SYSTEMS)

TELEMETRY_FILES = {
    "metrics": "metrics.parquet",
    "logs": "logs.parquet",
    "traces": "traces.parquet",
}
INJECT_TIME_FILE = "inject_time.txt"
# Present in 8 RE3-SS cases: the log line that *is* the root cause. Label, not telemetry.
ROOT_CAUSE_FILE = "root_cause.txt"
INDEX_FILE = "cases.parquet"  # per-case index, including the ground truth

_CASE_RE = re.compile(
    r"^(?P<suite>re[123])(?P<system>ob|ss|tt)_(?P<service>.+)_(?P<fault>[a-z]+\d*)_(?P<repetition>\d+)$"
)


@dataclass(frozen=True)
class DataLayout:
    """Where each zone of the pipeline lives under one data directory.

    ``raw`` and ``labels`` are label-bearing; ``blind`` is the only zone an RCA
    method may read.
    """

    root: Path

    @classmethod
    def default(cls) -> DataLayout:
        return cls(Path(os.environ.get("AIOPS_DATA_DIR", "data")))

    @property
    def raw(self) -> Path:
        return self.root / "raw" / "rcaeval"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def blind(self) -> Path:
        return self.root / "blind"


def normalize_dataset(name: str) -> str:
    """Accept ``RE2-OB``, ``re2-ob`` or ``re2ob`` and return ``RE2-OB``."""
    compact = name.strip().lower().replace("-", "").replace("_", "")
    match = re.fullmatch(r"(re[123])(ob|ss|tt)", compact)
    if not match:
        raise ValueError(f"unknown dataset {name!r}; expected one of {', '.join(DATASETS)}")
    return f"{match[1].upper()}-{match[2].upper()}"


def dataset_prefix(dataset: str) -> str:
    """``RE2-OB`` -> ``re2ob``, the prefix of that dataset's case directories."""
    return normalize_dataset(dataset).lower().replace("-", "")


@dataclass(frozen=True, order=True)
class CaseName:
    """The fields encoded in a raw case directory name (root cause included)."""

    suite: str
    system: str
    service: str
    fault: str
    repetition: int

    @classmethod
    def parse(cls, name: str) -> CaseName:
        match = _CASE_RE.match(name)
        if not match:
            raise ValueError(f"not an RCAEval case directory name: {name!r}")
        return cls(
            suite=match["suite"],
            system=match["system"],
            service=match["service"],
            fault=match["fault"],
            repetition=int(match["repetition"]),
        )

    @property
    def dataset(self) -> str:
        return f"{self.suite.upper()}-{self.system.upper()}"

    def __str__(self) -> str:
        return f"{self.suite}{self.system}_{self.service}_{self.fault}_{self.repetition}"


@dataclass(frozen=True)
class RawCase:
    """One downloaded case directory in the raw (label-bearing) mirror."""

    path: Path

    @property
    def name(self) -> CaseName:
        return CaseName.parse(self.path.name)

    def file(self, kind: str) -> Path | None:
        """Path of a telemetry file (``metrics``/``logs``/``traces``), or None if absent."""
        path = self.path / TELEMETRY_FILES[kind]
        return path if path.is_file() else None

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(kind for kind in TELEMETRY_FILES if self.file(kind) is not None)

    @property
    def is_complete(self) -> bool:
        """Metrics and inject time are the minimum every RCAEval case has."""
        return self.file("metrics") is not None and (self.path / INJECT_TIME_FILE).is_file()

    @property
    def inject_time(self) -> int:
        return read_inject_time(self.path / INJECT_TIME_FILE)


def read_inject_time(path: Path) -> int:
    text = Path(path).read_text().strip()
    try:
        return int(text.split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"{path}: expected a unix timestamp, got {text[:40]!r}") from None


def discover_raw_cases(raw_dir: Path, datasets: list[str] | None = None) -> list[RawCase]:
    """Case directories under ``raw_dir``, sorted by name, optionally for some datasets only.

    Incomplete directories (an interrupted download) are included; check
    :attr:`RawCase.is_complete` before reading them.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    wanted = {normalize_dataset(d) for d in datasets} if datasets else None
    cases = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir() or not _CASE_RE.match(entry.name):
            continue
        case = RawCase(entry)
        if wanted is None or case.name.dataset in wanted:
            cases.append(case)
    return cases
