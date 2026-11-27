"""Reader invariants on the real RCAEval mirror, one case per downloaded dataset.

Skipped unless the data has been downloaded (``aiops-data download``).
"""

import pytest

from aiops.data.catalog import DATASETS, DataLayout, discover_raw_cases
from aiops.data.readers import metric_services, read_logs, read_metrics, read_traces

LAYOUT = DataLayout.default()
FIRST_CASE = {}
for _case in discover_raw_cases(LAYOUT.raw):
    if _case.is_complete:
        FIRST_CASE.setdefault(_case.name.dataset, _case)

pytestmark = pytest.mark.skipif(not FIRST_CASE, reason=f"no RCAEval data under {LAYOUT.raw}")


@pytest.mark.parametrize("dataset", [d for d in DATASETS if d in FIRST_CASE])
def test_sources_share_one_clock_and_one_set_of_names(dataset):
    case = FIRST_CASE[dataset]
    metrics = read_metrics(case.file("metrics"))
    assert metrics.index.is_unique and metrics.index.is_monotonic_increasing
    assert metrics.index.min() < case.inject_time < metrics.index.max()
    services = set(metric_services(metrics))
    start, end = metrics.index.min(), metrics.index.max()
    for path, reader in ((case.file("logs"), read_logs), (case.file("traces"), read_traces)):
        if path is None:
            continue
        frame = reader(path)
        assert frame["time"].is_monotonic_increasing
        assert start - 1 <= frame["time"].min() and frame["time"].max() <= end + 1
        assert set(frame["service"].dropna()) <= services
