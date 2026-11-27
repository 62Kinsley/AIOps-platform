"""The seven tools on the real RCAEval blind view, one case per built dataset.

Skipped unless the blind view has been built (``aiops-data build``).
"""

import pytest

from aiops.data.catalog import DATASETS, DataLayout
from aiops.data.dataset import MANIFEST_FILE, BlindDataset
from aiops.knowledge import KnowledgeBase
from aiops.knowledge.topology import resolve_system
from aiops.mcp_support import to_json
from aiops.monitoring import MonitoringService
from aiops.monitoring import traces as tracetools

LAYOUT = DataLayout.default()
BUILT = [d for d in DATASETS if (LAYOUT.blind / d / MANIFEST_FILE).is_file()]
MAX_CHARS = 16_000  # about 4k tokens: what one tool result may cost an agent

pytestmark = pytest.mark.skipif(not BUILT, reason=f"no blind view under {LAYOUT.blind}")


@pytest.fixture(scope="module")
def service():
    return MonitoringService(LAYOUT, cache_size=1)


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.mark.parametrize("dataset", BUILT)
def test_monitoring_tools_answer_within_budget(service, kb, dataset):
    case = BlindDataset.open(dataset, LAYOUT)[0]
    overview = service.incident_overview(case.case_id)
    results = {
        "get_incident_overview": overview,
        "detect_anomalies": service.detect_anomalies(case.case_id),
        "query_metrics": service.query_metrics(case.case_id, metric="cpu"),
        "search_logs": service.search_logs(case.case_id),
        "search_logs(query)": service.search_logs(case.case_id, query="error|exception"),
        "query_traces": service.query_traces(case.case_id),
        "get_service_topology": kb.service_topology(case.case_id),
        "search_runbooks": kb.search_runbooks("latency up, cpu normal"),
    }
    for name, result in results.items():
        assert len(to_json(result)) < MAX_CHARS, f"{dataset} {name}: {len(to_json(result))} chars"
    assert results["detect_anomalies"]["series_scored"] > 0
    assert overview["window_s"][0] < 0 < overview["window_s"][1]
    has_logs, has_traces = case.logs is not None, case.traces is not None
    assert ("logs_available" not in results["search_logs"]) == has_logs
    assert ("traces_available" not in results["query_traces"]) == has_traces
    if has_traces:
        trace_id = results["query_traces"]["operations"][0]["example_trace"]
        tree = service.query_traces(case.case_id, trace_id=trace_id)
        assert tree["spans"] >= 1 and tree["tree"]


@pytest.mark.parametrize("dataset", BUILT)
def test_topology_knows_every_candidate_service(kb, dataset):
    topology = resolve_system(dataset, kb.topologies)
    others = {s for t in kb.topologies.values() if t is not topology for s in t.services}
    for name in BlindDataset.open(dataset, LAYOUT).candidate_services:
        assert (name in topology.services or name in topology.telemetry_names
                or name in others), f"{dataset}: {name} is unknown to the knowledge base"


@pytest.mark.parametrize("dataset", [d for d in BUILT if d[-2:] in ("OB", "TT") and d[:3] != "RE1"])
def test_traced_calls_are_in_the_topology(kb, dataset):
    """Every service-to-service call in the pre-alert traces is a call of the topology."""
    topology = resolve_system(dataset, kb.topologies)
    known = {(c["from"], c["to"]) for c in topology.calls}
    for case in list(BlindDataset.open(dataset, LAYOUT))[:3]:
        spans = tracetools.prepare(case.traces, case.inject_time)
        pairs = tracetools.calls(spans[spans["t"] < 0])
        observed = set(zip(pairs["caller"].astype(str), pairs["callee"].astype(str)))
        assert observed <= known, f"{case.case_id}: unknown calls {sorted(observed - known)}"
