"""Knowledge Server tools: topology queries and runbook search over the packaged knowledge."""

import re

import pytest

from aiops.knowledge import KnowledgeBase
from aiops.knowledge import runbooks as runbook_module
from aiops.knowledge import topology as topology_module
from aiops.knowledge.runbooks import Runbook, RunbookIndex, tokenize
from aiops.knowledge.topology import EVIDENCE, resolve_system
from aiops.mcp_support import InvalidRequest


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def test_every_system_loads_consistently(kb):
    assert set(kb.topologies) == {"online-boutique", "sock-shop", "train-ticket"}
    for topology in kb.topologies.values():
        assert topology.entry_points and set(topology.entry_points) <= set(topology.services)
        assert {c["evidence"] for c in topology.calls} <= set(EVIDENCE)
        assert all(info["kind"] and info["description"] for info in topology.services.values())
        pairs = [(c["from"], c["to"]) for c in topology.calls]
        assert len(pairs) == len(set(pairs)), f"duplicate calls in {topology.key}"


@pytest.mark.parametrize("name, key", [
    ("Online Boutique", "online-boutique"), ("online_boutique", "online-boutique"), ("OB", "online-boutique"),
    ("RE2-OB", "online-boutique"), ("re3ob", "online-boutique"), ("RE2-OB-0078b4ee", "online-boutique"),
    ("sock shop", "sock-shop"), ("RE1-SS", "sock-shop"), ("Train Ticket", "train-ticket"), ("tt", "train-ticket"),
])
def test_resolve_system(kb, name, key):
    assert resolve_system(name, kb.topologies).key == key


def test_resolve_unknown_system(kb):
    with pytest.raises(InvalidRequest, match="unknown system 'hipster'.*Online Boutique"):
        resolve_system("hipster", kb.topologies)


def test_service_view(kb):
    out = kb.service_topology("Online Boutique", "cartservice")
    assert out["kind"] == "service" and out["tech"] == "C#"
    assert [c["service"] for c in out["calls"]] == ["redis"]
    assert {c["service"] for c in out["called_by"]} == {"frontend", "checkoutservice"}
    assert out["dependencies"] == [{"service": "redis", "hops": 1}]
    assert {d["service"] for d in out["dependents"]} == {"frontend", "checkoutservice"}
    assert set(out["evidence"]) == {"documented", "traces"}
    assert out["entry_point"] is False
    assert kb.service_topology("OB", "frontendservice")["service"] == "frontend"  # trace alias


def test_transitive_reach_grows_with_depth(kb):
    near = kb.service_topology("TT", "ts-station-service", depth=1)["dependents"]
    far = kb.service_topology("TT", "ts-station-service", depth=4)["dependents"]
    assert {d["hops"] for d in near} == {1} and len(far) > len(near)
    assert {"ts-basic-service", "ts-order-service"} <= {d["service"] for d in near}
    assert "ts-ui-dashboard" in {d["service"] for d in far}
    assert [d["hops"] for d in far] == sorted(d["hops"] for d in far)


def test_system_view(kb):
    out = kb.service_topology("Sock Shop")
    services = {s["service"]: s for s in out["services"]}
    assert set(services["orders"]["calls"]) == {"user", "carts", "payment", "shipping", "orders-db"}
    assert "calls" not in services["payment"]
    assert out["entry_points"] == ["front-end"]
    assert {n["service"] for n in out["known_noise"]} == {"queue-master", "carts"}


def test_names_outside_the_architecture(kb):
    istio = kb.service_topology("RE2-TT", "istio-init")
    assert istio["in_architecture"] is False and "init container" in istio["note"]
    foreign = kb.service_topology("RE3-OB", "carts")
    assert foreign["in_architecture"] is False and "belongs to Sock Shop" in foreign["note"]
    with pytest.raises(InvalidRequest, match="did you mean productcatalogservice"):
        kb.service_topology("OB", "productcatalog")


def test_runbooks_parse_and_are_complete():
    index = RunbookIndex.load()
    ids = [r.id for r in index.runbooks]
    assert len(ids) == 9
    for runbook in index.runbooks:
        path = runbook_module.DATA_DIR / f"{runbook.id}.md"
        assert path.is_file(), "a runbook's file name must be its id"
        assert runbook.keywords and runbook.summary and "## " in runbook.body
    with pytest.raises(ValueError, match="header"):
        Runbook.parse("no header here")
    with pytest.raises(ValueError, match="duplicate"):
        RunbookIndex(index.runbooks[:1] * 2)


@pytest.mark.parametrize("query, expected", [
    ("cpu hog, callers slow", "cpu-saturation"),
    ("memory leak oom restart", "memory-pressure"),
    ("disk io stress slow database", "disk-io-stress"),
    ("latency up on callers but callee cpu normal", "network-delay"),
    ("p90 much higher than p50 errors", "packet-loss"),
    ("connection refused too many open files", "socket-exhaustion"),
    ("new exception stack trace in logs", "code-level-fault"),
    ("which service is the root cause", "triage-workflow"),
    ("victim or cause? upstream callers", "fault-propagation"),
    ("cpu-saturation", "cpu-saturation"),
])
def test_runbook_search_ranks_the_matching_runbook_first(kb, query, expected):
    results = kb.search_runbooks(query)["results"]
    assert results[0]["id"] == expected
    assert results[0]["body"].startswith("## ")


def test_runbook_search_edge_cases(kb):
    catalogue = kb.search_runbooks()
    assert len(catalogue["runbooks"]) == 9 and "body" not in catalogue["runbooks"][0]
    nothing = kb.search_runbooks("zzzz qqqq")
    assert nothing["results"] == [] and len(nothing["other_runbooks"]) == 9
    exact = kb.search_runbooks("packet-loss", top_k=1)
    assert [r["id"] for r in exact["results"]] == ["packet-loss"] and exact["results"][0]["score"] == 1000
    assert tokenize("The errors, timeouts and process") == ["error", "timeout", "process"]


def test_knowledge_files_hold_no_case_specific_data():
    """The knowledge base must stay case-independent: no case names, ids or label tables."""
    raw_name = re.compile(r"re[123](ob|ss|tt)_[\w-]+_[a-z]+\d*_\d+")
    case_id = re.compile(r"RE[123]-(OB|SS|TT)-[0-9a-f]{8}")
    files = [*topology_module.DATA_DIR.glob("*.json"), *runbook_module.DATA_DIR.glob("*.md")]
    assert len(files) == 12
    for path in files:
        text = path.read_text()
        assert not raw_name.search(text) and not case_id.search(text), path.name
        assert "root_cause_service" not in text, path.name
