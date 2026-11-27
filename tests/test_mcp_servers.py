"""The two MCP servers end to end: tool listing, calls and errors through an MCP client,
in-process and over stdio (the way an agent runtime launches them)."""

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters

from aiops.knowledge.server import create_server as knowledge_server
from aiops.monitoring import MonitoringService
from aiops.monitoring.server import create_server as monitoring_server

MONITORING_TOOLS = ["get_incident_overview", "detect_anomalies", "query_metrics", "search_logs",
                    "query_traces"]
KNOWLEDGE_TOOLS = ["get_service_topology", "search_runbooks"]
SRC = Path(__file__).resolve().parents[1] / "src"


def run(coroutine_fn, *args):
    return anyio.run(coroutine_fn, *args)


async def _list_and_call(server, calls):
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        results = [await client.call_tool(name, args) for name, args in calls]
        return client.instructions, tools, results


def _text(result) -> str:
    (block,) = result.content
    return block.text


@pytest.fixture(scope="module")
def monitoring(incident_layout):
    return monitoring_server(MonitoringService(incident_layout))


def test_the_seven_tools(monitoring):
    _, mon_tools, _ = run(_list_and_call, monitoring, [])
    _, kb_tools, _ = run(_list_and_call, knowledge_server(), [])
    assert [t.name for t in mon_tools] == MONITORING_TOOLS
    assert [t.name for t in kb_tools] == KNOWLEDGE_TOOLS
    for tool in mon_tools + kb_tools:
        assert tool.annotations.read_only_hint is True and tool.annotations.destructive_hint is False
        assert len(tool.description) > 200 and "\n    " not in tool.description  # dedented
        assert tool.output_schema is None  # plain JSON text, no duplicate structured copy
        for name, prop in tool.input_schema["properties"].items():
            assert prop.get("description"), f"{tool.name}.{name} has no description"
    required = {t.name: t.input_schema.get("required", []) for t in mon_tools + kb_tools}
    assert all(required[name] == ["case_id"] for name in MONITORING_TOOLS)
    assert required == {**{n: ["case_id"] for n in MONITORING_TOOLS},
                        "get_service_topology": ["system"], "search_runbooks": []}


def test_monitoring_calls(monitoring, planted_case_id):
    calls = [(name, {"case_id": planted_case_id}) for name in MONITORING_TOOLS]
    calls[2][1]["service"] = "cartservice"  # query_metrics needs a service or a metric
    instructions, _, results = run(_list_and_call, monitoring, calls)
    assert "root_cause_candidates" in instructions
    for (name, _), result in zip(calls, results):
        assert not result.is_error, f"{name}: {_text(result)}"
        assert json.loads(_text(result))["case_id"] == planted_case_id
    anomalies = json.loads(_text(results[1]))["anomalies"]
    assert anomalies[0]["service"] == "cartservice"


def test_monitoring_errors_reach_the_client(monitoring, planted_case_id):
    calls = [("get_incident_overview", {"case_id": "nonsense"}),
             ("query_metrics", {"case_id": planted_case_id, "service": "cartsevice"}),
             ("detect_anomalies", {"case_id": planted_case_id, "top_k": 0}),
             ("search_logs", {})]
    _, _, results = run(_list_and_call, monitoring, calls)
    assert all(r.is_error for r in results)
    assert "is not a case id" in _text(results[0])
    assert "did you mean cartservice" in _text(results[1])
    assert "greater than or equal to 1" in _text(results[2])
    assert "case_id" in _text(results[3])


def test_knowledge_calls():
    calls = [("get_service_topology", {"system": "Train Ticket", "service": "ts-route-service"}),
             ("search_runbooks", {"query": "memory leak"}),
             ("get_service_topology", {"system": "Kubernetes"})]
    instructions, _, results = run(_list_and_call, knowledge_server(), calls)
    assert "triage-workflow" in instructions
    route = json.loads(_text(results[0]))
    assert {c["service"] for c in route["called_by"]} >= {"ts-travel-service", "ts-basic-service"}
    assert json.loads(_text(results[1]))["results"][0]["id"] == "memory-pressure"
    assert results[2].is_error and "unknown system" in _text(results[2])


async def _over_stdio(args, calls):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    async with Client(params, read_timeout_seconds=60) as client:
        names = [t.name for t in (await client.list_tools()).tools]
        return names, [await client.call_tool(name, a) for name, a in calls]


def test_servers_over_stdio(incident_layout, planted_case_id):
    names, results = run(_over_stdio,
                         ["-m", "aiops.monitoring.server", "--data-dir", str(incident_layout.root)],
                         [("get_incident_overview", {"case_id": planted_case_id})])
    assert names == MONITORING_TOOLS
    assert json.loads(_text(results[0]))["system"] == "Online Boutique"
    names, results = run(_over_stdio, ["-m", "aiops.knowledge.server"],
                         [("get_service_topology", {"system": "OB"})])
    assert names == KNOWLEDGE_TOOLS
    assert json.loads(_text(results[0]))["entry_points"] == ["frontend"]
