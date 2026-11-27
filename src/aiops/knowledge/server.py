"""``aiops-knowledge-server``: the Knowledge MCP server, two tools over static knowledge.

    aiops-knowledge-server                         # stdio: an MCP client launches it
    aiops-knowledge-server --transport streamable-http --port 8002

Tools: get_service_topology, search_runbooks. The server reads no case data.
"""

import argparse
import sys
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from aiops.knowledge.service import KnowledgeBase
from aiops.mcp_support import add_transport_args, call, package_version, register, serve

INSTRUCTIONS = """\
Static operations knowledge about the three microservice systems of the RCAEval benchmark \
(Online Boutique, Sock Shop, Train Ticket): their architecture and call graph \
(get_service_topology) and runbooks describing how each kind of fault shows up in metrics, logs \
and traces, and how to tell the faulty service from its victims (search_runbooks). Use it with \
the Monitoring server: map the services that look anomalous onto the call graph, then check the \
runbook of the fault you suspect. Start with search_runbooks('triage-workflow') if unsure how to \
proceed."""


def create_server(knowledge: KnowledgeBase | None = None) -> MCPServer:
    knowledge = knowledge or KnowledgeBase()
    server = MCPServer("aiops-knowledge", instructions=INSTRUCTIONS, version=package_version())

    def get_service_topology(
        system: Annotated[str, Field(
            description="System name (Online Boutique, Sock Shop, Train Ticket, or OB/SS/TT), "
                        "a dataset such as RE2-OB, or a case id")],
        service: Annotated[str | None, Field(
            description="A service of that system; omit for the whole call graph")] = None,
        depth: Annotated[int, Field(ge=1, le=6,
                                    description="Hops for dependencies/dependents")] = 2,
    ) -> str:
        """Architecture of a system: its services (kind, role), who calls whom, the entry
        point, known background noise in the logs, and names in the telemetry that are not
        services. With service: its direct callers and callees, its transitive dependencies
        (services whose faults can make it look faulty) and dependents (services that show
        symptoms when it fails), each call with its evidence (traces, documented, naming)."""
        return call(knowledge.service_topology, system=system, service=service, depth=depth)

    def search_runbooks(
        query: Annotated[str | None, Field(
            description="Symptoms or a fault type in English, e.g. 'latency up but cpu normal', "
                        "'new exceptions in logs', 'memory leak', or a runbook id. Omit to list "
                        "all runbooks")] = None,
        top_k: Annotated[int, Field(ge=1, le=10, description="Runbooks to return")] = 3,
    ) -> str:
        """Search the runbooks: how CPU saturation, memory pressure, disk I/O stress, network
        delay, packet loss, socket exhaustion and code-level faults show up in metrics, logs and
        traces, how to confirm each with the monitoring tools, and how to tell them apart; plus
        the triage workflow and how faults propagate between services. Returns the best
        matches with their full text (BM25 ranking)."""
        return call(knowledge.search_runbooks, query=query, top_k=top_k)

    for tool in (get_service_topology, search_runbooks):
        register(server, tool)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiops-knowledge-server", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_transport_args(parser, default_port=8002)
    args = parser.parse_args(argv)
    serve(create_server(), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
