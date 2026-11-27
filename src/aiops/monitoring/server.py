"""``aiops-monitoring-server``: the Monitoring MCP server, five tools over the blind view.

    aiops-monitoring-server                        # stdio: an MCP client launches it
    aiops-monitoring-server --common-metrics       # hide the metric-column shortcut
    aiops-monitoring-server --transport streamable-http --port 8001

Tools: get_incident_overview, detect_anomalies, query_metrics, search_logs, query_traces.
"""

import argparse
import sys
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from aiops.data.catalog import DataLayout
from aiops.mcp_support import add_transport_args, call, package_version, register, serve
from aiops.monitoring.service import MonitoringService

INSTRUCTIONS = """\
Telemetry of microservice failure cases from the RCAEval benchmark: one case is one incident, \
identified by a case id such as RE2-OB-0123abcd. Start with get_incident_overview, then \
detect_anomalies to see what changed, then drill down with query_metrics, search_logs and \
query_traces. All times are seconds relative to the alert (negative = before it); the window \
before the alert is the baseline. The goal of an investigation is the root-cause service, one of \
the overview's root_cause_candidates. A service with strong anomalies can be a victim of a slow \
or failing dependency, so compare onsets and follow the call graph (the Knowledge server's \
get_service_topology) before concluding."""

CaseId = Annotated[str, Field(description="Case id, e.g. RE2-OB-0123abcd")]
Service = Annotated[str | None, Field(
    description="Service name as listed by get_incident_overview, e.g. cartservice")]
Start = Annotated[float | None, Field(
    description="Window start in seconds relative to the alert (negative = before it). "
                "Default: start of the case")]
End = Annotated[float | None, Field(
    description="Window end in seconds relative to the alert. Default: end of the case")]


def create_server(monitoring: MonitoringService | None = None) -> MCPServer:
    monitoring = monitoring or MonitoringService()
    server = MCPServer("aiops-monitoring", instructions=INSTRUCTIONS, version=package_version())

    def get_incident_overview(case_id: CaseId) -> str:
        """Start here. What one failure case contains: the system, the alert time, the time
        window, which telemetry exists (metrics, logs, traces), the services with their metric
        kinds, log and span counts, and the root-cause candidates (the services the answer is
        chosen from). All monitoring tools take times in seconds relative to the alert."""
        return call(monitoring.incident_overview, case_id=case_id)

    def detect_anomalies(
        case_id: CaseId,
        service: Service = None,
        metric: Annotated[str | None, Field(
            description="Metric kind (cpu, mem, diskio, socket, workload, error, latency-50, "
                        "latency-90; 'latency' means both) or a full series name")] = None,
        top_k: Annotated[int, Field(ge=1, le=50, description="Series to list")] = 15,
    ) -> str:
        """Rank metric series by how abnormal they are after the alert, each against its own
        baseline before it (robust z-score: >=3 notable, >=10 strong; 'mixed' = rose and fell),
        with when each anomaly began (onset_s) and a per-service summary. Use it to decide where
        to look. The most anomalous service is not necessarily the cause: callers slow down
        when a callee is slow, so compare onsets and check dependencies."""
        return call(monitoring.detect_anomalies, case_id=case_id, service=service, metric=metric,
                    top_k=top_k)

    def query_metrics(
        case_id: CaseId,
        service: Service = None,
        metric: Annotated[str | None, Field(
            description="Metric kind across services (cpu, mem, diskio, socket, workload, error, "
                        "latency-50, latency-90; 'latency' means both) or a full series name")] = None,
        start: Start = None,
        end: End = None,
        points: Annotated[int, Field(ge=4, le=120, description="Buckets per series")] = 24,
    ) -> str:
        """Metric time series, averaged into `points` buckets over the window (t = bucket start),
        each with its anomaly score, baseline median and onset. Give a service (all its
        metrics), a metric kind (that metric for every service), or both. Series come most
        anomalous first; at most 30 are summarised and the first 12 include values."""
        return call(monitoring.query_metrics, case_id=case_id, service=service, metric=metric,
                    start=start, end=end, points=points)

    def search_logs(
        case_id: CaseId,
        service: Service = None,
        query: Annotated[str | None, Field(
            description="Case-insensitive regex (RE2 syntax) the lines must match, e.g. "
                        "'error|exception' or 'timeout'")] = None,
        start: Start = None,
        end: End = None,
        limit: Annotated[int, Field(ge=1, le=50, description="Templates (and examples) to list")] = 15,
    ) -> str:
        """Compare log messages before and after the alert. Lines are grouped into templates
        (ids, numbers and timestamps masked). Lists per-service log volume and error-like line
        counts, templates that are new or more frequent after the alert (score = log2 of the
        frequency ratio, error-like first) and templates that became rarer. With query, only
        matching lines count and `examples` shows raw lines near the alert. Some services log
        errors all the time, so a change matters more than a count."""
        return call(monitoring.search_logs, case_id=case_id, service=service, query=query,
                    start=start, end=end, limit=limit)

    def query_traces(
        case_id: CaseId,
        service: Service = None,
        operation: Annotated[str | None, Field(
            description="Case-insensitive substring of the operation name")] = None,
        trace_id: Annotated[str | None, Field(
            description="Show this one trace as a call tree (other filters are ignored)")] = None,
        start: Start = None,
        end: End = None,
        limit: Annotated[int, Field(ge=1, le=50, description="Operations and calls to list")] = 10,
    ) -> str:
        """Compare span latency and errors before and after the alert, per operation (p50/p90 in
        ms, error rate, p90 ratio) and per call between two services, measured on the caller's
        and the callee's side: when only the caller side slows down, the time is lost between
        them (network, or a queue at an overloaded callee). Also lists operations and calls that
        appeared or stopped after the alert, with example trace ids; pass trace_id to see one
        trace as an indented call tree. Online Boutique and Train Ticket record traces; Sock
        Shop and RE1 cases do not."""
        return call(monitoring.query_traces, case_id=case_id, service=service, operation=operation,
                    trace_id=trace_id, start=start, end=end, limit=limit)

    for tool in (get_incident_overview, detect_anomalies, query_metrics, search_logs, query_traces):
        register(server, tool)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiops-monitoring-server", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="data directory (default: $AIOPS_DATA_DIR or ./data)")
    parser.add_argument("--common-metrics", action="store_true",
                        help="only expose metric columns present in every case of a dataset, which "
                             "removes the column-presence shortcut (see README)")
    add_transport_args(parser, default_port=8001)
    args = parser.parse_args(argv)
    layout = DataLayout(args.data_dir) if args.data_dir else DataLayout.default()
    if not layout.blind.is_dir():
        print(f"warning: no blind view under {layout.blind}; set --data-dir or $AIOPS_DATA_DIR",
              file=sys.stderr)
    serve(create_server(MonitoringService(layout, common_metrics_only=args.common_metrics)), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
