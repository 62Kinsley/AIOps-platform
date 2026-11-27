"""The Knowledge Server's tools: system architecture and fault runbooks.

Everything here is static knowledge shipped with the package. The Knowledge
Server never opens case data, so it cannot leak labels, and its answers are the
same for every case of a system.
"""

from __future__ import annotations

from pathlib import Path

from aiops.data.readers import SERVICE_ALIASES
from aiops.knowledge import runbooks as runbook_module
from aiops.knowledge import topology as topology_module
from aiops.knowledge.runbooks import RunbookIndex
from aiops.knowledge.topology import EVIDENCE, Topology, load_topologies, resolve_system
from aiops.mcp_support import InvalidRequest, close_matches


class KnowledgeBase:
    def __init__(self, topology_dir: Path | None = None, runbook_dir: Path | None = None):
        self.topologies = load_topologies(topology_dir or topology_module.DATA_DIR)
        self.runbooks = RunbookIndex.load(runbook_dir or runbook_module.DATA_DIR)

    # ------------------------------------------------------------------ topology

    def service_topology(self, system: str, service: str | None = None, depth: int = 2) -> dict:
        topology = resolve_system(system, self.topologies)
        if not service or not service.strip():
            return self._system_view(topology)
        name = SERVICE_ALIASES.get(service.strip(), service.strip())
        if name not in topology.services:
            return self._outside_architecture(topology, name, service)
        callees, callers = topology.callees(name), topology.callers(name)
        used = {c["evidence"] for c in callees + callers}
        return {
            "system": topology.system,
            "service": name,
            **topology.services[name],
            "entry_point": name in topology.entry_points,
            "calls": [{"service": c["to"], "protocol": c["protocol"], "evidence": c["evidence"]}
                      for c in callees],
            "called_by": [{"service": c["from"], "protocol": c["protocol"],
                           "evidence": c["evidence"]} for c in callers],
            "dependencies": topology.reachable(name, depth, towards="callees"),
            "dependents": topology.reachable(name, depth, towards="callers"),
            "meaning": {
                "dependencies": f"services {name} calls, directly or within {depth} hops: a fault "
                                f"in one of them can make {name} look faulty",
                "dependents": f"services that call {name} within {depth} hops: they show "
                              f"symptoms when {name} fails",
            },
            "evidence": {k: v for k, v in EVIDENCE.items() if k in used},
            "known_noise": [n for n in topology.known_noise if n["service"] == name],
            "notes": [n for n in topology.notes if name in n],
        }

    def _system_view(self, topology: Topology) -> dict:
        calls: dict[str, list[str]] = {}
        for call in topology.calls:
            calls.setdefault(call["from"], []).append(call["to"])
        services = []
        for name, info in topology.services.items():
            row = {"service": name, "kind": info["kind"], "description": info["description"]}
            if calls.get(name):
                row["calls"] = calls[name]
            services.append(row)
        return {
            "system": topology.system,
            "datasets": list(topology.datasets),
            "source": topology.source,
            "entry_points": list(topology.entry_points),
            "services": services,
            "evidence": {k: v for k, v in EVIDENCE.items()
                         if any(c["evidence"] == k for c in topology.calls)},
            "telemetry_names": topology.telemetry_names,
            "known_noise": list(topology.known_noise),
            "notes": list(topology.notes),
            "hint": "pass service=<name> for its callers and its transitive dependencies",
        }

    def _outside_architecture(self, topology: Topology, name: str, asked: str) -> dict:
        note = topology.telemetry_names.get(name)
        if note is None:
            owner = next((t for t in self.topologies.values()
                          if t is not topology and name in t.services), None)
            if owner is None:
                raise InvalidRequest(f"no service {asked!r} in {topology.system}"
                                     + close_matches(name, list(topology.services)))
            note = (f"{name} belongs to {owner.system}, not {topology.system}; its telemetry in "
                    f"{topology.system} cases comes from outside the system under test")
        return {"system": topology.system, "service": name, "in_architecture": False, "note": note}

    # ------------------------------------------------------------------ runbooks

    def search_runbooks(self, query: str | None = None, top_k: int = 3) -> dict:
        catalogue = [{"id": r.id, "title": r.title, "applies_to": r.applies_to, "summary": r.summary}
                     for r in self.runbooks.runbooks]
        if not query or not query.strip():
            return {"runbooks": catalogue,
                    "hint": "pass a query (symptoms or a fault type) or a runbook id to read runbooks"}
        hits = self.runbooks.search(query, top_k)
        found = {r.id for r, _ in hits}
        return {
            "query": query,
            "results": [{"id": r.id, "title": r.title, "applies_to": r.applies_to,
                         "score": round(score, 2), "summary": r.summary, "body": r.body}
                        for r, score in hits],
            "other_runbooks": [{"id": c["id"], "title": c["title"]} for c in catalogue
                               if c["id"] not in found],
        }
