"""Architecture knowledge: the services of the three RCAEval systems and who calls whom.

One JSON file per system under ``data/topology/``, curated from each system's
published architecture and, for Train Ticket, from the calls observed in the
pre-alert traces of every RE2-TT and RE3-TT case (see each file's ``source``;
``tests/test_knowledge_real_data.py`` checks the files against the traces).
Each call records its evidence. The files hold static, case-independent
knowledge only: nothing in them depends on where a fault was injected.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from aiops.mcp_support import InvalidRequest

DATA_DIR = Path(__file__).parent / "data" / "topology"

EVIDENCE = {
    "traces": "observed in the pre-alert traces of the benchmark",
    "documented": "from the system's published architecture",
    "db-client-spans": "the service's database client spans, plus the naming convention",
    "naming": "inferred from the naming convention ts-X-service -> ts-X-mongo only",
    "gateway": "the service's spans start traces, so its requests come from the gateway",
}

_ABBREVIATIONS = {"ob": "online-boutique", "ss": "sock-shop", "tt": "train-ticket"}


@dataclass(frozen=True)
class Topology:
    system: str
    key: str
    datasets: tuple[str, ...]
    source: str
    services: dict[str, dict]
    calls: tuple[dict, ...]
    entry_points: tuple[str, ...]
    telemetry_names: dict[str, str]
    known_noise: tuple[dict, ...]
    notes: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> Topology:
        data = json.loads(Path(path).read_text())
        topology = cls(
            system=data["system"], key=data["key"], datasets=tuple(data["datasets"]),
            source=data["source"], services=dict(data["services"]),
            calls=tuple(data["calls"]), entry_points=tuple(data["entry_points"]),
            telemetry_names=dict(data.get("telemetry_names", {})),
            known_noise=tuple(data.get("known_noise", [])), notes=tuple(data.get("notes", [])),
        )
        unknown = {n for c in topology.calls for n in (c["from"], c["to"])} - set(topology.services)
        if unknown:
            raise ValueError(f"{path}: calls name services that are not listed: {sorted(unknown)}")
        return topology

    def callees(self, service: str) -> list[dict]:
        return [c for c in self.calls if c["from"] == service]

    def callers(self, service: str) -> list[dict]:
        return [c for c in self.calls if c["to"] == service]

    def reachable(self, service: str, depth: int, *, towards: str) -> list[dict]:
        """Services within ``depth`` calls, following callees (``towards="callees"``) or
        callers (``towards="callers"``), nearest first: ``[{"service", "hops"}]``."""
        step = self.callees if towards == "callees" else self.callers
        end = "to" if towards == "callees" else "from"
        seen, found = {service}, []
        queue = deque([(service, 0)])
        while queue:
            current, hops = queue.popleft()
            if hops == depth:
                continue
            for call in step(current):
                nxt = call[end]
                if nxt not in seen:
                    seen.add(nxt)
                    found.append({"service": nxt, "hops": hops + 1})
                    queue.append((nxt, hops + 1))
        return found


def load_topologies(directory: Path = DATA_DIR) -> dict[str, Topology]:
    topologies = [Topology.load(path) for path in sorted(Path(directory).glob("*.json"))]
    return {t.key: t for t in topologies}


def resolve_system(name: str, topologies: dict[str, Topology]) -> Topology:
    """Accept 'Online Boutique', 'online-boutique', 'OB', a dataset ('RE2-OB') or a case id."""
    text = str(name).strip().lower()
    match = re.match(r"^re[123][-_]?(ob|ss|tt)\b", text)
    key = _ABBREVIATIONS[match[1]] if match else _ABBREVIATIONS.get(text, re.sub(r"[\s_]+", "-", text))
    if key in topologies:
        return topologies[key]
    for topology in topologies.values():
        if topology.system.lower() == text:
            return topology
    systems = ", ".join(f"{t.system} ({k})" for k, t in topologies.items())
    raise InvalidRequest(f"unknown system {name!r}; known systems: {systems}. A dataset such as "
                         "RE2-OB or a case id also works")
