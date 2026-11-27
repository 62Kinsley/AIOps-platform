"""Phase 2 Knowledge Server: system architecture and fault runbooks through MCP.

The tools are methods of :class:`KnowledgeBase`, usable without MCP::

    from aiops.knowledge import KnowledgeBase
    kb = KnowledgeBase()
    kb.service_topology("Online Boutique", "cartservice")
    kb.search_runbooks("latency up, cpu normal")

``aiops-knowledge-server`` serves them over MCP (see :mod:`aiops.knowledge.server`).
"""

from aiops.knowledge.service import KnowledgeBase

__all__ = ["KnowledgeBase"]
