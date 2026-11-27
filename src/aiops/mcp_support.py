"""Plumbing shared by the platform's MCP servers: errors, JSON output, command line.

Tool logic lives in plain classes (``MonitoringService``, ``KnowledgeBase``) whose
methods return dicts, so it can be tested and reused without MCP. Each server
wraps those methods in tools that serialise the dict with :func:`to_json` and
turn :class:`InvalidRequest` into an MCP tool error the agent can read and fix.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

# Every tool only reads data the server already has.
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)

SIGNIFICANT_DIGITS = 4


class InvalidRequest(ValueError):
    """An argument the caller can fix: unknown case or service, empty window, bad trace id."""


def to_json(result: Any) -> str:
    """Compact JSON for an LLM reader: floats to 4 significant digits, NaN as null."""
    return json.dumps(_clean(result), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return None
        rounded = float(f"{value:.{SIGNIFICANT_DIGITS}g}")
        return int(rounded) if rounded.is_integer() and abs(rounded) < 2**53 else rounded
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def call(method: Callable[..., dict], **kwargs: Any) -> str:
    """Run one tool method and serialise its result; input errors become tool errors."""
    try:
        return to_json(method(**kwargs))
    except InvalidRequest as error:
        raise ToolError(str(error)) from None


def register(server: MCPServer, fn: Callable[..., str]) -> None:
    """Add ``fn`` as a read-only tool described by its (dedented) docstring."""
    server.add_tool(fn, description=inspect.cleandoc(fn.__doc__ or ""), annotations=READ_ONLY,
                    structured_output=False)


def close_matches(name: str, known: Sequence[str], limit: int = 3) -> str:
    """Error-message suffix: '; did you mean a, b?', or the full list when nothing is close."""
    close = difflib.get_close_matches(name, list(known), n=limit, cutoff=0.5)
    if close:
        return f"; did you mean {', '.join(close)}?"
    return f"; known: {', '.join(known)}" if known else ""


def package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("aiops-platform")
    except PackageNotFoundError:  # running from a source tree that was never installed
        return "dev"


def add_transport_args(parser: argparse.ArgumentParser, default_port: int) -> None:
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio",
                        help="stdio (default: the MCP client launches the server) or streamable-http")
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http bind address")
    parser.add_argument("--port", type=int, default=default_port, help="streamable-http port")


def serve(server: MCPServer, args: argparse.Namespace) -> None:
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)
