"""Span statistics, service-to-service calls and trace trees.

Online Boutique and Train Ticket record traces; Sock Shop and RE1 don't.
Latency is compared before and after the alert per operation, and per call from
one service to another. A call is measured twice: by the caller's span, which
includes the network and any queueing in front of the callee, and by the
callee's own span. When the caller-side latency grows while the callee's own
stays flat, the delay sits between the two: on the network, or in a queue at a
callee too busy to start the work.

Error rates come from span status codes (gRPC codes 1-16, HTTP codes >= 400).
Train Ticket records no status codes, so its error rates are unknown (null).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiops.mcp_support import InvalidRequest

MIN_SPANS = 5  # spans per phase before a latency comparison is reported
TREE_LINES = 60


def prepare(traces: pd.DataFrame, alert: int) -> pd.DataFrame:
    """The reader's span frame, reduced to what the tools use, with ``t`` in seconds
    relative to the alert and an ``error`` flag."""
    status = traces["status_code"]
    has_status = status.notna()
    error = has_status & (((status > 0) & (status < 100)) | (status >= 400)).fillna(False)
    return pd.DataFrame({
        "t": (traces["start_time_us"] - alert * 1_000_000) / 1e6,
        "start_us": traces["start_time_us"],
        "duration_us": traces["duration_us"],
        "trace_id": traces["trace_id"],
        "span_id": traces["span_id"],
        "parent_span_id": traces["parent_span_id"],
        "service": traces["service"].astype("category"),
        "operation": traces["operation"].fillna(traces["method"]).fillna("").astype("category"),
        "status_code": status,
        "has_status": has_status.to_numpy(dtype=bool),
        "error": error.to_numpy(dtype=bool),
    })


def calls(spans: pd.DataFrame) -> pd.DataFrame:
    """One row per span whose parent span belongs to another service: a call from
    ``caller`` to ``callee``, with both sides' durations."""
    parents = (spans[["span_id", "service", "duration_us"]].drop_duplicates("span_id")
               .rename(columns={"span_id": "parent_span_id", "service": "caller",
                                "duration_us": "caller_us"}))
    children = spans.loc[spans["parent_span_id"].notna(),
                         ["parent_span_id", "service", "t", "duration_us", "has_status", "error"]]
    pairs = children.merge(parents, on="parent_span_id", how="inner")
    pairs = pairs[pairs["caller"] != pairs["service"]]
    return pd.DataFrame({
        "t": pairs["t"].to_numpy(),
        "caller": pairs["caller"].to_numpy(),
        "callee": pairs["service"].to_numpy(),
        "caller_us": pairs["caller_us"].to_numpy(),
        "callee_us": pairs["duration_us"].to_numpy(),
        "has_status": pairs["has_status"].to_numpy(),
        "error": pairs["error"].to_numpy(),
    }).astype({"caller": "category", "callee": "category"})


def _phase_stats(frame: pd.DataFrame, keys: list[str], durations: dict[str, str]) -> pd.DataFrame:
    """Per key and phase (before/after the alert): count, error rate, p50/p90 of each
    duration column, in ms. Columns are (stat, phase) with phase in {"before", "after"}."""
    frame = frame.assign(phase=np.where(frame["t"] >= 0, "after", "before"))
    grouped = frame.groupby(keys + ["phase"], observed=True)
    stats = {"count": grouped.size(),
             "error_rate": grouped["error"].sum() / grouped["has_status"].sum().replace(0, np.nan)}
    for name, column in durations.items():
        stats[f"{name}p50_ms"] = grouped[column].quantile(0.5) / 1000
        stats[f"{name}p90_ms"] = grouped[column].quantile(0.9) / 1000
    table = pd.DataFrame(stats).unstack("phase")
    for stat in stats:  # make sure both phases exist as columns
        for phase in ("before", "after"):
            if (stat, phase) not in table.columns:
                table[(stat, phase)] = np.nan
    table[("count", "before")] = table[("count", "before")].fillna(0)
    table[("count", "after")] = table[("count", "after")].fillna(0)
    return table


def _side(table: pd.DataFrame, key, phase: str, count_name: str) -> dict | None:
    row = table.loc[key]
    if row[("count", phase)] == 0:
        return None
    side = {count_name: row[("count", phase)]}
    for stat in table.columns.get_level_values(0).unique():
        if stat != "count":
            side[stat] = row[(stat, phase)]
    return side


def _ratio(after, before) -> float:
    return float(after / before) if before and before > 0 and pd.notna(after) else np.nan


def _degradation(p90_ratios: list[float], before_err, after_err) -> float:
    worst = max([r for r in p90_ratios if pd.notna(r)], default=1.0)
    if pd.notna(before_err) and pd.notna(after_err):
        worst = max(worst, 1.0 + 20.0 * max(after_err - before_err, 0.0))
    return worst


def operation_report(spans: pd.DataFrame, *, limit: int) -> dict:
    """Operations ranked by how much they degraded after the alert."""
    table = _phase_stats(spans, ["service", "operation"], {"": "duration_us"})
    compared, new, stopped = [], [], []
    for key in table.index:
        nb, na = table.loc[key, ("count", "before")], table.loc[key, ("count", "after")]
        row = {"service": key[0], "operation": key[1],
               "before": _side(table, key, "before", "spans"),
               "after": _side(table, key, "after", "spans")}
        if nb >= MIN_SPANS and na >= MIN_SPANS:
            row["p90_ratio"] = _ratio(table.loc[key, ("p90_ms", "after")],
                                      table.loc[key, ("p90_ms", "before")])
            row["_rank"] = _degradation([row["p90_ratio"]], table.loc[key, ("error_rate", "before")],
                                        table.loc[key, ("error_rate", "after")])
            compared.append(row)
        elif na >= MIN_SPANS and nb == 0:
            new.append(row)
        elif nb >= MIN_SPANS and na == 0:
            stopped.append(row)
    compared.sort(key=lambda r: r["_rank"], reverse=True)
    top = compared[:limit]
    for row in top:
        row.pop("_rank")
    for row in top[:3]:
        row["example_trace"] = _example_trace(spans, row)
    new.sort(key=lambda r: r["after"]["spans"], reverse=True)
    stopped.sort(key=lambda r: r["before"]["spans"], reverse=True)
    return {"operations": top, "operations_compared": len(compared),
            "operations_new_after_alert": new[:5], "operations_stopped_after_alert": stopped[:5]}


def _example_trace(spans: pd.DataFrame, row: dict) -> str | None:
    """Trace id of an after-alert span of this operation whose duration is near the after-p90."""
    target = (row["after"] or {}).get("p90_ms")
    candidates = spans[(spans["service"] == row["service"]) & (spans["operation"] == row["operation"])
                       & (spans["t"] >= 0)]
    if candidates.empty or target is None or pd.isna(target):
        return None
    closest = (candidates["duration_us"] - target * 1000).abs().idxmin()
    return str(candidates.loc[closest, "trace_id"])


def call_report(pairs: pd.DataFrame, *, limit: int) -> dict:
    """Service-to-service calls ranked by how much they degraded after the alert."""
    if pairs.empty:
        return {"calls": [], "calls_new_after_alert": [], "calls_stopped_after_alert": []}
    table = _phase_stats(pairs, ["caller", "callee"], {"caller_": "caller_us", "callee_": "callee_us"})
    compared, new, stopped = [], [], []
    for key in table.index:
        nb, na = table.loc[key, ("count", "before")], table.loc[key, ("count", "after")]
        row = {"caller": key[0], "callee": key[1],
               "before": _side(table, key, "before", "calls"),
               "after": _side(table, key, "after", "calls")}
        if nb >= MIN_SPANS and na >= MIN_SPANS:
            ratios = {side: _ratio(table.loc[key, (f"{side}_p90_ms", "after")],
                                   table.loc[key, (f"{side}_p90_ms", "before")])
                      for side in ("caller", "callee")}
            row["caller_p90_ratio"], row["callee_p90_ratio"] = ratios["caller"], ratios["callee"]
            row["_rank"] = _degradation(list(ratios.values()), table.loc[key, ("error_rate", "before")],
                                        table.loc[key, ("error_rate", "after")])
            compared.append(row)
        elif na >= MIN_SPANS and nb == 0:
            new.append(row)
        elif nb >= MIN_SPANS and na == 0:
            stopped.append(row)
    compared.sort(key=lambda r: r["_rank"], reverse=True)
    for row in compared:
        row.pop("_rank")
    return {"calls": compared[:limit], "calls_new_after_alert": new[:5],
            "calls_stopped_after_alert": stopped[:5]}


def trace_tree(spans: pd.DataFrame, trace_id: str, *, max_lines: int = TREE_LINES) -> dict:
    """The spans of one trace as an indented call tree, one line per span:
    ``service: operation | +start offset ms | duration ms``."""
    trace = spans[spans["trace_id"] == trace_id].sort_values("start_us", kind="stable")
    if trace.empty:
        raise InvalidRequest(f"trace {trace_id!r} is not in this case")
    ids = set(trace["span_id"].dropna())
    children: dict[str, list] = {}
    roots = []
    for span in trace.itertuples(index=False):
        parent = span.parent_span_id
        if isinstance(parent, str) and parent in ids and parent != span.span_id:
            children.setdefault(parent, []).append(span)
        else:
            roots.append(span)
    t0 = int(trace["start_us"].min())
    lines: list[str] = []
    stack = [(span, 0) for span in reversed(roots)]
    seen = set()
    while stack and len(lines) < max_lines:
        span, depth = stack.pop()
        if span.span_id in seen:  # malformed traces can contain parent cycles
            continue
        seen.add(span.span_id)
        status = f" | status {span.status_code}" if span.error else ""
        lines.append(f"{'  ' * depth}{span.service}: {span.operation} | "
                     f"+{(span.start_us - t0) / 1000:.1f} ms | {span.duration_us / 1000:.2f} ms{status}")
        stack.extend((child, depth + 1) for child in reversed(children.get(span.span_id, [])))
    end = int((trace["start_us"] + trace["duration_us"]).max())
    return {
        "trace_id": trace_id,
        "start_s": float(trace["t"].min()),
        "duration_ms": (end - t0) / 1000,
        "spans": len(trace),
        "services": sorted(set(trace["service"].astype(str))),
        "tree": lines,
        "lines_omitted": len(trace) - len(lines),
    }
