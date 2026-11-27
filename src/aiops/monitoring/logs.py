"""Log templates and before/after log summaries.

A template is a log message with its variable parts masked: timestamps, ids,
numbers, IPs, e-mail addresses, and the values of identifying fields such as
``username=`` or ``"firstName":``. The 270k lines of a Train Ticket case
collapse into a few hundred templates this way. Comparing how often each
template occurs before and after the alert then shows what changed: new error
messages, retries, restarts, or a service that went quiet. Raw counts of
"error" lines are a poor signal by comparison, because some services log errors
all the time (Sock Shop's queue-master fails to reach the Docker socket every
few seconds, in every case).

The masking runs as RE2 regexes in pyarrow: about 2-3 s for the largest case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

TEMPLATE_CHARS = 160
SAMPLE_CHARS = 300
_MASK_INPUT_CHARS = 600  # templates keep only the first 160 chars; don't mask whole JSON bodies

_ID_KEYS = (r"user(?:name)?|first_?name|last_?name|name|password|e?mail|_?id|customer_?id|"
            r"item_?id|order_?id|session|token|card|long_?num|ccv|expires|street|city|postcode|"
            r"country|number")

# (RE2 pattern, replacement), applied in order.
MASKS = [
    # Spring Boot: "2024-02-24 05:08:30.701  INFO 1 --- [io-18898-exec-7] a.b.SomeClass  : msg"
    # -> "INFO a.b.SomeClass: msg" (the pid and the truncated thread name vary per line)
    (r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s+([A-Z]+)\s+(?:\[[^\]]*\]\s+)?\d+\s+---\s+"
     r"\[[^\]]*\]\s+(\S+)\s*:\s*", r"\1 \2: "),
    (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<TS>"),
    (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "<UUID>"),
    (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "<EMAIL>"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "<IP>"),
    # identifying JSON fields, also in escaped JSON: "firstName":"x" or \"firstName\":\"x\"
    (r'(?i)(\\?"(?:' + _ID_KEYS + r')\\?")(\s*:\s*)\\?"[^"\\]*\\?"', r'\1\2"<*>"'),
    # identifying key=value fields
    (r"(?i)\b(" + _ID_KEYS + r")=[^\s,;)\]}]+", r"\1=<*>"),
    (r"\b[0-9a-fA-F]{12,}\b", "<HEX>"),
    (r"[\w.:/-]*\d[\w.:/-]*", "<*>"),  # any remaining token containing a digit
]

ERROR_PATTERN = (r"(?i)\b(?:error|errors|exception|fail|failed|failure|fatal|panic|timeout|"
                 r"timed out|refused|unavailable|unreachable|denied|traceback|critical|severe|"
                 r"warn|warning)\b|\berr=[^n\s]")


def templates(messages: pd.Series) -> pd.Series:
    """Masked template of each message (same index), truncated to TEMPLATE_CHARS."""
    arr = pc.utf8_slice_codeunits(_arrow(messages), 0, _MASK_INPUT_CHARS)
    for pattern, replacement in MASKS:
        arr = pc.replace_substring_regex(arr, pattern=pattern, replacement=replacement)
    arr = pc.utf8_trim_whitespace(pc.utf8_slice_codeunits(arr, 0, TEMPLATE_CHARS))
    return pd.Series(pd.Categorical(arr.to_pandas()), index=messages.index)


def error_like(messages: pd.Series) -> pd.Series:
    """True for lines that mention an error, exception, failure, timeout, refusal or warning."""
    return _bool_series(pc.match_substring_regex(_arrow(messages), pattern=ERROR_PATTERN), messages)


def matches(messages: pd.Series, query: str) -> pd.Series:
    """Case-insensitive regex search (RE2 syntax); a pattern RE2 rejects is searched literally."""
    arr = _arrow(messages)
    try:
        hits = pc.match_substring_regex(arr, pattern=query, ignore_case=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        hits = pc.match_substring(arr, pattern=query, ignore_case=True)
    return _bool_series(hits, messages)


def _arrow(messages: pd.Series) -> pa.Array:
    return pa.array(messages.astype("string").fillna(""), type=pa.large_string())


def _bool_series(hits: pa.Array, like: pd.Series) -> pd.Series:
    return pd.Series(hits.to_numpy(zero_copy_only=False), index=like.index, dtype=bool)


def change_score(before: pd.Series | np.ndarray, after, before_len: float, after_len: float):
    """log2 of (after + 1) / (expected + 1), where expected scales ``before`` to the length of
    the after window: +1 is about twice as frequent, -1 about half, 0 unchanged."""
    if before_len <= 0:
        return np.log2(np.asarray(after, dtype=float) + 1)
    expected = np.asarray(before, dtype=float) * (after_len / before_len)
    return np.log2((np.asarray(after, dtype=float) + 1) / (expected + 1))


def summarize(logs: pd.DataFrame, lo: float, hi: float, *, limit: int = 15) -> dict:
    """Before/after comparison of ``logs`` over [lo, hi].

    ``logs`` has columns ``t`` (seconds relative to the alert), ``service``,
    ``message``, ``template`` and ``error`` (error-like), sorted by ``t``.
    """
    sel = logs[(logs["t"] >= lo) & (logs["t"] <= hi)]
    before_len, after_len = max(0.0, min(hi, 0.0) - lo), max(0.0, hi - max(lo, 0.0))
    after = sel["t"] >= 0
    out: dict = {"lines": {"before_alert": int((~after).sum()), "after_alert": int(after.sum())},
                 "services": [], "templates_increased": [], "templates_decreased": []}
    if sel.empty:
        return out

    frame = pd.DataFrame({
        "service": sel["service"], "template": sel["template"], "message": sel["message"],
        "t": sel["t"], "after": after, "before_n": (~after).astype(int),
        "after_n": after.astype(int), "error": sel["error"],
        "error_before": sel["error"] & ~after, "error_after": sel["error"] & after,
    })

    keys = ["service", "template"]
    table = frame.groupby(keys, observed=True).agg(
        before=("before_n", "sum"), after=("after_n", "sum"), error_like=("error", "any"))
    table["first_after_s"] = frame[frame["after"]].groupby(keys, observed=True)["t"].min()
    # one example line per template: its first line after the alert, else its first line
    first = frame.sort_values("after", ascending=False, kind="stable").drop_duplicates(keys)
    table["sample"] = first.set_index(keys)["message"].str.slice(0, SAMPLE_CHARS)
    table["score"] = change_score(table["before"], table["after"], before_len, after_len)
    table = table.reset_index()

    increased = table[table["score"] >= 1]
    increased = (increased.assign(rank=increased["score"] + increased["error_like"])
                 .sort_values(["rank", "after"], ascending=False).head(limit))
    decreased = table[(table["score"] <= -1) & (table["before"] >= 10)]
    decreased = decreased.sort_values("score").head(max(3, limit // 3))

    per = frame.groupby("service", observed=True).agg(
        before=("before_n", "sum"), after=("after_n", "sum"),
        error_like_before=("error_before", "sum"), error_like_after=("error_after", "sum"))
    per["new_templates_after"] = (table[(table["before"] == 0) & (table["after"] > 0)]
                                  .groupby("service", observed=True).size())
    per["new_templates_after"] = per["new_templates_after"].fillna(0).astype(int)
    per["score"] = change_score(per["before"], per["after"], before_len, after_len)
    per["interest"] = (per["new_templates_after"]
                       + (per["error_like_after"] - per["error_like_before"]).clip(lower=0)
                       + per["score"].abs())
    per = per.sort_values("interest", ascending=False).drop(columns="interest").reset_index()

    out["services"] = _records(per)
    out["templates_increased"] = _records(increased[
        ["service", "template", "before", "after", "score", "error_like", "first_after_s", "sample"]])
    out["templates_decreased"] = _records(decreased[
        ["service", "template", "before", "after", "score", "sample"]])
    return out


def _records(frame: pd.DataFrame) -> list[dict]:
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    return [{k: v for k, v in row.items()} for row in rows]
