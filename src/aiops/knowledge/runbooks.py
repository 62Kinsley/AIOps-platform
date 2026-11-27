"""Runbooks: how each kind of fault shows up in the telemetry, searched with BM25.

Each runbook is a Markdown file under ``data/runbooks/`` with a small header::

    ---
    id: cpu-saturation
    title: CPU saturation (CPU hog, CPU stress)
    applies_to: cpu
    keywords: cpu, hog, stress, ...
    summary: One sentence.
    ---

The search ranks runbooks with BM25 over their title (weighted x3), keywords and
summary (x2) and body. Queries are symptoms or fault types in English, e.g.
"latency up but cpu normal" or "connection refused".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "runbooks"

K1, B = 1.5, 0.75
EXACT_ID_SCORE = 1000.0  # a query that is exactly a runbook id
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in into is it its of on or that the their "
    "then there these this to was were when where which while with without".split())


@dataclass(frozen=True)
class Runbook:
    id: str
    title: str
    applies_to: str
    keywords: tuple[str, ...]
    summary: str
    body: str

    @classmethod
    def parse(cls, text: str, source: str = "<runbook>") -> Runbook:
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"{source}: missing '---' header")
        fields = {}
        for line in match[1].splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
        missing = {"id", "title", "applies_to", "keywords", "summary"} - set(fields)
        if missing:
            raise ValueError(f"{source}: header lacks {sorted(missing)}")
        return cls(id=fields["id"], title=fields["title"], applies_to=fields["applies_to"],
                   keywords=tuple(k.strip() for k in fields["keywords"].split(",") if k.strip()),
                   summary=fields["summary"], body=match[2].strip())


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]  # crude plural folding: errors -> error, timeouts -> timeout
        tokens.append(token)
    return tokens


class RunbookIndex:
    def __init__(self, runbooks: list[Runbook]):
        ids = [r.id for r in runbooks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate runbook ids: {sorted(i for i in ids if ids.count(i) > 1)}")
        self.runbooks = list(runbooks)
        self._docs = [Counter(tokenize(" ".join([r.title] * 3 + [" ".join(r.keywords), r.summary] * 2
                                                + [r.id.replace("-", " "), r.body])))
                      for r in self.runbooks]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._avg = sum(self._lengths) / max(len(self._lengths), 1)
        df = Counter(token for doc in self._docs for token in doc)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def load(cls, directory: Path = DATA_DIR) -> RunbookIndex:
        return cls([Runbook.parse(p.read_text(), str(p)) for p in sorted(Path(directory).glob("*.md"))])

    def get(self, runbook_id: str) -> Runbook | None:
        return next((r for r in self.runbooks if r.id == runbook_id), None)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Runbook, float]]:
        """Runbooks with a positive BM25 score, best first. An exact runbook id comes first."""
        terms = set(tokenize(query))
        scored = []
        for runbook, doc, length in zip(self.runbooks, self._docs, self._lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self._avg))
                    score += self._idf[term] * norm
            if runbook.id == query.strip().lower():
                score = EXACT_ID_SCORE
            if score > 0:
                scored.append((runbook, score))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:top_k]
