"""Delimiter inference.

For every candidate delimiter we count its occurrences on each non-blank
sample row, then pick the candidate whose count is positive on every
row and whose count is most consistent (lowest variance). Ties are
broken by total occurrence count, so a column-heavy file picks the
delimiter that genuinely separates fields rather than one that happens
to appear once on every row.
"""

from __future__ import annotations

import statistics
from typing import Iterable

CANDIDATE_DELIMITERS: tuple[str, ...] = (",", "\t", ";", "|", ":", " ")


def _strip_quoted_runs(text: str) -> str:
    """Replace quoted regions with the empty string so that delimiter
    counting isn't confused by commas / tabs that appear inside quoted
    fields. Doubled-quote escapes are handled the same way csv does.
    """
    out: list[str] = []
    state = _StripState()
    i = 0
    while i < len(text):
        i = state.step(text, i, out)
    return "".join(out)


class _StripState:
    """Tiny state object that owns the in-quote bit and quote char."""

    def __init__(self) -> None:
        self.in_quote = False
        self.quote: str | None = None

    def step(self, text: str, i: int, out: list[str]) -> int:
        ch = text[i]
        if self.in_quote:
            return self._inside(text, i, ch)
        if ch in ('"', "'"):
            self.in_quote = True
            self.quote = ch
            return i + 1
        out.append(ch)
        return i + 1

    def _inside(self, text: str, i: int, ch: str) -> int:
        assert self.quote is not None
        if ch != self.quote:
            return i + 1
        if i + 1 < len(text) and text[i + 1] == self.quote:
            return i + 2
        self.in_quote = False
        self.quote = None
        return i + 1


def _candidate_score(rows: Iterable[str], delim: str) -> tuple[int, float, int] | None:
    counts = [r.count(delim) for r in rows if r]
    if not counts or any(c == 0 for c in counts):
        return None
    spread = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    return (len(counts), spread, sum(counts))


def infer_delimiter(text: str, *, sample_rows: list[str] | None = None) -> str | None:
    """Return the inferred single-character delimiter, or ``None`` if
    none of the candidate delimiters appear in *text*.
    """
    if not text:
        return None
    if sample_rows is None:
        cleaned = _strip_quoted_runs(text)
        sample_rows = [r for r in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n") if r]
    scored: list[tuple[float, int, str]] = []
    for delim in CANDIDATE_DELIMITERS:
        score = _candidate_score(sample_rows, delim)
        if score is None:
            continue
        rows, spread, total = score
        scored.append((spread - rows * 1000.0, -total, delim))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]
