"""Header detection.

The heuristic compares the *type signature* of the first row against
the type signature of the rows that follow. A row's type signature is
the sequence of inferred types for each cell, taken from
``{int, float, bool, date, blank, string}``. If the first row is
all-string while the body rows have any non-string, non-blank types,
we consider it a header.
"""

from __future__ import annotations

import re
from typing import Sequence

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+|\d+\.\d+[eE][+-]?\d+|\d+[eE][+-]?\d+)$")
_BOOL_VALUES = {"true", "false", "yes", "no"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?$")


def cell_type(value: str) -> str:
    """Classify *value* into ``"blank" | "int" | "float" | "bool" | "date" | "string"``."""
    stripped = value.strip()
    if not stripped:
        return "blank"
    if _INT_RE.match(stripped):
        return "int"
    if _FLOAT_RE.match(stripped):
        return "float"
    if stripped.lower() in _BOOL_VALUES:
        return "bool"
    if _DATE_RE.match(stripped):
        return "date"
    return "string"


def has_header(rows: Sequence[Sequence[str]]) -> bool:
    """Return ``True`` if the first row of *rows* looks like a header.

    Returns ``False`` if there are fewer than two rows.
    """
    if len(rows) < 2:
        return False
    first_types = [cell_type(c) for c in rows[0]]
    if not first_types or any(t != "string" for t in first_types):
        # Header rows are typically all strings; if even one cell in the
        # first row parses as int/float/bool/date, treat it as data.
        return False
    body_types = [[cell_type(c) for c in row] for row in rows[1:]]
    # body_types is non-empty here because we already guarded len(rows) < 2 above.
    # at least one body row must contain a non-string, non-blank cell
    return any(t not in ("string", "blank") for row in body_types for t in row)
