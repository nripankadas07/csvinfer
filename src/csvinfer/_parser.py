"""A small RFC-4180-style row parser used by the inference helpers.

The standard library's :func:`csv.reader` would do, but it locks the
quote character at construction time and treats unmatched quotes
fairly leniently. csvinfer needs to be able to test multiple
quote-character candidates against the same input, so a tiny
hand-rolled state machine is included here.
"""

from __future__ import annotations

from typing import Iterator

from ._types import Dialect


def split_rows(text: str, terminator: str) -> list[str]:
    """Split *text* on the given line terminator, dropping a trailing
    empty entry if present (the result of a file ending in a newline).
    Other empty lines (interior blank lines) are preserved.
    """
    if not text:
        return []
    rows = text.split(terminator)
    if rows and rows[-1] == "":
        rows.pop()
    return rows


def detect_terminator(text: str) -> str:
    """Pick the dominant line terminator in *text*.

    Counts CRLF first because LF would otherwise win even on CRLF
    inputs. Falls back to ``"\\n"`` when no terminator is present (a
    single-line sample is still a CSV).
    """
    crlf = text.count("\r\n")
    if crlf > 0:
        return "\r\n"
    if text.count("\r") > text.count("\n"):
        return "\r"
    return "\n"


def iter_fields(row: str, *, delimiter: str, quote_char: str | None) -> Iterator[str]:
    """Yield the fields of *row* respecting the supplied quote char.

    The parser handles doubled quotes (``""``) as literal escapes and
    treats unmatched quotes by falling back to a delimiter-only split
    of the unquoted remainder.
    """
    if quote_char is None or quote_char not in row:
        yield from row.split(delimiter)
        return
    yield from _stateful_iter(row, delimiter, quote_char)


def _stateful_iter(row: str, delimiter: str, quote_char: str) -> Iterator[str]:
    """Quote-aware split, kept under the 30-line / 3-nesting gate by
    delegating per-position decisions to :func:`_advance`.
    """
    field: list[str] = []
    in_quote = False
    i = 0
    while i < len(row):
        in_quote, i, emit, end_field = _advance(row, i, in_quote, delimiter, quote_char)
        if emit is not None:
            field.append(emit)
        if end_field:
            yield "".join(field)
            field = []
    yield "".join(field)


def _advance(
    row: str, i: int, in_quote: bool, delimiter: str, quote_char: str,
) -> tuple[bool, int, str | None, bool]:
    """Make one transition. Returns
    ``(new_in_quote, new_index, char_to_emit_or_None, end_field)``.
    """
    ch = row[i]
    if in_quote:
        return _step_inside_quote(row, i, quote_char)
    if ch == quote_char:
        return True, i + 1, None, False
    if ch == delimiter:
        return False, i + 1, None, True
    return False, i + 1, ch, False


def _step_inside_quote(
    row: str, i: int, quote_char: str,
) -> tuple[bool, int, str | None, bool]:
    """Advance one position inside a quoted field."""
    ch = row[i]
    if ch != quote_char:
        return True, i + 1, ch, False
    if i + 1 < len(row) and row[i + 1] == quote_char:
        return True, i + 2, quote_char, False
    return False, i + 1, None, False


def parse_rows(text: str, dialect: Dialect) -> list[list[str]]:
    """Parse *text* into rows of fields using *dialect*."""
    raw = split_rows(text, dialect.line_terminator)
    return [list(iter_fields(r, delimiter=dialect.delimiter, quote_char=dialect.quote_char))
            for r in raw]
