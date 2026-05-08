"""Public top-level API for csvinfer."""

from __future__ import annotations

from ._delimiter import infer_delimiter as _infer_delim
from ._errors import EmptyInputError
from ._header import has_header as _row_has_header
from ._parser import detect_terminator, parse_rows, split_rows
from ._quote import infer_quote_char as _infer_quote
from ._types import Dialect

_BOM = "﻿"


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith(_BOM) else text


def infer_delimiter(text: str) -> str | None:
    """Return the inferred delimiter or ``None`` if undetectable."""
    return _infer_delim(_strip_bom(text))


def infer_quote_char(text: str) -> str | None:
    """Return the inferred quote character or ``None`` if no quoting
    is present in the sample.
    """
    return _infer_quote(_strip_bom(text))


def has_header(text: str, *, dialect: Dialect | None = None) -> bool:
    """Return ``True`` if the first row of *text* looks like a header.

    *dialect* is optional — if omitted, the dialect is inferred from
    the same text first.
    """
    cleaned = _strip_bom(text)
    if not cleaned:
        raise EmptyInputError("cannot detect header on empty input")
    chosen = dialect if dialect is not None else infer(cleaned)
    rows = parse_rows(cleaned, chosen)
    return _row_has_header(rows)


def infer(text: str, *, sample_lines: int = 50) -> Dialect:
    """Infer a complete :class:`Dialect` from *text*.

    Args:
        text: the raw CSV text. May be prefixed with a BOM (stripped
            automatically).
        sample_lines: hard cap on the number of lines used for
            inference. Larger samples are accurate but slower.

    Raises:
        EmptyInputError: if *text* is empty after BOM removal.
        ValueError: if *sample_lines* is not a positive integer.
    """
    if not isinstance(sample_lines, int) or isinstance(sample_lines, bool) or sample_lines < 1:
        raise ValueError("sample_lines must be a positive integer")
    cleaned = _strip_bom(text)
    if not cleaned:
        raise EmptyInputError("cannot infer dialect from empty input")
    terminator = detect_terminator(cleaned)
    sample = "\n".join(split_rows(cleaned, terminator)[:sample_lines])
    delim = _infer_delim(sample) or ","
    quote = _infer_quote(sample)
    pre = Dialect(delimiter=delim, quote_char=quote, has_header=False, line_terminator=terminator)
    rows = parse_rows(cleaned, pre)
    return Dialect(
        delimiter=delim,
        quote_char=quote,
        has_header=_row_has_header(rows),
        line_terminator=terminator,
    )


def read_rows(text: str, *, dialect: Dialect | None = None) -> list[list[str]]:
    """Parse *text* into rows using *dialect* or an inferred dialect.

    Returns the full list of rows, including any header row. Use
    :attr:`Dialect.has_header` to know whether to skip the first row.
    """
    cleaned = _strip_bom(text)
    if not cleaned:
        return []
    chosen = dialect if dialect is not None else infer(cleaned)
    return parse_rows(cleaned, chosen)
