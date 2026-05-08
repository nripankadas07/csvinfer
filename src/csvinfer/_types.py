"""The :class:`Dialect` value type returned by inference.

A dialect is a small, frozen record describing how to interpret a CSV
text. It is deliberately narrower than :class:`csv.Dialect` from the
standard library — only the parameters that csvinfer can actually
infer are present, so the type cannot lie about what was discovered.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._errors import InvalidDialectError

_LEGAL_TERMINATORS = ("\n", "\r\n", "\r")


@dataclass(frozen=True)
class Dialect:
    """The inferred shape of a CSV text.

    Attributes:
        delimiter: The single character separating fields on a row.
        quote_char: The character used to quote fields, or ``None`` if
            no quoting was detected.
        has_header: ``True`` if the first row appears to be a header.
        line_terminator: The dominant line terminator (``"\\n"``,
            ``"\\r\\n"``, or ``"\\r"``).
    """

    delimiter: str
    quote_char: str | None
    has_header: bool
    line_terminator: str

    def __post_init__(self) -> None:
        if len(self.delimiter) != 1:
            raise InvalidDialectError(
                f"delimiter must be a single character, got {self.delimiter!r}"
            )
        if self.quote_char is not None and len(self.quote_char) != 1:
            raise InvalidDialectError(
                f"quote_char must be a single character or None, got {self.quote_char!r}"
            )
        if self.line_terminator not in _LEGAL_TERMINATORS:
            raise InvalidDialectError(
                "line_terminator must be one of \\n, \\r\\n, or \\r"
            )
