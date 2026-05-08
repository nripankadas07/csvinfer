"""Exception hierarchy used by :mod:`csvinfer`.

Every error raised by the library inherits from :class:`CsvInferError`,
which itself extends :class:`ValueError`. Catching ``ValueError`` is
the cheapest way to catch any csvinfer-specific failure; catching the
subclasses lets you handle each cause individually.
"""

from __future__ import annotations


class CsvInferError(ValueError):
    """Base class for every error raised by csvinfer."""


class EmptyInputError(CsvInferError):
    """Raised when inference is asked to operate on an empty sample."""


class AmbiguousDialectError(CsvInferError):
    """Raised when the heuristic cannot pick a single best dialect.

    Carries the candidates considered so callers can display them or
    fall back to a default.
    """

    def __init__(self, message: str, *, candidates: list[str]) -> None:
        super().__init__(message)
        self.candidates: list[str] = list(candidates)


class InvalidDialectError(CsvInferError):
    """Raised when constructing a :class:`Dialect` with invalid fields."""
