"""Tests for the Dialect frozen dataclass and its validation."""

from __future__ import annotations

import pytest

from csvinfer import Dialect, InvalidDialectError


def test_construct_minimal() -> None:
    d = Dialect(delimiter=",", quote_char='"', has_header=True, line_terminator="\n")
    assert d.delimiter == ","
    assert d.quote_char == '"'
    assert d.has_header is True
    assert d.line_terminator == "\n"


def test_quote_char_may_be_none() -> None:
    d = Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\n")
    assert d.quote_char is None


def test_rejects_multi_char_delimiter() -> None:
    with pytest.raises(InvalidDialectError):
        Dialect(delimiter=",,", quote_char=None, has_header=False, line_terminator="\n")


def test_rejects_empty_delimiter() -> None:
    with pytest.raises(InvalidDialectError):
        Dialect(delimiter="", quote_char=None, has_header=False, line_terminator="\n")


def test_rejects_multi_char_quote() -> None:
    with pytest.raises(InvalidDialectError):
        Dialect(delimiter=",", quote_char='""', has_header=False, line_terminator="\n")


def test_rejects_invalid_terminator() -> None:
    with pytest.raises(InvalidDialectError):
        Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\t")


def test_immutable() -> None:
    d = Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\n")
    with pytest.raises(Exception):
        d.delimiter = ";"  # type: ignore[misc]


def test_equality_and_hash() -> None:
    a = Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\n")
    b = Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\n")
    assert a == b
    assert hash(a) == hash(b)
