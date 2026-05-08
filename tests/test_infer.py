"""End-to-end tests for the top-level `infer` and `read_rows` calls."""

from __future__ import annotations

import pytest

from csvinfer import (
    Dialect, EmptyInputError, has_header, infer, read_rows,
)


def test_infer_comma_with_header(comma_text: str) -> None:
    d = infer(comma_text)
    assert d.delimiter == ","
    assert d.quote_char is None
    assert d.has_header is True
    assert d.line_terminator == "\n"


def test_infer_tab_with_header(tab_text: str) -> None:
    d = infer(tab_text)
    assert d.delimiter == "\t"
    assert d.has_header is True


def test_infer_quoted_double_quote(quoted_text: str) -> None:
    d = infer(quoted_text)
    assert d.delimiter == ","
    assert d.quote_char == '"'
    assert d.has_header is True


def test_infer_crlf_terminator(crlf_text: str) -> None:
    d = infer(crlf_text)
    assert d.line_terminator == "\r\n"


def test_infer_cr_terminator(cr_text: str) -> None:
    d = infer(cr_text)
    assert d.line_terminator == "\r"


def test_infer_strips_bom(bom_text: str) -> None:
    d = infer(bom_text)
    assert d.delimiter == ","
    assert d.has_header is True


def test_infer_rejects_empty_input() -> None:
    with pytest.raises(EmptyInputError):
        infer("")


def test_infer_rejects_bom_only() -> None:
    with pytest.raises(EmptyInputError):
        infer("﻿")


def test_infer_default_delimiter_when_none_present() -> None:
    text = "alpha\nbeta\ngamma\n"
    d = infer(text)
    assert d.delimiter == ","


def test_infer_rejects_non_int_sample_lines() -> None:
    with pytest.raises(ValueError):
        infer("a,b\n1,2\n", sample_lines="50")  # type: ignore[arg-type]


def test_infer_rejects_zero_sample_lines() -> None:
    with pytest.raises(ValueError):
        infer("a,b\n1,2\n", sample_lines=0)


def test_infer_rejects_bool_sample_lines() -> None:
    with pytest.raises(ValueError):
        infer("a,b\n1,2\n", sample_lines=True)  # type: ignore[arg-type]


def test_read_rows_with_default_dialect(comma_text: str) -> None:
    rows = read_rows(comma_text)
    assert rows[0] == ["name", "age", "active"]
    assert rows[1] == ["Ada", "30", "true"]


def test_read_rows_with_explicit_dialect() -> None:
    d = Dialect(delimiter=";", quote_char=None, has_header=False, line_terminator="\n")
    rows = read_rows("a;b;c\n1;2;3\n", dialect=d)
    assert rows == [["a", "b", "c"], ["1", "2", "3"]]


def test_read_rows_strips_bom(bom_text: str) -> None:
    rows = read_rows(bom_text)
    assert rows[0][0] == "name"


def test_read_rows_empty_returns_empty() -> None:
    assert read_rows("") == []


def test_has_header_with_explicit_dialect(comma_text: str) -> None:
    d = Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\n")
    assert has_header(comma_text, dialect=d) is True


def test_has_header_with_inferred_dialect(comma_text: str) -> None:
    assert has_header(comma_text) is True


def test_has_header_rejects_empty() -> None:
    with pytest.raises(EmptyInputError):
        has_header("")
