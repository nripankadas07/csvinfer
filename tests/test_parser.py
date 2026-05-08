"""Tests for the small RFC-4180-style parser."""

from __future__ import annotations

from csvinfer import Dialect
from csvinfer._parser import detect_terminator, iter_fields, parse_rows, split_rows


def test_split_rows_drops_trailing_empty_only() -> None:
    assert split_rows("a\nb\n", "\n") == ["a", "b"]


def test_split_rows_preserves_interior_blanks() -> None:
    assert split_rows("a\n\nb\n", "\n") == ["a", "", "b"]


def test_split_rows_no_trailing_terminator() -> None:
    assert split_rows("a\nb", "\n") == ["a", "b"]


def test_split_rows_empty_input() -> None:
    assert split_rows("", "\n") == []


def test_split_rows_crlf() -> None:
    assert split_rows("a\r\nb\r\n", "\r\n") == ["a", "b"]


def test_detect_terminator_lf() -> None:
    assert detect_terminator("a\nb\n") == "\n"


def test_detect_terminator_crlf() -> None:
    assert detect_terminator("a\r\nb\r\n") == "\r\n"


def test_detect_terminator_cr_only() -> None:
    assert detect_terminator("a\rb\rc\r") == "\r"


def test_detect_terminator_single_line() -> None:
    assert detect_terminator("a,b,c") == "\n"


def test_iter_fields_no_quotes() -> None:
    assert list(iter_fields("a,b,c", delimiter=",", quote_char=None)) == ["a", "b", "c"]


def test_iter_fields_quote_char_not_in_row_short_circuits() -> None:
    assert list(iter_fields("a,b,c", delimiter=",", quote_char='"')) == ["a", "b", "c"]


def test_iter_fields_quoted_simple() -> None:
    out = list(iter_fields('"a","b","c"', delimiter=",", quote_char='"'))
    assert out == ["a", "b", "c"]


def test_iter_fields_quoted_with_embedded_delim() -> None:
    out = list(iter_fields('"a,b","c"', delimiter=",", quote_char='"'))
    assert out == ["a,b", "c"]


def test_iter_fields_doubled_quote_escape() -> None:
    out = list(iter_fields('"a""b","c"', delimiter=",", quote_char='"'))
    assert out == ['a"b', "c"]


def test_iter_fields_mixed_quoted_and_bare() -> None:
    out = list(iter_fields('a,"b,c",d', delimiter=",", quote_char='"'))
    assert out == ["a", "b,c", "d"]


def test_iter_fields_preserves_empty_field() -> None:
    out = list(iter_fields("a,,b", delimiter=",", quote_char=None))
    assert out == ["a", "", "b"]


def test_parse_rows_full_pipeline() -> None:
    d = Dialect(delimiter=",", quote_char='"', has_header=False, line_terminator="\n")
    rows = parse_rows('a,b\n"c, c","d"\n', d)
    assert rows == [["a", "b"], ["c, c", "d"]]


def test_parse_rows_handles_empty_text() -> None:
    d = Dialect(delimiter=",", quote_char=None, has_header=False, line_terminator="\n")
    assert parse_rows("", d) == []
