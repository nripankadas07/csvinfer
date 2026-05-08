"""Tests for header detection and the cell_type classifier."""

from __future__ import annotations

import pytest

from csvinfer import cell_type, has_header
from csvinfer._header import has_header as raw_has_header


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", "blank"),
        ("   ", "blank"),
        ("0", "int"),
        ("-42", "int"),
        ("3.14", "float"),
        (".5", "float"),
        ("1.5e3", "float"),
        ("1e3", "float"),
        ("true", "bool"),
        ("FALSE", "bool"),
        ("YES", "bool"),
        ("no", "bool"),
        ("2026-05-08", "date"),
        ("2026-05-08T12:30", "date"),
        ("2026-05-08 12:30:45", "date"),
        ("Ada", "string"),
        ("hello world", "string"),
    ],
)
def test_cell_type_classification(value: str, expected: str) -> None:
    assert cell_type(value) == expected


def test_has_header_true_for_string_then_typed_body(comma_text: str) -> None:
    assert has_header(comma_text) is True


def test_has_header_false_for_pure_string_body() -> None:
    text = "name,city\nAda,London\nGrace,Boston\n"
    assert has_header(text) is False


def test_has_header_false_for_typed_first_row() -> None:
    text = "1,2,3\n4,5,6\n"
    assert has_header(text) is False


def test_has_header_false_for_single_row() -> None:
    text = "name,age\n"
    assert has_header(text) is False


def test_raw_has_header_handles_empty_input() -> None:
    assert raw_has_header([]) is False


def test_raw_has_header_handles_first_row_empty() -> None:
    # If the first row has zero columns there's nothing to compare.
    assert raw_has_header([[], ["a", "b"]]) is False


def test_raw_has_header_skips_when_body_all_strings() -> None:
    rows = [["a", "b"], ["c", "d"], ["e", "f"]]
    assert raw_has_header(rows) is False


# (the 'body has no rows but len(rows)>=2' branch was removed as
# unreachable: len(rows)>=2 already implies rows[1:] is non-empty.)

