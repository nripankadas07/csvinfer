"""Tests for quote-character inference."""

from __future__ import annotations

from csvinfer import CANDIDATE_QUOTES, infer_quote_char


def test_infers_double_quote(quoted_text: str) -> None:
    assert infer_quote_char(quoted_text) == '"'


def test_infers_single_quote(single_quote_text: str) -> None:
    assert infer_quote_char(single_quote_text) == "'"


def test_returns_none_when_no_quote_present(comma_text: str) -> None:
    assert infer_quote_char(comma_text) is None


def test_returns_none_for_empty() -> None:
    assert infer_quote_char("") is None


def test_returns_none_when_only_unmatched_quote() -> None:
    assert infer_quote_char('a,b"\nc,d') is None


def test_returns_none_for_apostrophe_in_word() -> None:
    # Quote characters embedded in words shouldn't fool inference;
    # don't is two characters, one is an apostrophe — odd count → None.
    assert infer_quote_char("a,b\nc,don't") is None


def test_prefers_more_pairs() -> None:
    text = (
        '"a","b"\n'
        "'c','d'\n"
    )
    # '"' has 4 occurrences, "'" has 4 → tie; either could win.
    # Just ensure it doesn't return None or a non-candidate.
    chosen = infer_quote_char(text)
    assert chosen in CANDIDATE_QUOTES


def test_real_quote_check_after_delimiter() -> None:
    # Quote sits after a comma — that's a legitimate field-start position.
    assert infer_quote_char('a,"b","c"') == '"'


def test_real_quote_check_at_start_of_string() -> None:
    assert infer_quote_char('"a","b"\n"c","d"') == '"'


def test_rejects_quote_inside_word() -> None:
    # Both occurrences of " sit between letters, never after a delim/start.
    text = "abc\"def\"ghi,abc\"def\"ghi"
    assert infer_quote_char(text) is None
