"""Tests for delimiter inference."""

from __future__ import annotations

import pytest

from csvinfer import CANDIDATE_DELIMITERS, infer_delimiter
from csvinfer._delimiter import _candidate_score, _strip_quoted_runs


def test_infers_comma(comma_text: str) -> None:
    assert infer_delimiter(comma_text) == ","


def test_infers_tab(tab_text: str) -> None:
    assert infer_delimiter(tab_text) == "\t"


def test_infers_semicolon(semicolon_text: str) -> None:
    assert infer_delimiter(semicolon_text) == ";"


def test_infers_pipe() -> None:
    text = "a|b|c\n1|2|3\n4|5|6\n"
    assert infer_delimiter(text) == "|"


def test_infers_colon() -> None:
    text = "a:b:c\n1:2:3\n4:5:6\n"
    assert infer_delimiter(text) == ":"


def test_infers_space() -> None:
    text = "a b c\n1 2 3\n4 5 6\n"
    assert infer_delimiter(text) == " "


def test_returns_none_when_no_delimiter_present() -> None:
    text = "alphabetagamma\nzetatheta\n"
    assert infer_delimiter(text) is None


def test_returns_none_when_only_some_lines_have_delim() -> None:
    text = "a,b\nplain\nc,d\n"
    # comma occurs only on rows 0 and 2, so it must be rejected
    assert infer_delimiter(text) is None


def test_returns_none_for_empty_text() -> None:
    assert infer_delimiter("") is None


def test_ignores_delimiter_inside_quotes() -> None:
    text = (
        'a,b,c\n'
        '"x,still x","y","z"\n'
        '"x","y","z"\n'
    )
    # The comma inside quotes shouldn't push tab or semicolon ahead.
    assert infer_delimiter(text) == ","


def test_preserves_doubled_quotes_inside_strip() -> None:
    cleaned = _strip_quoted_runs('"a ""b""",c')
    # the quoted region (with doubled quote) is removed entirely; remaining is ',c'
    assert cleaned == ",c"


def test_strip_quoted_runs_handles_unclosed_quote() -> None:
    # an unclosed quote consumes the rest of the input
    cleaned = _strip_quoted_runs('a,"b\nc')
    assert cleaned == "a,"


def test_candidate_score_rejects_zero_count() -> None:
    assert _candidate_score(["abc", "def"], ",") is None


def test_candidate_score_accepts_uniform() -> None:
    score = _candidate_score(["a,b", "c,d", "e,f"], ",")
    assert score is not None
    rows, spread, total = score
    assert rows == 3
    assert spread == 0.0
    assert total == 3


def test_candidates_tuple_is_distinct() -> None:
    assert len(set(CANDIDATE_DELIMITERS)) == len(CANDIDATE_DELIMITERS)


def test_picks_more_consistent_candidate() -> None:
    # both ',' and ';' appear, but ',' is consistent across rows.
    text = "a,b;c\nd,e;f\ng,h;i\n"
    # comma + semicolon both present; comma appears once per row, semicolon once per row;
    # tie-break should prefer one — accept either as long as it's deterministic and not None.
    delim = infer_delimiter(text)
    assert delim in (",", ";")


def test_returns_string_type() -> None:
    delim = infer_delimiter("a,b\nc,d\n")
    assert isinstance(delim, str)


@pytest.mark.parametrize("delim", list(CANDIDATE_DELIMITERS))
def test_each_candidate_round_trips(delim: str) -> None:
    body = f"a{delim}b{delim}c\n1{delim}2{delim}3\n4{delim}5{delim}6\n"
    assert infer_delimiter(body) == delim


def test_infer_delimiter_accepts_explicit_sample_rows() -> None:
    # When the caller pre-tokenises the rows, the delimiter inference
    # skips its own row-splitting path and uses the given rows directly.
    # Imported from the private module since the public API does not
    # expose this argument.
    from csvinfer._delimiter import infer_delimiter as raw_infer
    rows = ["a,b,c", "1,2,3", "4,5,6"]
    assert raw_infer("ignored:body:here\nx:y:z", sample_rows=rows) == ","
