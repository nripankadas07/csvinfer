"""Public-API surface checks: every advertised name is reachable."""

from __future__ import annotations

import csvinfer


def test_module_exports_expected_names() -> None:
    expected = {
        "CANDIDATE_DELIMITERS", "CANDIDATE_QUOTES",
        "AmbiguousDialectError", "CsvInferError", "Dialect",
        "EmptyInputError", "InvalidDialectError",
        "cell_type", "has_header", "infer", "infer_delimiter",
        "infer_quote_char", "read_rows",
    }
    assert expected.issubset(set(csvinfer.__all__))
    for name in expected:
        assert hasattr(csvinfer, name), f"missing export: {name}"


def test_version_string() -> None:
    assert isinstance(csvinfer.__version__, str)
    assert csvinfer.__version__.count(".") == 2


def test_error_hierarchy() -> None:
    assert issubclass(csvinfer.CsvInferError, ValueError)
    assert issubclass(csvinfer.EmptyInputError, csvinfer.CsvInferError)
    assert issubclass(csvinfer.AmbiguousDialectError, csvinfer.CsvInferError)
    assert issubclass(csvinfer.InvalidDialectError, csvinfer.CsvInferError)


def test_ambiguous_dialect_carries_candidates() -> None:
    err = csvinfer.AmbiguousDialectError("multi", candidates=[",", ";"])
    assert err.candidates == [",", ";"]
    assert "multi" in str(err)
