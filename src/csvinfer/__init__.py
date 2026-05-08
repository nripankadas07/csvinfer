"""csvinfer — infer CSV dialect (delimiter, quote, header) from raw text.

Synopsis:

    >>> import csvinfer
    >>> sample = "name,age,active\\nAda,30,true\\nGrace,40,false\\n"
    >>> dialect = csvinfer.infer(sample)
    >>> dialect.delimiter, dialect.has_header
    (',', True)

Without ``csv.Sniffer``, in pure Python.
"""

from ._api import has_header, infer, infer_delimiter, infer_quote_char, read_rows
from ._delimiter import CANDIDATE_DELIMITERS
from ._errors import (
    AmbiguousDialectError,
    CsvInferError,
    EmptyInputError,
    InvalidDialectError,
)
from ._header import cell_type
from ._quote import CANDIDATE_QUOTES
from ._types import Dialect

__all__ = [
    "CANDIDATE_DELIMITERS",
    "CANDIDATE_QUOTES",
    "AmbiguousDialectError",
    "CsvInferError",
    "Dialect",
    "EmptyInputError",
    "InvalidDialectError",
    "cell_type",
    "has_header",
    "infer",
    "infer_delimiter",
    "infer_quote_char",
    "read_rows",
]

__version__ = "0.1.0"
