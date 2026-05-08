# csvinfer

Infer CSV dialect — delimiter, quote character, header presence, line terminator — from raw text. Pure Python, zero runtime dependencies, no `csv.Sniffer`.

## Why

`csv.Sniffer` is part of the standard library, but it's quirky on uncommon delimiters, can hang on adversarial inputs, and gives no easy way to inspect *why* it picked one dialect over another. `csvinfer` is a small, transparent alternative built around a frozen `Dialect` value type and a handful of well-tested heuristics.

## Install

```bash
pip install csvinfer
```

Requires Python 3.10 or newer (uses PEP 604 union types and `dataclass(frozen=True)` strictness).

## Usage

```python
import csvinfer

raw = open("data.csv").read()
dialect = csvinfer.infer(raw)
print(dialect)
# Dialect(delimiter=',', quote_char='"', has_header=True, line_terminator='\n')

rows = csvinfer.read_rows(raw, dialect=dialect)
header, body = (rows[0], rows[1:]) if dialect.has_header else (None, rows)
```

You can also call each piece on its own:

```python
csvinfer.infer_delimiter("a,b,c\n1,2,3\n")        # ','
csvinfer.infer_quote_char('"a","b"\n"c","d"\n')   # '"'
csvinfer.has_header("name,age\nAda,30\n")          # True
csvinfer.cell_type("3.14")                         # 'float'
```

## API

### Top-level functions

- `infer(text: str, *, sample_lines: int = 50) -> Dialect`
- `infer_delimiter(text: str) -> str | None`
- `infer_quote_char(text: str) -> str | None`
- `has_header(text: str, *, dialect: Dialect | None = None) -> bool`
- `read_rows(text: str, *, dialect: Dialect | None = None) -> list[list[str]]`
- `cell_type(value: str) -> str` — one of `"blank" | "int" | "float" | "bool" | "date" | "string"`

### Dialect

```python
@dataclass(frozen=True)
class Dialect:
    delimiter: str
    quote_char: str | None
    has_header: bool
    line_terminator: str   # "\n", "\r\n", or "\r"
```

Validates its fields on construction; mismatched values raise `InvalidDialectError`.

### Errors

`CsvInferError` is the root (also a `ValueError`). Specific subclasses: `EmptyInputError`, `AmbiguousDialectError` (carries a `candidates` list), `InvalidDialectError`.

### Candidates

Constants exposing the candidate sets so callers can adjust expectations:

- `CANDIDATE_DELIMITERS = (',', '\t', ';', '|', ':', ' ')`
- `CANDIDATE_QUOTES = ('"', "'")`

## Heuristics, in plain English

- **Delimiter**: count occurrences of each candidate per non-blank row, ignoring quoted regions; pick the one whose count is positive on every row and most consistent (lowest variance, then highest total).
- **Quote**: pick the candidate whose count is even and that appears at a legitimate field-start position (start of input, after a newline, or after a delimiter).
- **Header**: classify cell types via lightweight regexes; if the first row is all-string and the body has any non-string cells, mark it as a header.
- **Line terminator**: prefer `\r\n` if present, else `\r` if it dominates `\n`, else `\n`.

## Running tests

```bash
pip install -e ".[test]"
pytest --cov=src/csvinfer --cov-branch     # 107 tests, 100% line + 100% branch
mypy --strict src/csvinfer                 # 0 issues
```

## License

MIT
