"""Quote-character inference.

For each candidate quote character we count its occurrences in *text*.
A quote character is plausible if it appears an even number of times
(every open quote has a matching close); the candidate with the most
matched pairs wins. ``None`` is returned if no candidate appears at
all, which is a strong signal that the text is unquoted.
"""

from __future__ import annotations

CANDIDATE_QUOTES: tuple[str, ...] = ('"', "'")


def infer_quote_char(text: str) -> str | None:
    """Return the inferred quote character, or ``None`` if no quoting
    is detected in *text*.
    """
    if not text:
        return None
    best: tuple[int, str] | None = None
    for q in CANDIDATE_QUOTES:
        count = text.count(q)
        if count == 0 or count % 2 != 0:
            continue
        # only count it if at least one quote follows the start of a field
        # (immediately after start-of-string, newline, or a likely delimiter)
        if not _looks_like_real_quote(text, q):
            continue
        score = count // 2
        if best is None or score > best[0]:
            best = (score, q)
    return best[1] if best else None


def _looks_like_real_quote(text: str, quote: str) -> bool:
    """Heuristic: at least one occurrence should sit at a position
    where a CSV quote could legally start (beginning of string, after a
    newline, or after a non-alphanumeric character).
    """
    for i, ch in enumerate(text):
        if ch != quote:
            continue
        if i == 0:
            return True
        prev = text[i - 1]
        if prev in "\n\r,\t;|: ":
            return True
    return False
