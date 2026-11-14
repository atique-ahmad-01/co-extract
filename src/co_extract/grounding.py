"""Locate an extracted quote in the source text (exact, then fuzzy).

A value whose evidence cannot be found in the document is treated as a guess and dropped.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

_TRANS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


@dataclass
class Span:
    quote: str
    start: int
    end: int
    match: str  # "exact" | "fuzzy"
    score: float
    page: int = 1

    def to_json(self) -> dict:
        return {"quote": self.quote, "page": self.page, "start": self.start, "end": self.end,
                "match": self.match}


def _normalize(text: str) -> tuple[str, list[int]]:
    """Lowercase, unify punctuation, collapse whitespace. Returns (normalized, index_map)
    where index_map[i] is the offset in the original text of normalized char i."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # drops leading whitespace
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        for c in ch.translate(_TRANS).lower():
            out.append(c)
            idx.append(i)
        prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _span(text: str, idx: list[int], a: int, b: int, match: str, score: float) -> Span:
    start, end = idx[a], idx[b - 1] + 1
    page = text.count("\f", 0, start) + 1
    return Span(text[start:end], start, end, match, round(score, 3), page)


def locate(quote, text: str, fuzzy_threshold: float = 0.85) -> Span | None:
    if quote is None or not str(quote).strip() or not text:
        return None
    nq, _ = _normalize(str(quote))
    nt, idx = _normalize(text)
    if not nq or not nt:
        return None
    pos = nt.find(nq)
    if pos >= 0:
        return _span(text, idx, pos, pos + len(nq), "exact", 1.0)
    if len(nq) < 6:
        return None  # too short for a meaningful fuzzy match

    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(nq)
    L = len(nq)
    step = max(1, L // 8)
    best = (0.0, 0, 0)

    def score(s: int, w: int) -> float:
        sm.set_seq1(nt[s:s + w])
        if sm.real_quick_ratio() <= best[0] or sm.quick_ratio() <= best[0]:
            return 0.0
        return sm.ratio()

    widths = sorted({L, max(1, int(L * 0.9)), int(L * 1.1) + 1})
    for w in widths:
        for s in range(0, max(1, len(nt) - w + 1), step):
            r = score(s, w)
            if r > best[0]:
                best = (r, s, w)
    r0, s0, w0 = best
    for w in range(max(1, w0 - 3), w0 + 4):
        for s in range(max(0, s0 - step), min(len(nt), s0 + step + 1)):
            r = score(s, w)
            if r > best[0]:
                best = (r, s, w)
    r, s, w = best
    if r >= fuzzy_threshold:
        return _span(text, idx, s, min(len(nt), s + w), "fuzzy", r)
    return None
