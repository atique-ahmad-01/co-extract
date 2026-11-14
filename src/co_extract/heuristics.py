"""Regex extractors for labelled fields.

Used as a second opinion when scoring LLM output, and as the fallback when no LLM is
available.
"""

from __future__ import annotations

import re

from .normalize import (
    DATE_PATTERN, MONEY_PATTERN, NEG_CUES, NO_CHANGE_RE, NUM_WORDS,
    normalize_id, normalize_status, normalize_text, parse_amount, parse_date, parse_days,
)

SEP = r"[ \t]*(?::|#|-|=|\.{2,})[ \t.:#=]*"
_ID = r"([A-Z0-9][A-Z0-9\-_/.]*)"
_LABEL_NO = r"[ \t]*(?:no\.?|number|num\.?|#|id)[ \t.:#]*"

PROJECT_RE = re.compile(r"(?im)^[ \t]*(?:project[ \t]+name|project|job[ \t]+name|job)" + SEP + r"(.+)$")
REQUESTED_RE = re.compile(
    r"(?im)\b(?:requested|initiated|originated|submitted|proposed)[ \t]+by[ \t]*(?::|-|\.{2,})?[ \t.]*(.+)$")
DESC_RE = re.compile(
    r"(?im)^[ \t]*(?:description(?:[ \t]+of[ \t]+(?:change|work))?|scope(?:[ \t]+of[ \t]+(?:change|work))?"
    r"|reason[ \t]+for[ \t]+change|the[ \t]+contract[ \t]+is[ \t]+changed[ \t]+as[ \t]+follows)"
    + SEP + r"(.*)$")
CO_RE = re.compile(
    r"(?i)\b(?:change[ \t]+order|c\.?[ \t]?o\.?|p\.?c\.?o\.?|c\.?o\.?r\.?)(?:[ \t]+request)?" + _LABEL_NO + _ID)
CONTRACT_RE = re.compile(
    r"(?i)\b(?:contract|subcontract|agreement|p\.?o\.?|purchase[ \t]+order)" + _LABEL_NO + _ID)
DATE_LABEL_RE = re.compile(
    r"(?i)\b(?:dated|date(?:[ \t]+issued)?|issue[ \t]+date)" + SEP + "(" + DATE_PATTERN + ")")
AMOUNT_LABEL = (
    r"(?:change[ \t]+order[ \t]+(?:amount|total|value|sum)"
    r"|amount[ \t]+of[ \t]+(?:this[ \t]+)?change(?:[ \t]+order)?"
    r"|net[ \t]+change"
    r"|total[ \t]+(?:of[ \t]+)?(?:this[ \t]+)?change(?:[ \t]+order)?(?:[ \t]+amount)?"
    r"|contract[ \t]+(?:sum|amount|price)[ \t]+(?:will[ \t]+be|is|shall[ \t]+be)[ \t]+"
    r"(?:increased|decreased|adjusted|reduced)[ \t]+by"
    r"|total[ \t]+(?:add|deduct|credit)"
    r"|this[ \t]+change[ \t]+order(?:[ \t]+amount)?)"
)
AMOUNT_RE = re.compile(r"(?i)" + AMOUNT_LABEL + r"[^\n$\d(]{0,60}?(" + MONEY_PATTERN + ")")
AMOUNT_SKIP = re.compile(r"(?i)\b(prior|previous(?:ly)?|original|new|revised)\b")
DAYS_LINE = re.compile(
    r"(?i)\b(contract[ \t]+time|schedule|time[ \t]+extension|time|completion|duration)\b")
_WORDS = "|".join(sorted(NUM_WORDS, key=len, reverse=True))
DAYS_VALUE = re.compile(
    r"(?i)(?:[-+−]?[ \t]*\d+|\b(?:" + _WORDS + r")\b)[ \t]*(?:\([ \t]*\d+[ \t]*\)[ \t]*)?"
    r"(?:calendar|working|business|work)?[ \t]*(?:days?|weeks?|months?)\b")
DAYS_NEG = re.compile(r"(?i)\b(reduc\w*|decreas\w*|accelerat\w*|shorten\w*)")
STATUS_RE = re.compile(
    r"(?im)\b(?:approval[ \t]+status|status|owner[ \t]+decision|decision)" + SEP + r"(.+)$")
STATUS_BY_RE = re.compile(r"(?im)^[ \t]*(approved|rejected)[ \t]+by\b.*$")


def _cell(v: str) -> str | None:
    """Cut a value at layout column gaps (3+ spaces or a tab)."""
    v = re.split(r" {3,}|\t", v.strip())[0]
    v = v.strip(" .:;,")
    return v or None


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start: end if end != -1 else len(text)]


def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def _ids(rx: re.Pattern, text: str) -> tuple[str, str] | None:
    for m in rx.finditer(text):
        v = normalize_id(m.group(1).rstrip(".-/"))
        if v and _has_digit(v):
            return v, m.group(0).strip()
    return None


def extract_all(text: str) -> dict[str, tuple[object, str] | None]:
    """Return {field: (value, quote) | None}."""
    out: dict[str, tuple[object, str] | None] = {}

    m = PROJECT_RE.search(text)
    v = normalize_text(_cell(m.group(1))) if m else None
    out["project_name"] = (v, m.group(0).strip()) if v else None

    out["change_order_number"] = _ids(CO_RE, text)
    contract = _ids(CONTRACT_RE, text)
    if contract and out["change_order_number"] and contract[0] == out["change_order_number"][0]:
        contract = None
    out["contract_number"] = contract

    m = DATE_LABEL_RE.search(text)
    d = parse_date(m.group(1)) if m else None
    out["date"] = (d.isoformat(), m.group(0).strip()) if d else None

    m = REQUESTED_RE.search(text)
    v = normalize_text(_cell(m.group(1))) if m else None
    out["requested_by"] = (v, m.group(0).strip()) if v else None

    out["description"] = None
    m = DESC_RE.search(text)
    if m:
        body = m.group(1).strip()
        if not body:  # value is on the following lines, up to a blank line
            rest = text[m.end():].lstrip("\n")
            body = rest.split("\n\n", 1)[0]
        v = normalize_text(body, max_len=600)
        if v:
            out["description"] = (v, body.strip())

    out["amount"] = None
    for m in AMOUNT_RE.finditer(text):
        line = _line_at(text, m.start())
        if AMOUNT_SKIP.search(line):
            continue
        val = parse_amount(m.group(1))
        if val is None:
            continue
        if val > 0 and NEG_CUES.search(line):
            val = -val
        out["amount"] = (val, m.group(0).strip())
        break

    out["schedule_impact_days"] = None
    for line in text.split("\n"):
        if not DAYS_LINE.search(line):
            continue
        m = DAYS_VALUE.search(line)
        if m:
            n = parse_days(m.group(0))
            if n is not None:
                if n > 0 and DAYS_NEG.search(line):
                    n = -n
                out["schedule_impact_days"] = (n, line.strip())
                break
        elif NO_CHANGE_RE.search(line):
            out["schedule_impact_days"] = (0, line.strip())
            break

    out["approval_status"] = None
    m = STATUS_RE.search(text)
    if m and (s := normalize_status(_cell(m.group(1)) or "")):
        out["approval_status"] = (s, m.group(0).strip())
    else:
        m = STATUS_BY_RE.search(text)
        if m:
            out["approval_status"] = (m.group(1).lower(), m.group(0).strip())

    return out
