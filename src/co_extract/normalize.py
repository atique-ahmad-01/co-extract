"""Turn raw extracted strings into typed, canonical values.

Every parser returns ``None`` when it cannot produce a value; it never guesses.
"""

from __future__ import annotations

import datetime as dt
import re

NULLISH = {
    "", "n/a", "na", "none", "null", "unknown", "not specified", "not stated",
    "not provided", "-", "--", "tbd", "tba",
}

# Dates

MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_PATTERN = (
    r"(?:\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    rf"|{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\.?,?\s+\d{{4}})"
)
DATE_RE = re.compile(r"\b" + DATE_PATTERN + r"\b", re.I)

_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%m.%d.%Y", "%m.%d.%y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
)


def _try_date_formats(s: str) -> dt.date | None:
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s, flags=re.I)
    s = re.sub(r"\bsept\b", "sep", s, flags=re.I)
    s = re.sub(r"\b([A-Za-z]{3,9})\.", r"\1", s)
    s = re.sub(r"\s+", " ", s.replace(",", " ")).strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_date(raw) -> dt.date | None:
    """Parse a date. Numeric dates are read as US month/day."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, dt.datetime):
        return raw.date()
    if isinstance(raw, dt.date):
        return raw
    s = str(raw).strip()
    if s.lower() in NULLISH:
        return None
    d = _try_date_formats(s)
    if d:
        return d
    m = DATE_RE.search(s)
    return _try_date_formats(m.group()) if m else None


# Money

MONEY_PATTERN = r"\(?[ \t]*[-−]?[ \t]*(?:USD|US\$|\$)?[ \t]*\d[\d,]*(?:\.\d{1,2})?[ \t]*\)?"
NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
NEG_CUES = re.compile(r"\b(credit|deduct\w*|decrease[ds]?|reduc\w*|omit\w*)\b", re.I)
POS_CUES = re.compile(r"\b(increase[ds]?|add|added|addition\w*|extra)\b", re.I)


def parse_amount(raw) -> float | None:
    """Parse a money amount; decreases/credits come back negative."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 2)
    s = str(raw).strip()
    if s.lower() in NULLISH:
        return None
    compact = re.sub(r"\s+", "", s).replace("−", "-").replace("–", "-")
    m = NUM_RE.search(compact)
    if not m:
        return None
    value = float(m.group().replace(",", ""))
    head, tail = compact[: m.start()], compact[m.end():]
    if tail[:1].lower() == "k" and not tail[1:2].isalpha():
        value *= 1000
    negative = bool(re.search(r"(\(|-)(USD|US\$|\$)?$", head)) or bool(NEG_CUES.search(s))
    return round(-value if negative else value, 2)


# Days

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "sixty": 60, "ninety": 90,
}
NO_CHANGE_RE = re.compile(
    r"\b(no\s+(?:change|impact|extension|adjustment|additional\s+(?:time|days))|none|unchanged"
    r"|not\s+(?:be\s+)?(?:changed|affected|extended|impacted))\b",
    re.I,
)
_DAYS_NEG = re.compile(r"\b(reduc|decreas|accelerat|shorten)", re.I)


def parse_days(raw) -> int | None:
    """Parse a schedule impact into whole days (weeks x7, months x30); reductions negative."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(round(raw))
    s = str(raw).strip().lower().replace("−", "-")
    if not s or (s in NULLISH and s != "none"):
        return None
    n: float | None = None
    m = re.search(r"[-+]?\s*\d+(?:\.\d+)?", s)
    if m:
        n = float(m.group().replace(" ", ""))
    else:
        for word, val in sorted(NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{word}\b", s):
                n = float(val)
                break
    if n is None:
        return 0 if NO_CHANGE_RE.search(s) else None
    if re.search(r"\bweeks?\b", s):
        n *= 7
    elif re.search(r"\bmonths?\b", s):
        n *= 30
    if n > 0 and _DAYS_NEG.search(s):
        n = -n
    return int(round(n))


# Status

STATUSES = ("approved", "pending", "rejected")
_STATUS_RULES = (
    ("rejected", re.compile(
        r"\b(reject\w*|denied|deny|declin\w*|void\w*|withdrawn|not\s+approved|disapprov\w*)\b", re.I)),
    ("pending", re.compile(
        r"\b(pending|submitted|await\w*|under\s+review|in\s+review|proposed|draft|requested|unsigned"
        r"|not\s+yet|for\s+approval|_{3,})", re.I)),
    ("approved", re.compile(
        r"(\b(approved|approve|accepted|executed|authori[sz]ed|signed)\b|/s/)", re.I)),
)


def normalize_status(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in NULLISH:
        return None
    if s in STATUSES:
        return s
    for label, rx in _STATUS_RULES:
        if rx.search(s):
            return label
    return None


# Identifiers and text

def normalize_id(raw) -> str | None:
    if raw is None or isinstance(raw, bool):
        return None
    s = str(raw).strip()
    s = re.sub(r"^(?:no\.?|number|#)\s*[:#]?\s*", "", s, flags=re.I).strip(" .:#")
    if s.lower() in NULLISH:
        return None
    return s


def id_key(s) -> str:
    """Comparison key for identifiers: case/punctuation-insensitive, O/0 and l/1 folded,
    leading zeros dropped ("CO-009" == "C0-9")."""
    k = re.sub(r"[^a-z0-9]", "", str(s).lower()).replace("o", "0").replace("l", "1")
    return re.sub(r"(?<!\d)0+(?=\d)", "", k)


def normalize_text(raw, max_len: int = 200) -> str | None:
    if raw is None or isinstance(raw, bool):
        return None
    s = re.sub(r"\s+", " ", str(raw)).strip().strip("\"'")
    if s.lower() in NULLISH:
        return None
    return s[:max_len].rstrip()


_STOP = {"the", "of", "and", "for", "inc", "llc", "a", "an", "by", "to", "at", "on", "in"}


def tokens(s) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if len(t) > 1 and t not in _STOP}


def token_overlap(a, b) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# Dispatch

ID_FIELDS = ("contract_number", "change_order_number")


def normalize_field(name: str, raw):
    if name == "amount":
        return parse_amount(raw)
    if name == "date":
        d = parse_date(raw)
        return d.isoformat() if d else None
    if name == "schedule_impact_days":
        return parse_days(raw)
    if name == "approval_status":
        return normalize_status(raw)
    if name in ID_FIELDS:
        return normalize_id(raw)
    if name == "description":
        return normalize_text(raw, max_len=600)
    return normalize_text(raw, max_len=200)


def same_value(name: str, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if name == "amount":
        return abs(float(a) - float(b)) < 0.01
    if name in ID_FIELDS:
        return id_key(a) == id_key(b)
    if name == "description":
        return True
    if name in ("project_name", "requested_by"):
        return token_overlap(a, b) >= 0.6
    return a == b
