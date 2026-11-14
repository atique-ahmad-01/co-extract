"""Business-rule validation beyond types: ranges and cross-field / cross-document checks."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from .grounding import Span
from .normalize import MONEY_PATTERN, NEG_CUES, POS_CUES, id_key, parse_amount

REQUIRED = ("change_order_number", "amount")
MAX_ABS_AMOUNT = 100_000_000
MIN_DATE = dt.date(1990, 1, 1)
FUTURE_SLACK_DAYS = 60
DAYS_RANGE = (-365, 730)

_GAP = r"[^\n$\d(]{0,60}?"
_PRIOR = (r"contract[ \t]+(?:sum|amount|value|price)[ \t]+prior[ \t]+to[ \t]+this[ \t]+"
          r"(?:change[ \t]+order|co)\b(?:[ \t]+was)?")
_CURRENT = r"(?:current|previously[ \t]+(?:revised|adjusted))[ \t]+contract[ \t]+(?:sum|amount|value|price)"
_ORIGINAL = r"original[ \t]+contract[ \t]+(?:sum|amount|value|price)(?:[ \t]+was)?"
_PREVIOUS = (r"(?:net[ \t]+change[ \t]+by[ \t]+)?previous(?:ly)?[ \t]+(?:authorized[ \t]+|approved[ \t]+)?"
             r"change[ \t]+orders?")
_NEW = (r"(?<!previously[ ])(?<!previous[ ])(?:new|revised|adjusted)[ \t]+contract[ \t]+"
        r"(?:sum|amount|value|price)")
_TIME_NO_CHANGE = re.compile(
    r"(?i)(contract[ \t]+time|schedule)[^\n]{0,60}\b(no[ \t]+change|unchanged"
    r"|not[ \t]+(?:be[ \t]+)?(?:changed|affected|extended))")


@dataclass
class Issue:
    field: str | None
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}" if self.field else self.message


def _money_after(label: str, text: str) -> float | None:
    m = re.search(r"(?i)" + label + _GAP + "(" + MONEY_PATTERN + ")", text)
    return parse_amount(m.group(1)) if m else None


def contract_sums(text: str) -> dict[str, float | None]:
    original = _money_after(_ORIGINAL, text)
    previous = _money_after(_PREVIOUS, text)
    prior = _money_after(_PRIOR, text)
    if prior is None:
        prior = _money_after(_CURRENT, text)
    if prior is None and original is not None and (
            previous is not None or not re.search(r"(?i)previous", text)):
        prior = round(original + (previous or 0.0), 2)
    return {"original": original, "previous": previous, "prior": prior,
            "new": _money_after(_NEW, text)}


def validate(values: dict, sources: dict[str, Span | None], text: str,
             today: dt.date | None = None) -> list[Issue]:
    today = today or dt.date.today()
    issues: list[Issue] = []

    for f in REQUIRED:
        if values.get(f) is None:
            issues.append(Issue(f, "required field missing"))
    if values.get("project_name") is None and values.get("contract_number") is None:
        issues.append(Issue(None, "neither project_name nor contract_number found"))

    # date range
    d = values.get("date")
    if d is not None:
        dd = dt.date.fromisoformat(d)
        if dd < MIN_DATE:
            issues.append(Issue("date", f"{d} is implausibly old"))
        elif dd > today + dt.timedelta(days=FUTURE_SLACK_DAYS):
            issues.append(Issue("date", f"{d} is in the future"))

    # amount: range, sign vs wording, contract-sum arithmetic
    amount = values.get("amount")
    if amount is not None:
        if abs(amount) > MAX_ABS_AMOUNT:
            issues.append(Issue("amount", f"{amount} outside plausible range"))
        src = sources.get("amount")
        if src is not None:
            ctx = text[max(0, src.start - 80): src.end + 20]
            neg, pos = NEG_CUES.search(ctx), POS_CUES.search(ctx)
            if amount > 0 and neg and not pos:
                issues.append(Issue("amount", f"positive but document says '{neg.group()}'"))
            elif amount < 0 and pos and not neg:
                issues.append(Issue("amount", f"negative but document says '{pos.group()}'"))

    sums = contract_sums(text)
    if amount is not None and sums["prior"] is not None and sums["new"] is not None:
        expected = round(sums["new"] - sums["prior"], 2)
        if abs(expected - amount) > 1.0:
            issues.append(Issue("amount",
                                f"contract math: prior {sums['prior']:,.2f} + change {amount:,.2f} "
                                f"!= new {sums['new']:,.2f} (implied change {expected:,.2f})"))
    if None not in (sums["original"], sums["previous"]) and sums["prior"] is not None:
        if abs(sums["original"] + sums["previous"] - sums["prior"]) > 1.0:
            issues.append(Issue(None, "document's own contract sums do not add up"))

    # schedule
    days = values.get("schedule_impact_days")
    if days is not None:
        if not DAYS_RANGE[0] <= days <= DAYS_RANGE[1]:
            issues.append(Issue("schedule_impact_days", f"{days} outside plausible range"))
        if days != 0 and _TIME_NO_CHANGE.search(text):
            issues.append(Issue("schedule_impact_days", f"{days} but document says no time change"))

    # identifiers
    co, ct = values.get("change_order_number"), values.get("contract_number")
    if co is not None and ct is not None and id_key(co) == id_key(ct):
        issues.append(Issue("contract_number", "same as change_order_number"))

    return issues
