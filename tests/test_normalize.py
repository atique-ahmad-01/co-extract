import pytest

from co_extract.normalize import (
    id_key, normalize_field, normalize_status, parse_amount, parse_date, parse_days, same_value,
)


@pytest.mark.parametrize("raw, expected", [
    ("$18,740.00", 18740.00),
    ("($10,630.00)", -10630.00),
    ("-$2,500", -2500.00),
    ("Credit of $1,200", -1200.00),
    ("USD 3,250.5", 3250.50),
    ("$12.5k", 12500.00),
    (-450, -450.00),
    ("TBD", None),
    ("n/a", None),
    (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("March 12, 2025", "2025-03-12"),
    ("Mar. 12th, 2025", "2025-03-12"),
    ("7/2/2025", "2025-07-02"),
    ("2025-05-20", "2025-05-20"),
    ("12 Sept 2024", "2024-09-12"),
    ("Sent: Thursday, January 9, 2025 4:12 PM", "2025-01-09"),
    ("1l/04/2024", None),  # OCR noise: refuse rather than guess
    ("not stated", None),
])
def test_parse_date(raw, expected):
    d = parse_date(raw)
    assert (d.isoformat() if d else None) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Five (5) days", 5),
    ("two (2) weeks", 14),
    ("+7 calendar days", 7),
    ("decreased by 3 working days", -3),
    ("-5", -5),
    ("No change", 0),
    ("none", 0),
    ("TBD", None),
    (12, 12),
])
def test_parse_days(raw, expected):
    assert parse_days(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("APPROVED", "approved"),
    ("Pending approval", "pending"),
    ("Not approved", "rejected"),
    ("SUBMITTED - awaiting GC / Owner approval", "pending"),
    ("Owner: Signed: ______________", "pending"),
    ("By: /s/ L. Grant", "approved"),
    ("void", "rejected"),
    ("whatever", None),
])
def test_normalize_status(raw, expected):
    assert normalize_status(raw) == expected


def test_id_key_folds_ocr_and_zeros():
    assert id_key("C0-09") == id_key("CO-9")
    assert id_key("004") == id_key("4")
    assert id_key("RMOB-2024-117") != id_key("RMOB-2024-171")


def test_normalize_field_date_returns_iso_string():
    assert normalize_field("date", "March 12, 2025") == "2025-03-12"


def test_same_value_names_tolerate_suffixes():
    assert same_value("requested_by", "Fenwick Studio", "Fenwick Studio (Architect) - ASI-22")
    assert not same_value("amount", 100.0, 100.5)
