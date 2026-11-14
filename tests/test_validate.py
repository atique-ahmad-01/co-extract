from co_extract.grounding import locate
from co_extract.validate import contract_sums, validate
from tests.conftest import TODAY

BASE = {
    "project_name": "P", "contract_number": "C-1", "change_order_number": "7", "date": "2025-03-04",
    "requested_by": None, "description": None, "amount": 146250.0, "schedule_impact_days": 7,
    "approval_status": "pending",
}


def messages(issues):
    return [str(i) for i in issues]


def test_contract_math_conflict_is_flagged(doc6):
    sums = contract_sums(doc6)
    assert sums["prior"] == 19112400.00 and sums["new"] == 19285650.00
    issues = validate(BASE, {}, doc6, today=TODAY)
    assert any("contract math" in m for m in messages(issues))


def test_consistent_contract_math_passes(doc1):
    values = dict(BASE, amount=18740.0, schedule_impact_days=5, date="2025-03-12")
    assert not [m for m in messages(validate(values, {}, doc1, today=TODAY)) if "math" in m]


def test_sign_contradicting_document_wording(doc2):
    src = locate("TOTAL CREDIT THIS COR", doc2)
    issues = validate(dict(BASE, amount=10630.0, schedule_impact_days=-3), {"amount": src}, doc2, today=TODAY)
    assert any("positive but document says" in m for m in messages(issues))


def test_ranges_and_required():
    values = dict(BASE, amount=None, date="2031-01-01", schedule_impact_days=5000)
    msgs = messages(validate(values, {}, "", today=TODAY))
    assert any("amount: required" in m for m in msgs)
    assert any("in the future" in m for m in msgs)
    assert any("outside plausible range" in m for m in msgs)


def test_schedule_contradicts_no_change_text():
    text = "Contract time: no change.\nAmount of this change: $500.00"
    msgs = messages(validate(dict(BASE, amount=500.0, schedule_impact_days=10), {}, text, today=TODAY))
    assert any("no time change" in m for m in msgs)


def test_contract_number_equal_to_co_number():
    msgs = messages(validate(dict(BASE, contract_number="CO-7", change_order_number="co 7"), {}, "", today=TODAY))
    assert any("same as change_order_number" in m for m in msgs)
