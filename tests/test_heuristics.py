from co_extract.heuristics import extract_all
from tests.conftest import read_sample


def val(h, name):
    return h[name][0] if h.get(name) else None


def test_aia_style(doc1):
    h = extract_all(doc1)
    assert val(h, "project_name") == "Riverside Medical Office Building"
    assert val(h, "contract_number") == "RMOB-2024-117"
    assert val(h, "change_order_number") == "004"
    assert val(h, "date") == "2025-03-12"
    assert val(h, "amount") == 18740.00  # not the prior/new contract sums
    assert val(h, "schedule_impact_days") == 5


def test_credit_and_reduction(doc2):
    h = extract_all(doc2)
    assert val(h, "amount") == -10630.00
    assert val(h, "schedule_impact_days") == -3
    assert val(h, "approval_status") == "pending"
    assert val(h, "change_order_number") == "17"
    assert val(h, "contract_number") == "LV-ELEC-0392"


def test_weeks_and_rejected():
    h = extract_all(read_sample("co_05_multi_amount_rejected.txt"))
    assert val(h, "schedule_impact_days") == 14
    assert val(h, "approval_status") == "rejected"
    assert val(h, "date") == "2025-05-20"


def test_email_has_no_schedule_impact():
    # "on site 2 weeks after release" is a lead time, not a schedule change
    h = extract_all(read_sample("co_04_email_pasted.txt"))
    assert val(h, "schedule_impact_days") is None
    assert val(h, "contract_number") is None
    assert val(h, "amount") == 9988.00
