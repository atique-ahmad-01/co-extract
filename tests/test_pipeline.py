import json
from pathlib import Path

import jsonschema
import pytest

from co_extract import FIELDS
from co_extract.pipeline import extract_file, extract_text
from tests.conftest import ROOT, SYNTH, TODAY, FakeClient

SCHEMA = json.loads((ROOT / "schema" / "change_order.schema.json").read_text())


def f(value, quote):
    return {"value": value, "quote": quote}


GOOD_DOC1 = {
    "project_name": f("Riverside Medical Office Building", "PROJECT: Riverside Medical Office Building"),
    "contract_number": f("RMOB-2024-117", "CONTRACT NO.: RMOB-2024-117"),
    "change_order_number": f("004", "CHANGE ORDER NUMBER: 004"),
    "date": f("2025-03-12", "DATE: March 12, 2025"),
    "requested_by": f("Riverside Health Partners", "Requested by Owner (Riverside Health Partners)"),
    "description": f("Upgraded LED corridor fixtures 2A-2F per ASI-09 with added circuiting.",
                     "Furnish and install upgraded LED fixtures in corridors 2A-2F per ASI-09"),
    "amount": f(18740.0, "The Contract Sum will be increased by this Change Order in the amount of $ 18,740.00"),
    "schedule_impact_days": f(5, "The Contract Time will be increased by Five (5) days"),
    "approval_status": f("approved", "By: /s/ L. Grant"),
}


def run(text, samples, **kw):
    return extract_text(text, client=FakeClient(samples), n_samples=len(samples), today=TODAY, **kw)


def test_clean_document_with_consistent_samples_is_auto_accepted(doc1):
    out = run(doc1, [GOOD_DOC1] * 3)
    assert out["needs_review"] is False, out["review_reasons"]
    assert out["amount"]["value"] == 18740.0
    assert out["date"]["value"] == "2025-03-12"
    assert all(out[name]["confidence"] >= 0.8 for name in FIELDS)
    src = out["contract_number"]["source"]
    assert doc1[src["start"]:src["end"]] == "CONTRACT NO.: RMOB-2024-117"
    jsonschema.validate(out, SCHEMA)


def test_hallucinated_value_is_dropped_to_null(doc1):
    bad = dict(GOOD_DOC1, contract_number=f("ZX-555-0001", "Contract ZX-555-0001"))
    out = run(doc1, [bad] * 3)
    assert out["contract_number"]["value"] is None
    assert out["contract_number"]["source"] is None
    assert out["needs_review"] is True
    assert any("no supporting evidence" in r for r in out["review_reasons"])


def test_sample_disagreement_lowers_confidence(doc1):
    consistent = run(doc1, [GOOD_DOC1] * 3)["amount"]["confidence"]
    noisy = [GOOD_DOC1, dict(GOOD_DOC1, amount=f(17840.0, None)), dict(GOOD_DOC1, amount=f(2559990.0, None))]
    disagree = run(doc1, noisy)["amount"]["confidence"]
    assert disagree < consistent


def test_absent_field_is_null_with_high_confidence():
    text = ("Change Order No. 12\nProject: Test Tower\nChange order amount: $1,000.00\n"
            "Description: add one door.")
    sample = {name: f(None, None) for name in FIELDS}
    sample.update(change_order_number=f("12", "Change Order No. 12"),
                  project_name=f("Test Tower", "Project: Test Tower"),
                  amount=f(1000.0, "Change order amount: $1,000.00"))
    out = run(text, [sample] * 3)
    assert out["schedule_impact_days"] == {"value": None, "confidence": 0.9, "source": None}
    assert out["contract_number"]["value"] is None


def test_null_contradicted_by_pattern_triggers_review(doc1):
    out = run(doc1, [dict(GOOD_DOC1, amount=f(None, None))] * 3)
    assert out["amount"]["value"] is None  # we report null, never the pattern's guess
    assert out["amount"]["confidence"] < 0.6
    assert out["needs_review"] is True


def test_wrong_sign_is_flagged(doc2):
    sample = {name: f(None, None) for name in FIELDS}
    sample.update(change_order_number=f("17", "COR No.  17"),
                  amount=f(10630.0, "TOTAL CREDIT THIS COR"))
    out = run(doc2, [sample])
    assert out["needs_review"] is True
    assert any("positive but document says" in r for r in out["review_reasons"])


def test_contract_math_conflict_flags_review(doc6):
    sample = {
        "change_order_number": f("GP-1187-CO-03", "CO No.: GP-1187-CO-03"),
        "project_name": f("Mesa Verde Parking Structure", "Job: Mesa Verde Parking Structure"),
        "amount": f(146250.0, "This change order:                       $    146,250.00"),
    }
    out = run(doc6, [sample] * 3)
    assert out["needs_review"] is True
    assert any("contract math" in r for r in out["review_reasons"])


def test_llm_failure_falls_back_to_patterns(doc1):
    out = extract_text(doc1, client=FakeClient(error=RuntimeError("boom")), today=TODAY)
    assert out["meta"]["mode"] == "llm_failed_fallback"
    assert out["needs_review"] is True
    assert out["amount"]["value"] == 18740.0


def test_unparseable_llm_output_falls_back(doc1):
    out = run(doc1, [{}, {}])
    assert out["meta"]["mode"] == "llm_failed_fallback"
    assert out["needs_review"] is True


def test_missing_file_returns_review_not_crash(tmp_path):
    out = extract_file(tmp_path / "nope.pdf")
    assert out["needs_review"] is True
    assert all(out[name]["value"] is None for name in FIELDS)
    jsonschema.validate(out, SCHEMA)


@pytest.mark.parametrize("path", sorted(SYNTH.glob("*.txt")), ids=lambda p: p.name)
def test_offline_output_matches_schema(path):
    jsonschema.validate(extract_file(path, today=TODAY), SCHEMA)


@pytest.mark.parametrize("path", sorted((ROOT / "outputs").rglob("*.json")), ids=lambda p: p.name)
def test_committed_outputs_match_schema(path: Path):
    jsonschema.validate(json.loads(path.read_text()), SCHEMA)
