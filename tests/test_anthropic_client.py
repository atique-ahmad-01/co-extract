"""Claude client tests with a stubbed SDK: no network, no key, no cost."""
import json
from types import SimpleNamespace

import jsonschema
import pytest

pytest.importorskip("anthropic")

from co_extract import FIELDS  # noqa: E402
from co_extract.anthropic_client import EXTRACTION_SCHEMA, AnthropicClient, ExtractionRefused, _to_pipeline  # noqa: E402
from co_extract.pipeline import extract_text  # noqa: E402
from tests.conftest import TODAY  # noqa: E402

ANSWER = {name: {"present": False, "value": "", "quote": ""} for name in FIELDS} | {
    "change_order_number": {"present": True, "value": "17", "quote": "COR No.  17"},
    "amount": {"present": True, "value": "-10630.00",
               "quote": "TOTAL CREDIT THIS COR                                 ($10,630.00)"},
    "approval_status": {"present": True, "value": "none", "quote": ""},
    "notes": [],
}


def test_schema_has_no_union_types():
    assert "anyOf" not in json.dumps(EXTRACTION_SCHEMA)
    assert '"type": [' not in json.dumps(EXTRACTION_SCHEMA)


def test_absent_fields_become_null():
    out = _to_pipeline(ANSWER)
    assert out["project_name"] == {"value": None, "quote": None}
    assert out["approval_status"] == {"value": None, "quote": None}  # "none" means no indication
    assert out["amount"] == {"value": "-10630.00", "quote": ANSWER["amount"]["quote"]}


class StubMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kw):
        self.calls.append(kw)
        return self.response


def make_client(monkeypatch, response):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c = AnthropicClient()
    stub = StubMessages(response)
    c._client = SimpleNamespace(beta=SimpleNamespace(messages=stub))
    return c, stub


def ok_response(payload):
    return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                           content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def test_schema_accepts_a_full_answer():
    jsonschema.validate(ANSWER, EXTRACTION_SCHEMA)


def test_request_uses_structured_output_and_defaults(monkeypatch):
    c, stub = make_client(monkeypatch, ok_response(ANSWER))
    out = c.extract("doc text", 2)
    assert out == [_to_pipeline(ANSWER)] * 2 and len(stub.calls) == 2
    kw = stub.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in kw
    assert "<document>" in kw["messages"][0]["content"]


def test_refusal_raises(monkeypatch):
    refused = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"), content=[])
    c, _ = make_client(monkeypatch, refused)
    with pytest.raises(ExtractionRefused):
        c.extract("doc", 1)


def test_claude_output_flows_through_pipeline(monkeypatch, doc2):
    c, _ = make_client(monkeypatch, ok_response(ANSWER))
    c.provider = "anthropic"
    out = extract_text(doc2, client=c, n_samples=1, today=TODAY)
    assert out["amount"]["value"] == -10630.0
    assert out["meta"]["provider"] == "anthropic" and out["meta"]["model"] == "claude-opus-5-5"
