"""Claude client using the official Anthropic SDK with structured outputs."""

from __future__ import annotations

import base64
import json
import os

from .llm import SYSTEM_PROMPT, document_message

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

FIELD_NAMES = (
    "project_name", "contract_number", "change_order_number", "date", "requested_by",
    "description", "amount", "schedule_impact_days", "approval_status",
)

# Structured outputs limit how many nullable fields a schema may have, so every field uses
# plain types plus a `present` flag, and absent fields are converted to null afterwards.
FORMAT_NOTE = """

Response format for this request: every field is {"present": bool, "value": string, "quote": string}.
If the document does not state the field, set present=false, value="" and quote="" (never guess).
Write numbers as plain strings: amount like "-10630.00", schedule_impact_days like "-3" or "0".
approval_status "none" means there is no indication."""

TRANSCRIBE_PROMPT = """Transcribe ALL text in this scanned document page exactly as written, \
preserving line breaks and the reading order of tables (one row per line, cells separated by 3+ spaces).
Copy numbers, IDs and amounts character-for-character. Do not correct, summarize or explain anything.
Write [illegible] for text you cannot read and [signature] for handwritten signatures.
Output only the transcription."""


def _field(value_schema: dict) -> dict:
    return {
        "type": "object",
        "properties": {"present": {"type": "boolean"}, "value": value_schema, "quote": {"type": "string"}},
        "required": ["present", "value", "quote"],
        "additionalProperties": False,
    }


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        **{name: _field({"type": "string"}) for name in FIELD_NAMES if name != "approval_status"},
        "approval_status": _field({"type": "string", "enum": ["approved", "pending", "rejected", "none"]}),
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*FIELD_NAMES, "notes"],
    "additionalProperties": False,
}


def _to_pipeline(data: dict) -> dict:
    out: dict = {"notes": data.get("notes", [])}
    for name in FIELD_NAMES:
        field = data.get(name) or {}
        value = field.get("value")
        if not field.get("present") or value in ("", "none", None):
            out[name] = {"value": None, "quote": None}
        else:
            out[name] = {"value": value, "quote": field.get("quote") or None}
    return out


class ExtractionRefused(RuntimeError):
    pass


class AnthropicClient:
    """Reads ANTHROPIC_API_KEY, ANTHROPIC_MODEL and LLM_EFFORT."""

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 120.0,
                 effort: str | None = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.effort = effort or os.environ.get("LLM_EFFORT", "medium")
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=3)

    def extract(self, text: str, n: int) -> list[dict]:
        return [self._extract_once(text) for _ in range(n)]

    def _extract_once(self, text: str) -> dict:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT + FORMAT_NOTE,
            messages=[{"role": "user", "content": document_message(text)}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
            },
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise ExtractionRefused(f"model declined: {getattr(response.stop_details, 'category', None)}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("response truncated at max_tokens")
        body = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(body)
        return _to_pipeline(data) if isinstance(data, dict) else {}

    def transcribe_image(self, data: bytes, media_type: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": base64.standard_b64encode(data).decode()}},
                {"type": "text", "text": TRANSCRIBE_PROMPT},
            ]}],
        )
        if response.stop_reason == "refusal":
            raise ExtractionRefused("model declined to transcribe the page")
        return "\n".join(b.text for b in response.content if b.type == "text")
