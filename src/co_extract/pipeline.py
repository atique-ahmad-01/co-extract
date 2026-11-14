"""ingest -> extract (LLM samples + patterns) -> ground -> score -> validate."""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass

from . import FIELDS, __version__
from .grounding import Span, locate
from .heuristics import extract_all
from .ingest import load_text
from .normalize import (
    DATE_RE, ID_FIELDS, NO_CHANGE_RE, NUM_RE, NUM_WORDS,
    id_key, normalize_field, normalize_status, parse_date, same_value, token_overlap,
)
from .validate import validate

# Confidence factors; see "Confidence" in the README.
F_FUZZY = 0.85              # quote found only approximately (OCR noise)
F_UNSUPPORTED = 0.6         # quote found, but the value can't be re-derived from it
F_PATTERN_SILENT = 0.9      # regex found nothing to compare against
F_PATTERN_DISAGREE = 0.7
F_OFFLINE = 0.8             # pattern-only mode has no second opinion
F_VISION = 0.85             # source text is an LLM transcription of a scan
NULL_CONFIDENCE = 0.9       # LLM samples agree the field is absent
NULL_UNKNOWN = 0.5          # pattern-only mode: no match is not evidence of absence
NULL_CONTRADICTED = 0.35
UNGROUNDED_CONFIDENCE = 0.2
ISSUE_PENALTY = 0.5


def review_threshold() -> float:
    return float(os.environ.get("REVIEW_THRESHOLD", "0.6"))


@dataclass
class FieldOut:
    value: object
    confidence: float
    source: Span | None = None

    def to_json(self) -> dict:
        return {"value": self.value, "confidence": round(min(1.0, max(0.0, self.confidence)), 3),
                "source": self.source.to_json() if self.source else None}


def _unpack(entry) -> tuple[object, str | None]:
    if isinstance(entry, dict):
        quote = entry.get("quote") or entry.get("source")
        return entry.get("value"), quote if isinstance(quote, str) else None
    return entry, None


def _supported(name: str, value, quote: str) -> bool:
    """Can the normalized value be re-derived from its own evidence span?"""
    q = quote or ""
    if name == "amount":
        nums = [float(m.group().replace(",", "")) for m in NUM_RE.finditer(re.sub(r"\s+", "", q))]
        return any(abs(n - abs(value)) < 0.01 for n in nums)
    if name == "date":
        return any((d := parse_date(m.group())) and d.isoformat() == value for m in DATE_RE.finditer(q))
    if name == "schedule_impact_days":
        if value == 0:
            return bool(NO_CHANGE_RE.search(q) or re.search(r"\b0\b", q) or "zero" in q.lower())
        nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", q)}
        nums |= {float(v) for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\b", q.lower())}
        a = abs(value)
        return a in nums or a / 7 in nums or a / 30 in nums
    if name in ID_FIELDS:
        return id_key(value) in id_key(q)
    if name == "approval_status":
        return normalize_status(q) == value
    if name in ("project_name", "requested_by"):
        return token_overlap(value, q) >= 0.6
    return True


def _resolve(name: str, samples: list[dict], heur, text: str, offline: bool) -> tuple[FieldOut, list[str]]:
    cands = []
    for s in samples:
        raw, quote = _unpack(s.get(name) if isinstance(s, dict) else None)
        try:
            val = normalize_field(name, raw)
        except (TypeError, ValueError):
            val = None
        cands.append((val, quote, raw))
    val, quote, raw = cands[0]
    agreement = sum(same_value(name, v, val) for v, _, _ in cands) / len(cands)
    hval = heur[0] if heur else None

    if val is None:
        if hval is not None and name != "description":
            return FieldOut(None, NULL_CONTRADICTED), [
                f"{name}: extractor returned null but a pattern match found {hval!r}"]
        return FieldOut(None, (NULL_UNKNOWN if offline else NULL_CONFIDENCE) * agreement), []

    span = locate(quote, text) or (locate(raw, text) if isinstance(raw, str) else None)
    if span is None and heur and same_value(name, val, hval):
        span = locate(heur[1], text)
    if span is None:
        return FieldOut(None, UNGROUNDED_CONFIDENCE), [
            f"{name}: value {raw!r} has no supporting evidence in the document; returned null"]

    conf = 1.0 if span.match == "exact" else F_FUZZY
    if not _supported(name, val, span.quote):
        conf *= F_UNSUPPORTED
    if name != "description":
        if hval is None:
            conf *= F_PATTERN_SILENT
        elif not same_value(name, val, hval):
            conf *= F_PATTERN_DISAGREE
    conf *= 0.5 + 0.5 * agreement
    if offline:
        conf *= F_OFFLINE
    return FieldOut(val, conf, span), []


def extract_text(text: str, client=None, n_samples: int = 3, today: dt.date | None = None,
                 source_file: str | None = None, ingest: str | None = None) -> dict:
    reasons: list[str] = []
    samples: list[dict] = []
    has_text = bool(text.strip())
    heur = extract_all(text) if has_text else {}

    mode = "offline"
    if not has_text:
        reasons.append("no text could be extracted from the document")
    elif client is not None:
        try:
            samples = [s for s in client.extract(text, max(1, n_samples)) if isinstance(s, dict) and s]
        except Exception as e:  # a failed call must not stop a batch
            reasons.append(f"LLM call failed ({type(e).__name__}: {str(e)[:200]}); used pattern fallback")
        else:
            if not samples:
                reasons.append("LLM returned no usable JSON; used pattern fallback")
        mode = "llm" if samples else "llm_failed_fallback"
    if not samples:
        samples = [{k: {"value": v[0], "quote": v[1]} for k, v in heur.items() if v}]
    offline = mode != "llm"

    fields: dict[str, FieldOut] = {}
    for name in FIELDS:
        out, notes = _resolve(name, samples, heur.get(name), text, offline)
        fields[name] = out if has_text else FieldOut(None, 0.0)
        reasons.extend(notes)

    if has_text:
        issues = validate({k: f.value for k, f in fields.items()},
                          {k: f.source for k, f in fields.items()}, text, today=today)
        for issue in issues:
            if issue.field in fields:
                fields[issue.field].confidence *= ISSUE_PENALTY
            reasons.append(str(issue))

    if ingest and "vision" in ingest:
        for f in fields.values():
            if f.source is not None:
                f.confidence *= F_VISION

    threshold = review_threshold()
    for name, f in fields.items():
        if f.confidence < threshold and not any(r.startswith(name + ":") for r in reasons):
            reasons.append(f"{name}: low confidence ({f.confidence:.2f})")

    result: dict = {"source_file": source_file}
    result.update({name: fields[name].to_json() for name in FIELDS})
    result["needs_review"] = bool(reasons)
    result["review_reasons"] = reasons
    result["meta"] = {
        "pipeline_version": __version__,
        "mode": mode,
        "provider": getattr(client, "provider", None) if mode == "llm" else None,
        "model": getattr(client, "model", None) if mode == "llm" else None,
        "samples": len(samples) if mode == "llm" else 0,
        "ingest": ingest,
        "chars": len(text),
        "review_threshold": threshold,
    }
    return result


def extract_file(path, client=None, n_samples: int = 3, today: dt.date | None = None) -> dict:
    vision = getattr(client, "transcribe_image", None) if client is not None else None
    try:
        text, method = load_text(path, vision=vision)
    except Exception as e:
        result = extract_text("", source_file=str(path), ingest="failed")
        result["review_reasons"].insert(0, f"ingest failed: {e}")
        return result
    return extract_text(text, client=client, n_samples=n_samples, today=today,
                        source_file=str(path), ingest=method)
