"""Streamlit app: upload or paste a change order and get validated JSON. Run with `make app`."""

from __future__ import annotations

import html
import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from co_extract import FIELDS  # noqa: E402
from co_extract.ingest import IngestError, load_text  # noqa: E402
from co_extract.pipeline import extract_text, review_threshold  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

LABELS = {
    "project_name": "Project", "contract_number": "Contract #", "change_order_number": "Change order #",
    "date": "Date", "requested_by": "Requested by", "description": "Description", "amount": "Amount",
    "schedule_impact_days": "Schedule impact", "approval_status": "Approval status",
}
SAMPLES = sorted((ROOT / "samples").glob("*/*.*"))
SAMPLES = [p for p in SAMPLES if p.suffix.lower() in {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}]

st.set_page_config(page_title="Change Order Extractor", page_icon="🏗️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
section[data-testid="stSidebar"] {min-width: 360px !important; max-width: 360px !important;}
.doc {font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; line-height: 1.5;
      white-space: pre-wrap; background: var(--secondary-background-color); padding: 14px;
      border-radius: 8px; max-height: 640px; overflow: auto;}
.doc mark {border-radius: 3px; padding: 0 2px;}
.doc mark sup {font-size: 9px; font-weight: 600; margin-left: 2px; opacity: .8;}
.hi {background: #c8f0d0; color: #0b3d17;} .mid {background: #fde8b4; color: #4a3500;}
.lo {background: #f8c9c4; color: #5c120b;}
.pill {display:inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600;}
.engine-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; margin: 6px 0 14px;
              background: var(--secondary-background-color); border-left: 4px solid var(--accent);}
.engine-card .title {font-weight: 700; font-size: 15px; display: flex; align-items: center; gap: 8px;}
.engine-card .logo {width: 26px; height: 26px; border-radius: 7px; display: inline-flex; align-items: center;
                    justify-content: center; color: #fff; font-size: 14px; background: var(--accent);}
.engine-card .sub {font-size: 12.5px; opacity: .75; margin: 4px 0 8px;}
.engine-card ul {margin: 0; padding-left: 18px; font-size: 12.5px;} .engine-card li {margin: 2px 0;}
.status {display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
         padding: 2px 9px; border-radius: 999px; margin-top: 8px;}
.status.ok {background: rgba(34,160,80,.14); color: #17803d;} .status.bad {background: rgba(220,60,50,.14); color: #c23a2c;}
.status.neutral {background: rgba(128,128,128,.15);}
.status .dot {width: 7px; height: 7px; border-radius: 50%; background: currentColor;}
.badge {display:inline-flex; align-items:center; gap:6px; padding: 3px 11px; border-radius: 999px; font-size: 13px;
        font-weight: 600; border: 1px solid rgba(128,128,128,.3); background: var(--secondary-background-color);}
</style>""", unsafe_allow_html=True)


def conf_class(c: float, threshold: float) -> str:
    return "hi" if c >= 0.85 else ("mid" if c >= threshold else "lo")


def fmt_value(name: str, v) -> str:
    if v is None:
        return "—"
    if name == "amount":
        return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
    if name == "schedule_impact_days":
        return f"{v:+d} days" if v else "no change"
    return str(v)


def highlight(text: str, result: dict, threshold: float) -> str:
    spans = []
    for i, name in enumerate(FIELDS, 1):
        src = result[name]["source"]
        if src:
            spans.append((src["start"], src["end"], i, name, result[name]["confidence"]))
    spans.sort()
    out, pos = [], 0
    for start, end, i, name, conf in spans:
        if start < pos:
            continue
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark class="{conf_class(conf, threshold)}" title="{LABELS[name]} · {conf:.2f}">'
                   f'{html.escape(text[start:end])}<sup>{i}</sup></mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return '<div class="doc">' + "".join(out).replace("\f", "\n──── page break ────\n") + "</div>"


@st.cache_resource(show_spinner=False)
def get_client(provider: str, model: str, effort: str | None):
    from co_extract.providers import make_client
    return make_client(provider, model=model, effort=effort,
                       cache_dir=os.environ.get("LLM_CACHE_DIR", ROOT / ".cache/llm"))


def read_upload(upload, vision=None) -> tuple[str, str]:
    suffix = Path(upload.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(upload.getvalue())
        tmp = f.name
    try:
        return load_text(tmp, vision=vision)
    finally:
        os.unlink(tmp)


from co_extract.providers import available_providers, default_provider  # noqa: E402

ENGINES = {
    "anthropic": {
        "label": "Claude", "name": "Anthropic Claude", "logo": "✳", "accent": "#d97757",
        "key_env": "ANTHROPIC_API_KEY",
        "sub": "Official Anthropic SDK · structured outputs",
        "points": ["JSON schema enforced by the API: never malformed",
                   "Reads scans & photos (vision) when there's no text layer",
                   "Every value grounded in a verbatim quote"],
        "models": {"claude-opus-5-5": "Opus 5.5 · best default · $4 / $20 per M tokens",
                   "claude-sonnet-5-5": "Sonnet 5.5 · faster · $2 / $10",
                   "claude-haiku-5-5": "Haiku 5.5 · cheapest · $0.10 / $0.50",
                   "claude-fable-5-2": "Fable 5.2 · most capable · $8 / $40"},
        "default_model": os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"),
    },
    "openai": {
        "label": "OpenAI", "name": "OpenAI-compatible", "logo": "◎", "accent": "#10a37f",
        "key_env": "LLM_API_KEY",
        "sub": f"LLM_BASE_URL · {os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')}",
        "points": ["Any OpenAI-compatible endpoint (OpenAI, Azure, Groq, vLLM…)",
                   "JSON mode with tolerant parsing"],
        "models": {"gpt-4o-mini": "gpt-4o-mini · cheap & fast", "gpt-4o": "gpt-4o",
                   "gpt-4.1-mini": "gpt-4.1-mini"},
        "default_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    },
    "offline": {
        "label": "Patterns", "name": "Patterns only", "logo": "⚙", "accent": "#6b7280",
        "key_env": None, "sub": "Deterministic regex extraction · no API call",
        "points": ["Free, instant, works offline", "Lower recall on messy documents",
                   "Same validation; absent fields scored as unknown"],
        "models": {}, "default_model": None,
    },
}
CUSTOM = "Custom…"
providers = available_providers()


def engine_card(key: str, has_key: bool, model: str | None) -> str:
    e = ENGINES[key]
    if e["key_env"] is None:
        status = '<span class="status neutral"><span class="dot"></span>No key needed</span>'
    elif has_key:
        status = f'<span class="status ok"><span class="dot"></span>{e["key_env"]} configured</span>'
    else:
        status = f'<span class="status bad"><span class="dot"></span>{e["key_env"]} missing in .env</span>'
    model_line = f"<div class='sub'><b>Model:</b> <code>{html.escape(model)}</code></div>" if model else ""
    points = "".join(f"<li>{html.escape(p)}</li>" for p in e["points"])
    return (f'<div class="engine-card" style="--accent:{e["accent"]}"><div class="title">'
            f'<span class="logo">{e["logo"]}</span>{e["name"]}</div><div class="sub">{html.escape(e["sub"])}</div>'
            f'{model_line}<ul>{points}</ul>{status}</div>')


with st.sidebar:
    st.markdown("### ⚡ Extraction engine")
    default_engine = (os.environ.get("LLM_PROVIDER") and default_provider()) or \
        ("anthropic" if "anthropic" in providers else default_provider()) or "offline"
    engine = st.segmented_control(
        "Engine", list(ENGINES), default=default_engine, label_visibility="collapsed",
        format_func=lambda k: f'{ENGINES[k]["logo"]} {ENGINES[k]["label"]}', key="engine",
    ) or default_engine
    has_key = engine == "offline" or engine in providers

    model, effort, n_samples = None, None, 1
    if engine != "offline":
        e = ENGINES[engine]
        options = list(e["models"]) + ([e["default_model"]] if e["default_model"] not in e["models"] else []) + [CUSTOM]
        picked = st.selectbox("Model", options, index=options.index(e["default_model"]), key=f"model_{engine}",
                              format_func=lambda m: e["models"].get(m, m))
        model = st.text_input("Model ID", key=f"custom_{engine}") if picked == CUSTOM else picked
        if engine == "anthropic":
            effort = st.select_slider("Effort", ["low", "medium", "high", "xhigh", "max"],
                                      value=os.environ.get("LLM_EFFORT", "medium"),
                                      help="How much the model thinks. Higher = slower and costlier.")
        n_samples = st.slider("Self-consistency samples", 1, 5, int(os.environ.get("LLM_SAMPLES", "3")),
                              help="Independent extractions; disagreement lowers confidence.")

    st.markdown(engine_card(engine, has_key, model), unsafe_allow_html=True)
    if not has_key:
        st.warning(f"Add `{ENGINES[engine]['key_env']}` to `.env` and restart, or pick another engine. "
                   "Running with patterns only for now.")
    st.divider()
    st.caption("Confidence: 🟩 ≥ 0.85 · 🟨 ≥ review threshold · 🟥 below → needs review")

use_llm = engine != "offline" and has_key and bool(model)


client = None
if use_llm:
    try:
        client = get_client(engine, model, effort)
    except Exception as e:
        st.warning(f"LLM unavailable ({e}); using pattern extraction.")

vision = getattr(client, "transcribe_image", None)

st.title("🏗️ Change Order Extractor")
_e = ENGINES[engine if use_llm else "offline"]
st.markdown(f'<span class="badge"><span style="color:{_e["accent"]}">{_e["logo"]}</span>'
            f'{_e["name"]}{" · " + html.escape(model) if use_llm else ""}'
            f'{" · effort " + effort if use_llm and effort else ""}'
            f'{" · " + str(n_samples) + "× samples" if use_llm else ""}</span>', unsafe_allow_html=True)
st.caption("Messy change order in → validated JSON out. Every value is grounded in a quote from the document; "
           "anything not in the document is `null`, never a guess.")

tab_up, tab_paste, tab_sample = st.tabs(["📄 Upload", "📋 Paste text", "🧪 Sample documents"])
source_name, text, ingest = None, None, None
with tab_up:
    upload = st.file_uploader("PDF, scan/photo, or text file", type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp"])
    if not vision:
        st.caption("Tip: choose the **Claude** engine to read scanned pages and photos (vision).")
    if upload:
        try:
            with st.spinner("Reading document" + (" (Claude vision for scanned pages)…" if vision else "…")):
                text, ingest = read_upload(upload, vision)
            source_name = upload.name
        except Exception as e:
            st.error(f"Could not read the file: {e}")
with tab_paste:
    pasted = st.text_area("Paste a change order, email or log entry", height=220)
    if pasted.strip() and st.button("Use pasted text"):
        st.session_state["pasted"] = pasted
    if "pasted" in st.session_state and not upload:
        text, ingest, source_name = st.session_state["pasted"], "text", "pasted text"
with tab_sample:
    choice = st.selectbox("Sample", ["—"] + [str(p.relative_to(ROOT)) for p in SAMPLES])
    if choice != "—" and text is None:
        try:
            with st.spinner("Reading document…"):
                text, ingest = load_text(ROOT / choice, vision=vision)
        except IngestError as e:
            st.error(f"Could not read the file: {e}")
        source_name = choice

if text is None:
    st.info("Upload a file, paste text, or pick a sample to start.")
    st.stop()


with st.spinner("Extracting…"):
    result = extract_text(text, client=client, n_samples=n_samples, source_file=source_name, ingest=ingest)
threshold = review_threshold()


if result["needs_review"]:
    st.error("**Needs review** — " + f"{len(result['review_reasons'])} issue(s)")
    for r in result["review_reasons"]:
        st.markdown(f"- {r}")
else:
    st.success("**Auto-accepted** — all checks passed and every field is above the confidence threshold.")

meta = result["meta"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Amount", fmt_value("amount", result["amount"]["value"]))
m2.metric("Schedule", fmt_value("schedule_impact_days", result["schedule_impact_days"]["value"]))
m3.metric("Status", fmt_value("approval_status", result["approval_status"]["value"]))
m4.metric("Mode", f"{meta['mode']}" + (f" · {meta['samples']}×" if meta["samples"] else ""))

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Extracted fields")
    rows = []
    for i, name in enumerate(FIELDS, 1):
        f = result[name]
        rows.append({
            "#": i, "field": LABELS[name], "value": fmt_value(name, f["value"]),
            "confidence": f["confidence"],
            "evidence": (f["source"]["quote"][:90] + ("…" if len(f["source"]["quote"]) > 90 else ""))
            if f["source"] else "",
            "page": f["source"]["page"] if f["source"] else None,
        })
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={
            "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f"),
            "#": st.column_config.NumberColumn(width="small"),
            "page": st.column_config.NumberColumn(width="small"),
        },
    )
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button("⬇️ Download JSON", json_str, file_name=f"{Path(str(source_name)).stem}.json",
                       mime="application/json", type="primary")
    with st.expander("Raw JSON", expanded=False):
        st.code(json_str, language="json")

with right:
    st.subheader("Document · evidence highlighted")
    st.markdown(highlight(text, result, threshold), unsafe_allow_html=True)
    st.caption(f"Ingest: `{ingest}` · {meta['chars']:,} chars · highlight numbers match the field table")
