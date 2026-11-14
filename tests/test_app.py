"""Smoke test for the Streamlit UI (skipped if streamlit isn't installed)."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from tests.conftest import ROOT  # noqa: E402


def test_app_renders_a_sample_offline(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.selectbox[0].select("samples/synthetic/co_06_math_conflict.txt").run()
    assert not at.exception
    assert any("Needs review" in e.value for e in at.error)
    assert any("contract math" in m.value for m in at.markdown)
