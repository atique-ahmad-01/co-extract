import pytest

from co_extract.grounding import locate
from co_extract.ingest import IngestError, load_text
from tests.conftest import SYNTH


def test_text_file_loads():
    text, method = load_text(SYNTH / "co_01_aia_style.txt")
    assert method == "text" and "RMOB-2024-117" in text


def test_missing_file_raises(tmp_path):
    with pytest.raises(IngestError):
        load_text(tmp_path / "missing.pdf")


def test_evidence_spans_cite_page_numbers():
    text = "Page one header\n\f\nCOR No. 17\nTOTAL CREDIT ($500.00)"
    span = locate("TOTAL CREDIT ($500.00)", text)
    assert span.page == 2
    assert locate("Page one header", text).page == 1


def _scanned_pdf(path, text_lines):
    """A PDF whose only page is an image (no text layer), like a scan."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (900, 400), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(text_lines):
        d.text((20, 20 + 30 * i), line, fill="black")
    img.save(path, "PDF")
    return path


def test_scanned_pdf_without_vision_fails_cleanly(tmp_path):
    pdf = _scanned_pdf(tmp_path / "scan.pdf", ["CHANGE ORDER NO. 5"])
    with pytest.raises(IngestError):
        load_text(pdf)


def test_scanned_pdf_pages_are_transcribed_by_vision(tmp_path):
    pdf = _scanned_pdf(tmp_path / "scan.pdf", ["CHANGE ORDER NO. 5"])
    calls = []

    def fake_vision(data, media):
        calls.append(media)
        return "CHANGE ORDER NO. 5\nChange order amount: $1,250.00\nProject: Test Plaza"

    text, method = load_text(pdf, vision=fake_vision)
    assert method == "vision" and calls == ["image/jpeg"]
    assert "$1,250.00" in text


def test_image_file_needs_vision(tmp_path):
    from PIL import Image
    png = tmp_path / "photo.png"
    Image.new("RGB", (50, 50), "white").save(png)
    with pytest.raises(IngestError):
        load_text(png)
    text, method = load_text(png, vision=lambda d, m: "CO #9 amount of this change $900.00 project Alpha")
    assert method == "vision" and "CO #9" in text


def test_vision_sources_get_lower_confidence(tmp_path):
    from co_extract.pipeline import extract_file
    from tests.conftest import FakeClient
    pdf = _scanned_pdf(tmp_path / "scan.pdf", ["x"])
    sample = {"change_order_number": {"value": "5", "quote": "CHANGE ORDER NO. 5"},
              "amount": {"value": 1250.0, "quote": "Change order amount: $1,250.00"},
              "project_name": {"value": "Test Plaza", "quote": "Project: Test Plaza"}}
    client = FakeClient([sample])
    client.transcribe_image = lambda d, m: "CHANGE ORDER NO. 5\nChange order amount: $1,250.00\nProject: Test Plaza"
    out = extract_file(pdf, client=client, n_samples=1)
    assert out["meta"]["ingest"] == "vision"
    assert out["amount"]["value"] == 1250.0
    assert out["amount"]["confidence"] <= 0.85
