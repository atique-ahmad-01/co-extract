"""Read a PDF, image or text file into plain text.

PDF pages come from the text layer. Pages with no text but with images (scans), and image
files, are transcribed by a vision-capable LLM when one is supplied. Pages are joined with a
form feed so evidence can be cited by page.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable

Vision = Callable[[bytes, str], str]

MIN_CHARS = 40
SCANNED_PAGE_MAX_CHARS = 20
MAX_IMAGE_SIDE = 2000
MAX_IMAGE_BYTES = 4_000_000
PAGE_BREAK = "\n\f\n"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class IngestError(Exception):
    pass


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = "\n".join(line.rstrip(" \t") for line in s.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def load_text(path, vision: Vision | None = None) -> tuple[str, str]:
    """Return (text, method), where method is text, pdf_text, pdf_text+vision or vision."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(p, vision)
    if suffix in IMAGE_TYPES:
        return _image_text(p, vision)
    raw = p.read_bytes()
    try:
        return clean_text(raw.decode("utf-8")), "text"
    except UnicodeDecodeError:
        return clean_text(raw.decode("cp1252", errors="replace")), "text"


def _pdf_text(p: Path, vision: Vision | None) -> tuple[str, str]:
    import pdfplumber

    with pdfplumber.open(p) as pdf:
        pages = [clean_text(page.extract_text(layout=True) or "") for page in pdf.pages]
        scanned = [i for i, page in enumerate(pdf.pages)
                   if len(pages[i]) < SCANNED_PAGE_MAX_CHARS and page.images]
        if scanned and vision is not None:
            for i in scanned:
                image = pdf.pages[i].to_image(resolution=150).original
                pages[i] = clean_text(vision(_jpeg(image), "image/jpeg"))

    text = PAGE_BREAK.join(pages).strip()
    if len(text.replace("\f", "").strip()) < MIN_CHARS:
        raise IngestError("PDF has no text layer; use the Claude engine to read scanned pages")
    if scanned and vision is not None:
        return text, "vision" if len(scanned) == len(pages) else "pdf_text+vision"
    return text, "pdf_text"


def _image_text(p: Path, vision: Vision | None) -> tuple[str, str]:
    if vision is None:
        raise IngestError("images need the Claude engine (vision)")
    data, media_type = p.read_bytes(), IMAGE_TYPES[p.suffix.lower()]
    if len(data) > MAX_IMAGE_BYTES:
        from PIL import Image
        data, media_type = _jpeg(Image.open(p)), "image/jpeg"
    text = clean_text(vision(data, media_type))
    if len(text) < MIN_CHARS:
        raise IngestError("no readable text in image")
    return text, "vision"


def _jpeg(image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
