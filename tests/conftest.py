import datetime as dt
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "samples" / "synthetic"
TODAY = dt.date(2025, 12, 1)  # pinned so date-range checks are deterministic


def read_sample(name: str) -> str:
    from co_extract.ingest import load_text
    return load_text(SYNTH / name)[0]


class FakeClient:
    """Stands in for the LLM: returns canned samples, records calls."""
    model = "fake-model"

    def __init__(self, samples=None, error=None):
        self.samples = samples or []
        self.error = error
        self.calls = 0

    def extract(self, text, n):
        self.calls += 1
        if self.error:
            raise self.error
        return self.samples[:n]


@pytest.fixture
def doc1():
    return read_sample("co_01_aia_style.txt")


@pytest.fixture
def doc2():
    return read_sample("co_02_credit_pending.txt")


@pytest.fixture
def doc6():
    return read_sample("co_06_math_conflict.txt")
