from co_extract.grounding import locate
from tests.conftest import read_sample


def test_exact_match_is_case_and_whitespace_insensitive(doc1):
    span = locate("contract   no.: rmob-2024-117", doc1)
    assert span is not None and span.match == "exact"
    assert doc1[span.start:span.end] == "CONTRACT NO.: RMOB-2024-117"


def test_fuzzy_match_tolerates_ocr_noise():
    doc3 = read_sample("co_03_ocr_noisy.txt")
    span = locate("Amount of this change: $ 8,410.00", doc3)
    assert span is not None and span.match == "fuzzy"
    assert "8,4l0.00" in span.quote


def test_invented_quote_is_not_found(doc1):
    assert locate("Contract No. ZZZ-999-000", doc1) is None
    assert locate("", doc1) is None
    assert locate(None, doc1) is None
