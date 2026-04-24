import pytest

from src.json_validator import extract_json_object, validate_extraction


def test_extract_json_object_from_fenced_response() -> None:
    data = extract_json_object("""
        Here is the extraction:

        ```json
        {"paper_id": "paper-1", "title": "A Test Paper", "datasets": ["Cora"]}
        ```
        """)

    assert data["paper_id"] == "paper-1"
    assert data["datasets"] == ["Cora"]


def test_validate_extraction_applies_defaults() -> None:
    extraction = validate_extraction({"paper_id": "paper-1", "title": "A Test Paper"})

    assert extraction.paper_id == "paper-1"
    assert extraction.title == "A Test Paper"
    assert extraction.datasets == []
    assert extraction.metrics == []


def test_validate_extraction_requires_paper_id() -> None:
    with pytest.raises(ValueError, match="Invalid paper extraction JSON"):
        validate_extraction({"title": "Missing ID"})
