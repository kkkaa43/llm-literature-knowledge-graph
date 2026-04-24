import json
from pathlib import Path

from src import llm_extractor
from src.json_validator import PaperExtraction
from src.llm_extractor import (
    build_quality_report,
    detect_sections,
    extract_directory,
    extract_one_paper,
    merge_extraction_dicts,
    missing_schema_fields,
    quality_warnings,
)


def full_extraction_payload(**overrides):
    payload = {
        "paper_id": "paper-1",
        "arxiv_id": "",
        "venue_or_source": "arXiv",
        "title": "A Paper",
        "authors": [],
        "year": "2024",
        "abstract": "",
        "task": "",
        "method_name": "",
        "contributions": [],
        "research_problem": "",
        "motivation": "",
        "methodology": "",
        "datasets": [],
        "baselines": [],
        "metrics": [],
        "main_results": "",
        "limitations": "",
        "future_work": "",
        "citation_context": "",
        "related_papers": [],
        "keywords": [],
        "confidence": {"datasets": 0.0, "baselines": 0.0, "metrics": 0.0, "methodology": 0.0},
    }
    payload.update(overrides)
    return payload


def test_missing_schema_fields_detects_model_omissions() -> None:
    data = {
        "paper_id": "paper-1",
        "title": "A Paper",
        "authors": [],
    }

    missing = missing_schema_fields(data)

    assert "methodology" in missing
    assert "datasets" in missing
    assert "paper_id" not in missing


def test_quality_warnings_flag_empty_datasets_and_short_methodology() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        title="A Paper",
        methodology="short method",
        datasets=[],
    )

    warnings = quality_warnings(extraction)

    assert {warning.field for warning in warnings} == {"datasets", "methodology"}


def test_detect_sections_splits_common_paper_headings() -> None:
    sections = detect_sections(
        """
        ABSTRACT
        A short abstract.

        INTRODUCTION
        Motivation text.

        METHOD
        Method text.

        EXPERIMENTS
        Experimental text.

        REFERENCES
        [1] Ignored.
        """,
        max_chars=1000,
    )

    assert [section.name for section in sections] == ["Abstract", "Introduction", "Method", "Experiments"]


def test_merge_extraction_dicts_deduplicates_lists_and_maxes_confidence() -> None:
    merged = merge_extraction_dicts(
        paper_id="paper-1",
        metadata={"title": "Metadata Title", "published": "2024-01-01T00:00:00"},
        chunks=[
            full_extraction_payload(
                title="Short",
                datasets=["Cora", "cora", "PubMed"],
                metrics=["Accuracy"],
                confidence={"datasets": 0.4, "metrics": 0.7, "baselines": 0.0, "methodology": 0.2},
            ),
            full_extraction_payload(
                title="A Much Longer Paper Title",
                datasets=["Cora", "Citeseer"],
                baselines=["GCN"],
                confidence={"datasets": 0.9, "metrics": 0.2, "baselines": 0.8, "methodology": 0.6},
            ),
        ],
    )

    assert merged["title"] == "A Much Longer Paper Title"
    assert merged["datasets"] == ["Cora", "PubMed", "Citeseer"]
    assert merged["baselines"] == ["GCN"]
    assert merged["confidence"]["datasets"] == 0.9
    assert merged["confidence"]["metrics"] == 0.7


def test_extract_one_paper_uses_targeted_retry_for_weak_fields(tmp_path: Path, monkeypatch) -> None:
    text_path = tmp_path / "paper-1.txt"
    text_path.write_text(
        """
        ABSTRACT
        This paper studies graph learning.

        METHOD
        Method.
        """,
        encoding="utf-8",
    )
    responses = [
        full_extraction_payload(methodology="short", datasets=[], confidence={"methodology": 0.2}),
        full_extraction_payload(
            methodology="The method uses a careful graph neural architecture with attention and calibration.",
            datasets=["Cora"],
            confidence={"datasets": 0.8, "methodology": 0.9},
        ),
    ]

    def fake_call_provider(**_kwargs):
        return json.dumps(responses.pop(0))

    monkeypatch.setattr(llm_extractor, "call_provider", fake_call_provider)

    extraction = extract_one_paper(
        text_path=text_path,
        provider="openrouter",
        model="test-model",
        prompt_template=Path("prompts/extraction_prompt.txt").read_text(encoding="utf-8"),
        metadata_index={},
        max_chars=1000,
        retries=1,
        sleep_seconds=0,
    )

    assert extraction.datasets == ["Cora"]
    assert extraction.confidence.datasets == 0.8
    assert "careful graph neural architecture" in extraction.methodology


def test_quality_report_scores_extractions() -> None:
    extraction = PaperExtraction(
        paper_id="paper-1",
        title="A Paper",
        methodology="A detailed method with enough words to pass the short methodology check.",
        datasets=["Cora"],
        metrics=["Accuracy"],
        keywords=["graph learning"],
    )
    warnings = quality_warnings(extraction)

    report = build_quality_report([extraction], warnings)

    assert report[0].paper_id == "paper-1"
    assert report[0].quality_score > 0
    assert report[0].warning_count == 0


def test_extract_directory_saves_failures_and_warnings(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "texts"
    input_dir.mkdir()
    (input_dir / "good.txt").write_text("GOOD", encoding="utf-8")
    (input_dir / "bad.txt").write_text("BAD", encoding="utf-8")

    def fake_extract_one_paper(**kwargs):
        text_path = Path(kwargs["text_path"])
        if text_path.stem == "bad":
            raise RuntimeError("simulated extraction failure")
        return PaperExtraction(
            paper_id=text_path.stem,
            title="Good Paper",
            methodology="short",
            datasets=[],
        )

    monkeypatch.setattr(llm_extractor, "extract_one_paper", fake_extract_one_paper)

    output_json = tmp_path / "papers.json"
    output_csv = tmp_path / "papers.csv"
    failed_json = tmp_path / "failed_papers.json"
    warnings_json = tmp_path / "warnings.json"

    extractions = extract_directory(
        input_dir=input_dir,
        output_json=output_json,
        output_csv=output_csv,
        failed_json=failed_json,
        warnings_json=warnings_json,
        provider="mock",
    )

    failures = json.loads(failed_json.read_text(encoding="utf-8"))
    output_document = json.loads(output_json.read_text(encoding="utf-8"))
    warnings = json.loads(warnings_json.read_text(encoding="utf-8"))

    assert output_document["run_metadata"]["schema_version"] == "paper_extraction.v1"
    assert output_document["papers"][0]["paper_id"] == "good"
    assert [extraction.paper_id for extraction in extractions] == ["good"]
    assert failures[0]["paper_id"] == "bad"
    assert "simulated extraction failure" in failures[0]["error"]
    assert {warning["field"] for warning in warnings} == {"datasets", "methodology"}
