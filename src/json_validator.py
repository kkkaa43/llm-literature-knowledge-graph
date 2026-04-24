from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class FieldConfidence(BaseModel):
    datasets: float = 0.0
    baselines: float = 0.0
    metrics: float = 0.0
    methodology: float = 0.0


class PaperExtraction(BaseModel):
    paper_id: str = Field(description="Unique paper ID used by this project.")
    arxiv_id: str = ""
    venue_or_source: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    abstract: str = ""
    task: str = ""
    method_name: str = ""
    contributions: list[str] = Field(default_factory=list)
    research_problem: str = ""
    motivation: str = ""
    methodology: str = ""
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    main_results: str = ""
    limitations: str = ""
    future_work: str = ""
    citation_context: str = ""
    related_papers: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    confidence: FieldConfidence = Field(default_factory=FieldConfidence)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    text = raw_text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        text = text[start : end + 1]

    return json.loads(text)


def validate_extraction(data: dict[str, Any]) -> PaperExtraction:
    """Validate extracted data against the project schema."""
    try:
        return PaperExtraction.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"Invalid paper extraction JSON: {error}") from error


def save_extractions_json(
    extractions: list[PaperExtraction],
    output_path: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    """Save validated extractions to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.model_dump() for item in extractions]
    document: Any = rows
    if run_metadata is not None:
        document = {"run_metadata": run_metadata, "papers": rows}
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
