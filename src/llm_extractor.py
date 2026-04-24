from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.json_validator import (
    FieldConfidence,
    PaperExtraction,
    extract_json_object,
    save_extractions_json,
    validate_extraction,
)
from src.run_metadata import extraction_run_metadata
from src.utils import ensure_dir

DEFAULT_PROMPT_PATH = Path("prompts/extraction_prompt.txt")
DEFAULT_METADATA_PATH = Path("data/metadata/arxiv_metadata.json")
REQUIRED_EXTRACTION_FIELDS = [
    "paper_id",
    "arxiv_id",
    "venue_or_source",
    "title",
    "authors",
    "year",
    "abstract",
    "task",
    "method_name",
    "contributions",
    "research_problem",
    "motivation",
    "methodology",
    "datasets",
    "baselines",
    "metrics",
    "main_results",
    "limitations",
    "future_work",
    "citation_context",
    "related_papers",
    "keywords",
    "confidence",
]

SECTION_PATTERNS: dict[str, list[str]] = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction", "intro"],
    "Related Work": ["related work", "background", "prior work"],
    "Method": ["method", "methods", "methodology", "approach", "model", "proposed method"],
    "Experiments": ["experiments", "experimental setup", "evaluation", "results"],
    "Limitations": ["limitations", "limitation", "discussion"],
    "Conclusion": ["conclusion", "conclusions", "future work"],
}

STOP_SECTION_NAMES = {"references", "bibliography", "appendix", "acknowledgements", "acknowledgments"}


@dataclass
class ExtractionFailure:
    paper_id: str
    text_path: str
    error: str


@dataclass
class ExtractionQualityWarning:
    paper_id: str
    field: str
    warning: str


@dataclass
class PaperSection:
    name: str
    text: str


@dataclass
class PaperQualityReport:
    paper_id: str
    quality_score: float
    warning_count: int
    warnings: list[dict[str, str]]
    confidence: dict[str, float]
    extracted_field_count: int
    total_field_count: int


def load_prompt_template(prompt_path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


def load_metadata(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, dict[str, Any]]:
    """Load arXiv metadata and index it by PDF/text file stem."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}

    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        pdf_path = row.get("pdf_path", "")
        if pdf_path:
            indexed[Path(pdf_path).stem] = row
        paper_id = row.get("paper_id")
        if paper_id:
            indexed[paper_id] = row
    return indexed


def truncate_text(text: str, max_chars: int = 25000) -> str:
    """Keep extraction affordable and leave room for instructions."""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3) :]
    return f"{head}\n\n[... middle content truncated ...]\n\n{tail}"


def section_heading_name(line: str) -> str | None:
    """Return a canonical section name when a line looks like a paper heading."""
    cleaned = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+", "", line.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .:-\t").lower()
    if not cleaned or len(cleaned) > 80:
        return None
    if cleaned in STOP_SECTION_NAMES:
        return "References"

    for canonical, aliases in SECTION_PATTERNS.items():
        if cleaned in aliases:
            return canonical
    return None


def split_long_section(section: PaperSection, max_chars: int) -> list[PaperSection]:
    """Split an oversized section on paragraph boundaries without losing its section label."""
    if len(section.text) <= max_chars:
        return [section]

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", section.text) if paragraph.strip()]
    chunks: list[PaperSection] = []
    current: list[str] = []
    current_size = 0
    part = 1

    for paragraph in paragraphs:
        paragraph_size = len(paragraph) + 2
        if current and current_size + paragraph_size > max_chars:
            chunks.append(PaperSection(name=f"{section.name} part {part}", text="\n\n".join(current)))
            current = []
            current_size = 0
            part += 1
        current.append(paragraph)
        current_size += paragraph_size

    if current:
        chunks.append(
            PaperSection(name=f"{section.name} part {part}" if part > 1 else section.name, text="\n\n".join(current))
        )
    return chunks


def detect_sections(text: str, max_chars: int = 25000) -> list[PaperSection]:
    """Detect common academic paper sections and return chunkable section objects."""
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        section_name = section_heading_name(line)
        if section_name == "References":
            headings.append((index, section_name))
            break
        if section_name:
            headings.append((index, section_name))

    if not headings:
        return split_long_section(PaperSection(name="Full Paper", text=text), max_chars=max_chars)

    sections: list[PaperSection] = []
    first_heading = headings[0][0]
    if first_heading > 0:
        front_matter = "\n".join(lines[:first_heading]).strip()
        if front_matter:
            sections.append(PaperSection(name="Front Matter", text=front_matter))

    for position, (start, section_name) in enumerate(headings):
        if section_name == "References":
            break
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[start + 1 : end]).strip()
        if section_text:
            sections.append(PaperSection(name=section_name, text=section_text))

    chunks: list[PaperSection] = []
    for section in sections:
        chunks.extend(split_long_section(section, max_chars=max_chars))
    return chunks or split_long_section(PaperSection(name="Full Paper", text=text), max_chars=max_chars)


def build_prompt(
    paper_id: str,
    paper_text: str,
    metadata: dict[str, Any] | None,
    prompt_template: str,
    max_chars: int,
    section_name: str = "Full Paper",
    extraction_mode: str = "full_paper",
    previous_error: str = "",
) -> str:
    metadata = metadata or {}
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    return prompt_template.format(
        paper_id=paper_id,
        metadata_json=metadata_json,
        paper_text=truncate_text(paper_text, max_chars=max_chars),
        section_name=section_name,
        extraction_mode=extraction_mode,
        previous_error=previous_error,
    )


def missing_schema_fields(data: dict[str, Any]) -> list[str]:
    """Return schema fields missing from a raw model JSON object."""
    return [field for field in REQUIRED_EXTRACTION_FIELDS if field not in data]


def confidence_as_dict(confidence: FieldConfidence | dict[str, Any] | None) -> dict[str, float]:
    if isinstance(confidence, FieldConfidence):
        return confidence.model_dump()
    if isinstance(confidence, dict):
        values: dict[str, float] = {}
        for field in FieldConfidence.model_fields:
            try:
                values[field] = float(confidence.get(field, 0.0) or 0.0)
            except (TypeError, ValueError):
                values[field] = 0.0
        return values
    return FieldConfidence().model_dump()


def normalize_list_items(items: list[Any] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in items or []:
        value = str(item).strip()
        if not value:
            continue
        key = re.sub(r"\s+", " ", value).casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def metadata_seed(paper_id: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    published = str(metadata.get("published") or "")
    return {
        "paper_id": paper_id,
        "arxiv_id": str(metadata.get("arxiv_id") or metadata.get("paper_id") or ""),
        "venue_or_source": str(
            metadata.get("venue_or_source") or metadata.get("entry_url") or "arXiv" if metadata else ""
        ),
        "title": str(metadata.get("title") or ""),
        "authors": normalize_list_items(metadata.get("authors") or []),
        "year": published[:4] if published else "",
        "abstract": str(metadata.get("summary") or ""),
        "task": "",
        "method_name": "",
        "contributions": [],
        "research_problem": "",
        "motivation": str(metadata.get("summary") or ""),
        "methodology": "",
        "datasets": [],
        "baselines": [],
        "metrics": [],
        "main_results": "",
        "limitations": "",
        "future_work": "",
        "citation_context": "",
        "related_papers": [],
        "keywords": normalize_list_items(metadata.get("categories") or []),
        "confidence": FieldConfidence().model_dump(),
    }


def merge_extraction_dicts(
    paper_id: str, metadata: dict[str, Any] | None, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Merge per-section extraction records into one paper-level JSON object."""
    merged = metadata_seed(paper_id, metadata)
    list_fields = {"authors", "contributions", "datasets", "baselines", "metrics", "related_papers", "keywords"}
    text_fields = [
        "arxiv_id",
        "venue_or_source",
        "title",
        "year",
        "abstract",
        "task",
        "method_name",
        "research_problem",
        "motivation",
        "methodology",
        "main_results",
        "limitations",
        "future_work",
        "citation_context",
    ]
    confidence_values = confidence_as_dict(merged.get("confidence"))

    for chunk in chunks:
        for field in text_fields:
            value = str(chunk.get(field) or "").strip()
            current = str(merged.get(field) or "").strip()
            if value and (not current or len(value) > len(current)):
                merged[field] = value

        for field in list_fields:
            merged[field] = normalize_list_items(
                [*merged.get(field, []), *normalize_list_items(chunk.get(field) or [])]
            )

        chunk_confidence = confidence_as_dict(chunk.get("confidence"))
        for field, value in chunk_confidence.items():
            confidence_values[field] = max(confidence_values.get(field, 0.0), value)

    merged["paper_id"] = paper_id
    merged["confidence"] = confidence_values
    return merged


def quality_warnings(extraction: PaperExtraction, methodology_min_words: int = 8) -> list[ExtractionQualityWarning]:
    """Run lightweight quality checks on a validated extraction."""
    warnings: list[ExtractionQualityWarning] = []

    if not extraction.datasets:
        warnings.append(
            ExtractionQualityWarning(
                paper_id=extraction.paper_id,
                field="datasets",
                warning="No datasets were extracted.",
            )
        )

    methodology_words = extraction.methodology.split()
    if len(methodology_words) < methodology_min_words:
        warnings.append(
            ExtractionQualityWarning(
                paper_id=extraction.paper_id,
                field="methodology",
                warning=f"Methodology is shorter than {methodology_min_words} words.",
            )
        )

    return warnings


def quality_score(extraction: PaperExtraction, warnings: list[ExtractionQualityWarning]) -> float:
    """Score extraction completeness and field confidence on a 0-1 scale."""
    important_fields = [
        "title",
        "research_problem",
        "methodology",
        "datasets",
        "baselines",
        "metrics",
        "main_results",
        "limitations",
        "keywords",
    ]
    present = 0
    for field in important_fields:
        value = getattr(extraction, field)
        if isinstance(value, list):
            present += 1 if value else 0
        else:
            present += 1 if str(value).strip() else 0

    completeness = present / len(important_fields)
    confidence = extraction.confidence.model_dump()
    confidence_average = sum(confidence.values()) / len(confidence) if confidence else 0.0
    warning_penalty = min(0.35, 0.08 * len(warnings))
    return round(max(0.0, min(1.0, completeness * 0.75 + confidence_average * 0.25 - warning_penalty)), 3)


def build_quality_report(
    extractions: list[PaperExtraction], warnings: list[ExtractionQualityWarning]
) -> list[PaperQualityReport]:
    warnings_by_paper: dict[str, list[ExtractionQualityWarning]] = {}
    for warning in warnings:
        warnings_by_paper.setdefault(warning.paper_id, []).append(warning)

    reports: list[PaperQualityReport] = []
    total_fields = len(REQUIRED_EXTRACTION_FIELDS)
    for extraction in extractions:
        paper_warnings = warnings_by_paper.get(extraction.paper_id, [])
        dumped = extraction.model_dump()
        extracted_field_count = 0
        for value in dumped.values():
            if isinstance(value, list):
                extracted_field_count += 1 if value else 0
            elif isinstance(value, dict):
                extracted_field_count += 1 if any(float(item or 0.0) > 0 for item in value.values()) else 0
            else:
                extracted_field_count += 1 if str(value).strip() else 0
        reports.append(
            PaperQualityReport(
                paper_id=extraction.paper_id,
                quality_score=quality_score(extraction, paper_warnings),
                warning_count=len(paper_warnings),
                warnings=[{"field": warning.field, "warning": warning.warning} for warning in paper_warnings],
                confidence=extraction.confidence.model_dump(),
                extracted_field_count=extracted_field_count,
                total_field_count=total_fields,
            )
        )
    return reports


def call_openrouter(prompt: str, model: str, temperature: float = 0.0) -> str:
    """Call OpenRouter through the OpenAI-compatible chat completions API."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract structured research-paper information as valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def call_gemini(prompt: str, model: str, temperature: float = 0.0) -> str:
    """Call Gemini with the google-generativeai package."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=api_key)
    generation_config = {"temperature": temperature, "response_mime_type": "application/json"}
    gemini_model = genai.GenerativeModel(model, generation_config=generation_config)
    response = gemini_model.generate_content(prompt)
    return response.text or ""


def mock_extract(paper_id: str, paper_text: str, metadata: dict[str, Any] | None) -> PaperExtraction:
    """Create a deterministic placeholder extraction for pipeline testing."""
    metadata = metadata or {}
    title = metadata.get("title") or first_non_empty_line(paper_text) or paper_id
    authors = metadata.get("authors") or []
    year = ""
    if metadata.get("published"):
        year = str(metadata["published"])[:4]

    datasets = find_known_terms(
        paper_text,
        [
            "Cora",
            "Citeseer",
            "PubMed",
            "CIFAR-10",
            "CIFAR-100",
            "ImageNet",
            "MNIST",
            "COCO",
            "SQuAD",
            "MMLU",
            "GSM8K",
        ],
    )
    baselines = find_known_terms(paper_text, ["GCN", "GraphSAGE", "GAT", "BERT", "ResNet", "LSTM"])
    metrics = find_known_terms(paper_text, ["Accuracy", "F1", "Precision", "Recall", "BLEU", "ROUGE", "AUC"])

    words = [word.lower().strip(".,:;()[]{}") for word in paper_text.split() if len(word) > 5]
    common_keywords = sorted(set(words[:200]))[:8]

    return PaperExtraction(
        paper_id=paper_id,
        arxiv_id=str(metadata.get("arxiv_id") or metadata.get("paper_id") or ""),
        venue_or_source=str(metadata.get("entry_url") or "arXiv" if metadata else ""),
        title=title,
        authors=authors,
        year=year,
        abstract=str(metadata.get("summary") or ""),
        task="node classification" if re_search(r"node classification", paper_text, ignore_case=True) else "",
        method_name=(
            "lightweight attention message passing" if re_search(r"attention", paper_text, ignore_case=True) else ""
        ),
        contributions=["Mock extraction: configure an LLM provider for full contribution extraction."],
        research_problem="Mock extraction: configure an LLM provider for full analysis.",
        motivation="Mock extraction generated without calling an external API.",
        methodology=(
            "The paper proposes a message passing architecture with a lightweight attention mechanism."
            if re_search(r"message passing|attention", paper_text, ignore_case=True)
            else ""
        ),
        datasets=datasets,
        baselines=baselines,
        metrics=metrics,
        main_results="",
        limitations="",
        future_work="",
        citation_context="",
        related_papers=[],
        keywords=common_keywords,
        confidence=FieldConfidence(
            datasets=0.7 if datasets else 0.0,
            baselines=0.7 if baselines else 0.0,
            metrics=0.7 if metrics else 0.0,
            methodology=0.45,
        ),
    )


def find_known_terms(text: str, terms: list[str]) -> list[str]:
    found = []
    for term in terms:
        pattern = r"(?<![A-Za-z0-9-])" + re_escape(term) + r"(?![A-Za-z0-9-])"
        if re_search(pattern, text, ignore_case=True):
            found.append(term)
    return found


def re_escape(value: str) -> str:
    import re

    return re.escape(value)


def re_search(pattern: str, text: str, ignore_case: bool = False) -> bool:
    import re

    flags = re.IGNORECASE if ignore_case else 0
    return bool(re.search(pattern, text, flags=flags))


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def call_provider(provider: str, prompt: str, model: str) -> str:
    if provider == "openrouter":
        return call_openrouter(prompt=prompt, model=model)
    if provider == "gemini":
        return call_gemini(prompt=prompt, model=model)
    raise ValueError("provider must be one of: mock, openrouter, gemini")


def extract_prompt_with_retries(
    *,
    paper_id: str,
    provider: str,
    model: str,
    prompt_builder,
    retries: int,
    sleep_seconds: float,
    require_all_fields: bool = True,
) -> dict[str, Any]:
    last_error: Exception | None = None
    previous_error = ""
    for attempt in range(1, retries + 1):
        try:
            prompt = prompt_builder(previous_error)
            raw_response = call_provider(provider=provider, prompt=prompt, model=model)
            data = extract_json_object(raw_response)
            data["paper_id"] = data.get("paper_id") or paper_id
            if require_all_fields:
                missing_fields = missing_schema_fields(data)
                if missing_fields:
                    raise ValueError(f"Model JSON missing required fields: {', '.join(missing_fields)}")
            return data
        except Exception as error:
            last_error = error
            previous_error = str(error)
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Failed to extract {paper_id}: {last_error}") from last_error


def targeted_retry(
    *,
    paper_id: str,
    paper_text: str,
    current_data: dict[str, Any],
    metadata: dict[str, Any] | None,
    provider: str,
    model: str,
    prompt_template: str,
    max_chars: int,
    retries: int,
    sleep_seconds: float,
    target_fields: list[str],
    reason: str,
) -> dict[str, Any]:
    current_json = json.dumps(current_data, ensure_ascii=False, indent=2)

    def build_retry_prompt(previous_error: str) -> str:
        base_prompt = build_prompt(
            paper_id=paper_id,
            paper_text=paper_text,
            metadata=metadata,
            prompt_template=prompt_template,
            max_chars=max_chars,
            section_name="Targeted Retry",
            extraction_mode="targeted_retry",
            previous_error=previous_error or reason,
        )
        return (
            f"{base_prompt}\n\n"
            f"Current merged extraction:\n{current_json}\n\n"
            f"Target fields to improve: {', '.join(target_fields)}.\n"
            "Return the full JSON schema again, preserving reliable existing values and only changing fields that the text supports."
        )

    return extract_prompt_with_retries(
        paper_id=paper_id,
        provider=provider,
        model=model,
        prompt_builder=build_retry_prompt,
        retries=retries,
        sleep_seconds=sleep_seconds,
        require_all_fields=True,
    )


def extract_one_paper(
    text_path: str | Path,
    provider: str,
    model: str,
    prompt_template: str,
    metadata_index: dict[str, dict[str, Any]],
    max_chars: int,
    retries: int,
    sleep_seconds: float,
) -> PaperExtraction:
    text_path = Path(text_path)
    paper_id = text_path.stem
    paper_text = text_path.read_text(encoding="utf-8", errors="ignore")
    metadata = metadata_index.get(paper_id, {})

    if provider == "mock":
        return mock_extract(paper_id=paper_id, paper_text=paper_text, metadata=metadata)

    sections = detect_sections(paper_text, max_chars=max_chars)
    chunk_results: list[dict[str, Any]] = []
    for section in sections:

        def build_chunk_prompt(previous_error: str, section: PaperSection = section) -> str:
            return build_prompt(
                paper_id=paper_id,
                paper_text=section.text,
                metadata=metadata,
                prompt_template=prompt_template,
                max_chars=max_chars,
                section_name=section.name,
                extraction_mode="section_chunk",
                previous_error=previous_error,
            )

        chunk_results.append(
            extract_prompt_with_retries(
                paper_id=paper_id,
                provider=provider,
                model=model,
                prompt_builder=build_chunk_prompt,
                retries=retries,
                sleep_seconds=sleep_seconds,
                require_all_fields=True,
            )
        )

    merged_data = merge_extraction_dicts(paper_id=paper_id, metadata=metadata, chunks=chunk_results)
    extraction = validate_extraction(merged_data)
    warnings = quality_warnings(extraction)
    targeted_fields = [warning.field for warning in warnings if warning.field in {"datasets", "methodology"}]
    if targeted_fields:
        retry_data = targeted_retry(
            paper_id=paper_id,
            paper_text=paper_text,
            current_data=merged_data,
            metadata=metadata,
            provider=provider,
            model=model,
            prompt_template=prompt_template,
            max_chars=max_chars,
            retries=max(1, min(2, retries)),
            sleep_seconds=sleep_seconds,
            target_fields=sorted(set(targeted_fields)),
            reason="; ".join(warning.warning for warning in warnings if warning.field in targeted_fields),
        )
        merged_data = merge_extraction_dicts(paper_id=paper_id, metadata=metadata, chunks=[merged_data, retry_data])
        extraction = validate_extraction(merged_data)

    return extraction


def save_extractions_csv(extractions: list[PaperExtraction], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    rows = [item.model_dump() for item in extractions]
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            row = row.copy()
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = "; ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)


def save_failures_json(failures: list[ExtractionFailure], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rows = [asdict(item) for item in failures]
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def save_quality_warnings_json(warnings: list[ExtractionQualityWarning], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rows = [asdict(item) for item in warnings]
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def save_quality_report_json(reports: list[PaperQualityReport], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rows = [asdict(item) for item in reports]
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_directory(
    input_dir: str | Path = "data/text",
    output_json: str | Path = "data/extracted/papers.json",
    output_csv: str | Path = "data/extracted/papers.csv",
    failed_json: str | Path = "data/extracted/failed_papers.json",
    warnings_json: str | Path = "data/extracted/extraction_warnings.json",
    quality_report_json: str | Path = "data/extracted/extraction_quality_report.json",
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    provider: str = "mock",
    model: str | None = None,
    max_chars: int = 25000,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    metadata_only: bool = False,
) -> list[PaperExtraction]:
    load_dotenv()

    input_dir = Path(input_dir)
    text_files = sorted(input_dir.glob("*.txt"))
    prompt_template = load_prompt_template(prompt_path)
    metadata_index = load_metadata(metadata_path)
    if metadata_only and metadata_index:
        metadata_stems = {Path(row["pdf_path"]).stem for row in metadata_index.values() if row.get("pdf_path")}
        text_files = [path for path in text_files if path.stem in metadata_stems]

    if provider == "mock":
        model = "mock"
    elif model is None:
        model = {
            "openrouter": "openai/gpt-4o-mini",
            "gemini": "gemini-1.5-flash",
        }[provider]

    extractions: list[PaperExtraction] = []
    failures: list[ExtractionFailure] = []
    warnings: list[ExtractionQualityWarning] = []
    for text_path in tqdm(text_files, desc="Extracting papers"):
        try:
            extraction = extract_one_paper(
                text_path=text_path,
                provider=provider,
                model=model,
                prompt_template=prompt_template,
                metadata_index=metadata_index,
                max_chars=max_chars,
                retries=retries,
                sleep_seconds=sleep_seconds,
            )
            extractions.append(extraction)
            warnings.extend(quality_warnings(extraction))
        except Exception as error:
            failures.append(
                ExtractionFailure(
                    paper_id=Path(text_path).stem,
                    text_path=str(text_path),
                    error=str(error),
                )
            )

    save_extractions_json(
        extractions,
        output_json,
        run_metadata=extraction_run_metadata(provider=provider, model=model, prompt_path=prompt_path),
    )
    save_extractions_csv(extractions, output_csv)
    save_failures_json(failures, failed_json)
    save_quality_warnings_json(warnings, warnings_json)
    save_quality_report_json(build_quality_report(extractions, warnings), quality_report_json)
    return extractions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured information from cleaned paper text.")
    parser.add_argument("--input-dir", default="data/text", help="Directory containing cleaned .txt files.")
    parser.add_argument("--output-json", default="data/extracted/papers.json", help="Output JSON path.")
    parser.add_argument("--output-csv", default="data/extracted/papers.csv", help="Output CSV path.")
    parser.add_argument("--failed-json", default="data/extracted/failed_papers.json", help="Failed papers JSON path.")
    parser.add_argument(
        "--warnings-json",
        default="data/extracted/extraction_warnings.json",
        help="Extraction quality warnings JSON path.",
    )
    parser.add_argument(
        "--quality-report-json",
        default="data/extracted/extraction_quality_report.json",
        help="Per-paper extraction quality report JSON path.",
    )
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH), help="arXiv metadata JSON path.")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH), help="Prompt template path.")
    parser.add_argument("--provider", default="mock", choices=["mock", "openrouter", "gemini"])
    parser.add_argument("--model", default=None, help="Model name for the selected provider.")
    parser.add_argument("--max-chars", type=int, default=25000, help="Maximum paper text characters sent to LLM.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for LLM/API failures.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Base delay between retries.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only process text files referenced by the metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractions = extract_directory(
        input_dir=args.input_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
        failed_json=args.failed_json,
        warnings_json=args.warnings_json,
        quality_report_json=args.quality_report_json,
        metadata_path=args.metadata_path,
        prompt_path=args.prompt_path,
        provider=args.provider,
        model=args.model,
        max_chars=args.max_chars,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        metadata_only=args.metadata_only,
    )
    print(f"Extracted structured information for {len(extractions)} papers.")
    print(f"Saved JSON to: {args.output_json}")
    print(f"Saved CSV to: {args.output_csv}")
    print(f"Saved failures to: {args.failed_json}")
    print(f"Saved warnings to: {args.warnings_json}")
    print(f"Saved quality report to: {args.quality_report_json}")


if __name__ == "__main__":
    main()
