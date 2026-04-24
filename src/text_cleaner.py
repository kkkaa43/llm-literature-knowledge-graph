from __future__ import annotations

import re
import unicodedata
from collections import Counter

SECTION_PATTERNS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    "appendix",
]


def normalize_unicode(text: str) -> str:
    """Normalize unicode while preserving readable scientific text."""
    return unicodedata.normalize("NFKC", text)


def remove_hyphenated_line_breaks(text: str) -> str:
    """Join words split across lines, e.g. 'represen-\ntation'."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_common_page_noise(pages: list[str], edge_lines: int = 3) -> list[str]:
    """
    Remove repeated lines that usually come from page headers or footers.

    The heuristic only considers the first and last few non-empty lines of each
    page, which keeps body text removal conservative.
    """
    if len(pages) < 3:
        return pages

    edge_candidates: list[str] = []
    page_lines: list[list[str]] = []

    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        page_lines.append(lines)
        edge_candidates.extend(lines[:edge_lines])
        edge_candidates.extend(lines[-edge_lines:])

    counts = Counter(edge_candidates)
    min_occurrences = max(3, int(len(pages) * 0.35))
    repeated_noise = {
        line
        for line, count in counts.items()
        if count >= min_occurrences and len(line) <= 140 and not looks_like_section_heading(line)
    }

    cleaned_pages = []
    for lines in page_lines:
        kept = [line for line in lines if line not in repeated_noise and not looks_like_page_number(line)]
        cleaned_pages.append("\n".join(kept))

    return cleaned_pages


def looks_like_page_number(line: str) -> bool:
    """Return True if a line is only a page number-like marker."""
    value = line.strip()
    return bool(re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", value))


def looks_like_section_heading(line: str) -> bool:
    """Detect common academic section headings."""
    normalized = re.sub(r"^\d+(\.\d+)*\s*", "", line.strip().lower())
    normalized = normalized.strip(".: ")
    return normalized in SECTION_PATTERNS


def normalize_section_headings(text: str) -> str:
    """
    Make major section headings easier for later chunking and LLM extraction.

    This keeps the original words but puts common headings on their own line.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if looks_like_section_heading(line):
            lines.append("")
            lines.append(line.upper())
            lines.append("")
        else:
            lines.append(raw_line)
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Clean one full paper text string."""
    text = normalize_unicode(text)
    text = remove_hyphenated_line_breaks(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = normalize_section_headings(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[str]) -> str:
    """Clean extracted page texts and join them into one paper text."""
    pages = [normalize_unicode(page) for page in pages]
    pages = remove_common_page_noise(pages)
    joined = "\n\n".join(pages)
    return clean_text(joined)
