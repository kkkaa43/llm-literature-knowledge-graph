from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json_file(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return [] if default is None else default
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_csv_file(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def unwrap_papers(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and isinstance(document.get("papers"), list):
        return document["papers"]
    return []


def unwrap_analysis(document: Any) -> dict[str, Any]:
    if isinstance(document, dict) and isinstance(document.get("analysis"), dict):
        return document["analysis"]
    return document if isinstance(document, dict) else {}


def list_values(papers: list[dict[str, Any]], field: str) -> list[str]:
    values: set[str] = set()
    for paper in papers:
        value = paper.get(field, [])
        if isinstance(value, str):
            value = [value]
        for item in value or []:
            item = str(item).strip()
            if item:
                values.add(item)
    return sorted(values, key=str.lower)


def filter_papers(
    papers: list[dict[str, Any]],
    query: str,
    datasets: list[str],
    baselines: list[str],
    metrics: list[str],
    keywords: list[str],
    tasks: list[str],
) -> list[dict[str, Any]]:
    filtered = []
    needle = query.strip().lower()
    for paper in papers:
        searchable = " ".join(
            str(paper.get(field, ""))
            for field in ["paper_id", "arxiv_id", "title", "abstract", "research_problem", "methodology", "task"]
        ).lower()
        if needle and needle not in searchable:
            continue
        if datasets and not set(datasets).intersection(paper.get("datasets", []) or []):
            continue
        if baselines and not set(baselines).intersection(paper.get("baselines", []) or []):
            continue
        if metrics and not set(metrics).intersection(paper.get("metrics", []) or []):
            continue
        if keywords and not set(keywords).intersection(paper.get("keywords", []) or []):
            continue
        if tasks and paper.get("task", "") not in tasks:
            continue
        filtered.append(paper)
    return filtered


def papers_to_frame(papers: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for paper in papers:
        rows.append(
            {
                "paper_id": paper.get("paper_id", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "task": paper.get("task", ""),
                "method": paper.get("method_name", ""),
                "datasets": ", ".join(paper.get("datasets", []) or []),
                "metrics": ", ".join(paper.get("metrics", []) or []),
                "keywords": ", ".join(paper.get("keywords", []) or []),
            }
        )
    return pd.DataFrame(rows)
