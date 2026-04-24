from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PAPER_EXTRACTION_SCHEMA_VERSION = "paper_extraction.v1"
GRAPH_ANALYSIS_SCHEMA_VERSION = "graph_analysis.v1"


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for generated artifacts."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def extraction_run_metadata(
    *,
    provider: str,
    model: str,
    prompt_path: str | Path,
    schema_version: str = PAPER_EXTRACTION_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "prompt_version": file_sha256(prompt_path),
        "prompt_path": str(prompt_path),
        "provider": provider,
        "model": model,
        "run_timestamp": utc_now_iso(),
    }


def graph_run_metadata(source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    source_metadata = source_metadata or {}
    return {
        "schema_version": GRAPH_ANALYSIS_SCHEMA_VERSION,
        "source_schema_version": source_metadata.get("schema_version", ""),
        "prompt_version": source_metadata.get("prompt_version", ""),
        "provider": source_metadata.get("provider", ""),
        "model": source_metadata.get("model", ""),
        "run_timestamp": utc_now_iso(),
    }
