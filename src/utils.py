from __future__ import annotations

import re
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(value: str, max_length: int = 120) -> str:
    """Convert a title or identifier into a filesystem-safe filename."""
    cleaned = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_length]
