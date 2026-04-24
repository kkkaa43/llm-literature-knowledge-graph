from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import requests
from tqdm import tqdm

from src.utils import ensure_dir, safe_filename


@dataclass
class PaperMetadata:
    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    categories: list[str]
    entry_url: str
    pdf_url: str
    pdf_path: str


@dataclass
class DownloadFailure:
    paper_id: str
    title: str
    pdf_url: str
    error: str


SORT_OPTIONS = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submittedDate": arxiv.SortCriterion.SubmittedDate,
    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
}


def retry_with_backoff(operation, *, attempts: int, sleep_seconds: float, operation_name: str):
    """Retry a network operation and raise a concise final error."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"{operation_name} failed after {attempts} attempts. Last error: {last_error}") from last_error


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    retries: int = 3,
    sleep_seconds: float = 3.0,
) -> list[arxiv.Result]:
    """Search arXiv and return result objects."""
    if sort_by not in SORT_OPTIONS:
        valid = ", ".join(SORT_OPTIONS)
        raise ValueError(f"Invalid sort_by='{sort_by}'. Choose one of: {valid}")

    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=SORT_OPTIONS[sort_by],
    )
    return retry_with_backoff(
        lambda: list(client.results(search)),
        attempts=retries,
        sleep_seconds=sleep_seconds,
        operation_name=f"arXiv search for query '{query}'",
    )


def search_arxiv_by_ids(
    paper_ids: Sequence[str],
    retries: int = 3,
    sleep_seconds: float = 3.0,
) -> list[arxiv.Result]:
    """Fetch specific arXiv papers by stable arXiv IDs."""
    cleaned_ids = [paper_id.strip() for paper_id in paper_ids if paper_id.strip()]
    if not cleaned_ids:
        raise ValueError("At least one arXiv ID is required.")

    client = arxiv.Client(page_size=min(len(cleaned_ids), 100), delay_seconds=3, num_retries=3)
    search = arxiv.Search(id_list=cleaned_ids, max_results=len(cleaned_ids))
    return retry_with_backoff(
        lambda: list(client.results(search)),
        attempts=retries,
        sleep_seconds=sleep_seconds,
        operation_name=f"arXiv lookup for IDs {', '.join(cleaned_ids)}",
    )


def load_paper_ids(id_file: str | Path) -> list[str]:
    """Load arXiv IDs from JSON or newline-delimited text."""
    id_file = Path(id_file)
    raw_text = id_file.read_text(encoding="utf-8")
    if id_file.suffix.lower() == ".json":
        rows = json.loads(raw_text)
        if not isinstance(rows, list):
            raise ValueError("ID file JSON must be a list of strings or objects.")

        paper_ids: list[str] = []
        for row in rows:
            if isinstance(row, str):
                paper_ids.append(row)
            elif isinstance(row, dict):
                paper_id = row.get("arxiv_id") or row.get("paper_id") or row.get("id")
                if paper_id:
                    paper_ids.append(str(paper_id))
            else:
                raise ValueError("ID file JSON entries must be strings or objects.")
        return [paper_id.strip() for paper_id in paper_ids if paper_id.strip()]

    return [line.strip() for line in raw_text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def download_pdf(
    pdf_url: str,
    output_path: Path,
    timeout: int = 60,
    retries: int = 3,
    sleep_seconds: float = 1.0,
) -> None:
    """Download one paper PDF."""

    def fetch_pdf() -> bytes:
        response = requests.get(pdf_url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            raise ValueError(f"Downloaded file does not look like a PDF: {pdf_url}")
        return response.content

    content = retry_with_backoff(
        fetch_pdf,
        attempts=retries,
        sleep_seconds=sleep_seconds,
        operation_name=f"PDF download {pdf_url}",
    )
    output_path.write_bytes(content)


def result_to_metadata(result: arxiv.Result, pdf_path: Path) -> PaperMetadata:
    """Convert an arXiv result to serializable metadata."""
    arxiv_id = result.get_short_id()
    return PaperMetadata(
        paper_id=safe_filename(arxiv_id),
        arxiv_id=arxiv_id,
        title=result.title,
        authors=[author.name for author in result.authors],
        summary=result.summary.replace("\n", " ").strip(),
        published=result.published.isoformat(),
        updated=result.updated.isoformat(),
        categories=list(result.categories),
        entry_url=result.entry_id,
        pdf_url=result.pdf_url,
        pdf_path=str(pdf_path),
    )


def save_metadata(metadata: Iterable[PaperMetadata], metadata_dir: Path) -> None:
    """Save metadata as JSON and CSV."""
    ensure_dir(metadata_dir)
    rows = [asdict(item) for item in metadata]

    json_path = metadata_dir / "arxiv_metadata.json"
    csv_path = metadata_dir / "arxiv_metadata.csv"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = row.copy()
            row["authors"] = "; ".join(row["authors"])
            row["categories"] = "; ".join(row["categories"])
            writer.writerow(row)


def save_download_failures(failures: Iterable[DownloadFailure], metadata_dir: Path) -> None:
    ensure_dir(metadata_dir)
    rows = [asdict(item) for item in failures]
    failure_path = metadata_dir / "arxiv_download_failures.json"
    failure_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def download_results(
    results: Iterable[arxiv.Result],
    output_dir: str | Path = "data",
    sleep_seconds: float = 1.0,
    retries: int = 3,
) -> list[PaperMetadata]:
    """Download PDFs for arXiv results and save metadata/failure reports."""
    output_dir = Path(output_dir)
    pdf_dir = ensure_dir(output_dir / "raw_pdfs")
    metadata_dir = ensure_dir(output_dir / "metadata")

    all_metadata: list[PaperMetadata] = []
    failures: list[DownloadFailure] = []

    for result in tqdm(results, desc="Downloading PDFs"):
        arxiv_id = safe_filename(result.get_short_id())
        title_stub = safe_filename(result.title, max_length=80)
        pdf_path = pdf_dir / f"{arxiv_id}_{title_stub}.pdf"

        if not pdf_path.exists():
            try:
                download_pdf(result.pdf_url, pdf_path, retries=retries, sleep_seconds=sleep_seconds)
                time.sleep(sleep_seconds)
            except Exception as error:
                failures.append(
                    DownloadFailure(
                        paper_id=arxiv_id,
                        title=result.title,
                        pdf_url=result.pdf_url,
                        error=str(error),
                    )
                )
                continue

        all_metadata.append(result_to_metadata(result, pdf_path))

    save_metadata(all_metadata, metadata_dir)
    save_download_failures(failures, metadata_dir)
    return all_metadata


def search_and_download(
    query: str | None = None,
    max_results: int = 10,
    output_dir: str | Path = "data",
    sort_by: str = "relevance",
    sleep_seconds: float = 1.0,
    retries: int = 3,
    paper_ids: Sequence[str] | None = None,
    id_file: str | Path | None = None,
) -> list[PaperMetadata]:
    """Search arXiv, download PDFs, and save metadata."""
    requested_ids = list(paper_ids or [])
    if id_file:
        requested_ids.extend(load_paper_ids(id_file))

    if requested_ids:
        results = search_arxiv_by_ids(
            requested_ids,
            retries=retries,
            sleep_seconds=max(sleep_seconds, 1.0),
        )
    else:
        if not query:
            raise ValueError("query is required unless paper_ids or id_file is provided.")
        results = search_arxiv(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            retries=retries,
            sleep_seconds=max(sleep_seconds, 1.0),
        )
    return download_results(
        results,
        output_dir=output_dir,
        sleep_seconds=sleep_seconds,
        retries=retries,
    )


def collect_cli_ids(arxiv_ids: Sequence[str] | None, id_file: str | Path | None) -> list[str]:
    paper_ids = list(arxiv_ids or [])
    if id_file:
        paper_ids.extend(load_paper_ids(id_file))
    return paper_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search arXiv papers and download PDFs.")
    parser.add_argument("--query", default=None, help="Search keyword, e.g. 'graph neural network'.")
    parser.add_argument(
        "--arxiv-id",
        dest="arxiv_ids",
        action="append",
        default=[],
        help="Specific arXiv ID to download. Repeat for multiple papers.",
    )
    parser.add_argument("--id-file", default=None, help="JSON or text file containing fixed arXiv IDs.")
    parser.add_argument("--max-results", type=int, default=10, help="Number of papers to download.")
    parser.add_argument("--output-dir", default="data", help="Output directory for PDFs and metadata.")
    parser.add_argument(
        "--sort-by",
        default="relevance",
        choices=sorted(SORT_OPTIONS),
        help="arXiv sorting criterion.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between PDF downloads.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retry count for arXiv search and PDF downloads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_ids = collect_cli_ids(args.arxiv_ids, args.id_file)
    papers = search_and_download(
        query=args.query,
        max_results=args.max_results,
        output_dir=args.output_dir,
        sort_by=args.sort_by,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        paper_ids=paper_ids,
    )
    print(f"Downloaded or found {len(papers)} papers.")
    print(f"Metadata saved to: {Path(args.output_dir) / 'metadata'}")


if __name__ == "__main__":
    main()
