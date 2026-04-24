from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineSummary:
    downloaded_papers: int = 0
    parsed_pdfs: int = 0
    extracted_records: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "query": None,
    "paper_ids": None,
    "id_file": None,
    "max_results": 5,
    "sort_by": "relevance",
    "data_dir": "data",
    "input_dir": None,
    "output_dir": "outputs",
    "log_path": "outputs/pipeline.log",
    "provider": "mock",
    "model": None,
    "max_chars": 25000,
    "retries": 3,
    "metadata_only": False,
    "overwrite_parse": False,
}


LOGGER_NAME = "literature_pipeline"


def setup_pipeline_logging(log_path: str | Path, console: bool = True) -> logging.Logger:
    """Configure pipeline logging to a file and optional console output."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(console_handler)

    return logger


def load_pipeline_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load pipeline defaults from YAML, falling back to built-in defaults."""
    config_path = Path(config_path)
    config = DEFAULT_PIPELINE_CONFIG.copy()
    if not config_path.exists():
        return config

    import yaml

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pipeline_config = raw_config.get("pipeline", raw_config)
    if not isinstance(pipeline_config, dict):
        raise ValueError("Pipeline config must be a mapping.")

    for key in DEFAULT_PIPELINE_CONFIG:
        if key in pipeline_config:
            config[key] = pipeline_config[key]
    return config


def override_config(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply non-None CLI overrides to a config dictionary."""
    merged = config.copy()
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def run_pipeline(
    query: str | None = None,
    paper_ids: Sequence[str] | None = None,
    id_file: str | Path | None = None,
    max_results: int = 5,
    sort_by: str = "relevance",
    data_dir: str | Path = "data",
    input_dir: str | Path | None = None,
    output_dir: str | Path = "outputs",
    log_path: str | Path | None = None,
    provider: str = "mock",
    model: str | None = None,
    max_chars: int = 25000,
    retries: int = 3,
    metadata_only: bool = False,
    overwrite_parse: bool = False,
    skip_search: bool = False,
    skip_parse: bool = False,
    skip_extract: bool = False,
    skip_graph: bool = False,
) -> PipelineSummary:
    """Run the literature mining pipeline from retrieval through graph export."""
    data_dir = Path(data_dir)
    input_dir = Path(input_dir) if input_dir else data_dir / "text"
    output_dir = Path(output_dir)
    logger = setup_pipeline_logging(log_path) if log_path else logging.getLogger(LOGGER_NAME)

    metadata_path = data_dir / "metadata" / "arxiv_metadata.json"
    extracted_json = data_dir / "extracted" / "papers.json"
    extracted_csv = data_dir / "extracted" / "papers.csv"
    extraction_quality_report = data_dir / "extracted" / "extraction_quality_report.json"

    summary = PipelineSummary()
    effective_model = "mock" if provider == "mock" else model or "default"
    logger.info(
        "Pipeline started: provider=%s model=%s query=%s paper_ids=%s max_results=%s",
        provider,
        effective_model,
        query or "",
        len(paper_ids or []),
        max_results,
    )

    if not skip_search:
        from src.arxiv_downloader import search_and_download

        if not query and not paper_ids and not id_file:
            raise ValueError("--query, --arxiv-id, or --id-file is required unless --skip-search or --sample is used.")
        logger.info("Search phase started.")
        papers = search_and_download(
            query=query,
            paper_ids=paper_ids,
            id_file=id_file,
            max_results=max_results,
            output_dir=data_dir,
            sort_by=sort_by,
            retries=retries,
        )
        summary.downloaded_papers = len(papers)
        logger.info("Search phase complete: downloaded_or_found=%s.", summary.downloaded_papers)
    else:
        logger.info("Search phase skipped.")

    if not skip_parse:
        from src.pdf_parser import parse_pdf_directory

        logger.info("Parse phase started.")
        parsed_paths = parse_pdf_directory(
            pdf_dir=data_dir / "raw_pdfs",
            output_dir=input_dir,
            overwrite=overwrite_parse,
        )
        summary.parsed_pdfs = len(parsed_paths)
        logger.info("Parse phase complete: parsed_pdfs=%s.", summary.parsed_pdfs)
    else:
        logger.info("Parse phase skipped.")

    if not skip_extract:
        from src.llm_extractor import extract_directory

        logger.info("Extract phase started.")
        extractions = extract_directory(
            input_dir=input_dir,
            output_json=extracted_json,
            output_csv=extracted_csv,
            quality_report_json=extraction_quality_report,
            metadata_path=metadata_path,
            provider=provider,
            model=model,
            max_chars=max_chars,
            retries=retries,
            metadata_only=metadata_only,
        )
        summary.extracted_records = len(extractions)
        logger.info("Extract phase complete: extracted_records=%s.", summary.extracted_records)
    else:
        logger.info("Extract phase skipped.")

    if not skip_graph:
        from src.knowledge_graph import build_and_export

        logger.info("Graph phase started.")
        graph = build_and_export(input_path=extracted_json, output_dir=output_dir)
        summary.graph_nodes = graph.number_of_nodes()
        summary.graph_edges = graph.number_of_edges()
        logger.info(
            "Graph phase complete: nodes=%s edges=%s.",
            summary.graph_nodes,
            summary.graph_edges,
        )
    else:
        logger.info("Graph phase skipped.")

    logger.info(
        "Pipeline complete: downloaded=%s parsed=%s extracted=%s graph_nodes=%s graph_edges=%s.",
        summary.downloaded_papers,
        summary.parsed_pdfs,
        summary.extracted_records,
        summary.graph_nodes,
        summary.graph_edges,
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the academic literature mining pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file with pipeline defaults.")
    parser.add_argument("--query", default=None, help="arXiv search query.")
    parser.add_argument(
        "--arxiv-id",
        dest="paper_ids",
        action="append",
        default=None,
        help="Specific arXiv ID to download. Repeat for multiple papers.",
    )
    parser.add_argument("--id-file", default=None, help="JSON or text file containing fixed arXiv IDs.")
    parser.add_argument("--max-results", type=int, default=None, help="Number of arXiv papers to download.")
    parser.add_argument(
        "--sort-by",
        default=None,
        choices=["relevance", "submittedDate", "lastUpdatedDate"],
        help="arXiv sorting criterion.",
    )
    parser.add_argument("--data-dir", default=None, help="Base directory for pipeline data.")
    parser.add_argument("--input-dir", default=None, help="Directory containing cleaned .txt files.")
    parser.add_argument("--output-dir", default=None, help="Directory for graph outputs.")
    parser.add_argument("--log-path", default=None, help="Path for pipeline run logs.")
    parser.add_argument("--provider", default=None, choices=["mock", "openrouter", "gemini"])
    parser.add_argument("--model", default=None, help="Model name for the selected provider.")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum paper text characters sent to LLM.")
    parser.add_argument("--retries", type=int, default=None, help="Retry count for LLM/API failures.")
    parser.add_argument("--metadata-only", action="store_true", help="Only extract papers referenced by metadata.")
    parser.add_argument("--overwrite-parse", action="store_true", help="Overwrite existing parsed text files.")
    parser.add_argument("--skip-search", action="store_true", help="Skip arXiv search and PDF download.")
    parser.add_argument("--skip-parse", action="store_true", help="Skip PDF parsing.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip LLM/mock extraction.")
    parser.add_argument("--skip-graph", action="store_true", help="Skip knowledge graph export.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run against examples/sample_paper.txt without downloading or parsing PDFs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    config = override_config(
        config,
        {
            "query": args.query,
            "paper_ids": args.paper_ids,
            "id_file": args.id_file,
            "max_results": args.max_results,
            "sort_by": args.sort_by,
            "data_dir": args.data_dir,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "log_path": args.log_path,
            "provider": args.provider,
            "model": args.model,
            "max_chars": args.max_chars,
            "retries": args.retries,
            "metadata_only": True if args.metadata_only else None,
            "overwrite_parse": True if args.overwrite_parse else None,
        },
    )

    skip_search = args.skip_search
    skip_parse = args.skip_parse
    input_dir = config["input_dir"]
    if args.sample:
        skip_search = True
        skip_parse = True
        input_dir = input_dir or "examples"

    summary = run_pipeline(
        query=config["query"],
        paper_ids=config["paper_ids"],
        id_file=config["id_file"],
        max_results=config["max_results"],
        sort_by=config["sort_by"],
        data_dir=config["data_dir"],
        input_dir=input_dir,
        output_dir=config["output_dir"],
        log_path=config["log_path"],
        provider=config["provider"],
        model=config["model"],
        max_chars=config["max_chars"],
        retries=config["retries"],
        metadata_only=config["metadata_only"],
        overwrite_parse=config["overwrite_parse"],
        skip_search=skip_search,
        skip_parse=skip_parse,
        skip_extract=args.skip_extract,
        skip_graph=args.skip_graph,
    )

    print("Pipeline complete.")
    print(f"Downloaded papers: {summary.downloaded_papers}")
    print(f"Parsed PDFs: {summary.parsed_pdfs}")
    print(f"Extracted records: {summary.extracted_records}")
    print(f"Graph nodes: {summary.graph_nodes}")
    print(f"Graph edges: {summary.graph_edges}")


if __name__ == "__main__":
    main()
