from pathlib import Path

from src.pipeline import load_pipeline_config, override_config, setup_pipeline_logging


def test_load_pipeline_config_reads_yaml_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        pipeline:
          query: graph neural network
          paper_ids:
            - "1706.03762"
          id_file: examples/regression_set.json
          max_results: 3
          provider: openrouter
          model: openai/gpt-4o-mini
          max_chars: 12000
          metadata_only: true
        """,
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config["query"] == "graph neural network"
    assert config["paper_ids"] == ["1706.03762"]
    assert config["id_file"] == "examples/regression_set.json"
    assert config["max_results"] == 3
    assert config["provider"] == "openrouter"
    assert config["model"] == "openai/gpt-4o-mini"
    assert config["max_chars"] == 12000
    assert config["metadata_only"] is True


def test_override_config_ignores_none_values() -> None:
    config = {"query": "rag", "max_results": 5, "provider": "mock"}

    merged = override_config(config, {"query": None, "max_results": 2, "provider": "gemini"})

    assert merged == {"query": "rag", "max_results": 2, "provider": "gemini"}


def test_setup_pipeline_logging_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    logger = setup_pipeline_logging(log_path, console=False)

    logger.info("hello pipeline")

    assert "hello pipeline" in log_path.read_text(encoding="utf-8")
