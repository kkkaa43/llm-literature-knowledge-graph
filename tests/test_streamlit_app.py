import json
from pathlib import Path

from src import streamlit_data


def test_streamlit_load_json_returns_papers(tmp_path: Path) -> None:
    path = tmp_path / "papers.json"
    path.write_text(json.dumps([{"paper_id": "paper-1", "title": "A Paper"}]), encoding="utf-8")

    papers = streamlit_data.load_json_file(path)

    assert papers[0]["paper_id"] == "paper-1"


def test_streamlit_unwraps_versioned_paper_document() -> None:
    papers = streamlit_data.unwrap_papers(
        {
            "run_metadata": {"schema_version": "paper_extraction.v1"},
            "papers": [{"paper_id": "paper-1"}],
        }
    )

    assert papers == [{"paper_id": "paper-1"}]


def test_streamlit_unwraps_versioned_analysis_document() -> None:
    analysis = streamlit_data.unwrap_analysis(
        {
            "run_metadata": {"schema_version": "graph_analysis.v1"},
            "analysis": {"most_used_datasets": [{"label": "Cora", "count": 1}]},
        }
    )

    assert analysis["most_used_datasets"][0]["label"] == "Cora"


def test_streamlit_papers_to_frame_includes_new_fields() -> None:
    frame = streamlit_data.papers_to_frame(
        [
            {
                "paper_id": "paper-1",
                "arxiv_id": "2401.00001",
                "title": "A Paper",
                "task": "node classification",
                "method_name": "Message Passing",
                "datasets": ["Cora"],
                "metrics": ["Accuracy"],
            }
        ]
    )

    assert frame.loc[0, "arxiv_id"] == "2401.00001"
    assert frame.loc[0, "task"] == "node classification"
    assert frame.loc[0, "method"] == "Message Passing"
