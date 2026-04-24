from __future__ import annotations

import html
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import streamlit_data  # noqa: E402

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
EXTRACTED_PATH = DATA_DIR / "extracted" / "papers.json"
EXTRACTED_CSV_PATH = DATA_DIR / "extracted" / "papers.csv"
FAILED_PAPERS_PATH = DATA_DIR / "extracted" / "failed_papers.json"
EXTRACTION_WARNINGS_PATH = DATA_DIR / "extracted" / "extraction_warnings.json"
EXTRACTION_QUALITY_PATH = DATA_DIR / "extracted" / "extraction_quality_report.json"
METADATA_PATH = DATA_DIR / "metadata" / "arxiv_metadata.json"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
TEXT_DIR = DATA_DIR / "text"
GRAPH_ANALYSIS_PATH = OUTPUT_DIR / "graph_analysis.json"
GRAPHML_PATH = OUTPUT_DIR / "graph.graphml"
NODES_CSV_PATH = OUTPUT_DIR / "nodes.csv"
EDGES_CSV_PATH = OUTPUT_DIR / "edges.csv"
NEO4J_CYPHER_PATH = OUTPUT_DIR / "neo4j_import.cypher"
NEO4J_EXAMPLES_PATH = OUTPUT_DIR / "neo4j_example_queries.cypher"
PIPELINE_LOG_PATH = OUTPUT_DIR / "pipeline.log"


st.set_page_config(
    page_title="Literature Knowledge Graph",
    page_icon=":material/library_books:",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetricValue"] { font-size: 1.32rem; }
        .paper-title { font-size: 1.08rem; font-weight: 650; margin-bottom: 0.2rem; }
        .paper-meta { color: #5f6673; font-size: 0.9rem; margin-bottom: 0.8rem; }
        .tag {
            display: inline-block;
            padding: 0.16rem 0.48rem;
            margin: 0.1rem 0.16rem 0.1rem 0;
            border: 1px solid #d8dde6;
            border-radius: 999px;
            background: #f7f8fa;
            font-size: 0.78rem;
            color: #303642;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json(path: str, modified_time: float) -> Any:
    file_path = Path(path)
    return streamlit_data.load_json_file(file_path, default=[] if file_path.suffix == ".json" else {})


@st.cache_data(show_spinner=False)
def load_csv(path: str, modified_time: float) -> pd.DataFrame:
    return streamlit_data.load_csv_file(path)


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


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


def render_tags(items: list[str]) -> None:
    if not items:
        st.caption("None")
        return
    tag_html = "".join(f'<span class="tag">{html.escape(str(item))}</span>' for item in items)
    st.markdown(tag_html, unsafe_allow_html=True)


def default_model(provider: str) -> str:
    defaults = {
        "mock": "mock",
        "openrouter": "openai/gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
    }
    return defaults[provider]


def render_download_button(path: Path, label: str, mime: str) -> None:
    if not path.exists():
        st.button(label, disabled=True, width="stretch")
        return

    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        width="stretch",
    )


def render_native_graph(nodes: pd.DataFrame, edges: pd.DataFrame, show_labels: bool) -> None:
    if nodes.empty:
        st.info("Build the graph to create graph data.")
        return

    graph = nx.Graph()
    visible_node_ids = set(nodes["id"].astype(str))
    for node_id in visible_node_ids:
        graph.add_node(node_id)
    if not edges.empty:
        for _, edge in edges.iterrows():
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in visible_node_ids and target in visible_node_ids:
                graph.add_edge(source, target)

    if graph.number_of_nodes() == 0:
        st.info("No graph nodes match the current filters.")
        return

    positions = nx.spring_layout(graph, seed=42, k=0.9)
    node_rows = []
    for _, row in nodes.iterrows():
        node_id = str(row.get("id", ""))
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_rows.append(
            {
                "id": node_id,
                "label": str(row.get("label", node_id)),
                "type": str(row.get("type", "")),
                "degree": int(row.get("degree", 0) or 0),
                "x": x,
                "y": y,
            }
        )

    edge_rows = []
    if not edges.empty:
        for edge_index, edge in edges.iterrows():
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in positions or target not in positions:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            edge_rows.append(
                {
                    "edge": edge_index,
                    "x": x1,
                    "y": y1,
                    "relation": str(edge.get("relation", "")),
                    "order": 0,
                }
            )
            edge_rows.append(
                {
                    "edge": edge_index,
                    "x": x2,
                    "y": y2,
                    "relation": str(edge.get("relation", "")),
                    "order": 1,
                }
            )

    node_frame = pd.DataFrame(node_rows)
    edge_frame = pd.DataFrame(edge_rows)

    edge_layer = (
        alt.Chart(edge_frame)
        .mark_line(color="#c6ccd6", opacity=0.55)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            detail="edge:N",
            order="order:Q",
            tooltip=["relation:N"],
        )
        if not edge_frame.empty
        else alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
    )
    node_layer = (
        alt.Chart(node_frame)
        .mark_circle(opacity=0.92, stroke="#ffffff", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            size=alt.Size("degree:Q", scale=alt.Scale(range=[120, 900]), legend=None),
            color=alt.Color("type:N", legend=alt.Legend(title="Node type")),
            tooltip=["label:N", "type:N", "degree:Q"],
        )
    )
    chart = edge_layer + node_layer
    if show_labels:
        text_layer = (
            alt.Chart(node_frame)
            .mark_text(dy=16, fontSize=11, color="#293241", limit=120)
            .encode(x="x:Q", y="y:Q", text="label:N")
        )
        chart += text_layer
    chart = chart.properties(height=620).interactive()
    st.altair_chart(chart, width="stretch")


def run_streamlit_pipeline(
    query: str,
    max_results: int,
    provider: str,
    model: str,
    max_chars: int,
    metadata_only: bool,
    sample_mode: bool,
) -> None:
    from src.pipeline import run_pipeline

    summary = run_pipeline(
        query=query,
        max_results=max_results,
        input_dir="examples" if sample_mode else None,
        provider=provider,
        model=model if provider != "mock" else None,
        max_chars=max_chars,
        metadata_only=metadata_only,
        skip_search=sample_mode,
        skip_parse=sample_mode,
        log_path=PIPELINE_LOG_PATH,
    )
    st.session_state["last_pipeline_summary"] = summary


def ranking_frame(analysis: dict[str, Any], key: str) -> pd.DataFrame:
    return pd.DataFrame(analysis.get(key, []) or [])


def fallback_ranking(papers: list[dict[str, Any]], field: str) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for paper in papers:
        for item in paper.get(field, []) or []:
            counter[str(item)] += 1
    return pd.DataFrame([{"label": label, "count": count} for label, count in counter.most_common(25)])


def render_detail(paper: dict[str, Any], quality_by_id: dict[str, dict[str, Any]]) -> None:
    title = html.escape(str(paper.get("title", "Untitled")))
    st.markdown(f'<div class="paper-title">{title}</div>', unsafe_allow_html=True)
    authors = html.escape(", ".join(paper.get("authors", []) or []))
    year = html.escape(str(paper.get("year", "")))
    arxiv_id = html.escape(str(paper.get("arxiv_id", "")))
    st.markdown(f'<div class="paper-meta">{year} · {arxiv_id} · {authors}</div>', unsafe_allow_html=True)

    quality = quality_by_id.get(str(paper.get("paper_id", "")), {})
    cols = st.columns(4)
    cols[0].metric("Quality", quality.get("quality_score", "n/a"))
    cols[1].metric("Warnings", quality.get("warning_count", 0))
    cols[2].metric("Datasets", len(paper.get("datasets", []) or []))
    cols[3].metric("Metrics", len(paper.get("metrics", []) or []))

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Abstract")
        st.write(paper.get("abstract") or paper.get("motivation") or "")
        st.subheader("Research Problem")
        st.write(paper.get("research_problem", ""))
        st.subheader("Methodology")
        st.write(paper.get("methodology", ""))
        st.subheader("Main Results")
        st.write(paper.get("main_results", ""))
        st.subheader("Limitations")
        st.write(paper.get("limitations", ""))
        st.subheader("Future Work")
        st.write(paper.get("future_work", ""))

    with right:
        st.subheader("Task")
        st.write(paper.get("task", ""))
        st.subheader("Method")
        st.write(paper.get("method_name", ""))
        st.subheader("Contributions")
        render_tags(paper.get("contributions", []) or [])
        st.subheader("Datasets")
        render_tags(paper.get("datasets", []) or [])
        st.subheader("Baselines")
        render_tags(paper.get("baselines", []) or [])
        st.subheader("Metrics")
        render_tags(paper.get("metrics", []) or [])
        st.subheader("Related Papers")
        render_tags(paper.get("related_papers", []) or [])
        st.subheader("Keywords")
        render_tags(paper.get("keywords", []) or [])
        st.subheader("Citation Context")
        st.write(paper.get("citation_context", ""))


def main() -> None:
    inject_css()

    st.title("LLM-Powered Academic Literature Knowledge Graph")

    with st.sidebar:
        st.header("Pipeline")
        sample_mode = st.checkbox("Use sample paper", value=True)
        query = st.text_input("arXiv keyword", value="retrieval augmented generation", disabled=sample_mode)
        max_results = st.number_input("Max papers", min_value=1, max_value=50, value=5, step=1, disabled=sample_mode)
        provider = st.selectbox("Provider", ["mock", "openrouter", "gemini"], index=0)
        model = st.text_input("Model", value=default_model(provider), disabled=provider == "mock")
        max_chars = st.number_input("Max chars per chunk", min_value=2000, max_value=100000, value=25000, step=1000)
        metadata_only = st.checkbox("Metadata-only extraction", value=provider != "mock", disabled=sample_mode)

        if st.button("Run Pipeline", type="primary", width="stretch"):
            with st.spinner("Running pipeline..."):
                run_streamlit_pipeline(
                    query=query,
                    max_results=int(max_results),
                    provider=provider,
                    model=model,
                    max_chars=int(max_chars),
                    metadata_only=metadata_only,
                    sample_mode=sample_mode,
                )
            st.success("Pipeline complete.")
            st.cache_data.clear()

        summary = st.session_state.get("last_pipeline_summary")
        if summary:
            st.caption(
                f"Last run: {summary.extracted_records} records, "
                f"{summary.graph_nodes} graph nodes, {summary.graph_edges} graph edges"
            )

    papers_document = load_json(str(EXTRACTED_PATH), mtime(EXTRACTED_PATH))
    papers = streamlit_data.unwrap_papers(papers_document)
    quality_report = load_json(str(EXTRACTION_QUALITY_PATH), mtime(EXTRACTION_QUALITY_PATH))
    warnings = load_json(str(EXTRACTION_WARNINGS_PATH), mtime(EXTRACTION_WARNINGS_PATH))
    failures = load_json(str(FAILED_PAPERS_PATH), mtime(FAILED_PAPERS_PATH))
    metadata = load_json(str(METADATA_PATH), mtime(METADATA_PATH))
    analysis_document = load_json(str(GRAPH_ANALYSIS_PATH), mtime(GRAPH_ANALYSIS_PATH))
    analysis = streamlit_data.unwrap_analysis(analysis_document)
    nodes = load_csv(str(NODES_CSV_PATH), mtime(NODES_CSV_PATH))
    edges = load_csv(str(EDGES_CSV_PATH), mtime(EDGES_CSV_PATH))
    quality_by_id = {str(row.get("paper_id", "")): row for row in quality_report or []}

    tab_search, tab_parse, tab_extract, tab_graph, tab_results = st.tabs(
        ["Search", "Parse", "Extract", "Graph", "Results"]
    )

    with tab_search:
        cols = st.columns(4)
        cols[0].metric("Metadata Records", len(metadata) if isinstance(metadata, list) else 0)
        cols[1].metric("Raw PDFs", len(list(RAW_PDF_DIR.glob("*.pdf"))) if RAW_PDF_DIR.exists() else 0)
        cols[2].metric("Parsed Texts", len(list(TEXT_DIR.glob("*.txt"))) if TEXT_DIR.exists() else 0)
        cols[3].metric("Extracted Papers", len(papers))
        if isinstance(metadata, list) and metadata:
            st.dataframe(
                pd.DataFrame(metadata)[["paper_id", "arxiv_id", "title", "published", "pdf_path"]],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Run a search or use the sample pipeline to populate metadata.")

    with tab_parse:
        raw_pdfs = sorted(RAW_PDF_DIR.glob("*.pdf")) if RAW_PDF_DIR.exists() else []
        text_files = sorted(TEXT_DIR.glob("*.txt")) if TEXT_DIR.exists() else []
        cols = st.columns(3)
        cols[0].metric("PDF Files", len(raw_pdfs))
        cols[1].metric("Text Files", len(text_files))
        cols[2].metric("Sample Available", "yes" if Path("examples/sample_paper.txt").exists() else "no")
        if text_files:
            selected_text = st.selectbox("Parsed text", [path.name for path in text_files])
            selected_path = TEXT_DIR / selected_text
            st.code(selected_path.read_text(encoding="utf-8", errors="ignore")[:5000], language="text")
        else:
            st.info("Parsed text files will appear in data/text after the parse phase.")

    with tab_extract:
        if papers:
            search_query = st.text_input("Search extracted papers", value="")
            filter_cols = st.columns(5)
            dataset_filter = filter_cols[0].multiselect("Dataset", list_values(papers, "datasets"))
            baseline_filter = filter_cols[1].multiselect("Baseline", list_values(papers, "baselines"))
            metric_filter = filter_cols[2].multiselect("Metric", list_values(papers, "metrics"))
            keyword_filter = filter_cols[3].multiselect("Keyword", list_values(papers, "keywords"))
            task_filter = filter_cols[4].multiselect("Task", list_values(papers, "task"))
        else:
            search_query = ""
            dataset_filter = []
            baseline_filter = []
            metric_filter = []
            keyword_filter = []
            task_filter = []

        filtered = filter_papers(
            papers,
            query=search_query,
            datasets=dataset_filter,
            baselines=baseline_filter,
            metrics=metric_filter,
            keywords=keyword_filter,
            tasks=task_filter,
        )

        cols = st.columns(4)
        cols[0].metric("Papers", len(filtered))
        cols[1].metric("Datasets", len(list_values(filtered, "datasets")))
        cols[2].metric("Metrics", len(list_values(filtered, "metrics")))
        cols[3].metric("Warnings", len(warnings or []))

        if filtered:
            st.dataframe(papers_to_frame(filtered), hide_index=True, width="stretch")
        else:
            st.info("Run extraction or adjust filters.")

        quality_tabs = st.tabs(["Quality Report", "Warnings", "Failures"])
        with quality_tabs[0]:
            if quality_report:
                st.dataframe(pd.DataFrame(quality_report), hide_index=True, width="stretch")
            else:
                st.info("Quality report will appear at data/extracted/extraction_quality_report.json.")
        with quality_tabs[1]:
            st.dataframe(pd.DataFrame(warnings or []), hide_index=True, width="stretch")
        with quality_tabs[2]:
            st.dataframe(pd.DataFrame(failures or []), hide_index=True, width="stretch")

    with tab_graph:
        graph_cols = st.columns(4)
        graph_cols[0].metric("Nodes", len(nodes))
        graph_cols[1].metric("Edges", len(edges))
        graph_cols[2].metric(
            "Datasets", int((nodes["type"] == "Dataset").sum()) if not nodes.empty and "type" in nodes else 0
        )
        graph_cols[3].metric(
            "Methods", int((nodes["type"] == "Method").sum()) if not nodes.empty and "type" in nodes else 0
        )

        graph_view, graph_analysis_tab, graph_exports = st.tabs(["Graph View", "Analysis", "Neo4j"])
        with graph_view:
            filter_cols = st.columns(5)
            available_types = sorted(nodes["type"].dropna().unique()) if not nodes.empty and "type" in nodes else []
            node_types = filter_cols[0].multiselect("Node type", available_types, default=available_types)
            keyword = filter_cols[1].selectbox("Keyword focus", [""] + list_values(papers, "keywords"))
            min_degree = filter_cols[2].number_input("Minimum degree", min_value=0, value=0, step=1)
            top_nodes = filter_cols[3].number_input("Top nodes", min_value=5, max_value=250, value=60, step=5)
            show_labels = filter_cols[4].checkbox("Show labels", value=False)

            filtered_nodes = nodes.copy()
            if not filtered_nodes.empty:
                if node_types:
                    filtered_nodes = filtered_nodes[filtered_nodes["type"].isin(node_types)]
                if "degree" in filtered_nodes:
                    filtered_nodes = filtered_nodes[filtered_nodes["degree"].fillna(0).astype(int) >= int(min_degree)]
                if keyword:
                    related_paper_ids = {
                        paper.get("paper_id") for paper in papers if keyword in (paper.get("keywords", []) or [])
                    }
                    filtered_nodes = filtered_nodes[
                        filtered_nodes["label"].astype(str).str.contains(keyword, case=False, na=False)
                        | filtered_nodes["paper_id"].isin(related_paper_ids)
                    ]
                if "degree" in filtered_nodes:
                    filtered_nodes = filtered_nodes.sort_values(["degree", "label"], ascending=[False, True]).head(
                        int(top_nodes)
                    )
                st.dataframe(filtered_nodes, hide_index=True, width="stretch")

            filtered_edges = edges.copy()
            if not filtered_nodes.empty and not filtered_edges.empty:
                visible_ids = set(filtered_nodes["id"].astype(str))
                filtered_edges = filtered_edges[
                    filtered_edges["source"].astype(str).isin(visible_ids)
                    & filtered_edges["target"].astype(str).isin(visible_ids)
                ]
            render_native_graph(filtered_nodes, filtered_edges, show_labels=show_labels)

            if not filtered_nodes.empty:
                labels = {
                    f"{row.get('label', '')} ({row.get('type', '')})": row.to_dict()
                    for _, row in filtered_nodes.iterrows()
                }
                selected_node = st.selectbox("Node detail", list(labels))
                st.json(labels[selected_node], expanded=False)

        with graph_analysis_tab:
            rank_tabs = st.tabs(["Datasets", "Metrics", "Keywords", "Connected Papers", "Shared Benchmarks"])
            with rank_tabs[0]:
                frame = ranking_frame(analysis, "most_used_datasets")
                st.dataframe(
                    frame if not frame.empty else fallback_ranking(papers, "datasets"),
                    hide_index=True,
                    width="stretch",
                )
            with rank_tabs[1]:
                frame = ranking_frame(analysis, "most_common_metrics")
                st.dataframe(
                    frame if not frame.empty else fallback_ranking(papers, "metrics"),
                    hide_index=True,
                    width="stretch",
                )
            with rank_tabs[2]:
                frame = ranking_frame(analysis, "most_common_keywords")
                st.dataframe(
                    frame if not frame.empty else fallback_ranking(papers, "keywords"),
                    hide_index=True,
                    width="stretch",
                )
            with rank_tabs[3]:
                st.dataframe(
                    pd.DataFrame(analysis.get("most_connected_papers", []) if isinstance(analysis, dict) else []),
                    hide_index=True,
                    width="stretch",
                )
            with rank_tabs[4]:
                st.dataframe(
                    pd.DataFrame(analysis.get("shared_benchmark_papers", []) if isinstance(analysis, dict) else []),
                    hide_index=True,
                    width="stretch",
                )

        with graph_exports:
            st.code(
                (
                    NEO4J_EXAMPLES_PATH.read_text(encoding="utf-8")
                    if NEO4J_EXAMPLES_PATH.exists()
                    else "Build graph outputs to create Neo4j query examples."
                ),
                language="cypher",
            )

    with tab_results:
        detail, downloads, logs = st.tabs(["Paper Detail", "Downloads", "Run Log"])
        with detail:
            options = {f"{paper.get('title', 'Untitled')} ({paper.get('paper_id', '')})": paper for paper in papers}
            if not options:
                st.warning("No extracted papers found.")
            else:
                selected_label = st.selectbox("Paper", list(options))
                render_detail(options[selected_label], quality_by_id)

        with downloads:
            download_cols = st.columns(3)
            with download_cols[0]:
                render_download_button(EXTRACTED_PATH, "Download JSON", "application/json")
            with download_cols[1]:
                render_download_button(EXTRACTED_CSV_PATH, "Download CSV", "text/csv")
            with download_cols[2]:
                render_download_button(EXTRACTION_QUALITY_PATH, "Download quality report", "application/json")

            graph_download_cols = st.columns(3)
            with graph_download_cols[0]:
                render_download_button(GRAPHML_PATH, "Download GraphML", "application/xml")
            with graph_download_cols[1]:
                render_download_button(NODES_CSV_PATH, "Download nodes CSV", "text/csv")
            with graph_download_cols[2]:
                render_download_button(EDGES_CSV_PATH, "Download edges CSV", "text/csv")

            extra_cols = st.columns(3)
            with extra_cols[0]:
                render_download_button(FAILED_PAPERS_PATH, "Download failures", "application/json")
            with extra_cols[1]:
                render_download_button(EXTRACTION_WARNINGS_PATH, "Download warnings", "application/json")
            with extra_cols[2]:
                render_download_button(NEO4J_CYPHER_PATH, "Download Neo4j Cypher", "text/plain")

        with logs:
            if PIPELINE_LOG_PATH.exists():
                st.code(PIPELINE_LOG_PATH.read_text(encoding="utf-8")[-6000:], language="text")
            else:
                st.info("Run the pipeline to create outputs/pipeline.log.")


if __name__ == "__main__":
    main()
