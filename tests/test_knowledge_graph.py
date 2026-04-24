import csv
import json

from src.knowledge_graph import (
    build_and_export,
    build_knowledge_graph,
    graph_analysis,
    load_alias_dictionary,
    load_papers,
    node_id,
)


def sample_papers() -> list[dict[str, object]]:
    return [
        {
            "paper_id": "paper-1",
            "title": "Graph Paper",
            "year": "2024",
            "methodology": "The paper proposes a message passing method. It has two stages.",
            "datasets": ["Cora", "Citeseer"],
            "baselines": ["GCN"],
            "metrics": ["Accuracy"],
            "keywords": ["node classification", "citation networks"],
        }
    ]


def test_build_knowledge_graph_counts_nodes_and_edges() -> None:
    graph = build_knowledge_graph(sample_papers())

    assert graph.number_of_nodes() == 8
    assert graph.number_of_edges() == 7


def test_build_knowledge_graph_adds_expected_relationships() -> None:
    graph = build_knowledge_graph(sample_papers())
    paper_node = node_id("Paper", "paper-1")

    assert graph.edges[paper_node, node_id("Dataset", "Cora")]["relation"] == "uses"
    assert graph.edges[paper_node, node_id("Baseline", "GCN")]["relation"] == "compares_with"
    assert graph.edges[paper_node, node_id("Metric", "Accuracy")]["relation"] == "evaluated_by"
    assert graph.edges[paper_node, node_id("Keyword", "node classification")]["relation"] == "related_to"


def test_entity_aliases_merge_close_labels() -> None:
    graph = build_knowledge_graph(
        [
            {"paper_id": "a", "title": "A", "datasets": ["CIFAR10"], "metrics": ["F1 score"]},
            {"paper_id": "b", "title": "B", "datasets": ["CIFAR-10"], "metrics": ["F1"]},
        ]
    )

    assert node_id("Dataset", "CIFAR-10") in graph
    assert node_id("Metric", "F1") in graph
    assert graph.number_of_nodes() == 4


def test_load_alias_dictionary_reads_manual_mapping(tmp_path) -> None:
    alias_path = tmp_path / "aliases.yaml"
    alias_path.write_text(
        """
        aliases:
          GPT-4 model: GPT-4
          CustomBench v1: CustomBench
        types:
          Dataset:
            COCO val: COCO
        """,
        encoding="utf-8",
    )

    aliases = load_alias_dictionary(alias_path)

    assert aliases["gpt 4 model"] == "GPT-4"
    assert aliases["custombench v1"] == "CustomBench"
    assert aliases["dataset::coco val"] == "COCO"


def test_type_specific_aliases_apply_to_entity_types(tmp_path) -> None:
    alias_path = tmp_path / "aliases.yaml"
    alias_path.write_text(
        """
        types:
          Dataset:
            GPT-4: CustomBench
        """,
        encoding="utf-8",
    )
    aliases = load_alias_dictionary(alias_path)

    graph = build_knowledge_graph(
        [{"paper_id": "a", "title": "A", "datasets": ["GPT-4"], "method_name": "GPT-4"}],
        alias_dictionary=aliases,
    )

    assert node_id("Dataset", "CustomBench") in graph
    assert node_id("Method", "GPT-4") in graph


def test_graph_analysis_ranks_entities() -> None:
    graph = build_knowledge_graph(sample_papers())
    analysis = graph_analysis(graph, sample_papers())

    assert analysis["most_used_datasets"][0]["label"] == "Cora"
    assert analysis["most_common_metrics"][0]["label"] == "Accuracy"


def test_build_and_export_writes_analysis_files(tmp_path) -> None:
    input_path = tmp_path / "papers.json"
    output_dir = tmp_path / "outputs"
    input_path.write_text(
        json.dumps({"run_metadata": {"schema_version": "paper_extraction.v1"}, "papers": sample_papers()}),
        encoding="utf-8",
    )

    graph = build_and_export(input_path=input_path, output_dir=output_dir)

    assert graph.number_of_nodes() == 8
    for filename in [
        "graph.json",
        "graph.gexf",
        "graph.graphml",
        "graph_analysis.json",
        "nodes.csv",
        "edges.csv",
        "neo4j_import.cypher",
        "neo4j_example_queries.cypher",
        "graph.html",
        "graph_interactive.html",
    ]:
        assert (output_dir / filename).exists()

    with (output_dir / "nodes.csv").open(encoding="utf-8", newline="") as csv_file:
        node_rows = list(csv.DictReader(csv_file))
    with (output_dir / "edges.csv").open(encoding="utf-8", newline="") as csv_file:
        edge_rows = list(csv.DictReader(csv_file))

    assert len(node_rows) == 8
    assert len(edge_rows) == 7
    assert {"id", "label", "type", "degree"}.issubset(node_rows[0])
    assert {"source", "target", "relation"}.issubset(edge_rows[0])

    analysis_document = json.loads((output_dir / "graph_analysis.json").read_text(encoding="utf-8"))
    assert analysis_document["run_metadata"]["schema_version"] == "graph_analysis.v1"
    assert analysis_document["analysis"]["most_used_datasets"][0]["label"] == "Cora"


def test_load_papers_accepts_legacy_and_versioned_documents(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.json"
    versioned_path = tmp_path / "versioned.json"
    legacy_path.write_text(json.dumps(sample_papers()), encoding="utf-8")
    versioned_path.write_text(json.dumps({"run_metadata": {}, "papers": sample_papers()}), encoding="utf-8")

    assert load_papers(legacy_path)[0]["paper_id"] == "paper-1"
    assert load_papers(versioned_path)[0]["paper_id"] == "paper-1"
