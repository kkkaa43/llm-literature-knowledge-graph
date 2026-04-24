from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import networkx as nx

from src.run_metadata import graph_run_metadata
from src.utils import ensure_dir

NODE_COLORS = {
    "Paper": "#4C78A8",
    "Dataset": "#F58518",
    "Method": "#54A24B",
    "Baseline": "#E45756",
    "Metric": "#72B7B2",
    "Keyword": "#B279A2",
}

DEFAULT_ALIAS_DICTIONARY = {
    "gpt4": "GPT-4",
    "gpt 4": "GPT-4",
    "gpt-4 model": "GPT-4",
    "imagenet1k": "ImageNet",
    "image net": "ImageNet",
    "cifar10": "CIFAR-10",
    "cifar 10": "CIFAR-10",
    "f1 score": "F1",
    "f1-score": "F1",
    "accuracy score": "Accuracy",
}

DEFAULT_ALIAS_PATH = Path("config/entity_aliases.yaml")


def normalize_entity(value: str) -> str:
    """Normalize entity labels so duplicate datasets/metrics merge better."""
    value = re.sub(r"\s+", " ", value.strip())
    return value.strip(" .,:;")


def canonical_key(value: str) -> str:
    value = normalize_entity(value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    return value


def type_alias_key(node_type: str, value: str) -> str:
    return f"{node_type.casefold()}::{canonical_key(value)}"


def canonical_entity_label(value: str, alias_dictionary: dict[str, str] | None = None, node_type: str = "") -> str:
    alias_dictionary = alias_dictionary or DEFAULT_ALIAS_DICTIONARY
    normalized = normalize_entity(value)
    alias_key = canonical_key(normalized)
    if node_type:
        typed_value = alias_dictionary.get(type_alias_key(node_type, normalized))
        if typed_value:
            return typed_value
    return alias_dictionary.get(alias_key, normalized)


def load_alias_dictionary(alias_path: str | Path | None = DEFAULT_ALIAS_PATH) -> dict[str, str]:
    """Load optional manual entity aliases and normalize their keys."""
    aliases = DEFAULT_ALIAS_DICTIONARY.copy()
    if not alias_path:
        return aliases

    path = Path(alias_path)
    if not path.exists():
        return aliases

    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mappings = raw.get("aliases", raw)
    if not isinstance(mappings, dict):
        raise ValueError("Entity alias mapping must be a YAML mapping.")

    for source, target in mappings.items():
        if source == "types" and isinstance(target, dict):
            continue
        aliases[canonical_key(str(source))] = normalize_entity(str(target))

    typed_mappings = raw.get("types", {})
    if typed_mappings:
        if not isinstance(typed_mappings, dict):
            raise ValueError("Entity type aliases must be a YAML mapping.")
        for node_type, node_aliases in typed_mappings.items():
            if not isinstance(node_aliases, dict):
                raise ValueError("Each entity type alias section must be a YAML mapping.")
            for source, target in node_aliases.items():
                aliases[type_alias_key(str(node_type), str(source))] = normalize_entity(str(target))
    return aliases


def fuzzy_existing_label(value: str, existing_labels: list[str], threshold: float = 0.94) -> str:
    value_key = canonical_key(value)
    for label in existing_labels:
        label_key = canonical_key(label)
        if value_key == label_key:
            return label
        if value_key and label_key and SequenceMatcher(None, value_key, label_key).ratio() >= threshold:
            return label
    return value


def node_id(node_type: str, label: str) -> str:
    normalized = normalize_entity(label).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return f"{node_type.lower()}:{normalized}"


def first_sentence(value: str, max_length: int = 100) -> str:
    value = normalize_entity(value)
    if not value:
        return ""

    match = re.search(r"(.+?[.!?])\s", value)
    sentence = match.group(1) if match else value
    if len(sentence) <= max_length:
        return sentence
    return sentence[: max_length - 3].rstrip() + "..."


def add_entity_node(
    graph: nx.DiGraph,
    node_type: str,
    label: str,
    existing_labels: dict[str, list[str]] | None = None,
    alias_dictionary: dict[str, str] | None = None,
) -> str | None:
    label = canonical_entity_label(label, alias_dictionary, node_type=node_type)
    if existing_labels is not None:
        label = fuzzy_existing_label(label, existing_labels.setdefault(node_type, []))
    if not label:
        return None

    entity_id = node_id(node_type, label)
    graph.add_node(
        entity_id,
        label=label,
        type=node_type,
        color=NODE_COLORS.get(node_type, "#999999"),
    )
    if existing_labels is not None and label not in existing_labels.setdefault(node_type, []):
        existing_labels[node_type].append(label)
    return entity_id


def add_relation(graph: nx.DiGraph, source: str, target: str | None, relation: str) -> None:
    if not target:
        return
    graph.add_edge(source, target, relation=relation, label=relation)


def build_knowledge_graph(
    papers: list[dict[str, Any]],
    alias_dictionary: dict[str, str] | None = None,
) -> nx.DiGraph:
    """Build a directed knowledge graph from extracted paper records."""
    graph = nx.DiGraph()
    existing_labels: dict[str, list[str]] = defaultdict(list)

    for paper in papers:
        paper_id = str(paper.get("paper_id") or paper.get("title") or "unknown")
        title = normalize_entity(str(paper.get("title") or paper_id))
        year = str(paper.get("year") or "")

        paper_node = node_id("Paper", paper_id)
        graph.add_node(
            paper_node,
            label=title,
            type="Paper",
            paper_id=paper_id,
            year=year,
            arxiv_id=str(paper.get("arxiv_id") or ""),
            task=str(paper.get("task") or ""),
            method_name=str(paper.get("method_name") or ""),
            color=NODE_COLORS["Paper"],
        )

        method_label = str(paper.get("method_name") or "").strip() or first_sentence(
            str(paper.get("methodology") or "")
        )
        add_relation(
            graph,
            paper_node,
            add_entity_node(graph, "Method", method_label, existing_labels, alias_dictionary),
            "proposes",
        )

        for dataset in paper.get("datasets", []) or []:
            add_relation(
                graph,
                paper_node,
                add_entity_node(graph, "Dataset", str(dataset), existing_labels, alias_dictionary),
                "uses",
            )

        for baseline in paper.get("baselines", []) or []:
            add_relation(
                graph,
                paper_node,
                add_entity_node(graph, "Baseline", str(baseline), existing_labels, alias_dictionary),
                "compares_with",
            )

        for metric in paper.get("metrics", []) or []:
            add_relation(
                graph,
                paper_node,
                add_entity_node(graph, "Metric", str(metric), existing_labels, alias_dictionary),
                "evaluated_by",
            )

        for keyword in paper.get("keywords", []) or []:
            add_relation(
                graph,
                paper_node,
                add_entity_node(graph, "Keyword", str(keyword), existing_labels, alias_dictionary),
                "related_to",
            )

    return graph


def load_paper_document(input_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = Path(input_path)
    document = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        return document, {}
    if isinstance(document, dict):
        papers = document.get("papers", [])
        if not isinstance(papers, list):
            raise ValueError("papers.json must contain a 'papers' list.")
        metadata = document.get("run_metadata", {})
        return papers, metadata if isinstance(metadata, dict) else {}
    raise ValueError("papers.json must be either a paper list or a document with a 'papers' list.")


def load_papers(input_path: str | Path) -> list[dict[str, Any]]:
    papers, _metadata = load_paper_document(input_path)
    return papers


def ranked_values(papers: list[dict[str, Any]], field: str, limit: int = 25) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for paper in papers:
        for value in paper.get(field, []) or []:
            label = canonical_entity_label(str(value))
            if label:
                counter[label] += 1
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def graph_analysis(graph: nx.DiGraph, papers: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_to_papers: dict[str, list[str]] = defaultdict(list)
    keyword_to_papers: dict[str, list[str]] = defaultdict(list)
    for paper in papers:
        paper_id = str(paper.get("paper_id") or paper.get("title") or "unknown")
        for dataset in paper.get("datasets", []) or []:
            dataset_to_papers[canonical_entity_label(str(dataset))].append(paper_id)
        for keyword in paper.get("keywords", []) or []:
            keyword_to_papers[canonical_entity_label(str(keyword))].append(paper_id)

    paper_nodes = [
        {
            "paper_id": attrs.get("paper_id", node),
            "title": attrs.get("label", node),
            "degree": graph.degree(node),
        }
        for node, attrs in graph.nodes(data=True)
        if attrs.get("type") == "Paper"
    ]
    paper_nodes.sort(key=lambda row: row["degree"], reverse=True)

    shared_benchmarks = [
        {"dataset": dataset, "paper_count": len(paper_ids), "papers": sorted(set(paper_ids))}
        for dataset, paper_ids in dataset_to_papers.items()
        if len(set(paper_ids)) > 1
    ]
    shared_benchmarks.sort(key=lambda row: row["paper_count"], reverse=True)

    keyword_clusters = [
        {"keyword": keyword, "paper_count": len(set(paper_ids)), "papers": sorted(set(paper_ids))}
        for keyword, paper_ids in keyword_to_papers.items()
    ]
    keyword_clusters.sort(key=lambda row: row["paper_count"], reverse=True)

    return {
        "most_used_datasets": ranked_values(papers, "datasets"),
        "most_common_metrics": ranked_values(papers, "metrics"),
        "most_common_keywords": ranked_values(papers, "keywords"),
        "most_connected_papers": paper_nodes[:25],
        "shared_benchmark_papers": shared_benchmarks[:25],
        "keyword_clusters": keyword_clusters[:50],
    }


def export_graph_analysis(
    analysis: dict[str, Any],
    output_path: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    document: Any = analysis
    if run_metadata is not None:
        document = {"run_metadata": run_metadata, "analysis": analysis}
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


def cypher_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def export_neo4j_cypher(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export a simple idempotent Cypher import script."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    lines = [
        "// Run in Neo4j Browser or cypher-shell.",
        "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE;",
    ]
    for node, attrs in graph.nodes(data=True):
        labels = [":KGNode", f":{re.sub(r'[^A-Za-z0-9_]', '', str(attrs.get('type') or 'Entity'))}"]
        properties = {
            "id": node,
            "label": attrs.get("label", node),
            "type": attrs.get("type", ""),
            "paper_id": attrs.get("paper_id", ""),
            "year": attrs.get("year", ""),
        }
        prop_text = ", ".join(f"{key}: {cypher_string(value)}" for key, value in properties.items())
        lines.append(f"MERGE (n{''.join(labels)} {{id: {cypher_string(node)}}}) SET n += {{{prop_text}}};")

    for source, target, attrs in graph.edges(data=True):
        relation = re.sub(r"[^A-Za-z0-9_]", "_", str(attrs.get("relation") or "RELATED_TO")).upper()
        lines.append(
            f"MATCH (a:KGNode {{id: {cypher_string(source)}}}), (b:KGNode {{id: {cypher_string(target)}}}) "
            f"MERGE (a)-[:{relation}]->(b);"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_neo4j_examples(output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    examples = """// Most used datasets
MATCH (p:Paper)-[:USES]->(d:Dataset)
RETURN d.label AS dataset, count(p) AS papers
ORDER BY papers DESC
LIMIT 20;

// Papers connected to a keyword
MATCH (p:Paper)-[:RELATED_TO]->(k:Keyword)
WHERE toLower(k.label) CONTAINS toLower($keyword)
RETURN p.paper_id, p.label, k.label;

// Shared benchmarks between papers
MATCH (p1:Paper)-[:USES]->(d:Dataset)<-[:USES]-(p2:Paper)
WHERE p1.id < p2.id
RETURN d.label AS dataset, p1.label AS paper_a, p2.label AS paper_b
LIMIT 50;
"""
    output_path.write_text(examples, encoding="utf-8")


def write_neo4j(
    graph: nx.DiGraph,
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
) -> None:
    """Write graph nodes and relationships directly to Neo4j."""
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the optional neo4j package to write directly to Neo4j.") from error

    driver = GraphDatabase.driver(uri, auth=(user, password))

    node_query = """
    MERGE (n:KGNode {id: $id})
    SET n.label = $label,
        n.type = $type,
        n.paper_id = $paper_id,
        n.year = $year
    """
    edge_query = """
    MATCH (a:KGNode {id: $source}), (b:KGNode {id: $target})
    MERGE (a)-[r:RELATED {relation: $relation}]->(b)
    SET r.label = $relation
    """

    def run(session, query: str, **kwargs: Any) -> None:
        session.run(query, **kwargs).consume()

    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        for node, attrs in graph.nodes(data=True):
            params = {
                "id": node,
                "label": attrs.get("label", node),
                "type": re.sub(r"[^A-Za-z0-9_]", "", str(attrs.get("type") or "Entity")),
                "paper_id": attrs.get("paper_id", ""),
                "year": attrs.get("year", ""),
            }
            run(session, node_query, **params)

        for source, target, attrs in graph.edges(data=True):
            run(
                session,
                edge_query,
                source=source,
                target=target,
                relation=str(attrs.get("relation") or "related_to"),
            )

    driver.close()


def export_graph_json(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export graph in a simple node-link JSON format."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    data = nx.node_link_data(graph, edges="edges")
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_graph_gexf(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export graph for tools such as Gephi."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    nx.write_gexf(graph, output_path)


def export_graph_graphml(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export graph in GraphML format for graph analysis tools."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    nx.write_graphml(graph, output_path)


def export_nodes_csv(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export graph nodes as a flat CSV table."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fieldnames = ["id", "label", "type", "paper_id", "year", "color", "degree"]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for node, attrs in graph.nodes(data=True):
            writer.writerow(
                {
                    "id": node,
                    "label": attrs.get("label", node),
                    "type": attrs.get("type", ""),
                    "paper_id": attrs.get("paper_id", ""),
                    "year": attrs.get("year", ""),
                    "color": attrs.get("color", ""),
                    "degree": graph.degree(node),
                }
            )


def export_edges_csv(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export graph edges as a flat CSV table."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fieldnames = ["source", "target", "relation", "label"]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for source, target, attrs in graph.edges(data=True):
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation", ""),
                    "label": attrs.get("label", ""),
                }
            )


def export_graph_html(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export an interactive PyVis HTML graph."""
    # PyVis can be fragile in minimal CI/sandbox environments. The static graph
    # keeps the pipeline deterministic while graph data remains available as JSON.
    export_static_graph_html(graph, output_path)
    return

    try:
        from pyvis.network import Network
    except ModuleNotFoundError:
        export_static_graph_html(graph, output_path)
        return

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    net = Network(
        height="760px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-2500, central_gravity=0.2, spring_length=160, spring_strength=0.04)

    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "")
        label = attrs.get("label", node)
        title = f"{node_type}: {label}"
        net.add_node(
            node,
            label=label,
            title=title,
            color=attrs.get("color", "#999999"),
            shape="dot" if node_type != "Paper" else "box",
        )

    for source, target, attrs in graph.edges(data=True):
        relation = attrs.get("relation", "")
        net.add_edge(source, target, label=relation, title=relation, arrows="to")

    net.write_html(str(output_path), notebook=False)


def export_static_graph_html(graph: nx.DiGraph, output_path: str | Path) -> None:
    """Export a dependency-light static SVG graph when PyVis is unavailable."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if graph.number_of_nodes() == 0:
        output_path.write_text("<html><body><p>No graph data.</p></body></html>", encoding="utf-8")
        return

    positions = nx.spring_layout(graph, seed=42, k=1.2)
    width = 1200
    height = 800
    margin = 80

    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value: float, min_value: float, max_value: float, size: int) -> float:
        if max_value == min_value:
            return size / 2
        return margin + (value - min_value) / (max_value - min_value) * (size - 2 * margin)

    scaled = {
        node: (
            scale(position[0], min_x, max_x, width),
            scale(position[1], min_y, max_y, height),
        )
        for node, position in positions.items()
    }

    edge_elements = []
    for source, target, attrs in graph.edges(data=True):
        x1, y1 = scaled[source]
        x2, y2 = scaled[target]
        relation = html_escape(str(attrs.get("relation", "")))
        edge_elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#b8b8b8" stroke-width="1.4"><title>{relation}</title></line>'
        )

    node_elements = []
    for node, attrs in graph.nodes(data=True):
        x, y = scaled[node]
        label = html_escape(str(attrs.get("label", node)))
        node_type = html_escape(str(attrs.get("type", "")))
        color = attrs.get("color", "#999999")
        radius = 18 if node_type != "Paper" else 24
        node_elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="0.92">'
            f"<title>{node_type}: {label}</title></circle>"
        )
        node_elements.append(
            f'<text x="{x:.1f}" y="{y + radius + 14:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#222">{label[:36]}</text>'
        )

    legend_items = []
    for index, (node_type, color) in enumerate(NODE_COLORS.items()):
        y = 24 + index * 22
        legend_items.append(f'<circle cx="24" cy="{y}" r="7" fill="{color}"></circle>')
        legend_items.append(f'<text x="40" y="{y + 4}" font-size="13">{node_type}</text>')

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Academic Literature Knowledge Graph</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #fafafa; color: #222; }}
    header {{ padding: 18px 24px 8px; }}
    h1 {{ font-size: 22px; margin: 0 0 6px; }}
    p {{ margin: 0; color: #555; }}
    svg {{ display: block; width: 100%; height: auto; background: #fff; border-top: 1px solid #eee; }}
    text {{ paint-order: stroke; stroke: #fff; stroke-width: 3px; stroke-linejoin: round; }}
  </style>
</head>
<body>
  <header>
    <h1>Academic Literature Knowledge Graph</h1>
  </header>
  <svg viewBox="0 0 {width} {height}" role="img">
    <g>{"".join(edge_elements)}</g>
    <g>{"".join(node_elements)}</g>
    <g>{"".join(legend_items)}</g>
  </svg>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def build_and_export(
    input_path: str | Path = "data/extracted/papers.json",
    output_dir: str | Path = "outputs",
    alias_path: str | Path | None = DEFAULT_ALIAS_PATH,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> nx.DiGraph:
    papers, source_metadata = load_paper_document(input_path)
    aliases = load_alias_dictionary(alias_path)
    graph = build_knowledge_graph(papers, alias_dictionary=aliases)
    analysis = graph_analysis(graph, papers)

    output_dir = ensure_dir(output_dir)
    export_graph_json(graph, output_dir / "graph.json")
    export_graph_analysis(
        analysis, output_dir / "graph_analysis.json", run_metadata=graph_run_metadata(source_metadata)
    )
    export_graph_gexf(graph, output_dir / "graph.gexf")
    export_graph_graphml(graph, output_dir / "graph.graphml")
    export_nodes_csv(graph, output_dir / "nodes.csv")
    export_edges_csv(graph, output_dir / "edges.csv")
    export_neo4j_cypher(graph, output_dir / "neo4j_import.cypher")
    export_neo4j_examples(output_dir / "neo4j_example_queries.cypher")
    export_static_graph_html(graph, output_dir / "graph.html")
    export_graph_html(graph, output_dir / "graph_interactive.html")

    if neo4j_uri and neo4j_user and neo4j_password:
        write_neo4j(
            graph,
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )

    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a knowledge graph from extracted paper JSON.")
    parser.add_argument("--input-path", default="data/extracted/papers.json", help="Path to extracted papers JSON.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for graph outputs.")
    parser.add_argument("--alias-path", default=str(DEFAULT_ALIAS_PATH), help="YAML file with manual entity aliases.")
    parser.add_argument("--write-neo4j", action="store_true", help="Write graph directly to Neo4j.")
    parser.add_argument("--neo4j-uri", default=None, help="Neo4j URI, e.g. bolt://localhost:7687.")
    parser.add_argument("--neo4j-user", default=None, help="Neo4j username.")
    parser.add_argument("--neo4j-password", default=None, help="Neo4j password.")
    parser.add_argument("--neo4j-database", default=None, help="Optional Neo4j database name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_and_export(
        input_path=args.input_path,
        output_dir=args.output_dir,
        alias_path=args.alias_path,
        neo4j_uri=args.neo4j_uri if args.write_neo4j else None,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Saved graph files to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
