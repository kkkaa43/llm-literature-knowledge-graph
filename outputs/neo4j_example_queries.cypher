// Most used datasets
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
