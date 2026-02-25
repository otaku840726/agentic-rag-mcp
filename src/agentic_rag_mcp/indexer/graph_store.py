"""
GraphStore — Neo4j/AuraDB graph storage for code relationships.

Stores AST-extracted symbols and relationships as a knowledge graph.
Provides traversal queries for graph-enhanced RAG.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Edge types supported by the graph
EDGE_TYPES = {
    "member_of",    # method/property belongs to class
    "inherits",     # class extends class
    "implements",   # class implements interface
    "imports",      # file/namespace imports
    "calls",        # method calls method
    "uses_type",    # property/param type reference
    "defined_in",   # symbol defined in file
}


class GraphStore:
    """Neo4j/AuraDB graph storage for code relationships."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j", project: str = "default"):
        try:
            import neo4j as neo4j_driver
        except ImportError:
            raise ImportError(
                "neo4j package is required for graph store. "
                "Install with: pip install neo4j  or  pip install agentic-rag-mcp[graph]"
            )
        self._neo4j = neo4j_driver
        self.driver = neo4j_driver.GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.default_project = project

        # Verify connection works
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"GraphStore connected to {uri} (database={database}, project={project})")
        except Exception as e:
            self.driver.close()
            raise ConnectionError(f"Failed to connect to Neo4j at {uri}: {e}") from e

    # ── Schema / Constraints ──────────────────────────────────────

    def ensure_schema(self):
        """Create indexes and constraints (idempotent)."""
        constraints_and_indexes = [
            # Unique constraint on (Symbol.fqn, Symbol.project) — supports multi-project isolation
            "CREATE CONSTRAINT symbol_fqn_project_unique IF NOT EXISTS FOR (s:Symbol) REQUIRE (s.fqn, s.project) IS UNIQUE",
            # Unique constraint on (File.path, File.project)
            "CREATE CONSTRAINT file_path_project_unique IF NOT EXISTS FOR (f:File) REQUIRE (f.path, f.project) IS UNIQUE",
            # Indexes for common lookups
            "CREATE INDEX symbol_name_idx IF NOT EXISTS FOR (s:Symbol) ON (s.name)",
            "CREATE INDEX symbol_file_path_idx IF NOT EXISTS FOR (s:Symbol) ON (s.file_path)",
            "CREATE INDEX symbol_kind_idx IF NOT EXISTS FOR (s:Symbol) ON (s.kind)",
            "CREATE INDEX symbol_project_idx IF NOT EXISTS FOR (s:Symbol) ON (s.project)",
            "CREATE INDEX file_project_idx IF NOT EXISTS FOR (f:File) ON (f.project)",
        ]
        with self.driver.session(database=self.database) as session:
            for stmt in constraints_and_indexes:
                try:
                    session.run(stmt)
                except Exception as e:
                    # Some constraints may already exist in different form
                    logger.debug(f"Schema statement skipped: {e}")
        logger.info("GraphStore schema ensured")

    # ── Write ─────────────────────────────────────────────────────

    def upsert_symbols(self, symbols: List[Dict[str, Any]]):
        """Batch upsert Symbol nodes from AST analysis.

        Each symbol dict should have: fqn, name, kind, file_path, namespace,
        start_line, end_line.
        """
        if not symbols:
            return

        query = """
        UNWIND $symbols AS sym
        MERGE (s:Symbol {fqn: sym.fqn, project: sym.project})
        SET s.name = sym.name,
            s.kind = sym.kind,
            s.file_path = sym.file_path,
            s.namespace = sym.namespace,
            s.start_line = sym.start_line,
            s.end_line = sym.end_line,
            s.project = sym.project
        WITH s, sym
        CALL apoc.create.addLabels(s, [sym.label]) YIELD node
        RETURN count(node)
        """
        # Fallback query without APOC (for environments without APOC plugin)
        query_no_apoc = """
        UNWIND $symbols AS sym
        MERGE (s:Symbol {fqn: sym.fqn, project: sym.project})
        SET s.name = sym.name,
            s.kind = sym.kind,
            s.file_path = sym.file_path,
            s.namespace = sym.namespace,
            s.start_line = sym.start_line,
            s.end_line = sym.end_line,
            s.project = sym.project
        RETURN count(s)
        """

        # Add kind-specific label
        KIND_LABEL_MAP = {
            "class": "Class",
            "interface": "Interface",
            "enum": "Enum",
            "struct": "Struct",
            "method": "Method",
            "constructor": "Constructor",
            "property": "Property",
            "record": "Record",
        }
        # Enrich each sym dict with project and label
        for sym in symbols:
            sym["label"] = KIND_LABEL_MAP.get(sym.get("kind", ""), "Symbol")
            if "project" not in sym:
                sym["project"] = self.default_project

        with self.driver.session(database=self.database) as session:
            try:
                session.run(query, symbols=symbols)
            except Exception:
                # APOC not available, use simpler query
                session.run(query_no_apoc, symbols=symbols)

        logger.debug(f"Upserted {len(symbols)} symbol nodes")

    def upsert_relationships(self, relationships: List[Dict[str, Any]], batch_size: int = 200):
        """Batch upsert edges from AST analysis.

        Each relationship dict should have: type, source (FQN), target (FQN or name).
        Optional: kind (for member_of edge).

        Args:
            relationships: List of relationship dicts.
            batch_size: Max relationships per Cypher UNWIND call. Keeps AuraDB
                        memory usage bounded (default 200, safe for Free/Shared tiers).

        Raises:
            RuntimeError: If any batch fails to write. Partial writes are logged
                          before raising so the caller knows how many succeeded.
        """
        if not relationships:
            return

        # Allowed relationship types (whitelist to prevent Cypher injection)
        ALLOWED_REL_TYPES = {
            "MEMBER_OF", "INHERITS", "IMPLEMENTS", "IMPORTS",
            "CALLS", "USES_TYPE", "DEFINED_IN",
            # Roslyn analyzer additional types
            "CREATES", "OVERRIDES", "REFERENCES",
            # Markdown analyzer types
            "SUBSECTION_OF",
            # Messaging relationships
            "PUBLISHES_TO", "SUBSCRIBES_TO",
            # Annotation usage: class/method/field → annotation type
            "ANNOTATED_BY",
        }

        # Group by relationship type
        by_type: Dict[str, List[Dict]] = {}
        for rel in relationships:
            rel_type = rel.get("type", "").upper()
            if rel_type not in ALLOWED_REL_TYPES:
                logger.warning(f"Skipping unknown relationship type: {rel_type}")
                continue
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        total_written = 0
        failures: List[str] = []

        with self.driver.session(database=self.database) as session:
            for rel_type, rels in by_type.items():
                if rel_type == "IMPORTS":
                    cypher = f"""
                    UNWIND $rels AS rel
                    MERGE (src:File {{path: rel.source}})
                    MERGE (tgt:Symbol {{fqn: rel.target, project: $default_project}})
                    ON CREATE SET tgt.name = rel.target, tgt.kind = 'external'
                    MERGE (src)-[r:IMPORTS]->(tgt)
                    RETURN count(r)
                    """
                elif rel_type in ("PUBLISHES_TO", "SUBSCRIBES_TO"):
                    cypher = f"""
                    UNWIND $rels AS rel
                    MERGE (src:Symbol {{fqn: rel.source, project: $default_project}})
                    MERGE (tgt:Symbol {{fqn: rel.target, project: $default_project}})
                    ON CREATE SET tgt.name = COALESCE(rel.target_name, rel.target), tgt.kind = 'external'
                    MERGE (src)-[r:{rel_type}]->(tgt)
                    SET r.queue_name = rel.metadata.queue_name,
                        r.line       = rel.metadata.line
                    RETURN count(r)
                    """
                else:
                    cypher = f"""
                    UNWIND $rels AS rel
                    MERGE (src:Symbol {{fqn: rel.source, project: $default_project}})
                    MERGE (tgt:Symbol {{fqn: rel.target, project: $default_project}})
                    ON CREATE SET tgt.name = COALESCE(rel.target_name, rel.target), tgt.kind = 'external'
                    MERGE (src)-[r:{rel_type}]->(tgt)
                    SET r.kind = rel.kind
                    RETURN count(r)
                    """

                # Split into batches to stay within AuraDB memory/timeout limits
                for batch_start in range(0, len(rels), batch_size):
                    batch = rels[batch_start:batch_start + batch_size]
                    batch_end = batch_start + len(batch)
                    try:
                        session.run(cypher, rels=batch, default_project=self.default_project)
                        total_written += len(batch)
                        logger.debug(
                            f"Upserted {rel_type} [{batch_start+1}–{batch_end}/{len(rels)}]"
                        )
                    except Exception as e:
                        msg = (
                            f"Failed to upsert {rel_type} batch "
                            f"[{batch_start+1}–{batch_end}/{len(rels)}]: {e}"
                        )
                        logger.error(msg)
                        failures.append(msg)

        if failures:
            raise RuntimeError(
                f"upsert_relationships: {len(failures)} batch(es) failed "
                f"({total_written}/{len(relationships)} relationships written).\n"
                + "\n".join(failures)
            )

        logger.info(f"Upserted {total_written} relationships ({len(by_type)} types)")

    def upsert_file_node(self, file_path: str, metadata: Dict[str, Any], project: Optional[str] = None, hash: Optional[str] = None):
        """Create/update a File node and link symbols to it.

        Args:
            file_path: Relative file path
            metadata: service, layer, category, language, etc.
            project: Project name for logical isolation (defaults to self.default_project)
        """
        proj = project if project is not None else self.default_project
        query = """
        MERGE (f:File {path: $path, project: $project})
        SET f.service = $service,
            f.layer = $layer,
            f.category = $category,
            f.language = $language,
            f.project = $project,
            f.hash = CASE WHEN $hash IS NOT NULL THEN $hash ELSE f.hash END
        WITH f
        MATCH (s:Symbol {file_path: $path, project: $project})
        MERGE (s)-[:DEFINED_IN]->(f)
        RETURN count(s)
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                path=file_path,
                project=proj,
                service=metadata.get("service", ""),
                layer=metadata.get("layer", ""),
                category=metadata.get("category", ""),
                language=metadata.get("language", ""),
                hash=hash,
            )

    def delete_by_file(self, file_path: str, project: Optional[str] = None):
        """Delete all symbols and relationships for a file (before re-indexing).

        Args:
            file_path: Relative file path
            project: Project name for logical isolation (defaults to self.default_project)
        """
        proj = project if project is not None else self.default_project
        query = """
        MATCH (s:Symbol {file_path: $path, project: $project})
        DETACH DELETE s
        WITH 1 AS dummy
        MATCH (f:File {path: $path, project: $project})
        DETACH DELETE f
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, path=file_path, project=proj)
        logger.debug(f"Deleted graph data for file: {file_path} (project={proj})")

    # ── Read ──────────────────────────────────────────────────────

    def get_neighbors(
        self,
        symbol_name: str,
        depth: int = 1,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get N-hop neighbors of a symbol.

        Args:
            symbol_name: Symbol name or FQN (uses fuzzy match on name if not FQN)
            depth: Number of hops (1-3)
            direction: "both", "incoming", "outgoing"
            relationship_types: Optional filter on edge types
            project: Optional project filter (defaults to self.default_project if not None)

        Returns:
            {"nodes": [...], "edges": [...]}
        """
        depth = min(max(depth, 1), 3)

        # Build relationship filter
        rel_filter = ""
        if relationship_types:
            types_str = "|".join(t.upper() for t in relationship_types)
            rel_filter = f":{types_str}"

        # Direction
        if direction == "outgoing":
            pattern = f"-[r{rel_filter}*1..{depth}]->"
        elif direction == "incoming":
            pattern = f"<-[r{rel_filter}*1..{depth}]-"
        else:
            pattern = f"-[r{rel_filter}*1..{depth}]-"

        proj = self.default_project if project is None else project

        query = f"""
        MATCH (start:Symbol)
        WHERE (start.fqn = $name OR start.name = $name
           OR start.fqn ENDS WITH ('.' + $name))
          AND ($project IS NULL OR start.project = $project)
        WITH start LIMIT 1
        MATCH path = (start){pattern}(neighbor:Symbol)
        WHERE neighbor.kind <> 'external'
        WITH start, neighbor,
             [rel IN relationships(path) | {{
                type: type(rel),
                source: startNode(rel).fqn,
                target: endNode(rel).fqn
             }}] AS edge_list
        RETURN DISTINCT
            neighbor.fqn AS fqn,
            neighbor.name AS name,
            neighbor.kind AS kind,
            neighbor.file_path AS file_path,
            neighbor.namespace AS namespace,
            edge_list
        LIMIT 50
        """

        nodes = []
        edges = set()
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=symbol_name, project=proj)
            for record in result:
                nodes.append({
                    "fqn": record["fqn"],
                    "name": record["name"],
                    "kind": record["kind"],
                    "file_path": record["file_path"],
                    "namespace": record["namespace"],
                })
                for edge in record["edge_list"]:
                    edges.add((edge["type"], edge["source"], edge["target"]))

        return {
            "nodes": nodes,
            "edges": [
                {"type": t, "source": s, "target": tgt}
                for t, s, tgt in edges
            ],
        }

    def get_call_chain(
        self, from_symbol: str, to_symbol: str, max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find shortest call path between two symbols."""
        query = """
        MATCH (start:Symbol), (end:Symbol)
        WHERE (start.fqn = $from OR start.name = $from)
          AND (end.fqn = $to OR end.name = $to)
        WITH start, end LIMIT 1
        MATCH path = shortestPath((start)-[:CALLS*1..{depth}]->(end))
        RETURN [n IN nodes(path) | {{
            fqn: n.fqn,
            name: n.name,
            kind: n.kind,
            file_path: n.file_path
        }}] AS chain
        """.replace("{depth}", str(min(max_depth, 10)))

        with self.driver.session(database=self.database) as session:
            result = session.run(query, **{"from": from_symbol, "to": to_symbol})
            record = result.single()
            if record:
                return record["chain"]
        return []

    def get_class_hierarchy(self, class_fqn: str) -> Dict[str, Any]:
        """Get inheritance/implementation tree for a class."""
        query = """
        MATCH (c:Symbol)
        WHERE c.fqn = $fqn OR c.name = $fqn
        WITH c LIMIT 1
        OPTIONAL MATCH parent_path = (c)-[:INHERITS*1..5]->(parent:Symbol)
        OPTIONAL MATCH impl_path = (c)-[:IMPLEMENTS*1..5]->(iface:Symbol)
        OPTIONAL MATCH child_path = (child:Symbol)-[:INHERITS*1..5]->(c)
        RETURN c.fqn AS root,
               collect(DISTINCT {fqn: parent.fqn, name: parent.name}) AS parents,
               collect(DISTINCT {fqn: iface.fqn, name: iface.name}) AS interfaces,
               collect(DISTINCT {fqn: child.fqn, name: child.name}) AS children
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, fqn=class_fqn)
            record = result.single()
            if record:
                return {
                    "root": record["root"],
                    "parents": [p for p in record["parents"] if p["fqn"]],
                    "interfaces": [i for i in record["interfaces"] if i["fqn"]],
                    "children": [c for c in record["children"] if c["fqn"]],
                }
        return {"root": class_fqn, "parents": [], "interfaces": [], "children": []}

    def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """Get all files connected to this file via symbol relationships."""
        query = """
        MATCH (s:Symbol {file_path: $path})-[r]-(other:Symbol)
        WHERE other.file_path <> $path
        RETURN DISTINCT other.file_path AS dep_file,
               collect(DISTINCT {type: type(r), symbol: other.name}) AS connections
        ORDER BY size(connections) DESC
        LIMIT 30
        """
        deps = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, path=file_path)
            for record in result:
                deps.append({
                    "file_path": record["dep_file"],
                    "connections": record["connections"],
                })
        return {"file": file_path, "dependencies": deps}

    def cypher_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute raw Cypher query (for advanced MCP tool).

        Returns list of record dicts.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def get_stats(self) -> Dict[str, Any]:
        """Return node/edge counts by type, plus per-project breakdown."""
        stats_query = """
        CALL () {
            MATCH (s:Symbol) RETURN 'Symbol' AS label, count(s) AS cnt
            UNION ALL
            MATCH (f:File) RETURN 'File' AS label, count(f) AS cnt
        }
        RETURN label, cnt
        """
        edge_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS cnt
        """
        projects_query = """
        MATCH (s:Symbol)
        WHERE s.project IS NOT NULL
        RETURN s.project AS project, count(s) AS symbol_count
        ORDER BY symbol_count DESC
        """
        node_counts = {}
        edge_counts = {}
        projects = []
        with self.driver.session(database=self.database) as session:
            for record in session.run(stats_query):
                node_counts[record["label"]] = record["cnt"]
            for record in session.run(edge_query):
                edge_counts[record["rel_type"]] = record["cnt"]
            for record in session.run(projects_query):
                projects.append({"project": record["project"], "symbol_count": record["symbol_count"]})

        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
            "projects": projects,
        }

    def get_all_file_hashes(self, project: Optional[str] = None) -> Dict[str, str]:
        """Return {file_path: hash} for all File nodes that have a stored hash.

        Used for incremental indexing — compare against current MD5 to detect changes.
        """
        proj = project if project is not None else self.default_project
        query = """
        MATCH (f:File {project: $project})
        WHERE f.hash IS NOT NULL
        RETURN f.path AS path, f.hash AS hash
        """
        result: Dict[str, str] = {}
        with self.driver.session(database=self.database) as session:
            for record in session.run(query, project=proj):
                result[record["path"]] = record["hash"]
        return result

    def list_projects(self) -> List[str]:
        """Return list of distinct project names in the graph."""
        query = """
        MATCH (s:Symbol)
        WHERE s.project IS NOT NULL
        RETURN DISTINCT s.project AS project
        ORDER BY project
        """
        projects = []
        with self.driver.session(database=self.database) as session:
            for record in session.run(query):
                projects.append(record["project"])
        return projects

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
        logger.info("GraphStore connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
