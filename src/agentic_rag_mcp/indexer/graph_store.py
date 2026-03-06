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
        
        // 【新增：強制建立 File 節點與實體關聯】
        WITH s, sym
        WHERE sym.file_path IS NOT NULL AND sym.file_path <> ''
        MERGE (f:File {path: sym.file_path, project: sym.project})
        MERGE (s)-[:DEFINED_IN]->(f)
        
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
            
        // 【新增：強制建立 File 節點與實體關聯】
        WITH s, sym
        WHERE sym.file_path IS NOT NULL AND sym.file_path <> ''
        MERGE (f:File {path: sym.file_path, project: sym.project})
        MERGE (s)-[:DEFINED_IN]->(f)
        
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

    # ── Community Detection ────────────────────────────────────────

    def compute_communities(self, project: Optional[str] = None):
        """Run Leiden community detection on the graph and persist results.

        Uses GDS to project the graph, run Leiden, and write back `communityId` to Symbol nodes.
        Then, uses LLM to generate readable names for these communities based on their central symbols.
        """
        proj = project if project is not None else self.default_project

        # 1. Clean up old community nodes for this project
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                MATCH (c:Community)
                WHERE c.project = $project
                DETACH DELETE c
                """, project=proj)
                logger.info(f"Cleaned up old Community nodes for project: {proj}")
        except Exception as e:
            logger.warning(f"Failed to clean up old Community nodes: {e}")

        # 2. Project graph and run Leiden
        try:
            with self.driver.session(database=self.database) as session:
                # Drop projection if exists
                session.run("CALL gds.graph.drop('codebase_graph', false) YIELD graphName")

                # Project graph
                # Note: Leiden requires UNDIRECTED orientation.
                project_query = """
                CALL gds.graph.project(
                  'codebase_graph',
                  ['Symbol', 'File'],
                  {
                    CALLS: {orientation: 'UNDIRECTED'},
                    USES_TYPE: {orientation: 'UNDIRECTED'},
                    MEMBER_OF: {orientation: 'UNDIRECTED'},
                    INHERITS: {orientation: 'UNDIRECTED'},
                    IMPLEMENTS: {orientation: 'UNDIRECTED'}
                  }
                )
                """
                session.run(project_query)

                # Run Leiden
                leiden_query = """
                CALL gds.leiden.write('codebase_graph', { writeProperty: 'communityId' })
                """
                session.run(leiden_query)

                logger.info(f"Successfully ran Leiden community detection via GDS for project: {proj}")
        except Exception as e:
            logger.error(f"Failed to run community detection: {e}")
            return

        # 3. Extract top symbols for each community
        communities_query = """
        MATCH (s:Symbol)
        WHERE s.communityId IS NOT NULL AND s.project = $project
        WITH s.communityId AS communityId, s
        ORDER BY communityId, COUNT { (s)-[]-() } DESC
        WITH communityId, collect(s)[..5] AS top_symbols
        RETURN communityId, [sym IN top_symbols | {name: sym.name, file_path: sym.file_path, fqn: sym.fqn}] AS top_symbols
        """
        communities = []
        with self.driver.session(database=self.database) as session:
            for record in session.run(communities_query, project=proj):
                communities.append({
                    "communityId": record["communityId"],
                    "top_symbols": record["top_symbols"]
                })

        # 4. Label communities using LLM (Optimized: Batching + Global Review)
        if not communities:
            return

        from ..provider import create_client_for, get_section_config
        import json

        try:
            client, comp_cfg = create_client_for("analyst")
            graph_cfg = get_section_config("graph_processing")
            label_cfg = graph_cfg.get("community_labeling", {})
            
            community_map = {}  # communityId -> {name, symbols}

            # Phase 1: Batch Drafting (Process communities in batches)
            batch_size = int(label_cfg.get("batch_size", 20))
            logger.info(f"Drafting names for {len(communities)} communities in batches of {batch_size}...")

            for i in range(0, len(communities), batch_size):
                batch = communities[i:i + batch_size]
                prompt = f"""
                You are a software architect. Analyze these {len(batch)} code communities.
                For each community, I provided the top central symbols and their file paths.
                Generate a short, precise name (3-5 words) for each that reflects its logical responsibility.

                DATA:
                {json.dumps(batch, indent=2)}

                Return ONLY a JSON object where keys are communityId (string) and values are names.
                """
                try:
                    response = client.chat.completions.create(
                        model=comp_cfg.model,
                        messages=[
                            {"role": "system", "content": "Reply with JSON ONLY."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"} if any(m in comp_cfg.model for m in ["gpt-4", "gpt-3.5", "minimax"]) else None
                    )
                    content = response.choices[0].message.content.strip()
                    # 去除可能的 markdown code blocks
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()
                    
                    batch_names = json.loads(content)
                    for comm in batch:
                        cid = str(comm["communityId"])
                        name = batch_names.get(cid, "Unnamed Module")
                        community_map[cid] = {"name": name, "symbols": comm["top_symbols"]}

                    logger.info(f"Drafted batch {i//batch_size + 1}/{(len(communities)-1)//batch_size + 1}")
                except Exception as batch_err:
                    logger.warning(f"Failed to draft names for batch {i}: {batch_err}")

            # Phase 2: Global Review (Consistency check)
            do_review = str(label_cfg.get("global_review", "true")).lower() == "true"
            if do_review and len(community_map) > 1:
                logger.info("Performing global consistency review for community names...")
                review_list = {cid: data["name"] for cid, data in community_map.items()}
                review_prompt = f"""
                You are a senior software architect. Review these {len(review_list)} detected module names for the project '{proj}'.
                Identify duplicate names, overly vague names (e.g., "Utilities"), or inconsistent naming styles.
                Provide a refined mapping that ensures each name is unique and descriptive within the project context.

                CURRENT NAMES:
                {json.dumps(review_list, indent=2)}

                Return ONLY a JSON object of refined names mapping: {{ communityId: refined_name }}
                """
                try:
                    review_resp = client.chat.completions.create(
                        model=comp_cfg.model,
                        messages=[
                            {"role": "system", "content": "You are an architect. Reply with JSON ONLY."},
                            {"role": "user", "content": review_prompt}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"} if any(m in comp_cfg.model for m in ["gpt-4", "gpt-3.5", "minimax"]) else None
                    )
                    content = review_resp.choices[0].message.content.strip()
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()
                        
                    refined_names = json.loads(content)
                    for cid, new_name in refined_names.items():
                        if cid in community_map:
                            community_map[cid]["name"] = new_name
                    logger.info("Global review completed.")
                except Exception as review_err:
                    logger.warning(f"Global review failed, using drafts: {review_err}")

            # Phase 3: Persistence (Create Community nodes in Neo4j)
            logger.info(f"Persisting {len(community_map)} Community nodes...")
            with self.driver.session(database=self.database) as session:
                for cid, data in community_map.items():
                    update_query = """
                    MATCH (s:Symbol {communityId: toInteger($comm_id), project: $project})
                    MERGE (c:Community {id: $comm_id, project: $project})
                    SET c.name = $comm_name
                    MERGE (s)-[:IN_COMMUNITY]->(c)
                    """
                    session.run(update_query, comm_id=cid, comm_name=data["name"], project=proj)

            logger.info(f"Successfully created and labeled {len(community_map)} Community nodes.")

        except Exception as e:
            logger.error(f"Failed to label communities: {e}")

    # ── Execution Flow Detection ───────────────────────────────────

    def compute_execution_flows(self, project: Optional[str] = None):
        """Detect execution flows from entry points and persist as Process nodes.

        Uses apoc.path.expandConfig to trace CALLS relationships from known entry points
        (e.g., C# [HttpGet], Java @RequestMapping, main methods) up to a max depth.
        """
        proj = project if project is not None else self.default_project

        # 1. Clean up old Process nodes for this project
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                MATCH (p:Process)
                WHERE p.project = $project
                DETACH DELETE p
                """, project=proj)
                logger.info(f"Cleaned up old Process nodes for project: {proj}")
        except Exception as e:
            logger.warning(f"Failed to clean up old Process nodes: {e}")

        # 2. Identify Entry Points
        # Using a simple heuristic for entry points: symbols with specific annotations/names
        # Note: Depending on the analyzer, annotations might be stored in metadata or fqn.
        # We also look for common method names like "main" or "Handler"
        entry_points_query = """
        MATCH (s:Symbol)
        WHERE (s.name = 'main' OR s.name ENDS WITH 'Controller'
           OR s.name ENDS WITH 'Handler'
           OR s.name ENDS WITH 'Listener')
          AND s.project = $project
        RETURN s.fqn AS fqn, s.name AS name
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(entry_points_query, project=proj)
                entry_points = [record["fqn"] for record in result]

                if not entry_points:
                    logger.info("No entry points found for execution flow detection.")
                    return

                logger.info(f"Found {len(entry_points)} potential entry points for project '{proj}'. Tracing execution flows...")

                # 3. Trace paths using APOC
                # We trace CALLS relationships up to maxLevel=10
                trace_query = """
                UNWIND $entry_points AS ep_fqn
                MATCH (start:Symbol {fqn: ep_fqn, project: $project})
                CALL apoc.path.expandConfig(start, {
                    relationshipFilter: "CALLS>",
                    labelFilter: "/Symbol",
                    minLevel: 1,
                    maxLevel: 10,
                    uniqueness: "NODE_GLOBAL"
                })
                YIELD path
                WITH start, path, length(path) as len
                // Only keep paths that end at leaf nodes or reach max depth
                WHERE len > 1
                WITH start, path
                ORDER BY length(path) DESC
                LIMIT 50 // limit per entry point to avoid explosion

                // Create Process node for each significant path
                WITH start, path,
                     [n IN nodes(path) | n.fqn] AS path_steps

                MERGE (p:Process {id: apoc.util.md5(path_steps), project: $project})
                ON CREATE SET p.name = 'Flow from ' + start.name,
                              p.entry_point = start.fqn,
                              p.file_path = start.file_path,
                              p.steps = path_steps

                WITH p, nodes(path) AS syms
                UNWIND range(0, size(syms)-1) AS idx
                WITH p, idx, syms[idx] AS sym
                MERGE (sym)-[r:STEP_IN_PROCESS]->(p)
                SET r.order = idx
                """

                session.run(trace_query, entry_points=entry_points, project=proj)
                logger.info("Successfully computed execution flows.")
        except Exception as e:
            logger.error(f"Failed to compute execution flows (APOC plugin might be missing): {e}")

    def delete_project(self, project: Optional[str] = None) -> int:
        """Delete all nodes and relationships associated with a project.

        Args:
            project: Project name (defaults to self.default_project)

        Returns:
            Number of nodes deleted.
        """
        proj = project if project is not None else self.default_project
        query = """
        MATCH (n)
        WHERE (n:Symbol OR n:File OR n:Community OR n:Process) AND n.project = $project
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, project=proj)
                record = result.single()
                deleted = record["deleted_count"] if record else 0
                logger.info(f"Deleted {deleted} nodes for project: {proj}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete project {proj}: {e}")
            raise

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
        OPTIONAL MATCH (start)-[:IN_COMMUNITY]->(c:Community)
        WITH start, c.name AS community_name
        MATCH path = (start){pattern}(neighbor:Symbol)
        WHERE neighbor.kind <> 'external'
        WITH start, community_name, neighbor,
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
            neighbor.start_line AS start_line,
            neighbor.end_line AS end_line,
            edge_list,
            community_name
        LIMIT 50
        """

        nodes = []
        edges = set()
        community_name = None
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=symbol_name, project=proj)
            for record in result:
                if community_name is None:
                    community_name = record["community_name"]
                nodes.append({
                    "fqn": record["fqn"],
                    "name": record["name"],
                    "kind": record["kind"],
                    "file_path": record["file_path"],
                    "namespace": record["namespace"],
                    "start_line": record["start_line"],
                    "end_line": record["end_line"],
                })
                for edge in record["edge_list"]:
                    edges.add((edge["type"], edge["source"], edge["target"]))

        return {
            "nodes": nodes,
            "edges": [
                {"type": t, "source": s, "target": tgt}
                for t, s, tgt in edges
            ],
            "community": community_name
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

    def get_project_tree(self, project: str = None) -> str:
        """從圖譜中提取並構建專案的完整目錄樹"""
        proj = project or self.default_project
        query = """
        MATCH (f:File {project: $project})
        RETURN f.path as path
        """
        try:
            with self.driver.session(database=self.database) as session:
                results = session.run(query, project=proj)
                paths = [r["path"] for r in results]
                
            if not paths:
                return "No files found in graph."

            # 構建樹狀結構
            tree_dict = {}
            for p in sorted(paths):
                parts = p.split('/')
                current = tree_dict
                for part in parts:
                    current = current.setdefault(part, {})

            # 轉換為格式化字串
            def render_tree(d, prefix=""):
                lines = []
                entries = list(d.keys())
                for i, key in enumerate(entries):
                    is_last = i == len(entries) - 1
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{key}")
                    if d[key]: # 如果有子目錄
                        extension = "    " if is_last else "│   "
                        lines.extend(render_tree(d[key], prefix + extension))
                return lines

            tree_str = "\n".join(render_tree(tree_dict))
            return f"[Project Directory Structure]\n{tree_str}"
        except Exception as e:
            logger.error(f"Failed to fetch project tree: {e}")
            return "Project tree unavailable."

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

    def find_bridge(self, start_name: str, end_name: str, project: str = None) -> List[List[str]]:
        """確定性地驗證兩個符號之間是否有調用或使用路徑"""
        proj = project or self.default_project
        query = """
        MATCH (start:Symbol) WHERE (start.name = $start OR start.fqn = $start) AND start.project = $project
        MATCH (end:Symbol) WHERE (end.name = $end OR end.fqn = $end) AND end.project = $project
        MATCH path = shortestPath((start)-[:CALLS|USES_TYPE|MEMBER_OF|IMPLEMENTS|INHERITS*1..5]-(end))
        RETURN [n IN nodes(path) | n.name] as steps
        """
        results = self.cypher_query(query, {"start": start_name, "end": end_name, "project": proj})
        return [r["steps"] for r in results]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
