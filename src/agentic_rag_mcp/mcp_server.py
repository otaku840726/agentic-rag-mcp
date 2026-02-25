"""
MCP Server - Agentic RAG 搜索服務
提供給 Claude Code 等工具調用
"""

import os
import json
import asyncio
import sys
import atexit
import logging
from pathlib import Path
from typing import Any
from dataclasses import asdict
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

# ── 配置 logging ──────────────────────────────────────────────
# 創建 .agentic-rag-cache 目錄（如果不存在）
cache_dir = Path.cwd() / ".agentic-rag-cache"
cache_dir.mkdir(exist_ok=True)

log_file = cache_dir / "mcp-server.log"
log_level = os.getenv('LOG_LEVEL', 'INFO')

# 配置 logging：同時輸出到文件和 stderr
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # 文件 handler (帶 rotation)
        RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        # stderr handler (供 MCP client 使用)
        logging.StreamHandler(sys.stderr)
    ]
)

# 降低第三方庫的日誌級別
logging.getLogger('qdrant_client').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"MCP Server starting, logs will be written to: {log_file}")
# ──────────────────────────────────────────────────────────────

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        CallToolResult,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP not installed. Run: pip install mcp")

from .agentic_search import AgenticSearch, AgenticSearchConfig
from .models import SearchResult
from .provider import get_neo4j_config


# 初始化搜索代理
search_agent = None

# 初始化索引服務
indexer_service = None


def get_indexer_service():
    """獲取或創建索引服務（單例）"""
    global indexer_service
    if indexer_service is None:
        from .indexer import IndexerService
        indexer_service = IndexerService()
    return indexer_service


def _is_neo4j_enabled() -> bool:
    """Check if Neo4j is enabled in config."""
    try:
        cfg = get_neo4j_config()
        return cfg["enabled"]
    except Exception:
        return False


# Graph store singleton
_graph_store = None


def get_graph_store():
    """Get or create graph store singleton. Returns None if Neo4j disabled."""
    global _graph_store
    if _graph_store is None:
        if not _is_neo4j_enabled():
            return None
        try:
            from .indexer.graph_store import GraphStore
            cfg = get_neo4j_config()
            _graph_store = GraphStore(
                uri=cfg["uri"],
                username=cfg["username"],
                password=cfg["password"],
                database=cfg["database"],
                project=os.getenv("GRAPH_PROJECT", "default"),
            )
        except Exception as e:
            logger.warning(f"Failed to create graph store: {e}")
            return None
    return _graph_store


@atexit.register
def _cleanup_graph_store():
    global _graph_store
    if _graph_store is not None:
        try:
            _graph_store.close()
        except Exception:
            pass


def get_search_agent() -> AgenticSearch:
    """獲取或創建搜索代理（單例）"""
    global search_agent
    if search_agent is None:
        search_agent = AgenticSearch()
    return search_agent


def format_response(result: SearchResult) -> str:
    """格式化搜索結果為可讀文本"""
    if not result.success:
        return f"Search failed: {result.error}"

    resp = result.response
    parts = []

    # Answer
    parts.append("## Answer\n")
    parts.append(resp.answer)
    parts.append("\n")

    # Flow
    if resp.flow:
        parts.append("\n## Flow\n")
        for step in resp.flow:
            parts.append(f"{step.step}. {step.description}")
            if step.code_ref:
                parts.append(f"   → {step.code_ref}")
            parts.append("\n")

    # Decision Points
    if resp.decision_points:
        parts.append("\n## Decision Points\n")
        for dp in resp.decision_points:
            parts.append(f"- **{dp.condition}**")
            parts.append(f"  - True: {dp.true_branch}")
            parts.append(f"  - False: {dp.false_branch}")
            parts.append("\n")

    # Config
    if resp.config:
        parts.append("\n## Configuration\n")
        for cfg in resp.config:
            parts.append(f"- `{cfg.key}`: {cfg.default_value or 'N/A'}")
            parts.append(f"  - Source: {cfg.source}")
            if cfg.description:
                parts.append(f"  - {cfg.description}")
            parts.append("\n")

    # Evidence
    if resp.evidence:
        parts.append("\n## Evidence\n")
        for i, e in enumerate(resp.evidence[:10], 1):
            parts.append(f"### {i}. {e.path}")
            parts.append(f"Span: {e.span}")
            parts.append(f"```")
            parts.append(e.quote)
            parts.append(f"```")
            parts.append("\n")

    # Metadata
    parts.append("\n---\n")
    parts.append(f"*Iterations: {resp.iterations} | Evidence found: {resp.total_evidence_found}*")

    return "\n".join(parts)


if MCP_AVAILABLE:
    # 創建 MCP Server
    server = Server("agentic-rag")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出可用的工具"""
        tools = [
            Tool(
                name="agentic-search",
                description="""
                Agentic RAG search for your codebase.

                This tool performs multi-hop search with:
                - Hybrid search (semantic + keyword)
                - Cross-encoder reranking
                - Iterative query planning
                - Evidence synthesis

                Use this when you need to:
                - Understand code flow and logic
                - Find related code across services
                - Trace configuration and state machines
                - Answer "how does X work?" questions
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query in natural language"
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum search iterations (default: 5)",
                            "default": 5
                        },
                        "llm_provider": {
                            "type": "string",
                            "description": "Override LLM provider for all components (analyst/planner/synthesizer). e.g. 'openai', 'local', 'gemini', 'openrouter', 'voyage'. Uses config default if not set."
                        },
                        "llm_model": {
                            "type": "string",
                            "description": "Override LLM model for all components. e.g. 'gpt-4o', 'gpt-4o-mini', 'qwen2.5-coder:32b'. Uses config default if not set."
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="quick-search",
                description="""
                Quick single-pass search without agentic iteration.
                Faster but less thorough than agentic-search.

                Use this for simple lookups like:
                - Find a specific class or method
                - Find configuration values
                - Quick code reference
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "operator": {
                            "type": "string",
                            "enum": ["hybrid", "semantic", "keyword", "exact"],
                            "description": "Search operator: hybrid (dense+sparse RRF fusion, recommended), semantic (dense only), keyword (BM25 sparse only), exact (string match). Default: hybrid",
                            "default": "hybrid"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="index-status",
                description="""
                Show the current indexing status.
                Returns Qdrant collection stats (points_count, by_category) and local state (indexed files count, last index time).
                """,
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            Tool(
                name="analyze-codebase",
                description="""
                Analyze code files in a directory and output structural analysis (AST, symbols) to JSON artifacts.
                
                Automatically detects project type and chooses appropriate analyzers:
                - Java/Spring Boot projects → Spoon (via Docker, if available)
                - .NET projects → Roslyn (via Docker, if available)
                - JavaScript, TypeScript, Python, Go, etc. → Tree-sitter
                
                Handles mixed codebases by using multiple analyzers as needed.
                This is the first step in the advanced indexing pipeline.
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory to analyze (relative to codebase root, e.g. 'src' or 'backend/services')"
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Optional: Directory to save analysis JSON artifacts (default: .agentic-rag-cache/analysis/)"
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Limit analysis to specific file extensions (e.g. ['.java', '.cs'])"
                        }
                    },
                    "required": ["directory"]
                }
            ),
            Tool(
                name="index-codebase",
                description="""
                Incrementally index an entire codebase into Qdrant + AuraDB with hash-based change detection.

                Two parallel pipelines:
                  - Pipeline A (Qdrant): ALL non-binary files → TreeSitter/Markdown/YAML semantic chunking → embed → Qdrant
                  - Pipeline B (AuraDB): .sln/.pom files whose source files changed → Roslyn/Spoon AST analysis → AuraDB graph

                Only changed/new files are reprocessed. Deleted files are removed from both databases.
                Re-running is safe and efficient — unchanged files are skipped automatically.
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Root directory of the codebase to index (absolute path)"
                        }
                    },
                    "required": ["directory"]
                }
            ),
            Tool(
                name="debug-env",
                description="Debug tool to check MCP server environment",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

        # Conditionally add graph tools if Neo4j is enabled
        if _is_neo4j_enabled():
            tools.extend([
                Tool(
                    name="graph-query",
                    description="""
                    Execute a Cypher query on the code knowledge graph (Neo4j).

                    The graph contains Symbol nodes (classes, methods, interfaces, etc.)
                    and File nodes, connected by relationships like MEMBER_OF, INHERITS,
                    IMPLEMENTS, IMPORTS, CALLS, USES_TYPE, DEFINED_IN.

                    Use this for advanced graph traversal queries.
                    """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Cypher query to execute"
                            },
                            "params": {
                                "type": "object",
                                "description": "Optional query parameters",
                                "default": {}
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="graph-neighbors",
                    description="""
                    Get related symbols for a given symbol name from the code knowledge graph.

                    Returns callers, callees, parent class, implementations, type references, etc.
                    Supports 1-3 hop depth and optional relationship type filtering.
                    """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol name or fully qualified name (e.g. 'DepositService' or 'OptimusPay.Internal.Business.DepositService')"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Number of hops (1-3, default 1)",
                                "default": 1
                            },
                            "relationship_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional: filter by relationship types (e.g. ['MEMBER_OF', 'INHERITS', 'CALLS'])"
                            },
                            "project": {
                                "type": "string",
                                "description": "Optional: project name to scope the search (defaults to GRAPH_PROJECT env var)"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="graph-list-projects",
                    description="List all projects indexed in the graph database.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="graph-status",
                    description="Show knowledge graph statistics: node counts and edge counts by type, with per-project breakdown.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project": {
                                "type": "string",
                                "description": "Optional: filter stats to a specific project"
                            }
                        }
                    }
                ),
            ])

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """處理工具調用"""

        if name == "agentic-search":
            query = arguments.get("query", "")
            max_iter = arguments.get("max_iterations", 5)
            llm_provider = arguments.get("llm_provider")
            llm_model = arguments.get("llm_model")

            if llm_provider or llm_model:
                # 有 override：建立臨時 agent，不影響 singleton
                tmp_config = AgenticSearch._load_default_config()
                tmp_config.max_iterations = max_iter
                agent = AgenticSearch(
                    config=tmp_config,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                )
                logger.info(f"agentic-search override: provider={llm_provider} model={llm_model}")
            else:
                agent = get_search_agent()
                agent.config.max_iterations = max_iter

            # 執行搜索
            result = agent.search(query)

            # 格式化回應
            text = format_response(result)

            return [TextContent(type="text", text=text)]

        elif name == "quick-search":
            query = arguments.get("query", "")
            operator = arguments.get("operator", "hybrid")
            top_k = arguments.get("top_k", 10)

            from .hybrid_search import HybridSearch
            from .reranker import create_reranker

            search = HybridSearch()
            reranker = create_reranker(use_cross_encoder=False)  # 用簡單 reranker 加速

            # 搜索
            results = search.search(query, operator=operator, top_n=top_k * 2)
            reranked = reranker.rerank(query, results, top_m=top_k)

            # 格式化
            parts = [f"## Quick Search Results for: {query}\n"]
            for i, r in enumerate(reranked, 1):
                parts.append(f"### {i}. {r.get('path', 'Unknown')}")
                parts.append(f"Score: {r.get('score_rerank', 0):.3f}")
                content = r.get('content', '')[:300]
                parts.append(f"```\n{content}\n```\n")

            return [TextContent(type="text", text="\n".join(parts))]

        elif name == "index-status":
            svc = get_indexer_service()
            result = svc.get_status()
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False, default=str))]

        elif name == "analyze-codebase":
            directory = arguments.get("directory", "")
            output_dir = arguments.get("output_dir")
            file_extensions = arguments.get("file_extensions")

            if not directory:
                return [TextContent(type="text", text="Error: directory is required")]

            svc = get_indexer_service()
            result = svc.run_analysis(directory, output_dir=output_dir, file_extensions=file_extensions)

            # Slim response: replace file lists with counts to avoid token overflow
            slim = {
                "directory": result.get("directory"),
                "output_dir": result.get("output_dir"),
                "total_items_analyzed": result.get("total_items_analyzed", 0),
                "total_items_skipped": result.get("total_items_skipped", 0),
                "results_by_analyzer": {
                    analyzer: {
                        k: (len(v) if isinstance(v, list) else v)
                        for k, v in info.items()
                    }
                    for analyzer, info in result.get("results_by_analyzer", {}).items()
                },
                "errors": result.get("errors", [])[:10],  # cap at 10
                "errors_total": len(result.get("errors", [])),
            }
            return [TextContent(type="text", text=json.dumps(slim, indent=2, ensure_ascii=False))]

        elif name == "index-codebase":
            directory = arguments.get("directory", "")
            if not directory:
                return [TextContent(type="text", text="Error: directory is required")]

            svc = get_indexer_service()
            result = svc.index_codebase(directory)

            slim = {
                "files_scanned": result.get("files_scanned", 0),
                "qdrant_indexed": result.get("qdrant_indexed", 0),
                "qdrant_skipped": result.get("qdrant_skipped", 0),
                "qdrant_deleted": result.get("qdrant_deleted", 0),
                "graph_solutions_indexed": result.get("graph_solutions_indexed", 0),
                "graph_deleted": result.get("graph_deleted", 0),
                "errors": result.get("errors", [])[:10],
                "errors_total": result.get("errors_total", 0),
            }
            if "warning" in result:
                slim["warning"] = result["warning"]
            return [TextContent(type="text", text=json.dumps(slim, indent=2, ensure_ascii=False))]

        elif name == "debug-env":
            import os
            debug_info = {
                "cwd": os.getcwd(),
                "env": dict(os.environ),
                "sys_path": sys.path if 'sys' in locals() else []
            }
            if 'sys' not in locals():
                import sys
                debug_info["sys_path"] = sys.path
            return [TextContent(type="text", text=json.dumps(debug_info, indent=2))]

        elif name == "graph-query":
            gs = get_graph_store()
            if gs is None:
                return [TextContent(type="text", text="Error: Neo4j is not enabled or not reachable")]
            query = arguments.get("query", "")
            params = arguments.get("params", {})
            try:
                results = gs.cypher_query(query, params)
                return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]
            except Exception as e:
                return [TextContent(type="text", text=f"Cypher query error: {e}")]

        elif name == "graph-neighbors":
            gs = get_graph_store()
            if gs is None:
                return [TextContent(type="text", text="Error: Neo4j is not enabled or not reachable")]
            symbol = arguments.get("symbol", "")
            depth = arguments.get("depth", 1)
            rel_types = arguments.get("relationship_types")
            project = arguments.get("project")
            try:
                result = gs.get_neighbors(symbol, depth=depth, relationship_types=rel_types, project=project)
                # Format for readability
                effective_project = project or gs.default_project
                parts = [f"## Neighbors of: {symbol} (project={effective_project})\n"]
                parts.append(f"Found {len(result['nodes'])} nodes, {len(result['edges'])} edges\n")
                if result["nodes"]:
                    parts.append("### Nodes")
                    for n in result["nodes"]:
                        parts.append(f"- **{n['name']}** ({n['kind']}) @ `{n['file_path']}`")
                if result["edges"]:
                    parts.append("\n### Edges")
                    for e in result["edges"]:
                        parts.append(f"- {e['source']} --[{e['type']}]--> {e['target']}")
                return [TextContent(type="text", text="\n".join(parts))]
            except Exception as e:
                return [TextContent(type="text", text=f"Graph neighbors error: {e}")]

        elif name == "graph-list-projects":
            gs = get_graph_store()
            if gs is None:
                return [TextContent(type="text", text="Error: Neo4j is not enabled or not reachable")]
            try:
                projects = gs.list_projects()
                parts = ["## Indexed Projects\n"]
                if projects:
                    for p in projects:
                        parts.append(f"- `{p}`")
                else:
                    parts.append("No projects indexed yet.")
                return [TextContent(type="text", text="\n".join(parts))]
            except Exception as e:
                return [TextContent(type="text", text=f"Graph list-projects error: {e}")]

        elif name == "graph-status":
            gs = get_graph_store()
            if gs is None:
                return [TextContent(type="text", text="Error: Neo4j is not enabled or not reachable")]
            try:
                stats = gs.get_stats()
                return [TextContent(type="text", text=json.dumps(stats, indent=2, default=str))]
            except Exception as e:
                return [TextContent(type="text", text=f"Graph status error: {e}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def main():
        """啟動 MCP Server"""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

else:
    async def main():
        print("MCP not available. Install with: pip install mcp")


# CLI 測試入口
def cli_test():
    """CLI 測試"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mcp_server.py <query>")
        print("Example: python mcp_server.py 'Robot 斷線後交易怎麼處理？'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Searching: {query}\n")

    agent = get_search_agent()
    result = agent.search(query)

    print(format_response(result))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] != "--server":
        # CLI 模式
        cli_test()
    else:
        # MCP Server 模式
        asyncio.run(main())
