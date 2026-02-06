"""
MCP Server - Agentic RAG 搜索服務
提供給 Claude Code 等工具調用
"""

import os
import json
import asyncio
from typing import Any
from dataclasses import asdict

from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

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


# 初始化搜索代理
search_agent = None


def get_search_agent() -> AgenticSearch:
    """獲取或創建搜索代理（單例）"""
    global search_agent
    if search_agent is None:
        config = AgenticSearchConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
            total_token_budget=int(os.getenv("TOKEN_BUDGET", "15000")),
            planner_model=os.getenv("PLANNER_MODEL", "gpt-4o-mini"),
            synthesizer_model=os.getenv("SYNTHESIZER_MODEL", "gpt-4o-mini"),
        )
        search_agent = AgenticSearch(config)
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
        return [
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
                            "enum": ["semantic", "keyword", "exact"],
                            "description": "Search operator (default: semantic)",
                            "default": "semantic"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """處理工具調用"""

        if name == "agentic-search":
            query = arguments.get("query", "")
            max_iter = arguments.get("max_iterations", 5)

            # 創建配置
            agent = get_search_agent()
            agent.config.max_iterations = max_iter

            # 執行搜索
            result = agent.search(query)

            # 格式化回應
            text = format_response(result)

            return [TextContent(type="text", text=text)]

        elif name == "quick-search":
            query = arguments.get("query", "")
            operator = arguments.get("operator", "semantic")
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
