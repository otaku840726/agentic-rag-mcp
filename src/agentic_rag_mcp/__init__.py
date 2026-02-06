"""
Agentic RAG MCP Server
Multi-hop retrieval-augmented generation for codebase search
"""

import asyncio
from .mcp_server import main as _main, cli_test

__version__ = "0.1.0"
__all__ = ["main", "cli_test"]


def main():
    """Entry point for the MCP server"""
    asyncio.run(_main())
