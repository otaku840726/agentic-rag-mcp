import os
from typing import Dict, Any, List

def semantic_search_tool(query: str, hybrid_search, query_builder, reranker, top_k: int = 20) -> List[Dict[str, Any]]:
    from .models import QueryIntent

    intent = QueryIntent(query=query, purpose="", query_type="semantic", operator="hybrid")
    queries = query_builder.build_from_intent(intent)

    all_raw_results = []
    for q in queries:
        raw = hybrid_search.search(q, operator="hybrid", top_n=50)
        all_raw_results.extend(raw)

    reranked = reranker.rerank(query, all_raw_results, top_m=top_k)
    return reranked

def graph_symbol_search_tool(symbol: str, graph_store, depth: int = 1) -> Dict[str, Any]:
    if not graph_store:
        return {"error": "Graph store not available or Neo4j disabled"}
    try:
        res = graph_store.get_neighbors(symbol, depth=depth)
        return res
    except Exception as e:
        return {"error": str(e)}

def read_exact_file_tool(path: str, lines: str = None) -> str:
    """Reads lines from a file. lines can be like '10-20'"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content_lines = f.readlines()

        if not lines:
            return "".join(content_lines)

        parts = lines.split("-")
        if len(parts) == 2:
            start = max(0, int(parts[0]) - 1)
            end = min(len(content_lines), int(parts[1]))
            return "".join(content_lines[start:end])
        else:
            return "".join(content_lines)
    except Exception as e:
        return f"Error reading file: {e}"

def list_directory_tool(path: str, max_items: int = 50) -> List[str]:
    ignore_dirs = {".git", "node_modules", "venv", "__pycache__", ".venv", ".idea", ".vscode", "target", "bin", "obj", "dist", "build"}
    try:
        all_items = os.listdir(path)
        filtered = [item for item in all_items if item not in ignore_dirs]

        # Sort so directories and files are somewhat consistent
        filtered.sort()

        result = filtered[:max_items]
        if len(filtered) > max_items:
            result.append(f"... (and {len(filtered) - max_items} more items hidden)")

        return result
    except Exception as e:
        return [f"Error listing directory: {e}"]
