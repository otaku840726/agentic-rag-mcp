import os
import json
from typing import Dict, Any, List

def semantic_search_tool(query: str, hybrid_search, query_builder, reranker, top_k: int = 20, cid: str = None, exclude_cids: List[int] = None) -> List[Dict[str, Any]]:
    from .models import QueryIntent

    intent = QueryIntent(query=query, purpose="", query_type="semantic", operator="hybrid")
    queries = query_builder.build_from_intent(intent)

    all_raw_results = []
    for q_dict in queries:
        text_query = q_dict["query"]
        op = q_dict.get("operator", "hybrid")
        filters = q_dict.get("filters", {})
        
        if cid and str(cid).isdigit():
            filters["community_id"] = int(cid)
            
        # ──【新增：物理層級過濾】──
        if exclude_cids:
            filters["exclude_community_ids"] = exclude_cids

        try:
            raw = hybrid_search.search(text_query, operator=op, filters=filters, top_n=50)
        except TypeError:
            raw = hybrid_search.search(text_query, operator=op, top_n=50)

        if exclude_cids:
            raw = [r for r in raw if r.get("payload", {}).get("community_id") not in exclude_cids]

        all_raw_results.extend(raw)

    reranked = reranker.rerank(query, all_raw_results, top_m=top_k)
    return reranked

def graph_symbol_search_tool(symbol: str, graph_store, depth: int = 1) -> List[Dict[str, Any]]:
    if not graph_store:
        return []
    try:
        results = graph_store.get_neighbors(symbol, depth=depth)
        return results
    except Exception as e:
        return [{"error": str(e)}]

def graph_list_files_tool(graph_store, dir_path: str = None, cid: str = None, pattern: str = None) -> List[Dict[str, Any]]:
    """利用 Neo4j 進行結構化檔案導航"""
    if not graph_store:
        return []
    try:
        project = getattr(graph_store, "default_project", "smilepay")
        
        conditions = ["f.project = $project"]
        params = {"project": project}
        
        if dir_path:
            conditions.append("f.path CONTAINS $dir_path")
            params["dir_path"] = dir_path
        if cid and str(cid).isdigit():
            conditions.append("f.communityId = $cid")
            params["cid"] = int(cid)
        if pattern:
            conditions.append("f.path =~ $pattern")
            params["pattern"] = f"(?i).*{pattern}.*"

        query = f"MATCH (f:File) WHERE {' AND '.join(conditions)} RETURN f.path as path, f.communityId as cid LIMIT 50"
        results = graph_store.cypher_query(query, params)
        return results
    except Exception as e:
        return [{"error": str(e)}]
def read_exact_file_tool(path: str, line_start: int = None, line_end: int = None, hybrid_search=None) -> str:
    """
    讀取檔案內容 (支援物理讀取與 Qdrant 虛擬備援)
    """
    # ──【資料淨化】防呆機制：移除可能被 LLM 誤加的 CID 前綴 ──
    if ":" in path:
        prefix = path.split(":")[0].strip()
        if prefix.isdigit() or prefix.lower() == "cid":
            path = path.split(":", 1)[1].strip()

    # 1. 嘗試物理路徑讀取
    try:
        potential_paths = [path, os.path.abspath(path)]
        
        # 支援從環境變數讀取目標專案根目錄
        target_dir = os.environ.get("TARGET_DIR", "")
        if target_dir and not path.startswith(target_dir):
            potential_paths.insert(0, os.path.join(target_dir, path))

        parts = path.split("/")
        if len(parts) > 1:
            potential_paths.append("/".join(parts[1:]))
            
        found_path = None
        for p in potential_paths:
            if os.path.exists(p) and os.path.isfile(p):
                found_path = p
                break
        
        if found_path:
            with open(found_path, "r", encoding="utf-8") as f:
                content_lines = f.readlines()
                
            # 根據行號精確提取 (1-based index)
            if line_start is not None and line_end is not None:
                start = max(1, int(line_start))
                end = min(len(content_lines), int(line_end))
                selected = content_lines[start-1:end]
                return f"[FILE: {path} | LINES: {start}-{end}]\n" + "".join(selected)
            elif line_start is not None:
                start = max(1, int(line_start))
                selected = content_lines[start-1:]
                return f"[FILE: {path} | LINES: {start}-END]\n" + "".join(selected)
                
            return f"[FILE: {path} | FULL CONTENT]\n" + "".join(content_lines)
            
    except Exception:
        pass # 轉向備援
        
    # 2. 物理讀取失敗，嘗試 Qdrant 虛擬讀取
    if hybrid_search and hasattr(hybrid_search, "client"):
        try:
            from qdrant_client import models
            collection = hybrid_search.collection_name
            
            # 使用 MatchValue 進行精確的完整路徑匹配，避免 MatchText 被 keyword 索引拒絕
            points, _ = hybrid_search.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=path)
                        )
                    ]
                ),
                limit=100,
                with_payload=True
            )
            
            if points:
                target_points = points
                target_points.sort(key=lambda p: p.payload.get("chunk_index", 0))
                
                full_text = []
                for p in target_points:
                    txt = p.payload.get("content") or p.payload.get("content_preview") or ""
                    full_text.append(str(txt))
                
                if full_text:
                    return f"[VIRTUAL READ from Qdrant: {target_points[0].payload.get('file_path')}]\n" + "\n".join(full_text)
        except Exception as qe:
            return f"Error: File not found physically, and Qdrant fallback failed: {qe}"
            
    return f"Error: File not found at {path}"

def list_directory_tool(path: str, max_items: int = 50) -> List[str]:
    ignore_dirs = {".git", ".agentic-rag-cache", "node_modules", "venv", "__pycache__", ".venv", ".idea", ".vscode", "target", "bin", "obj", "dist", "build"}
    try:
        all_items = os.listdir(path)
        filtered = [item for item in all_items if item not in ignore_dirs]
        result = sorted(filtered)[:max_items]
        if len(filtered) > max_items:
            result.append(f"... (and {len(filtered) - max_items} more items hidden)")
        return result
    except Exception as e:
        return [f"Error listing directory: {e}"]
