import time
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from .state import AgenticState
from .manager import ManagerAgent
from .worker import WorkerAgent
from .analyst import AnalystAgent
from .tools import (
    semantic_search_tool, 
    graph_symbol_search_tool, 
    read_exact_file_tool,
    graph_list_files_tool,
    community_search_tool,
    list_file_symbols_tool
)

logger = logging.getLogger(__name__)

class AgenticSearch:
    def __init__(self, hybrid_search, query_builder, reranker, graph_enhancer=None, config=None):
        self.hybrid_search = hybrid_search
        self.query_builder = query_builder
        self.reranker = reranker
        self.graph_store = graph_enhancer.graph if graph_enhancer else None
        
        self.manager = ManagerAgent(self._execute_tool)
        self.worker = WorkerAgent(self._execute_tool)
        
        self.graph = self._build_graph()

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        try:
            if tool_name == "semantic_search":
                # 【修改】將 top_k 改回 5，避免過多雜訊導致 LLM 產生幻覺或混亂
                raw_res = semantic_search_tool(args.get("query", ""), self.hybrid_search, self.query_builder, self.reranker, top_k=5, cid=args.get("cid"))
                clean_res = []
                for r in raw_res:
                    if isinstance(r, dict):
                        file_path = r.get("file_path") or r.get("payload", {}).get("file_path", "")
                        cid_val = r.get("community_id") or r.get("payload", {}).get("community_id", "Unknown")
                        content = r.get("content") or r.get("payload", {}).get("content", "")
                        if not content:
                            content = r.get("content_preview") or r.get("payload", {}).get("content_preview", "")
                    else:
                        payload = getattr(r, "payload", {})
                        file_path = getattr(r, "file_path", payload.get("file_path", ""))
                        cid_val = getattr(r, "community_id", payload.get("community_id", "Unknown"))
                        content = getattr(r, "content", payload.get("content", ""))
                        if not content:
                            content = getattr(r, "content_preview", payload.get("content_preview", ""))
                            
                    clean_res.append({
                        "file_path": file_path,
                        "community_id": cid_val,
                        "snippet": str(content)[:800]
                    })
                return clean_res
            elif tool_name == "graph_symbol_search":
                return graph_symbol_search_tool(args.get("symbol", ""), self.graph_store)
            elif tool_name == "list_file_symbols":
                return list_file_symbols_tool(args.get("file_path", ""), self.graph_store)
            elif tool_name == "graph_list_files":
                return graph_list_files_tool(self.graph_store, dir_path=args.get("dir_path"), cid=args.get("cid"), pattern=args.get("pattern"))
            elif tool_name == "read_exact_file":
                return read_exact_file_tool(
                    path=args.get("path", ""),
                    line_start=args.get("line_start"),
                    line_end=args.get("line_end"),
                    hybrid_search=self.hybrid_search
                )
            elif tool_name == "community_search":
                return community_search_tool(query=args.get("query", ""), graph_store=self.graph_store)
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"
        return "Unknown Tool."

    def _build_graph(self):
        builder = StateGraph(AgenticState)
        
        builder.add_node("manager", self.manager.act)
        builder.add_node("worker", self.worker.act)
        
        builder.set_entry_point("manager")
        
        # 路由邏輯
        def route_manager(state: AgenticState) -> str:
            if state.get("is_finished"):
                return END
            if state.get("current_task"):
                return "worker"
            # 如果 Manager 只是在做 Scouting，沒有派發新任務，則繼續留在 Manager
            return "manager"
            
        def route_worker(state: AgenticState) -> str:
            if not state.get("current_task"):
                # Worker 交卷了 (或放棄了)
                return "manager"
            return "worker" # 繼續執行自己發起的連續 Tool Call
            
        builder.add_conditional_edges("manager", route_manager, {"worker": "worker", "manager": "manager", END: END})
        builder.add_conditional_edges("worker", route_worker, {"manager": "manager", "worker": "worker"})
        
        return builder.compile()

    def search(self, query: str, project_context: str) -> str:
        initial_state = {
            "query": query,
            "project_context": project_context,
            "manager_thoughts": [],
            "investigation_log": [],
            "current_task": "",
            "target_cids": [],
            "current_worker_messages": [],
            "final_answer": "",
            "is_finished": False
        }
        
        # 設定執行次數上限，防止無限打乒乓球
        config = {"recursion_limit": 100}
        
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state.get("final_answer", "Investigation timed out without a final answer.")
