import time
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from .state import AgenticState
from .manager import ManagerAgent
from .worker import WorkerAgent
from src.agentic_rag_mcp.tools import (
    semantic_search_tool, 
    graph_symbol_search_tool, 
    read_exact_file_tool,
    graph_list_files_tool
)

logger = logging.getLogger(__name__)

class V3AgenticSearch:
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
                return semantic_search_tool(args.get("query", ""), self.hybrid_search, self.query_builder, self.reranker, top_k=5, cid=args.get("cid"))
            elif tool_name == "graph_symbol_search":
                return graph_symbol_search_tool(args.get("symbol", ""), self.graph_store)
            elif tool_name == "graph_list_files":
                return graph_list_files_tool(self.graph_store, dir_path=args.get("dir_path"), cid=args.get("cid"), pattern=args.get("pattern"))
            elif tool_name == "read_exact_file":
                return read_exact_file_tool(args.get("path", ""), hybrid_search=self.hybrid_search)
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
            return "worker"
            
        def route_worker(state: AgenticState) -> str:
            if not state.get("current_task"):
                # Worker 交卷了 (或放棄了)
                return "manager"
            return "worker" # 繼續執行自己發起的連續 Tool Call
            
        builder.add_conditional_edges("manager", route_manager, {"worker": "worker", END: END})
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
