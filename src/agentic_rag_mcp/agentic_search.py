"""
Agentic Search - 主循環控制
協調 Analyst, SubTaskDelegator (Map-Reduce), WorkerGraph, Synthesizer
"""

import os
import time
import logging
import uuid
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.graph import StateGraph, END

from .models import (
    GraphState, 
    EvidenceCard, 
    SynthesizedResponse, 
    SearchResult,
    AgenticSearchConfig,
    SynthesizerConfig,
    PlannerConfig,
    SubTask,
    SourceKind
)
from .provider import get_component_config, create_client_for, get_neo4j_config
from .analyst import Analyst
from .synthesizer import Synthesizer
from .evidence_store import EvidenceStore
from .search_logger import SearchTraceLogger
from .tools import (
    semantic_search_tool, 
    graph_symbol_search_tool, 
    read_exact_file_tool,
    graph_list_files_tool
)

logger = logging.getLogger(__name__)

class AgenticSearch:
    def __init__(self, hybrid_search=None, query_builder=None, reranker=None, graph_enhancer=None, config: Optional[AgenticSearchConfig] = None):
        self.logger = logger
        self.search_logger = SearchTraceLogger()
        self.config = config or AgenticSearchConfig()
        
        if hybrid_search is None:
            from .hybrid_search import HybridSearch
            hybrid_search = HybridSearch()
        if query_builder is None:
            from .query_builder import QueryBuilder
            query_builder = QueryBuilder()
        if reranker is None:
            # ──【修復：從配置讀取 Reranker】──
            from .reranker import Reranker, RerankerConfig
            r_base = get_component_config("reranker")
            r_config = RerankerConfig(
                provider=r_base.provider,
                model_name=r_base.model
            )
            reranker = Reranker(config=r_config)
            
        self.graph_store = None
        try:
            from .indexer.graph_store import GraphStore
            n_cfg = get_neo4j_config()
            self.graph_store = GraphStore(
                uri=n_cfg["uri"],
                username=n_cfg["username"],
                password=n_cfg["password"],
                database=n_cfg.get("database", "neo4j"),
                project=os.getenv("GRAPH_PROJECT", "smilepay")
            )
        except Exception as ge:
            logger.warning(f"GraphStore initialization failed: {ge}")

        if graph_enhancer is None:
            from .graph_search import GraphSearchEnhancer
            if self.graph_store:
                graph_enhancer = GraphSearchEnhancer(self.graph_store, hybrid_search)
            else:
                graph_enhancer = None

        self.hybrid_search = hybrid_search
        self.query_builder = query_builder
        self.reranker = reranker
        self.graph_enhancer = graph_enhancer
        
        self.evidence_store = EvidenceStore()
        self.analyst = Analyst()
        self.synthesizer = Synthesizer(self._build_llm_configs("synthesizer", SynthesizerConfig))
        self.budget = self._build_llm_configs("planner", PlannerConfig)
        
        self.graph = self._build_graph()

    def _build_llm_configs(self, component: str, config_cls):
        base = get_component_config(component)
        if config_cls.__name__ == "PlannerConfig":
            return config_cls(max_iterations=self.config.max_iterations, temperature=base.temperature)
        return config_cls(
            provider=base.provider,
            model=base.model,
            max_tokens=base.max_tokens,
            temperature=base.temperature
        )

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("context_awareness", self._node_context_awareness)
        builder.add_node("analyst", self._node_analyst)
        builder.add_node("sub_task_delegator", self._node_sub_task_delegator)
        builder.add_node("synthesizer", self._node_synthesizer)
        builder.set_entry_point("context_awareness")
        builder.add_edge("context_awareness", "analyst")
        builder.add_edge("analyst", "sub_task_delegator")
        builder.add_edge("sub_task_delegator", "synthesizer")
        builder.add_edge("synthesizer", END)
        return builder.compile()

    def _node_context_awareness(self, state: GraphState) -> Dict[str, Any]:
        return {}

    def _node_analyst(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"--- [Node: Analyst] Decomposing query ---")
        full_query = state["query"]
        if state.get("module_map"):
            full_query = f"{state['query']}\n\n[CRITICAL System Module Map]:\n{state['module_map']}\n(Investigate EACH core module listed above!)"
        out = self.analyst.analyze(full_query)
        sub_tasks = [SubTask(id=f"task_{i}", description=desc, assigned_domain="") for i, desc in enumerate(out.sub_tasks)]
        return {"intent": out.intent, "sub_tasks": sub_tasks}

    def _execute_tool(self, tool_name: str, args: Dict[str, Any], exclude_cids: List[int] = None) -> List[Dict[str, Any]]:
        if tool_name == "semantic_search":
            return semantic_search_tool(args.get("query", ""), self.hybrid_search, self.query_builder, self.reranker, top_k=self.config.top_n_search, cid=args.get("cid"), exclude_cids=exclude_cids)
        elif tool_name == "graph_symbol_search":
            return [{"path": "graph_search", "content": str(graph_symbol_search_tool(args.get("symbol", ""), self.graph_store, args.get("depth", 1)))}]
        elif tool_name == "graph_list_files":
            res = graph_list_files_tool(self.graph_store, dir_path=args.get("dir_path"), cid=args.get("cid"), pattern=args.get("pattern"))
            return [{"path": "graph_ls", "content": json.dumps(res, indent=2)}]
        elif tool_name == "read_exact_file":
            # ──【優化：傳入 hybrid_search 實例以支持 Qdrant 備援讀取】──
            return [{"path": args.get("path", ""), "content": read_exact_file_tool(args.get("path", ""), args.get("lines"), hybrid_search=self.hybrid_search)}]
        return []

    def _node_sub_task_delegator(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"--- [Node: Delegator] Dispatching workers ---")
        from .worker.worker_graph import WorkerGraph
        import concurrent.futures
        
        unique_tasks = []
        seen_intents = set()
        for t in state['sub_tasks']:
            intent_key = "".join(sorted(t.description.lower().split()))[:30]
            if intent_key not in seen_intents:
                unique_tasks.append(t)
                seen_intents.add(intent_key)
        
        final_tasks = unique_tasks[:2]
        logger.info(f"Consolidated into {len(final_tasks)} key investigations.")

        starting_summary = self.evidence_store.get_summary_for_planner()
        global_exclude_cids = []

        def execute_tools_callback(tool_calls, iteration, exclude_cids=None):
            new_cards = []
            for tc in tool_calls:
                try:
                    logger.info(f"Worker executing Tool: {tc['tool']} with args: {tc['args']}")
                    raw_result = self._execute_tool(tc["tool"], tc["args"], exclude_cids=exclude_cids)
                    cards = self._convert_to_evidence_cards(raw_result, tc, iteration)
                    new_cards.extend(cards)
                except Exception as e:
                    logger.error(f"Worker tool execution failed ({tc['tool']}): {e}")
            
            if self.graph_enhancer and new_cards:
                try:
                    expanded_cards = self.graph_enhancer.expand_evidence(new_cards, top_k=8)
                    if expanded_cards:
                        ecards = self._convert_to_evidence_cards(expanded_cards, {"tool":"graph_expansion"}, iteration)
                        new_cards.extend(ecards)
                except Exception as e:
                    logger.warning(f"Graph expansion in worker failed: {e}")
            return new_cards

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for task in final_tasks:
                worker = WorkerGraph(execute_tools_callback, self.budget)
                futures.append(executor.submit(worker.run, task.id, task.description, state["query"], starting_knowledge=starting_summary, exclude_cids=global_exclude_cids))
                
            reports = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_state = future.result()
                    reports.append(worker_state.get("final_report", "No report."))
                    if "local_evidence" in worker_state:
                        self.evidence_store.add(worker_state["local_evidence"])
                except Exception as e:
                    logger.error(f"Worker execution failed: {e}")
                    
        return {"worker_reports": reports}

    def _node_synthesizer(self, state: GraphState) -> Dict[str, Any]:
        logger.info(f"--- [Node: Synthesizer] Aggregating worker reports ---")
        all_valid_cards = [c for c in self.evidence_store.get_all_cards() if c.id not in state.get("rejected_ids", [])]
        all_valid_cards.sort(key=lambda c: c.score_rerank, reverse=True)
        
        reports = state.get("worker_reports", [])
        worker_context = "\n### Worker Reports Context:\n"
        for i, r in enumerate(reports):
            summary = r[:1500] + "..." if len(r) > 1500 else r
            worker_context += f"\n-- Report {i+1} --\n{summary}\n"

        response = self.synthesizer.synthesize(
            query=f"{state['query']}\n\n{worker_context}",
            evidence_cards=all_valid_cards[:25],
            search_history=state.get("search_history", []),
            iterations=1,
            logger=self.search_logger
        )
        return {"final_response": response}

    def _convert_to_evidence_cards(self, raw_results, tool_call, round_idx) -> List[EvidenceCard]:
        cards = []
        import hashlib
        for res in raw_results:
            text = res.get("content") or res.get("snippet") or ""
            fp = hashlib.md5(text.encode()).hexdigest()
            cards.append(EvidenceCard(
                id=res.get("id") or str(uuid.uuid4()),
                path=res.get("path") or "",
                symbol=res.get("symbol"),
                snippet=text[:300],
                chunk_text=text,
                score_hybrid=res.get("score", 0.5),
                score_rerank=res.get("score_rerank", 0.5),
                round_found=round_idx,
                source_kind=SourceKind.FILE if "path" in res else SourceKind.GRAPH,
                fingerprint=fp,
                community_id=res.get("community_id") or res.get("cid"),
                named_entities=[]
            ))
        return cards

    def search(self, query: str) -> SearchResult:
        start_time = time.time()
        module_map = ""
        if self.graph_store:
            try:
                stop_words = {"should", "what", "fill", "filled", "with", "api", "create"}
                keywords = [kw.lower() for kw in re.findall(r'\w+', query) if kw.lower() not in stop_words and len(kw) > 3]
                relevant_comms = []
                for kw in keywords:
                    res = self.graph_store.cypher_query(
                        "MATCH (c:Community {project: $project}) WHERE toLower(c.name) CONTAINS $kw RETURN c.id as id, c.name as name LIMIT 2",
                        {"kw": kw, "project": self.graph_store.default_project}
                    )
                    relevant_comms.extend(res)
                if relevant_comms:
                    unique_comms = {str(c['id']): c['name'] for c in relevant_comms}
                    module_map = "Highly relevant modules found in graph:\n" + "\n".join([f"- CID {cid}: {name}" for cid, name in unique_comms.items()])
                    logger.info(f"Pre-search guidance generated:\n{module_map}")
            except Exception as e:
                logger.warning(f"Pre-search failed: {e}")

        initial_state: GraphState = {"query": query, "module_map": module_map, "intent": "", "sub_tasks": [], "worker_reports": [], "search_history": [], "rejected_ids": [], "final_response": None}
        try:
            final_state = self.graph.invoke(initial_state)
            elapsed = int((time.time() - start_time) * 1000)
            return SearchResult(success=True, response=final_state.get("final_response"), debug_info={"time_ms": elapsed})
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return SearchResult(success=False, error=str(e))
