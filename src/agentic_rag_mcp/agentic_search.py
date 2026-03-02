"""
Agentic Search - 主循環控制
協調 Analyst, Planner, Hybrid Search, Reranker, Evidence Store, Synthesizer
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from .models import (
    SearchResult, SynthesizedResponse, GraphState,
    EvidenceCard, QueryIntent, MissingEvidence,
    Budget, QualityGate
)
from langgraph.graph import StateGraph, END

from .tools import (
    semantic_search_tool, graph_symbol_search_tool,
    read_exact_file_tool, list_directory_tool
)
from .utils import (
    compute_fingerprint, extract_named_entities,
    extract_tags, detect_source_kind, create_snippet
)
from .evidence_store import EvidenceStore
from .query_builder import QueryBuilder
from .budget import StopConditionChecker
from .hybrid_search import HybridSearch, HybridSearchBatch
from .reranker import create_reranker
from .analyst import Analyst, EnsembleAnalyst, AnalystConfig
from .planner import Planner, PlannerConfig
from .synthesizer import Synthesizer, SynthesizerConfig
from .search_logger import SearchTraceLogger
from .provider import get_section_config, get_neo4j_config
import uuid


@dataclass
class AgenticSearchConfig:
    """Agentic Search 配置"""
    # Budget
    max_iterations: int = 5
    total_token_budget: int = 15000

    # Evidence
    max_evidence_cards: int = 200
    working_set_size: int = 20

    # Search
    top_n_search: int = 100
    top_m_rerank: int = 20

    # Quality Gate
    min_code_evidence: int = 2
    min_tag_diversity: int = 2
    require_call_edge: bool = True
    require_named_entity: bool = True

    # Reranker
    use_cross_encoder: bool = True

    # LLM models are now configured in config.yaml (planner/synthesizer sections)


class AgenticSearch:
    """自主搜索代理"""

    def __init__(
        self,
        config: Optional[AgenticSearchConfig] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.config = config or self._load_default_config()

        # 初始化組件
        self.evidence_store = EvidenceStore(
            max_pool=self.config.max_evidence_cards,
            working_set_size=self.config.working_set_size
        )
        self.query_builder = QueryBuilder()
        self.hybrid_search = HybridSearch()
        self.batch_search = HybridSearchBatch(self.hybrid_search)
        self.reranker = create_reranker(self.config.use_cross_encoder)

        # LLM 組件 — 可被 llm_provider / llm_model 覆蓋
        llm_override = self._build_llm_configs(llm_provider, llm_model)
        self.analyst = EnsembleAnalyst(config=llm_override.get("analyst"))
        self.planner = Planner(config=llm_override.get("planner"))
        self.synthesizer = Synthesizer(config=llm_override.get("synthesizer"))
        self.logger = SearchTraceLogger()

        # Optional: Graph search enhancer
        self._graph_enhancer = None
        self._graph_enhancer_checked = False

        # 停機條件
        self.budget = Budget(
            max_iterations=self.config.max_iterations,
            total_token_budget=self.config.total_token_budget
        )
        quality_gate = QualityGate(
            min_code_evidence=self.config.min_code_evidence,
            min_tag_diversity=self.config.min_tag_diversity,
            require_call_edge=self.config.require_call_edge,
            require_named_entity=self.config.require_named_entity
        )
        self.stop_checker = StopConditionChecker(self.budget, quality_gate)

        self.graph = self._build_graph()

    @property
    def graph_enhancer(self):
        """Lazy-init graph search enhancer (returns None if Neo4j disabled)."""
        if not self._graph_enhancer_checked:
            self._graph_enhancer_checked = True
            try:
                neo4j_cfg = get_neo4j_config()
                if neo4j_cfg["enabled"]:
                    from .indexer.graph_store import GraphStore
                    from .graph_search import GraphSearchEnhancer
                    graph_store = GraphStore(
                        uri=neo4j_cfg["uri"],
                        username=neo4j_cfg["username"],
                        password=neo4j_cfg["password"],
                        database=neo4j_cfg["database"],
                    )
                    self._graph_enhancer = GraphSearchEnhancer(graph_store, self.hybrid_search)
            except Exception:
                pass
        return self._graph_enhancer

    @staticmethod
    def _build_llm_configs(
        provider: Optional[str],
        model: Optional[str],
    ) -> Dict[str, Any]:
        """當 llm_provider / llm_model 有值時，構建覆蓋各組件的 config 物件。
        否則回傳空 dict，讓各組件從 config.yaml 讀取預設值。"""
        if not provider and not model:
            return {}

        from .provider import get_component_config

        def make(component: str, config_cls):
            base = get_component_config(component)
            return config_cls(
                provider=provider or base.provider,
                model=model or base.model,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
            )

        return {
            "analyst": make("analyst", AnalystConfig),
            "planner": make("planner", PlannerConfig),
            "synthesizer": make("synthesizer", SynthesizerConfig),
        }

    @staticmethod
    def _load_default_config() -> AgenticSearchConfig:
        """從 config.yaml 的 budget/quality_gate/reranker 區塊讀取預設值"""
        budget_cfg = get_section_config("budget")
        qg_cfg = get_section_config("quality_gate")
        reranker_cfg = get_section_config("reranker")
        evidence_cfg = get_section_config("evidence_store")

        return AgenticSearchConfig(
            max_iterations=int(budget_cfg.get("max_iterations", 5)),
            total_token_budget=int(budget_cfg.get("total_token_budget", 15000)),
            max_evidence_cards=int(budget_cfg.get("max_evidence_cards", 200)),
            working_set_size=int(budget_cfg.get("working_set_size", 20)),
            top_n_search=int(reranker_cfg.get("top_n", 100)),
            top_m_rerank=int(reranker_cfg.get("top_m", 20)),
            min_code_evidence=int(qg_cfg.get("min_code_evidence", 2)),
            min_tag_diversity=int(qg_cfg.get("min_tag_diversity", 2)),
            require_call_edge=bool(qg_cfg.get("require_call_edge", True)),
            require_named_entity=bool(qg_cfg.get("require_named_entity", True)),
            use_cross_encoder=True,
        )

    def _build_graph(self):
        builder = StateGraph(GraphState)

        # 定義節點
        builder.add_node("context_awareness", self._node_context_awareness)
        builder.add_node("analyst", self._node_analyst)
        builder.add_node("planner", self._node_planner)
        builder.add_node("tool_executor", self._node_tool_executor)
        builder.add_node("synthesizer", self._node_synthesizer)

        # 定義邊
        builder.set_entry_point("context_awareness")
        builder.add_edge("context_awareness", "analyst")
        builder.add_edge("analyst", "planner")
        builder.add_edge("planner", "tool_executor")
        builder.add_conditional_edges("tool_executor", self._route_after_tools, {
            "planner": "planner",
            "synthesizer": "synthesizer"
        })
        builder.add_edge("synthesizer", END)

        return builder.compile()

    def _node_context_awareness(self, state: GraphState) -> Dict[str, Any]:
        tech_stack = "Unknown"
        if os.path.exists("pom.xml"):
            tech_stack = "Java/Spring"
        elif os.path.exists("package.json"):
            tech_stack = "Node.js"
        elif os.path.exists("pyproject.toml") or os.path.exists("requirements.txt"):
            tech_stack = "Python"
        elif os.path.exists("go.mod"):
            tech_stack = "Go"

        return {
            "tech_stack": tech_stack,
            "iteration": 0,
            "search_history": [],
            "completed_tasks": [],
            "tool_results": [],
            "consecutive_no_new": 0,
            "fallback_triggered": False,
            "should_stop": False,
            "sub_tasks": []
        }

    def _node_analyst(self, state: GraphState) -> Dict[str, Any]:
        evidence_summary = self.evidence_store.get_summary_for_planner()
        out = self.analyst.analyze(
            query=state["query"],
            evidence_summary=evidence_summary,
            iteration=state["iteration"]
        )
        return {
            "intent": out.intent,
            "sub_tasks": out.sub_tasks,
            "evidence_summary": evidence_summary
        }

    def _node_planner(self, state: GraphState) -> Dict[str, Any]:
        iteration = state["iteration"] + 1
        self.evidence_store.set_round(iteration)

        out = self.planner.plan(
            query=state["query"],
            evidence_summary=state.get("evidence_summary", ""),
            search_history=state.get("search_history", []),
            iteration=iteration,
            previous_missing=state.get("missing_evidence", []),
            sub_tasks=state.get("sub_tasks", []),
            tool_results=state.get("tool_results", [])
        )

        return {
            "iteration": iteration,
            "planner_tool_calls": out.tool_calls,
            "missing_evidence": out.missing_evidence,
            "should_stop": out.should_stop
        }

    def _node_tool_executor(self, state: GraphState) -> Dict[str, Any]:
        tool_calls = state.get("planner_tool_calls", [])
        search_history = list(state.get("search_history", []))

        all_raw_results = []
        new_results = []

        # Execute tool calls
        for tc in tool_calls:
            tool_name = tc.get("tool")
            args = tc.get("args", {})

            if tool_name == "semantic_search":
                q = args.get("query", "")
                if q:
                    search_history.append(f"semantic_search:{q}")
                    res = semantic_search_tool(q, self.hybrid_search, self.query_builder, self.reranker, top_k=self.config.top_n_search)
                    all_raw_results.extend(res)
            elif tool_name == "graph_symbol_search":
                symbol = args.get("symbol", "")
                if symbol:
                    search_history.append(f"graph:{symbol}")
                    res = graph_symbol_search_tool(symbol, self.graph_enhancer.graph_store if self.graph_enhancer else None, args.get("depth", 1))
                    new_results.append({"tool": "graph", "res": str(res)})
            elif tool_name == "read_exact_file":
                path = args.get("path", "")
                if path:
                    res = read_exact_file_tool(path, args.get("lines"))
                    new_results.append({"tool": "read_file", "path": path, "content": res})
            elif tool_name == "list_directory":
                path = args.get("path", "")
                if path:
                    res = list_directory_tool(path)
                    new_results.append({"tool": "list_dir", "path": path, "content": str(res)})

        new_count = 0

        # Helper to convert non-semantic tool results to evidence cards
        import hashlib
        for res in new_results:
            if res["tool"] in ["read_file", "list_dir"]:
                content = res.get("content", "")
                if not content: continue
                path = res.get("path", "unknown")
                snippet = create_snippet(content, 200)
                fingerprint = hashlib.md5((path + content).encode()).hexdigest()
                card = EvidenceCard(
                    id=f"{res['tool']}_{fingerprint[:8]}",
                    path=path,
                    symbol=None,
                    snippet=snippet,
                    chunk_text=content,
                    score_hybrid=1.0,
                    score_rerank=1.0,
                    tags=[res["tool"]],
                    round_found=state["iteration"],
                    source_kind=detect_source_kind(path),
                    span="full",
                    fingerprint=fingerprint
                )
                self.evidence_store.add([card])
                new_count += 1
            elif res["tool"] == "graph":
                content = res.get("res", "")
                if not content: continue
                snippet = create_snippet(content, 200)
                fingerprint = hashlib.md5(content.encode()).hexdigest()
                card = EvidenceCard(
                    id=f"graph_{fingerprint[:8]}",
                    path="graph_search",
                    symbol=None,
                    snippet=snippet,
                    chunk_text=content,
                    score_hybrid=1.0,
                    score_rerank=1.0,
                    tags=["graph_symbol"],
                    round_found=state["iteration"],
                    source_kind="code",
                    span="graph",
                    fingerprint=fingerprint
                )
                self.evidence_store.add([card])
                new_count += 1

        if all_raw_results:
            new_cards = self._convert_to_evidence_cards(all_raw_results, state["iteration"])
            new_count += self.evidence_store.add(new_cards)

            if self.graph_enhancer and new_cards:
                try:
                    graph_results = self.graph_enhancer.expand_evidence(new_cards, top_k=5)
                    if graph_results:
                        graph_cards = self._convert_to_evidence_cards(graph_results, state["iteration"])
                        new_count += self.evidence_store.add(graph_cards)
                except Exception:
                    pass

        consec_no_new = state.get("consecutive_no_new", 0)
        if new_count == 0 and not tool_calls:
            consec_no_new += 1
        else:
            consec_no_new = 0

        return {
            "search_history": search_history,
            "tool_results": state.get("tool_results", []) + new_results,
            "consecutive_no_new": consec_no_new,
            "evidence_summary": self.evidence_store.get_summary_for_planner()
        }

    def _route_after_tools(self, state: GraphState) -> str:
        if state.get("should_stop", False):
            return "synthesizer"

        all_cards = self.evidence_store.get_all_cards()

        # Create a mock SearchState for StopConditionChecker backward compatibility
        class MockState:
            def __init__(self, s):
                self.iteration = s.get("iteration", 0)
                self.consecutive_no_new = s.get("consecutive_no_new", 0)
                self.fallback_triggered = s.get("fallback_triggered", False)
                self.missing_evidence = s.get("missing_evidence", [])

        should_stop, _, _ = self.stop_checker.should_stop(MockState(state), all_cards)

        if should_stop or state["iteration"] >= self.budget.max_iterations:
            return "synthesizer"

        return "planner"

    def _node_synthesizer(self, state: GraphState) -> Dict[str, Any]:
        all_evidence = self.evidence_store.get_all_cards()
        all_evidence.sort(key=lambda x: x.score_rerank, reverse=True)
        top_evidence = all_evidence[:30]

        response = self.synthesizer.synthesize(
            query=state["query"],
            evidence_cards=top_evidence,
            search_history=state.get("search_history", []),
            iterations=state["iteration"],
            logger=self.logger
        )
        return {"final_response": response}

    def search(self, query: str) -> SearchResult:
        """
        執行自主搜索

        Args:
            query: 用戶查詢

        Returns:
            SearchResult
        """
        # 初始化狀態
        search_id = str(uuid.uuid4())
        self.evidence_store.clear()

        # Log Start
        self.logger.log_start(search_id, query, asdict(self.config))

        debug_info = {
            "search_id": search_id,
            "iterations": [],
            "search_history": [],
            "final_evidence_count": 0
        }

        usage_log: list = []
        search_start_time = time.time()

        try:
            # 初始狀態
            initial_state = {
                "query": query,
                "tech_stack": "Unknown",
                "intent": "",
                "sub_tasks": [],
                "completed_tasks": [],
                "evidence_summary": "",
                "iteration": 0,
                "search_history": [],
                "planner_tool_calls": [],
                "tool_results": [],
                "should_stop": False,
                "final_response": None,
                "consecutive_no_new": 0,
                "fallback_triggered": False,
                "missing_evidence": []
            }

            # 執行狀態圖
            final_state = self.graph.invoke(initial_state)

            response = final_state.get("final_response")

            # Aggregate performance stats
            elapsed_ms = round((time.time() - search_start_time) * 1000)

            # Collect metrics (mocked as Graph replaces usage_log)
            debug_info["perf_stats"] = {
                "elapsed_ms": elapsed_ms,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
                "breakdown": [],
            }

            if response:
                self.logger.log_end(search_id, True, asdict(response))

                return SearchResult(
                    success=True,
                    response=response,
                    debug_info=debug_info
                )
            else:
                return SearchResult(
                    success=False,
                    error="Synthesizer did not generate a response",
                    debug_info=debug_info
                )

        except Exception as e:
            import traceback
            # Log Error
            if 'search_id' in locals():
                 self.logger.log_end(search_id, False, None, str(e))
                 
            return SearchResult(
                success=False,
                error=str(e),
                debug_info={
                    "traceback": traceback.format_exc(),
                    "iterations": debug_info.get("iterations", [])
                }
            )

    def _convert_to_evidence_cards(
        self,
        results: List[Dict[str, Any]],
        round_num: int
    ) -> List[EvidenceCard]:
        """將搜索結果轉換為 EvidenceCard"""
        cards = []

        for r in results:
            path = r.get("path", "")
            content = r.get("content", "") or r.get("payload", {}).get("content_preview", "")

            if not content:
                continue

            # 提取信息
            tags = extract_tags(content, path)
            source_kind = detect_source_kind(path)
            named_entities = extract_named_entities(content)
            snippet = create_snippet(content, 200)
            fingerprint = compute_fingerprint(content, r.get("id", ""), path)

            # 從 payload 獲取 symbol
            payload = r.get("payload", {})
            symbol = payload.get("symbol") or payload.get("class_name") or payload.get("method_name")

            # 構建 span
            chunk_index = payload.get("chunk_index", 0)
            total_chunks = payload.get("total_chunks", 1)
            span = f"chunk {chunk_index + 1}/{total_chunks}"

            cards.append(EvidenceCard(
                id=r.get("id", ""),
                path=path,
                symbol=symbol,
                snippet=snippet,
                chunk_text=content,
                score_hybrid=r.get("score_hybrid", r.get("score", 0)),
                score_rerank=r.get("score_rerank", 0),
                tags=tags,
                round_found=round_num,
                source_kind=source_kind,
                span=span,
                fingerprint=fingerprint,
                named_entities=named_entities
            ))

        return cards


# 便捷函數
def search(query: str, config: Optional[AgenticSearchConfig] = None) -> SearchResult:
    """執行自主搜索的便捷函數"""
    agent = AgenticSearch(config)
    return agent.search(query)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # 測試
    result = search("Robot 斷線後交易會怎樣處理？")

    if result.success:
        print("=== Answer ===")
        print(result.response.answer)
        print("\n=== Flow ===")
        for step in result.response.flow:
            print(f"{step.step}. {step.description}")
        print("\n=== Evidence ===")
        for e in result.response.evidence[:5]:
            print(f"- {e.path}: {e.quote[:100]}...")
    else:
        print(f"Error: {result.error}")
