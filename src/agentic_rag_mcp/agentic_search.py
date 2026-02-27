"""
Agentic Search - 主循環控制
協調 Analyst, Planner, Hybrid Search, Reranker, Evidence Store, Synthesizer
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from .models import (
    SearchResult, SynthesizedResponse, InvestigationState,
    EvidenceCard, QueryIntent, MissingEvidence,
    Budget, QualityGate, Phase
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
from .planner import LLMJudge
from .synthesizer import Synthesizer, SynthesizerConfig
from .search_logger import SearchTraceLogger
from .provider import get_section_config, get_neo4j_config
from .specialists import (
    AnchorDetector, SubjectAnalyst, TechStackInferrer,
    CoverageAnalyst, SymmetryChecker, GapIdentifier, QueryGenerator,
    run_coverage_and_symmetry_parallel,
)
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

    def __init__(self, config: Optional[AgenticSearchConfig] = None):
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

        # LLM 組件
        self.synthesizer = Synthesizer()
        self.judge = LLMJudge()
        self.logger = SearchTraceLogger()

        # New Coordinator specialists
        self.anchor_detector = AnchorDetector()
        self.subject_analyst = SubjectAnalyst()
        self.tech_stack_inferrer = TechStackInferrer()
        self.coverage_analyst = CoverageAnalyst()
        self.symmetry_checker = SymmetryChecker()
        self.gap_identifier = GapIdentifier()
        self.query_generator = QueryGenerator()

        # Optional: Graph search enhancer
        self._graph_enhancer = None
        self._graph_enhancer_checked = False

        # 停機條件（注入 judge 以啟用語義判斷）
        budget = Budget(
            max_iterations=self.config.max_iterations,
            total_token_budget=self.config.total_token_budget
        )
        quality_gate = QualityGate(
            min_code_evidence=self.config.min_code_evidence,
            min_tag_diversity=self.config.min_tag_diversity,
            require_call_edge=self.config.require_call_edge,
            require_named_entity=self.config.require_named_entity,
        )
        self.stop_checker = StopConditionChecker(budget, quality_gate, judge=self.judge)

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
        state = InvestigationState(query=query)
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
            # 主循環
            while True:
                state.iteration += 1
                self.evidence_store.set_round(state.iteration)

                iteration_info = {
                    "iteration": state.iteration,
                    "queries": [],
                    "results_found": 0,
                    "new_evidence": 0
                }

                # ── Step 0: AnchorDetector (code) ──────────────────────────
                all_cards_so_far = self.evidence_store.get_all_cards()
                self.anchor_detector.detect(state, all_cards_so_far)

                # ── Step 1: SubjectAnalyst (once, before first search) ──────
                if not state.subject_analyst_done:
                    self.subject_analyst.analyze(
                        state=state,
                        usage_log=usage_log,
                        logger_obj=self.logger,
                        search_id=search_id,
                    )

                # ── Step 2: TechStackInferrer (once, at Phase1→Phase2) ──────
                if state.phase == Phase.TWO and not state.tech_stack_inferred:
                    self.tech_stack_inferrer.infer(
                        state=state,
                        cards=all_cards_so_far,
                        usage_log=usage_log,
                        logger_obj=self.logger,
                        search_id=search_id,
                    )

                # ── Step 3: CoverageAnalyst + SymmetryChecker (parallel) ────
                evidence_summary = self.evidence_store.get_summary_for_planner()
                run_coverage_and_symmetry_parallel(
                    coverage_analyst=self.coverage_analyst,
                    symmetry_checker=self.symmetry_checker,
                    state=state,
                    evidence_summary=evidence_summary,
                    usage_log=usage_log,
                    logger_obj=self.logger,
                    search_id=search_id,
                )

                # ── Step 4: GapIdentifier ───────────────────────────────────
                state.missing_evidence = self.gap_identifier.identify(
                    state=state,
                    usage_log=usage_log,
                    logger_obj=self.logger,
                    search_id=search_id,
                )

                # ── Step 5: QueryGenerator ──────────────────────────────────
                queries_to_execute = self.query_generator.generate(
                    state=state,
                    usage_log=usage_log,
                    logger_obj=self.logger,
                    search_id=search_id,
                )

                # Log specialist outputs for this iteration
                iteration_info["investigation_state"] = state.to_debug_dict()
                iteration_info["missing_evidence"] = [m.need for m in state.missing_evidence]
                if state.symmetry_gaps:
                    iteration_info["symmetry_gaps"] = state.symmetry_gaps

                # ── Step 6: Execute searches ────────────────────────────────

                # 如果是 fallback 觸發
                if state.consecutive_no_new == 1 and not state.fallback_triggered:
                    fallback_queries = self.query_builder.build_fallback_queries(
                        self.evidence_store.get_working_set(),
                        state.missing_evidence
                    )
                    queries_to_execute.extend(fallback_queries)
                    state.fallback_triggered = True
                    iteration_info["fallback_triggered"] = True

                # 構建實際查詢 — 分流：圖譜遍歷 vs 向量/關鍵字搜索
                all_queries = []
                graph_traverse_results: List[Dict[str, Any]] = []

                for intent in queries_to_execute:
                    state.search_history.append(intent.query)
                    iteration_info["queries"].append(intent.query)

                    # 圖譜遍歷：直接派發給 GraphSearchEnhancer
                    if intent.query_type.startswith("graph_traverse") and self.graph_enhancer:
                        direction = intent.query_type.replace("graph_traverse_", "")
                        try:
                            gr = self.graph_enhancer.traverse_direct(
                                symbol=intent.query,
                                direction=direction,
                                top_k=10,
                            )
                            graph_traverse_results.extend(gr)
                            iteration_info.setdefault("graph_traverse", []).append(
                                {"symbol": intent.query, "direction": direction, "found": len(gr)}
                            )
                        except Exception:
                            pass
                    else:
                        built = self.query_builder.build_from_intent(intent)
                        all_queries.extend(built)

                # 批量向量/關鍵字搜索
                raw_results = self.batch_search.search_batch(
                    all_queries,
                    top_n_per_query=self.config.top_n_search // len(all_queries) if all_queries else 50
                )

                # 合併圖譜遍歷結果
                if graph_traverse_results:
                    raw_results = raw_results + graph_traverse_results
                iteration_info["results_found"] = len(raw_results)

                # 3. Rerank
                reranked = self.reranker.rerank(
                    query=query,
                    candidates=raw_results,
                    top_m=self.config.top_m_rerank
                )

                # 4. 轉換為 EvidenceCard 並存入
                new_cards = self._convert_to_evidence_cards(reranked, state.iteration)
                new_count = self.evidence_store.add(new_cards)

                # 4b. Optional: Graph-based context expansion
                if self.graph_enhancer and new_cards:
                    try:
                        graph_results = self.graph_enhancer.expand_evidence(new_cards, top_k=5)
                        if graph_results:
                            graph_cards = self._convert_to_evidence_cards(graph_results, state.iteration)
                            graph_new = self.evidence_store.add(graph_cards)
                            new_count += graph_new
                            iteration_info["graph_expansion"] = len(graph_results)
                    except Exception:
                        pass  # Graph expansion is best-effort

                iteration_info["new_evidence"] = new_count

                # 更新 consecutive_no_new
                if new_count == 0:
                    state.consecutive_no_new += 1
                else:
                    state.consecutive_no_new = 0
                    state.fallback_triggered = False  # 重置 fallback 狀態

                # 5. 檢查停機條件
                all_cards = self.evidence_store.get_all_cards()
                should_stop, stop_reason, supplementary = self.stop_checker.should_stop(
                    state, all_cards, usage_log=usage_log
                )
                # ImpactReviewer 指出的補充缺口 → 存入 impact_reviewer_gaps（獨立欄位，不被 GapIdentifier 覆蓋）
                if not should_stop and supplementary and isinstance(supplementary[0], str):
                    state.impact_reviewer_gaps = supplementary  # GapIdentifier 下輪會讀取
                    iteration_info["impact_review_additions"] = supplementary

                iteration_info["stop_reason"] = stop_reason if should_stop else None
                debug_info["iterations"].append(iteration_info)
                
                # Log Iteration
                self.logger.log_iteration(search_id, state.iteration, state.to_debug_dict(), iteration_info)

                if should_stop:
                    break

            # 6. Synthesize 最終回應
            all_evidence = self.evidence_store.get_all_cards()
            # 按分數排序取 top
            all_evidence.sort(key=lambda x: x.score_rerank, reverse=True)
            top_evidence = all_evidence[:30]

            response = self.synthesizer.synthesize(
                query=query,
                evidence_cards=top_evidence,
                search_history=state.search_history,
                iterations=state.iteration,
                logger=self.logger,
                search_id=search_id,
                usage_log=usage_log
            )

            # Post-process: expand evidence refs that lack causal context
            card_map = {c.id: c for c in all_evidence}
            card_map.update({c.id[:8]: c for c in all_evidence})
            expand_count = 0
            for ev_ref in response.evidence:
                if ev_ref.needs_expand:
                    card = card_map.get(ev_ref.card_id) or card_map.get(ev_ref.card_id[:8])
                    if card:
                        try:
                            expanded = self.synthesizer.expand_evidence(card, query)
                            ev_ref.quote = expanded
                            ev_ref.needs_expand = False
                            expand_count += 1
                        except Exception:
                            pass  # Best-effort: keep original quote on failure
            if expand_count:
                debug_info["evidence_expanded"] = expand_count

            debug_info["search_history"] = state.search_history
            debug_info["final_evidence_count"] = len(all_evidence)

            # Aggregate performance stats
            elapsed_ms = round((time.time() - search_start_time) * 1000)
            total_prompt = sum(e.get("prompt_tokens", 0) for e in usage_log)
            total_completion = sum(e.get("completion_tokens", 0) for e in usage_log)
            debug_info["perf_stats"] = {
                "elapsed_ms": elapsed_ms,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "llm_calls": len(usage_log),
                "breakdown": usage_log,
            }

            # Log End
            self.logger.log_end(search_id, True, asdict(response))

            return SearchResult(
                success=True,
                response=response,
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
