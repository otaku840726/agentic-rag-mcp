"""
Agentic Search - 主循環控制
協調 Planner, Hybrid Search, Reranker, Evidence Store, Synthesizer
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .models import (
    SearchResult, SynthesizedResponse, SearchState,
    EvidenceCard, QueryIntent, MissingEvidence,
    Budget, QualityGate
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
from .planner import Planner
from .synthesizer import Synthesizer


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
        self.config = config or AgenticSearchConfig()

        # 初始化組件
        self.evidence_store = EvidenceStore(
            max_pool=self.config.max_evidence_cards,
            working_set_size=self.config.working_set_size
        )
        self.query_builder = QueryBuilder()
        self.hybrid_search = HybridSearch()
        self.batch_search = HybridSearchBatch(self.hybrid_search)
        self.reranker = create_reranker(self.config.use_cross_encoder)
        self.planner = Planner()
        self.synthesizer = Synthesizer()

        # 停機條件
        budget = Budget(
            max_iterations=self.config.max_iterations,
            total_token_budget=self.config.total_token_budget
        )
        quality_gate = QualityGate(
            min_code_evidence=self.config.min_code_evidence,
            min_tag_diversity=self.config.min_tag_diversity,
            require_call_edge=self.config.require_call_edge,
            require_named_entity=self.config.require_named_entity
        )
        self.stop_checker = StopConditionChecker(budget, quality_gate)

    def search(self, query: str) -> SearchResult:
        """
        執行自主搜索

        Args:
            query: 用戶查詢

        Returns:
            SearchResult
        """
        # 初始化狀態
        state = SearchState(query=query)
        self.evidence_store.clear()

        debug_info = {
            "iterations": [],
            "search_history": [],
            "final_evidence_count": 0
        }

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

                # 1. Planner 生成查詢計劃
                evidence_summary = self.evidence_store.get_summary_for_planner()
                plan = self.planner.plan(
                    query=query,
                    evidence_summary=evidence_summary,
                    search_history=state.search_history,
                    iteration=state.iteration,
                    previous_missing=state.missing_evidence
                )

                # 更新狀態
                state.missing_evidence = plan.missing_evidence
                iteration_info["missing_evidence"] = [m.need for m in plan.missing_evidence]

                # Planner 判斷應該停止
                if plan.should_stop:
                    iteration_info["stop_reason"] = "planner_decided"
                    debug_info["iterations"].append(iteration_info)
                    break

                # 2. 執行搜索
                queries_to_execute = plan.next_queries

                # 如果是 fallback 觸發
                if state.consecutive_no_new == 1 and not state.fallback_triggered:
                    fallback_queries = self.query_builder.build_fallback_queries(
                        self.evidence_store.get_working_set(),
                        state.missing_evidence
                    )
                    queries_to_execute.extend(fallback_queries)
                    state.fallback_triggered = True
                    iteration_info["fallback_triggered"] = True

                # 構建實際查詢
                all_queries = []
                for intent in queries_to_execute:
                    built = self.query_builder.build_from_intent(intent)
                    all_queries.extend(built)
                    state.search_history.append(intent.query)
                    iteration_info["queries"].append(intent.query)

                # 批量搜索
                raw_results = self.batch_search.search_batch(
                    all_queries,
                    top_n_per_query=self.config.top_n_search // len(all_queries) if all_queries else 50
                )
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
                iteration_info["new_evidence"] = new_count

                # 更新 consecutive_no_new
                if new_count == 0:
                    state.consecutive_no_new += 1
                else:
                    state.consecutive_no_new = 0
                    state.fallback_triggered = False  # 重置 fallback 狀態

                # 5. 檢查停機條件
                all_cards = self.evidence_store.get_all_cards()
                should_stop, stop_reason, _ = self.stop_checker.should_stop(state, all_cards)

                iteration_info["stop_reason"] = stop_reason if should_stop else None
                debug_info["iterations"].append(iteration_info)

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
                iterations=state.iteration
            )

            debug_info["search_history"] = state.search_history
            debug_info["final_evidence_count"] = len(all_evidence)

            return SearchResult(
                success=True,
                response=response,
                debug_info=debug_info
            )

        except Exception as e:
            import traceback
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
