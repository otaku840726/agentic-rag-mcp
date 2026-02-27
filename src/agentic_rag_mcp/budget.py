"""
Budget & Quality Gate - 預算控制和質量門檻
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .models import EvidenceCard, MissingEvidence, InvestigationState, Budget, QualityGate
from .utils import check_accept_coverage

if TYPE_CHECKING:
    from .planner import LLMJudge


class BudgetManager:
    """預算管理器"""

    def __init__(self, budget: Optional[Budget] = None):
        self.budget = budget or Budget()
        self.tokens_used = 0

    def can_continue(self, state: InvestigationState) -> Tuple[bool, str]:
        """檢查是否可以繼續"""
        if state.iteration >= self.budget.max_iterations:
            return False, "max_iterations_reached"

        if state.total_tokens >= self.budget.total_token_budget:
            return False, "token_budget_exceeded"

        return True, ""

    def add_tokens(self, count: int):
        """記錄使用的 token"""
        self.tokens_used += count

    def get_remaining_tokens(self) -> int:
        """獲取剩餘 token"""
        return max(0, self.budget.total_token_budget - self.tokens_used)

    def get_stats(self) -> dict:
        """獲取統計"""
        return {
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.get_remaining_tokens(),
            "budget": self.budget.total_token_budget
        }


class QualityGateChecker:
    """質量門檻檢查器"""

    def __init__(self, gate: Optional[QualityGate] = None):
        self.gate = gate or QualityGate()

    def check(self, cards: List[EvidenceCard]) -> Tuple[bool, str]:
        """
        檢查質量門檻
        Returns: (passed, reason)
        """
        if not cards:
            return False, "no_evidence"

        # 1. 最少 code 類型證據
        code_count = sum(1 for c in cards if c.source_kind == "code")
        if code_count < self.gate.min_code_evidence:
            return False, f"insufficient_code_evidence: {code_count}/{self.gate.min_code_evidence}"

        # 2. Tag 多樣性
        all_tags = set()
        for c in cards:
            all_tags.update(c.tags)
        if len(all_tags) < self.gate.min_tag_diversity:
            return False, f"insufficient_tag_diversity: {len(all_tags)}/{self.gate.min_tag_diversity}"

        # 3. Call edge 證據
        if self.gate.require_call_edge:
            has_call = any(
                'callsite' in c.tags or 'invocation' in c.tags
                for c in cards
            )
            if not has_call:
                return False, "missing_call_edge_evidence"

        # 4. 具名實體 (config key OR enum)
        if self.gate.require_named_entity:
            has_config = any(c.named_entities.get("config_keys") for c in cards)
            has_enum = any(c.named_entities.get("enums") for c in cards)
            if not (has_config or has_enum):
                return False, "missing_named_entity"

        return True, "passed"


class StopConditionChecker:
    """停機條件檢查器"""

    def __init__(
        self,
        budget: Optional[Budget] = None,
        quality_gate: Optional[QualityGate] = None,
        judge: Optional["LLMJudge"] = None,
    ):
        self.budget_manager = BudgetManager(budget)
        self.quality_checker = QualityGateChecker(quality_gate)
        self.judge = judge

    def should_stop(
        self,
        state: InvestigationState,
        evidence_cards: List[EvidenceCard],
        usage_log: Optional[list] = None,
    ) -> Tuple[bool, str, List]:
        """
        檢查是否應該停止
        Returns: (should_stop, reason, fallback_queries)
        """
        # 1. 預算硬上限 — 無條件強制停止
        can_continue, reason = self.budget_manager.can_continue(state)
        if not can_continue:
            return True, reason, []

        # 2. 所有 missing_evidence 語義滿足（兩段式：regex → LLMJudge）
        all_satisfied = self._check_missing_evidence_satisfied(
            state.missing_evidence, evidence_cards, usage_log=usage_log
        )
        if all_satisfied:
            # Fix B: 對稱性缺口仍存在時不允許停止
            if state.symmetry_gaps:
                return False, f"symmetry_gaps_pending({len(state.symmetry_gaps)})", []

            # Fix D: AI 整體完整性確認（取代數字閥值）
            if self.judge:
                evidence_summary = self._build_evidence_summary(evidence_cards)
                review = self.judge.review_completeness(
                    query=state.query,
                    evidence_summary=evidence_summary,
                    usage_log=usage_log,
                )
                if not review.get("complete", True):
                    missing_areas = review.get("missing_impact_areas", [])
                    reason = review.get("reason", "")
                    return False, f"impact_review_incomplete: {reason}", missing_areas

            return True, "all_evidence_found_and_quality_passed", []

        # 3. Stuck 處理
        if state.consecutive_no_new == 1 and not state.fallback_triggered:
            return False, "triggering_fallback", []

        if state.consecutive_no_new >= 2:
            return True, "stuck_after_fallback", []

        return False, "", []

    @staticmethod
    def _build_evidence_summary(cards: List[EvidenceCard], top_k: int = 15) -> str:
        """
        為 ImpactReviewer 建立簡潔的證據摘要
        取 rerank 分數最高的 top_k 張，每張取 path + snippet
        """
        sorted_cards = sorted(cards, key=lambda c: c.score_rerank, reverse=True)
        lines = []
        for c in sorted_cards[:top_k]:
            symbol_part = f" [{c.symbol}]" if c.symbol else ""
            snippet = c.snippet[:120].replace("\n", " ")
            lines.append(f"- {c.path}{symbol_part}: {snippet}")
        return "\n".join(lines) if lines else "（無證據）"

    def _check_missing_evidence_satisfied(
        self,
        missing_evidence: List[MissingEvidence],
        cards: List[EvidenceCard],
        usage_log: Optional[list] = None,
    ) -> bool:
        """檢查所有 missing evidence 是否滿足（兩段式）"""
        if not missing_evidence:
            return True

        for m in missing_evidence:
            if not self._is_satisfied(m, cards, usage_log=usage_log):
                return False

        return True

    def _is_satisfied(
        self,
        missing: MissingEvidence,
        cards: List[EvidenceCard],
        usage_log: Optional[list] = None,
    ) -> bool:
        """
        兩段式 missing evidence 滿足判斷：
        Stage 1: regex 快速通道（無 LLM 成本）
        Stage 2: regex 不確定時 → LLMJudge 語義判斷
        """
        if not missing.accept:
            return True

        # Stage 1: regex 掃描所有卡片
        covered_items: List[str] = []
        for item in missing.accept:
            for card in cards:
                matched = check_accept_coverage([item], card.chunk_text)
                if matched:
                    covered_items.append(item)
                    break

        coverage_rate = len(covered_items) / len(missing.accept)

        # 明確滿足 (>= 67%) → 不需要 LLM
        if coverage_rate >= 0.67:
            return True

        # regex 無法確認 → 交給 LLMJudge 語義判斷
        if self.judge and cards:
            summary = self._build_candidates_summary(missing, cards)
            result = self.judge.judge(
                need=missing.need,
                accept=missing.accept,
                covered=covered_items,
                candidates_summary=summary,
                usage_log=usage_log,
            )
            return bool(result.get("satisfied", False))

        # 沒有 judge → 維持原本 67% 規則
        return coverage_rate >= 0.67

    @staticmethod
    def _build_candidates_summary(
        missing: MissingEvidence,
        cards: List[EvidenceCard],
        top_k: int = 6,
    ) -> str:
        """
        為 LLMJudge 建立候選證據摘要
        策略：先按與 missing.need / missing.accept 的關鍵字重疊度排序，
        再按 rerank 分數兜底，取 top_k 張，每張截 250 字
        """
        import re

        # 從 need 和 accept 中提取關鍵詞（長度 >= 3 的非停用詞）
        _stopwords = {"the", "a", "an", "is", "in", "of", "and", "or", "for",
                      "to", "with", "how", "what", "where", "when", "which"}
        need_tokens = set(re.findall(r'\w{3,}', missing.need.lower())) - _stopwords
        accept_tokens: set = set()
        for item in missing.accept:
            accept_tokens.update(re.findall(r'\w{3,}', item.lower()))
        accept_tokens -= _stopwords
        query_tokens = need_tokens | accept_tokens

        def _relevance_score(card: EvidenceCard) -> float:
            """關鍵字重疊度 (0-1) + rerank 分數加成"""
            if not query_tokens:
                return card.score_rerank
            text_lower = card.chunk_text.lower()
            matched = sum(1 for t in query_tokens if t in text_lower)
            overlap = matched / len(query_tokens)
            return overlap * 0.7 + card.score_rerank * 0.3

        sorted_cards = sorted(cards, key=_relevance_score, reverse=True)
        lines = []
        for c in sorted_cards[:top_k]:
            snippet = c.chunk_text[:250].replace("\n", " ")
            lines.append(f"[{c.path}] {snippet}")
        return "\n".join(lines) if lines else "（無候選證據）"

    def add_tokens(self, count: int):
        """記錄 token 使用"""
        self.budget_manager.add_tokens(count)

    def get_stats(self) -> dict:
        """獲取統計"""
        return self.budget_manager.get_stats()
