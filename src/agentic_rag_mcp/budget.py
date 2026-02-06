"""
Budget & Quality Gate - 預算控制和質量門檻
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from .models import EvidenceCard, MissingEvidence, SearchState, Budget, QualityGate
from .utils import check_accept_coverage


class BudgetManager:
    """預算管理器"""

    def __init__(self, budget: Optional[Budget] = None):
        self.budget = budget or Budget()
        self.tokens_used = 0

    def can_continue(self, state: SearchState) -> Tuple[bool, str]:
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
        quality_gate: Optional[QualityGate] = None
    ):
        self.budget_manager = BudgetManager(budget)
        self.quality_checker = QualityGateChecker(quality_gate)

    def should_stop(
        self,
        state: SearchState,
        evidence_cards: List[EvidenceCard]
    ) -> Tuple[bool, str, List]:
        """
        檢查是否應該停止
        Returns: (should_stop, reason, fallback_queries)
        """
        # 1. 預算檢查
        can_continue, reason = self.budget_manager.can_continue(state)
        if not can_continue:
            return True, reason, []

        # 2. 所有 missing_evidence 都滿足 + 質量通過
        all_satisfied = self._check_missing_evidence_satisfied(
            state.missing_evidence, evidence_cards
        )
        if all_satisfied:
            quality_passed, quality_reason = self.quality_checker.check(evidence_cards)
            if quality_passed:
                return True, "all_evidence_found_and_quality_passed", []

        # 3. Stuck 處理
        if state.consecutive_no_new == 1 and not state.fallback_triggered:
            # 第一次 stuck → 觸發 fallback，不停止
            return False, "triggering_fallback", []

        if state.consecutive_no_new >= 2:
            # 連續 2 次 stuck → 停止
            return True, "stuck_after_fallback", []

        return False, "", []

    def _check_missing_evidence_satisfied(
        self,
        missing_evidence: List[MissingEvidence],
        cards: List[EvidenceCard]
    ) -> bool:
        """檢查所有 missing evidence 是否滿足"""
        if not missing_evidence:
            return True

        for m in missing_evidence:
            if not self._is_satisfied(m, cards):
                return False

        return True

    def _is_satisfied(
        self,
        missing: MissingEvidence,
        cards: List[EvidenceCard]
    ) -> bool:
        """
        檢查單個 missing evidence 是否滿足
        使用覆蓋率判斷
        """
        if not missing.accept:
            return True

        covered_count = 0
        for item in missing.accept:
            # 檢查是否有任何卡片滿足這個條件
            for card in cards:
                covered = check_accept_coverage([item], card.chunk_text)
                if covered:
                    covered_count += 1
                    break

        coverage_rate = covered_count / len(missing.accept)

        # 覆蓋率 >= 67% 視為滿足
        return coverage_rate >= 0.67

    def add_tokens(self, count: int):
        """記錄 token 使用"""
        self.budget_manager.add_tokens(count)

    def get_stats(self) -> dict:
        """獲取統計"""
        return self.budget_manager.get_stats()
