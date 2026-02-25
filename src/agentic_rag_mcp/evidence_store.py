"""
Evidence Store - 兩層證據管理
- pool: 全量去重池
- working_set: 多樣性 topM
"""

from collections import Counter
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .models import EvidenceCard


@dataclass
class EvictionWeights:
    """Eviction 權重配置"""
    score: float = 1.0
    recency: float = 0.2
    kind_rarity: float = 0.2
    tag_rarity: float = 0.15


class EvidenceStore:
    """兩層證據存儲"""

    def __init__(
        self,
        max_pool: int = 200,
        working_set_size: int = 20,
        eviction_weights: Optional[EvictionWeights] = None
    ):
        self.pool: Dict[str, EvidenceCard] = {}  # fingerprint → card
        self.max_pool = max_pool
        self.working_set_size = working_set_size
        self.eviction_weights = eviction_weights or EvictionWeights()
        self.current_round = 0

    def set_round(self, round_num: int):
        """設置當前輪數"""
        self.current_round = round_num

    def add(self, cards: List[EvidenceCard]) -> int:
        """
        加入新證據
        Returns: 新增數量
        """
        new_count = 0
        for card in cards:
            if card.fingerprint not in self.pool:
                self.pool[card.fingerprint] = card
                new_count += 1
            else:
                # 如果已存在，更新分數（取較高者）
                existing = self.pool[card.fingerprint]
                if card.score_rerank > existing.score_rerank:
                    card.round_found = existing.round_found  # 保留原始發現輪數
                    self.pool[card.fingerprint] = card

        # 超過上限時，移除低分卡片
        if len(self.pool) > self.max_pool:
            self._evict_lowest()

        return new_count

    def _calculate_evict_score(self, card: EvidenceCard) -> float:
        """計算 eviction 分數（分數越高越不容易被移除）"""
        w = self.eviction_weights

        # 基礎分
        score = w.score * card.score_rerank

        # 新近加成 (最近 2 輪)
        recency = max(0, 2 - (self.current_round - card.round_found))
        score += w.recency * recency

        # source_kind 稀有度
        kind_counts = Counter(c.source_kind for c in self.pool.values())
        kind_rarity = 1.0 / (kind_counts.get(card.source_kind, 0) + 1)
        score += w.kind_rarity * kind_rarity

        # tag 稀有度 (取最稀有的 tag)
        tag_counts = Counter(tag for c in self.pool.values() for tag in c.tags)
        tag_rarity = 0
        if card.tags:
            tag_rarity = max(1.0 / (tag_counts.get(t, 0) + 1) for t in card.tags)
        score += w.tag_rarity * tag_rarity

        return score

    def _evict_lowest(self):
        """移除最低分的卡片"""
        if len(self.pool) <= self.max_pool:
            return

        # 計算所有卡片的 evict score
        scored_cards = [
            (fp, self._calculate_evict_score(card))
            for fp, card in self.pool.items()
        ]

        # 按分數排序，移除最低的
        scored_cards.sort(key=lambda x: x[1])
        to_remove = len(self.pool) - self.max_pool

        for fp, _ in scored_cards[:to_remove]:
            del self.pool[fp]

    def get_working_set(self) -> List[EvidenceCard]:
        """
        取得多樣性 topM
        - 先取高分
        - 再保證不同 module/tag 覆蓋
        """
        cards = list(self.pool.values())

        if not cards:
            return []

        # 1. 先按 score_rerank 排序
        cards.sort(key=lambda c: c.score_rerank, reverse=True)

        # 2. 多樣性選擇
        selected = []
        seen_modules = set()
        seen_tags = set()

        for card in cards:
            if len(selected) >= self.working_set_size:
                break

            # 取得 module key
            module_key = self._get_module_key(card.path)

            # 優先選擇新 module 或新 tag
            new_module = module_key not in seen_modules
            new_tag = not seen_tags.issuperset(set(card.tags)) if card.tags else False

            # 前 10 張高分優先，之後要求多樣性
            if len(selected) < 10 or new_module or new_tag:
                selected.append(card)
                seen_modules.add(module_key)
                if card.tags:
                    seen_tags.update(card.tags)

        return selected

    def _get_module_key(self, path: str) -> str:
        """取 module 層級"""
        if not path:
            return "unknown"

        parts = path.split('/')

        # Pattern 1: Java monorepo
        for i, p in enumerate(parts):
            if p == 'modules' and i + 1 < len(parts):
                return f"modules/{parts[i+1]}"

        # Pattern 2: src/main
        for i, p in enumerate(parts):
            if p == 'src' and i + 1 < len(parts) and parts[i+1] in ['main', 'test']:
                if i >= 1:
                    return '/'.join(parts[max(0, i-1):i+1])

        # Pattern 3: 常見頂層
        for i, p in enumerate(parts):
            if p in ['services', 'apps', 'packages', 'libs', 'service']:
                if i + 1 < len(parts):
                    return f"{p}/{parts[i+1]}"

        # Pattern 4: .NET project directories (e.g., Project.Module)
        for i, p in enumerate(parts):
            if '.' in p and p[0].isupper() and not p.endswith(('.cs', '.py', '.js', '.ts')):
                return p

        # Fallback
        return '/'.join(parts[:min(3, len(parts))])

    def get_summary_for_planner(self) -> str:
        """
        給 Planner 的固定格式摘要
        每張卡一行，不含完整 snippet
        """
        working_set = self.get_working_set()

        if not working_set:
            return "No evidence collected yet."

        lines = ["card_id | path | symbol | tags | summary | score"]
        lines.append("-" * 80)

        for card in working_set:
            summary = card.snippet[:50] + "..." if len(card.snippet) > 50 else card.snippet
            summary = summary.replace('\n', ' ').replace('|', '/')
            tags_str = ','.join(card.tags[:3]) if card.tags else '-'

            lines.append(
                f"{card.id[:8]} | {card.path[-40:]} | {card.symbol or '-'} | "
                f"{tags_str} | {summary} | {card.score_rerank:.2f}"
            )

        return "\n".join(lines)

    def get_cards_by_ids(self, card_ids: List[str]) -> List[EvidenceCard]:
        """根據 ID 獲取卡片"""
        result = []
        for card in self.pool.values():
            if card.id in card_ids or card.id[:8] in card_ids:
                result.append(card)
        return result

    def get_all_cards(self) -> List[EvidenceCard]:
        """獲取所有卡片"""
        return list(self.pool.values())

    def get_stats(self) -> Dict:
        """獲取統計信息"""
        cards = list(self.pool.values())

        if not cards:
            return {
                "total": 0,
                "by_source_kind": {},
                "by_tag": {},
                "avg_score": 0
            }

        # 按 source_kind 統計
        by_kind = Counter(c.source_kind for c in cards)

        # 按 tag 統計
        by_tag = Counter(tag for c in cards for tag in c.tags)

        # 平均分數
        avg_score = sum(c.score_rerank for c in cards) / len(cards)

        return {
            "total": len(cards),
            "by_source_kind": dict(by_kind),
            "by_tag": dict(by_tag.most_common(10)),
            "avg_score": round(avg_score, 3),
            "working_set_size": len(self.get_working_set())
        }

    def clear(self):
        """清空存儲"""
        self.pool.clear()
        self.current_round = 0
