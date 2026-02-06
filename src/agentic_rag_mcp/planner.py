"""
Planner LLM - 生成下一輪查詢的短輸出結構化模型
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .provider import create_client_for
from .models import (
    PlannerOutput, QueryIntent, MissingEvidence,
    EvidenceCard, SearchState
)


PLANNER_SYSTEM_PROMPT = """你是代碼庫搜索 Planner。
你的任務是分析當前搜索狀態，決定下一步行動。

**職責:**
1. 分析當前證據是否足以回答問題
2. 識別還缺什麼關鍵證據
3. 生成 2-5 個下一輪查詢

**只輸出 JSON，不要任何解釋。**

**輸出格式:**
{
    "should_stop": false,
    "missing_evidence": [
        {
            "need": "描述缺少什麼",
            "accept": ["接受條件1", "接受條件2"],
            "priority": "high|medium|low"
        }
    ],
    "evidence_found": ["card_id1", "card_id2"],
    "next_queries": [
        {
            "query": "查詢內容",
            "purpose": "查詢目的",
            "query_type": "keyword|symbol|config|semantic",
            "operator": "semantic|keyword|exact|symbol_ref|callsite"
        }
    ],
    "rationale": "簡短的決策理由（1-2句）"
}

**查詢類型說明:**
- hybrid: 混合搜索 (dense + sparse BM25 RRF fusion)，推薦預設使用
- semantic: 純語義搜索，適合模糊概念
- keyword: BM25 關鍵字搜索，適合技術名詞、SP名、類名精確匹配
- exact: 精確匹配，適合類名、方法名
- symbol_ref: 符號引用搜索
- callsite: 調用點搜索，如 "methodName("

**Accept 條件常見類型:**
- "config key": 配置鍵名
- "default value": 預設值
- "where read": 讀取位置
- "enum definition": 枚舉定義
- "transition": 狀態轉換
- "entry point": 入口點
- "call edge": 調用關係
"""


@dataclass
class PlannerConfig:
    """Planner 配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 4000  # thinking model 需要更多 tokens
    temperature: float = 0.1
    base_url: Optional[str] = None


class Planner:
    """Planner LLM"""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        client, comp_cfg = create_client_for("planner")
        self.client = client
        # config.yaml 的值優先，但允許外部傳入覆蓋
        if not config:
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model
            self.config.max_tokens = comp_cfg.max_tokens
            self.config.temperature = comp_cfg.temperature

    def plan(
        self,
        query: str,
        evidence_summary: str,
        search_history: List[str],
        iteration: int,
        previous_missing: Optional[List[MissingEvidence]] = None
    ) -> PlannerOutput:
        """
        生成下一輪計劃

        Args:
            query: 原始查詢
            evidence_summary: 當前證據摘要
            search_history: 已搜索的查詢
            iteration: 當前迭代次數
            previous_missing: 上一輪的 missing evidence

        Returns:
            PlannerOutput
        """
        # 構建用戶提示
        user_prompt = self._build_user_prompt(
            query, evidence_summary, search_history, iteration, previous_missing
        )

        # 調用 LLM
        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # 非 local provider 才使用 response_format
        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        # 解析回應
        content = response.choices[0].message.content
        return self._parse_response(content)

    def _build_user_prompt(
        self,
        query: str,
        evidence_summary: str,
        search_history: List[str],
        iteration: int,
        previous_missing: Optional[List[MissingEvidence]]
    ) -> str:
        """構建用戶提示"""
        parts = [
            f"**原始問題:** {query}",
            f"**當前迭代:** {iteration}",
            "",
            "**當前證據:**",
            evidence_summary if evidence_summary else "尚未收集到證據",
            "",
            "**已搜索查詢:**",
            ", ".join(search_history) if search_history else "尚未進行搜索",
        ]

        if previous_missing:
            parts.append("")
            parts.append("**上一輪識別的缺失證據:**")
            for m in previous_missing:
                parts.append(f"- {m.need} (priority: {m.priority})")
                parts.append(f"  accept: {', '.join(m.accept)}")

        parts.append("")
        parts.append("請分析並輸出下一輪計劃（JSON 格式）。")

        return "\n".join(parts)

    def _parse_response(self, content: str) -> PlannerOutput:
        """解析 LLM 回應"""
        try:
            # 處理 thinking model 的 <think> 標籤
            if '<think>' in content:
                # 移除 thinking 部分，只保留 JSON
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # 嘗試提取 JSON 部分
            content = content.strip()
            if not content.startswith('{'):
                # 嘗試找到 JSON 開始位置
                json_start = content.find('{')
                if json_start >= 0:
                    json_end = content.rfind('}')
                    if json_end > json_start:
                        content = content[json_start:json_end + 1]

            data = json.loads(content)

            # 解析 missing_evidence
            missing_evidence = []
            for m in data.get("missing_evidence", []):
                missing_evidence.append(MissingEvidence(
                    need=m.get("need", ""),
                    accept=m.get("accept", []),
                    priority=m.get("priority", "medium")
                ))

            # 解析 next_queries
            next_queries = []
            for q in data.get("next_queries", []):
                next_queries.append(QueryIntent(
                    query=q.get("query", ""),
                    purpose=q.get("purpose", ""),
                    query_type=q.get("query_type", "semantic"),
                    operator=q.get("operator", "semantic"),
                    filters=q.get("filters")
                ))

            return PlannerOutput(
                next_queries=next_queries,
                missing_evidence=missing_evidence,
                evidence_found=data.get("evidence_found", []),
                rationale=data.get("rationale", ""),
                should_stop=data.get("should_stop", False)
            )

        except json.JSONDecodeError as e:
            # 解析失敗時返回默認輸出
            return PlannerOutput(
                next_queries=[],
                missing_evidence=[],
                evidence_found=[],
                rationale=f"Failed to parse LLM response: {e}",
                should_stop=True
            )

    def get_token_count(self, text: str) -> int:
        """估算 token 數量"""
        # 粗略估算: 1 token ≈ 4 字符
        return len(text) // 4


class LLMJudge:
    """
    LLM Judge - 用於 is_satisfied 的不確定情況
    極短輸出，只回答 true/false + 1句理由
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig(max_tokens=100)
        client, comp_cfg = create_client_for("judge")
        self.client = client
        if not config:
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model

    def judge(
        self,
        need: str,
        accept: List[str],
        covered: List[str],
        candidates_summary: str
    ) -> Dict[str, Any]:
        """
        判斷 missing evidence 是否被滿足

        Args:
            need: 需要什麼
            accept: 接受條件
            covered: 已覆蓋的條件
            candidates_summary: 候選卡片摘要

        Returns:
            {"satisfied": bool, "reason": str}
        """
        prompt = f"""判斷以下證據需求是否被滿足。

需求: {need}
接受條件: {', '.join(accept)}
已覆蓋: {', '.join(covered)}
候選證據:
{candidates_summary}

只回答 JSON: {{"satisfied": true/false, "reason": "一句話理由"}}"""

        kwargs = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0,
        }

        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        try:
            content = response.choices[0].message.content

            # 處理 thinking model 的 <think> 標籤
            if '<think>' in content:
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # 提取 JSON
            content = content.strip()
            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start >= 0:
                    json_end = content.rfind('}')
                    if json_end > json_start:
                        content = content[json_start:json_end + 1]

            return json.loads(content)
        except:
            return {"satisfied": False, "reason": "Failed to parse judge response"}
