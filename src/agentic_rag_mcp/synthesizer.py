"""
Synthesizer LLM - 最終整合回應
只在循環結束後調用一次
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from openai import OpenAI

from models import (
    SynthesizedResponse, FlowStep, DecisionPoint,
    ConfigItem, EvidenceRef, EvidenceCard
)
from utils import has_causal_verb, CAUSAL_VERBS


SYNTHESIZER_SYSTEM_PROMPT = """你是代碼庫專家。
根據收集到的證據，生成結構化的回答。

**輸出格式 (JSON):**
{
    "answer": "結論摘要（2-3句話）",
    "flow": [
        {
            "step": 1,
            "description": "步驟描述",
            "code_ref": "card_id | path | L120-L150"
        }
    ],
    "decision_points": [
        {
            "condition": "條件描述",
            "true_branch": "為真時的行為",
            "false_branch": "為假時的行為",
            "code_ref": "card_id | path | span"
        }
    ],
    "config": [
        {
            "key": "配置鍵名",
            "default_value": "預設值（如有）",
            "source": "來源文件",
            "description": "說明"
        }
    ],
    "evidence": [
        {
            "card_id": "id",
            "path": "文件路徑",
            "span": "L120-L150",
            "quote": "關鍵引用（120-200字）",
            "needs_expand": false
        }
    ]
}

**要求:**
1. answer 要簡潔但完整
2. flow 按實際執行順序排列
3. decision_points 列出關鍵分支邏輯
4. config 列出相關配置項
5. evidence 引用具體代碼，quote 控制在 120-200 字
6. 所有引用必須來自提供的證據，不要臆測
"""


@dataclass
class SynthesizerConfig:
    """Synthesizer 配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 6000  # thinking model 需要更多 tokens
    temperature: float = 0.2
    base_url: Optional[str] = None


class Synthesizer:
    """Synthesizer LLM"""

    def __init__(self, config: Optional[SynthesizerConfig] = None):
        self.config = config or SynthesizerConfig()

        # 檢查是否使用本地 LLM
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        if use_local:
            local_url = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
            local_model = os.getenv("LOCAL_LLM_MODEL", "qwen/qwen3-4b-thinking-2507")
            self.client = OpenAI(base_url=local_url, api_key="not-needed")
            self.config.model = local_model
            print(f"[Synthesizer] Using local LLM: {local_url} / {local_model}")
        elif self.config.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def synthesize(
        self,
        query: str,
        evidence_cards: List[EvidenceCard],
        search_history: List[str],
        iterations: int
    ) -> SynthesizedResponse:
        """
        整合證據生成最終回答

        Args:
            query: 原始查詢
            evidence_cards: 所有收集的證據卡片
            search_history: 搜索歷史
            iterations: 迭代次數

        Returns:
            SynthesizedResponse
        """
        # 構建提示
        user_prompt = self._build_user_prompt(query, evidence_cards)

        # 調用 LLM
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # 本地 LLM 可能不支援 response_format
        if not use_local:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        # 解析回應
        content = response.choices[0].message.content
        result = self._parse_response(content, evidence_cards)

        # 補充元數據
        result.search_history = search_history
        result.iterations = iterations
        result.total_evidence_found = len(evidence_cards)

        return result

    def _build_user_prompt(
        self,
        query: str,
        evidence_cards: List[EvidenceCard]
    ) -> str:
        """構建用戶提示"""
        parts = [
            f"**問題:** {query}",
            "",
            "**收集到的證據:**",
            ""
        ]

        for i, card in enumerate(evidence_cards[:30]):  # 最多 30 張卡
            parts.append(f"--- 證據 {i+1} ---")
            parts.append(f"ID: {card.id[:8]}")
            parts.append(f"Path: {card.path}")
            parts.append(f"Symbol: {card.symbol or '-'}")
            parts.append(f"Source: {card.source_kind}")
            parts.append(f"Tags: {', '.join(card.tags)}")
            parts.append(f"Score: {card.score_rerank:.2f}")
            parts.append(f"Content:")
            # 控制內容長度
            content = card.chunk_text[:500] if len(card.chunk_text) > 500 else card.chunk_text
            parts.append(content)
            parts.append("")

        parts.append("---")
        parts.append("請根據以上證據，生成結構化回答（JSON 格式）。")

        return "\n".join(parts)

    def _parse_response(
        self,
        content: str,
        evidence_cards: List[EvidenceCard]
    ) -> SynthesizedResponse:
        """解析 LLM 回應"""
        try:
            # 處理 thinking model 的 <think> 標籤
            import re
            if '<think>' in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # 嘗試提取 JSON 部分
            content = content.strip()
            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start >= 0:
                    json_end = content.rfind('}')
                    if json_end > json_start:
                        content = content[json_start:json_end + 1]

            data = json.loads(content)

            # 解析 flow
            flow = []
            for f in data.get("flow", []):
                flow.append(FlowStep(
                    step=f.get("step", 0),
                    description=f.get("description", ""),
                    code_ref=f.get("code_ref", "")
                ))

            # 解析 decision_points
            decision_points = []
            for d in data.get("decision_points", []):
                decision_points.append(DecisionPoint(
                    condition=d.get("condition", ""),
                    true_branch=d.get("true_branch", ""),
                    false_branch=d.get("false_branch", ""),
                    code_ref=d.get("code_ref", "")
                ))

            # 解析 config
            config = []
            for c in data.get("config", []):
                config.append(ConfigItem(
                    key=c.get("key", ""),
                    default_value=c.get("default_value"),
                    source=c.get("source", ""),
                    description=c.get("description", "")
                ))

            # 解析 evidence
            evidence = []
            for e in data.get("evidence", []):
                # 檢查是否需要擴展
                quote = e.get("quote", "")
                needs_expand = e.get("needs_expand", False)
                expand_reasons = []

                # 額外檢查：是否缺少因果動詞
                if not has_causal_verb(quote):
                    needs_expand = True
                    expand_reasons.append("causal_verb_missing")

                evidence.append(EvidenceRef(
                    card_id=e.get("card_id", ""),
                    path=e.get("path", ""),
                    span=e.get("span", ""),
                    quote=quote,
                    needs_expand=needs_expand,
                    expand_reasons=expand_reasons
                ))

            return SynthesizedResponse(
                answer=data.get("answer", ""),
                flow=flow,
                decision_points=decision_points,
                config=config,
                evidence=evidence,
                search_history=[],
                iterations=0,
                total_evidence_found=0
            )

        except json.JSONDecodeError as e:
            # 解析失敗時返回基本回答
            return SynthesizedResponse(
                answer=f"Failed to parse synthesis: {e}. Raw response: {content[:200]}...",
                flow=[],
                decision_points=[],
                config=[],
                evidence=[],
                search_history=[],
                iterations=0,
                total_evidence_found=0
            )

    def expand_evidence(
        self,
        card: EvidenceCard,
        context_query: str
    ) -> str:
        """
        擴展單個證據的詳細內容

        用於 needs_expand=True 的情況
        """
        prompt = f"""從以下代碼中提取與問題相關的關鍵部分。

問題: {context_query}

代碼內容:
{card.chunk_text}

請提取最相關的 200-300 字內容，包含:
1. 關鍵的函數/方法調用
2. 狀態變更或配置讀取
3. 條件判斷邏輯

只輸出提取的內容，不要解釋。"""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0
        )

        return response.choices[0].message.content
