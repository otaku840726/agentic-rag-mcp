"""
Synthesizer LLM - 最終整合回應
只在循環結束後調用一次
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .provider import create_client_for
from .models import (
    SynthesizedResponse, FlowStep, DecisionPoint,
    ConfigItem, EvidenceRef, EvidenceCard
)
from .utils import has_causal_verb, CAUSAL_VERBS, extract_json_from_response


SYNTHESIZER_SYSTEM_PROMPT = """You are an Expert Codebase Synthesizer.
Your task is to generate a structured, highly accurate technical response based STRICTLY on the provided evidence.

**Output Format (JSON):**
{
    "answer": "A concise and direct conclusion (2-3 sentences).",
    "flow": [
        {
            "step": 1,
            "description": "Step description",
            "code_ref": "card_id | path"
        }
    ],
    "decision_points": [
        {
            "condition": "Condition description",
            "true_branch": "Action if true",
            "false_branch": "Action if false",
            "code_ref": "card_id | path"
        }
    ],
    "config": [
        {
            "key": "Config key name",
            "default_value": "Default value if any",
            "source": "Source file",
            "description": "Explanation"
        }
    ],
    "evidence": [
        {
            "card_id": "id",
            "path": "File path",
            "span": "Chunk info",
            "quote": "Key quote (120-200 chars)",
            "needs_expand": false
        }
    ]
}

**CRITICAL RULES (ANTI-HALLUCINATION):**
1. **Strict Empiricism**: Every step in the `flow` MUST be directly backed by the provided snippets.
2. **Framework-Awareness (Implicit Execution)**: If you identify framework-level components that implicitly intercept or pre-process requests (such as middleware, interceptors, request filters, or global error handlers), you MUST include them as explicit steps in the `flow` (usually as early pre-processing steps). Describe any data transformations or mappings they perform. This does not violate the anti-hallucination rule as long as the component is present in the evidence.
3. **NO Hallucinated Links**: DO NOT connect Class A to Class B unless you explicitly see the call site in the evidence or recognize a standard implicit framework pattern. If they are in the same evidence pool but completely unconnected (e.g., v1 and v2 endpoints), treat them as separate or alternative flows.
4. **Domain Isolation**: If evidence contains mixed domains (e.g., Member-side vs Merchant-side), clearly state that there are multiple contexts, rather than forcing them into a single linear flow.
5. **Transparency**: If you only have file paths and no method bodies, state: "The exact logic is not available in the extracted snippets."
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
        if config:
            from .provider import create_client
            self.config = config
            self.client = create_client(config.provider)
        else:
            self.config = SynthesizerConfig()
            client, comp_cfg = create_client_for("synthesizer")
            self.client = client
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model
            self.config.max_tokens = comp_cfg.max_tokens
            self.config.temperature = comp_cfg.temperature

    def synthesize(
        self,
        query: str,
        evidence_cards: List[EvidenceCard],
        search_history: List[str],
        iterations: int,
        logger: Any = None,
        search_id: str = "",
        usage_log: Optional[list] = None
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
        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        import time
        start_time = time.time()
        
        response = self.client.chat.completions.create(**kwargs)
        
        latency = (time.time() - start_time) * 1000

        msg = response.choices[0].message
        
        # 1. 兼容性日誌：捕捉原生的 Reasoning 或 Content
        reasoning = getattr(msg, "reasoning", None)
        if reasoning:
            if logger:
                logger.log_llm_event(search_id, "synthesizer_reasoning", self.config.model, [], reasoning, 0)
            else:
                print(f"\n[DEBUG - Synthesizer Reasoning]\n{reasoning}\n")

        content = msg.content or ""

        if usage_log is not None and hasattr(response, "usage") and response.usage:
            usage_log.append({
                "component": "synthesizer",
                "model": self.config.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": round(latency),
            })

        # Log to trace if logger provided
        if logger and search_id:
            logger.log_llm_event(
                search_id=search_id,
                step="synthesizer",
                model=self.config.model,
                messages=kwargs["messages"],
                response=content,
                latency_ms=latency
            )

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
            data = extract_json_from_response(content)

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
