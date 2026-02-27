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


SYNTHESIZER_SYSTEM_PROMPT = """You are a codebase expert.
Based on the collected evidence, generate a structured answer.

**Output format (JSON):**
{
    "answer": "summary conclusion (2-3 sentences)",
    "flow": [
        {
            "step": 1,
            "description": "step description",
            "code_ref": "card_id | path | L120-L150"
        }
    ],
    "decision_points": [
        {
            "condition": "condition description",
            "true_branch": "behavior when true",
            "false_branch": "behavior when false",
            "code_ref": "card_id | path | span"
        }
    ],
    "config": [
        {
            "key": "config key name",
            "default_value": "default value (if any)",
            "source": "source file path",
            "description": "description"
        }
    ],
    "evidence": [
        {
            "card_id": "id",
            "path": "file path",
            "span": "L120-L150",
            "quote": "key excerpt (120-200 chars)",
            "needs_expand": false
        }
    ]
}

**Requirements:**
1. answer must be concise but complete
2. flow must be in actual execution order
3. decision_points must list key branching logic
4. config must list all relevant configuration items
5. evidence must reference specific code; quote must be 120-200 characters
6. all references must come from the provided evidence — do not speculate
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

        # 解析回應
        content = response.choices[0].message.content

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
            f"**Question:** {query}",
            "",
            "**Collected Evidence:**",
            ""
        ]

        for i, card in enumerate(evidence_cards[:30]):  # max 30 cards
            parts.append(f"--- Evidence {i+1} ---")
            parts.append(f"ID: {card.id[:8]}")
            parts.append(f"Path: {card.path}")
            parts.append(f"Span: {card.span}")
            parts.append(f"Symbol: {card.symbol or '-'}")
            parts.append(f"Source: {card.source_kind}")
            parts.append(f"Tags: {', '.join(card.tags)}")
            parts.append(f"Score: {card.score_rerank:.2f}")
            parts.append(f"Content:")
            content = card.chunk_text[:2000] if len(card.chunk_text) > 2000 else card.chunk_text
            parts.append(content)
            parts.append("")

        parts.append("---")
        parts.append("Based on the evidence above, generate a structured answer in JSON format.")

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
        prompt = f"""Extract the most relevant parts of the following code with respect to the question.

Question: {context_query}

Code:
{card.chunk_text}

Extract the 200-300 most relevant characters, including:
1. Key function/method calls
2. State changes or configuration reads
3. Conditional branching logic

Output the extracted content only, no explanation."""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0
        )

        return response.choices[0].message.content
