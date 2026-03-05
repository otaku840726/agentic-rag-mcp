"""
Analyst LLM - Decomposes user query into investigation sub-tasks.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .provider import create_client_for
from .models import AnalystOutput
from .utils import extract_json_from_response

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a Lead Software Architect. Your task is to decompose a user query into highly effective search tasks.

PROJECT CONTEXT:
You are analyzing a large codebase. Use the provided [System Module Map] if available to prioritize specific Community IDs (CIDs).

YOUR GOAL:
Identify the core entities, their interactions, and the missing segments in your current knowledge.

STRATEGY:
1. **Deduplication**: If multiple intents point to the same functional area, merge them into ONE task.
2. **Precision**: Target specific CIDs mentioned in the [System Module Map] for faster discovery.
3. **Coverage**: Ensure you cover both the request entry point and the core business logic.

OUTPUT FORMAT (Strict JSON):
{
    "subject": "The main entity being discussed",
    "actors": ["List of services/roles involved"],
    "covered": ["What is already known"],
    "gaps": ["What we still need to find"],
    "sub_tasks": [
        "search: 'specific keywords' in CID [X]",
        "investigate: 'component interaction'"
    ],
    "reasoning": "Brief technical logic for this plan",
    "intent": "Search/Repair/Audit"
}
"""

class Analyst:
    def __init__(self, config=None):
        # 使用專門的配置或預設配置
        self.client, self.comp_cfg = create_client_for("analyst")
        self.model = self.comp_cfg.model
        self.temperature = 0.0

    def analyze(self, query: str) -> AnalystOutput:
        try:
            # 判斷 provider 是否支援並行 (目前 minimax local 不支援，這裡我們循序執行)
            # 在 AgenticSearch 中已經處理了 personas 的循序呼叫，這裡簡化為單次呼叫或主架構分析
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Decompose this query: {query}"}
                ],
                temperature=self.temperature
            )
            msg = response.choices[0].message
            content = msg.content or ""
            
            # 捕捉可能的 Reasoning
            reasoning = getattr(msg, "reasoning", None)
            if reasoning:
                logger.info(f"🤔 [Analyst Reasoning]:\n{reasoning}")

            data = extract_json_from_response(content)

            return AnalystOutput(
                subject=data.get("subject", ""),
                actors=data.get("actors", []),
                covered=data.get("covered", []),
                gaps=data.get("gaps", []),
                sub_tasks=data.get("sub_tasks", []),
                reasoning=data.get("reasoning", ""),
                intent=data.get("intent", "Search")
            )
        except Exception as e:
            logger.error(f"Analyst failed: {e}")
            return AnalystOutput("error", [], [], [], [f"fallback search: {query}"], str(e))
