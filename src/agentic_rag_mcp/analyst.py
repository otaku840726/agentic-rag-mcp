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
1. **Task Type Identification (Crucial)**:
   - For **"Pinpoint Tracking"** questions (e.g., "What does field X mean?", "Where is API Y defined?"): Generate exactly ONE (1) consolidated investigation task. Combine all relevant CIDs into this single task's scope so a single Elite Worker can trace the logic deeply without parallel interference.
   - For **"Broad Discovery"** questions (e.g., "How does the entire architecture work?"): You may generate multiple parallel tasks to explore different domains.
2. **Deduplication**: Never assign multiple workers to search for the same entity in different folders.
3. **Precision**: Target specific CIDs mentioned in the [System Module Map].

OUTPUT FORMAT (Strict JSON):
{
    "subject": "The main entity being discussed",
    "actors": ["List of services/roles involved"],
    "covered": ["What is already known"],
    "gaps": ["What we still need to find"],
    "sub_tasks": [
        "investigate: 'component interaction' across CID [X, Y, Z]"
    ],
    "reasoning": "Brief technical logic for this plan, explicitly stating if this is Pinpoint Tracking or Broad Discovery.",
    "intent": "Search/Repair/Audit"
}
"""

class Analyst:
    def __init__(self, config=None, hybrid_search=None):
        self.client, self.comp_cfg = create_client_for("analyst")
        self.model = self.comp_cfg.model
        self.temperature = 0.0
        
        # 嘗試讀取全域專案上下文
        self.project_context = ""
        try:
            import os
            context_path = os.path.join(os.getcwd(), "PROJECT_CONTEXT.md")
            if os.path.exists(context_path):
                with open(context_path, "r", encoding="utf-8") as f:
                    self.project_context = f.read()
                    logger.info("Loaded PROJECT_CONTEXT.md from local file successfully.")
            elif hybrid_search and hasattr(hybrid_search, "client"):
                import uuid
                system_doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "system.project_context"))
                res = hybrid_search.client.retrieve(
                    collection_name=hybrid_search.collection_name,
                    ids=[system_doc_id]
                )
                if res:
                    self.project_context = res[0].payload.get("content", "")
                    logger.info("Loaded PROJECT_CONTEXT from Qdrant successfully.")
        except Exception as e:
            logger.warning(f"Could not load PROJECT_CONTEXT: {e}")

    def analyze(self, query: str) -> AnalystOutput:
        try:
            system_prompt = ANALYST_SYSTEM_PROMPT
            if self.project_context:
                system_prompt = f"--- GLOBAL PROJECT CONTEXT & RULES ---\n{self.project_context}\n\n===================\n\n{ANALYST_SYSTEM_PROMPT}"
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this query: {query}"}
                ],
                temperature=self.temperature
            )
            msg = response.choices[0].message
            content = msg.content or ""
            
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
