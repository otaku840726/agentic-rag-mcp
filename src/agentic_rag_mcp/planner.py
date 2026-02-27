"""
LLMJudge - Semantic evidence satisfaction judge and impact completeness reviewer.
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .provider import create_client_for
from .models import MissingEvidence
from .utils import extract_json_from_response


@dataclass
class PlannerConfig:
    """Judge / Planner LLM config"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 4000
    temperature: float = 0.1
    base_url: Optional[str] = None


IMPACT_REVIEW_PROMPT = """You are an Impact Completeness Reviewer.

Task: Before the search concludes, determine whether the impact scope analysis for this engineering requirement is complete.
You are NOT evaluating whether the answer is correct — you are asking: **Are there any code areas that should have been confirmed but have not yet been found?**

**Symmetric completeness checklist:**

Data layer:
- If an Entity/Table write (INSERT/UPDATE/DAO.Save) was found → Have all readers been confirmed? (SELECT stored procedures, GET APIs, all Jobs/Services that SELECT from this Entity)
- If a DAO change was found → Have all upstream Services that depend on this DAO been confirmed?

Business layer:
- If a Service method was found → Have all callers been confirmed? (Controller entry points + Background Jobs/Schedulers)
- If an Interface definition was found → Have all implementing classes and the services they belong to been confirmed?

Interface/API layer:
- If an internal Controller API was found → Has the external response format been confirmed? (Response DTO, Merchant Callback DTO)
- If a Callback push was found → Have the corresponding fields in the Merchant query API been confirmed?

Message layer:
- If a Queue/RabbitMQ publish was found → Has the corresponding Consumer/Listener/Subscriber been confirmed?

Cross-service:
- If a shared Entity/Interface is affected → Have all other services that use it been confirmed?

**Output JSON only, no explanation:**
{
    "complete": true/false,
    "missing_impact_areas": ["description 1: Found X but have not confirmed Y", "description 2: ..."],
    "reason": "one sentence explaining the overall judgment"
}
"""


class LLMJudge:
    """
    LLM Judge - semantic judgment for missing evidence satisfaction.
    Short output: only true/false + 1-sentence reason.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        if config:
            from .provider import create_client
            self.config = config
            self.client = create_client(config.provider)
        else:
            self.config = PlannerConfig(max_tokens=150)
            client, comp_cfg = create_client_for("judge")
            self.client = client
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model
            self.config.max_tokens = comp_cfg.max_tokens
            self.config.temperature = comp_cfg.temperature

    def judge(
        self,
        need: str,
        accept: List[str],
        covered: List[str],
        candidates_summary: str,
        usage_log: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Semantically judge whether a single missing evidence item is satisfied.

        Args:
            need: what is needed
            accept: acceptance criteria list
            covered: regex-confirmed covered criteria
            candidates_summary: relevant evidence card summary (path + snippet)
            usage_log: performance tracking list

        Returns:
            {"satisfied": bool, "reason": str}
        """
        prompt = f"""Determine whether the following evidence requirement has been satisfied.

Requirement: {need}
Acceptance criteria: {', '.join(accept)}
Already covered by regex: {', '.join(covered) if covered else 'none'}
Candidate evidence:
{candidates_summary}

Reply with JSON only: {{"satisfied": true/false, "reason": "one-sentence reason"}}"""

        kwargs = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
        }

        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        start_time = time.time()
        response = self.client.chat.completions.create(**kwargs)
        latency = (time.time() - start_time) * 1000

        if usage_log is not None and hasattr(response, "usage") and response.usage:
            usage_log.append({
                "component": "judge",
                "model": self.config.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": round(latency),
            })

        try:
            content = response.choices[0].message.content
            return extract_json_from_response(content)
        except Exception:
            return {"satisfied": False, "reason": "Failed to parse judge response"}

    def review_completeness(
        self,
        query: str,
        evidence_summary: str,
        usage_log: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Impact scope completeness review — final AI check before stopping.

        Not per-item judgment, but holistic: "Given all collected evidence,
        are there any missing symmetric counterparts in the impact scope?"

        Args:
            query: original requirement description (Jira ticket)
            evidence_summary: compact summary of all found evidence
            usage_log: performance tracking list

        Returns:
            {"complete": bool, "missing_impact_areas": [...], "reason": str}
        """
        user_prompt = f"Requirement: {query}\n\nCollected evidence:\n{evidence_summary}"

        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": IMPACT_REVIEW_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 600,
            "temperature": 0,
        }

        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        start_time = time.time()
        response = self.client.chat.completions.create(**kwargs)
        latency = (time.time() - start_time) * 1000

        if usage_log is not None and hasattr(response, "usage") and response.usage:
            usage_log.append({
                "component": "impact_reviewer",
                "model": self.config.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": round(latency),
            })

        try:
            return extract_json_from_response(response.choices[0].message.content)
        except Exception:
            # Conservative: assume complete on parse error to avoid infinite loop
            return {"complete": True, "missing_impact_areas": [], "reason": "parse_error_assume_complete"}
