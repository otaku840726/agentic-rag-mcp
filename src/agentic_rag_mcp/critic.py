"""
Critic LLM - Audits the collected evidence for completeness and domain consistency.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .provider import create_client_for
from .models import MissingEvidence
from .utils import extract_json_from_response

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """You are a Lead Code Auditor. Your job is to review the WORKER'S INVESTIGATION LOG and determine if the research is complete.

AUDIT CRITERIA:
1. **Fact Verification**: Does the log provide "Hard Proof" (direct code quotes) for its claims? 
2. **Logic Completeness**: Has the worker traced the full path? (e.g., if asking about a field, did they find the DTO *and* the Controller using it?)
3. **Domain Purity**: If the worker found files from a wrong module (e.g., Promo instead of Deposit), reject those specific findings.

DECISION LOGIC:
- If the worker's log provides clear code quotes and answers the user's query, set `is_complete` to true.
- If the worker only has file paths but no internal facts/quotes, set `is_complete` to false and demand `read_exact_file`.

Please output your audit in the following JSON format:
{
    "is_complete": false,
    "audit_report": "Summary of what is verified and what is still a 'trust-me' claim without proof.",
    "refinement_guidance": "Direction for the Planner: 'The findings in file X are good, but we still need the validation logic in file Y.'",
    "rejected_ids": ["IDs of evidence cards that were proven irrelevant during investigation"],
    "missing_elements": [
        {"need": "Specific fact or code block missing", "accept": ["keywords"]}
    ]
}
"""

@dataclass
class CriticOutput:
    is_complete: bool
    confidence_score: float
    critique: str
    missing_elements: List[Dict[str, Any]]
    suspicious_ids: List[str]
    rejected_ids: List[str]
    reasoning: str
    refinement_guidance: str = ""
    verification_tasks: List[Dict[str, Any]] = field(default_factory=list)

class Critic:
    def __init__(self, config=None):
        self.client, self.comp_cfg = create_client_for("analyst")
        self.model = self.comp_cfg.model
        self.temperature = 0.0

    def evaluate(self, query: str, structured_evidence: List[Dict[str, Any]], findings: str = "", intent: str = "", sub_tasks: List[str] = []) -> CriticOutput:
        try:
            summary = ""
            for e in structured_evidence[:15]:
                summary += f"ID: {e['id']} | Path: {e['path']} | Symbol: {e['symbol']}\n"
            
            user_prompt = f"""USER QUERY: {query}
TARGET INTENT: {intent}

--- WORKER'S LATEST INVESTIGATION LOG ---
{findings}

--- ATTACHED EVIDENCE CARDS (Reference only) ---
{summary}

Please audit the Investigation Log. Does it provide enough implementation proof?
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            msg = response.choices[0].message
            content = msg.content or ""
            data = extract_json_from_response(content)
            
            guidance = data.get("refinement_guidance", "")
            report = data.get("audit_report", "")
            final_critique = f"{guidance}\n\nAudit: {report}" if guidance else report

            return CriticOutput(
                is_complete=data.get("is_complete", False),
                confidence_score=0.5,
                critique=final_critique,
                missing_elements=data.get("missing_elements", []),
                suspicious_ids=[],
                rejected_ids=data.get("rejected_ids", []),
                reasoning=final_critique,
                refinement_guidance=guidance
            )
        except Exception as e:
            logger.error(f"Critic failed: {e}")
            return CriticOutput(True, 1.0, str(e), [], [], [], str(e))
