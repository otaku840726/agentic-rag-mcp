"""
Planner LLM - Decides the next tool to call based on current evidence.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .provider import create_client_for
from .models import PlannerOutput, MissingEvidence, PlannerConfig
from .utils import extract_json_from_response

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a Code Search Strategist. Your goal is to gather enough code evidence to fully answer the user's query.

AVAILABLE TOOLS:
1. `semantic_search`: Find initial entry points, APIs, or general concepts. 
   - Arguments: {"query": "search terms", "cid": "optional_community_id"}
2. `graph_list_files`: List files in a directory or community via Knowledge Graph. Use this to find specific controllers/DTOs when semantic search is too broad.
   - Arguments: {"dir_path": "optional_path_snippet", "cid": "optional_cid", "pattern": "optional_name_pattern"}
3. `graph_symbol_search`: Find callers/callees of a specific class or method. 
   - Arguments: {"symbol": "ClassName"}
4. `read_exact_file`: Read the full body of a file. 
   - Arguments: {"path": "absolute/path/to/file.java"}
5. `process_search`: Find existing execution flows. 
   - Arguments: {"query": "keyword"}

STRATEGIC RULES (EFFICIENCY FIRST):
1. **The "Rule of Content"**: If a file path is in 'Current Evidence' with 'Status: MISSING', and the Critic points it out as relevant, your VERY NEXT action MUST be `read_exact_file`. Do NOT perform more semantic searches until the core files are read.
2. **The "Graph Navigation"**: If `semantic_search` keeps hitting the wrong module, use `graph_list_files` with a `pattern` (e.g., "*Controller*") and a `cid` to see all relevant files in that community.
3. **The "Search Defense"**: If the Critic has REJECTED a domain (e.g., 'Stop searching in CID 73 Promo'), you MUST acknowledge this. In your next `semantic_search`, explicitly focus on different CIDs or paths. 
4. **No Redundancy**: Do not call the same tool with the same arguments if the previous result was empty or rejected.

Please output your plan in the following JSON format:
{
    "tool_calls": [
        {"tool": "tool_name", "args": {"arg_key": "arg_value"}}
    ],
    "rationale": "Execution Logic: (1) What files I am reading to fill gaps, (2) What rejected paths I am avoiding.",
    "should_stop": false
}
"""

class Planner:
    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.client, self.comp_cfg = create_client_for("planner")

    def plan(
        self,
        query: str,
        evidence_summary: str,
        search_history: List[str],
        iteration: int,
        previous_missing: List[MissingEvidence] = None,
        sub_tasks: List[str] = None,
        tool_results: List[Dict[str, Any]] = None,
        critic_feedback: str = ""
    ) -> PlannerOutput:
        
        user_prompt = f"User Query: {query}\nIteration: {iteration}\n\nTarget Sub-tasks: {sub_tasks}\n\n"
        
        if critic_feedback:
            user_prompt += f"CRITICAL - Critic Feedback & Guidance:\n{critic_feedback}\n\n"
            
        user_prompt += f"Current Evidence Summary:\n{evidence_summary}\n\n"
        
        if previous_missing:
            critic_notes = [m.need for m in previous_missing]
            user_prompt += f"Missing Elements to Find:\n{json.dumps(critic_notes, indent=2)}\n\n"
            
        user_prompt += "What is your next tool call? Remember the Rule of Content and Search Defense."

        try:
            response = self.client.chat.completions.create(
                model=self.comp_cfg.model,
                messages=[
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature
            )
            msg = response.choices[0].message

            reasoning = getattr(msg, "reasoning", None)
            if reasoning:
                logger.info(f"🤔 [Planner Reasoning]:\n{reasoning}")

            content = msg.content or ""
            print(f"\n[DEBUG - Planner Raw Content]\n{content}\n")

            data = extract_json_from_response(content)

            tool_calls = data.get("tool_calls") or data.get("plan") or data.get("tools") or []

            return PlannerOutput(
                next_queries=[], 
                missing_evidence=[],
                evidence_found=[],
                rationale=data.get("rationale", ""),
                should_stop=data.get("should_stop", False),
                tool_calls=tool_calls
            )
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return PlannerOutput([], [], [], str(e), True, [])
