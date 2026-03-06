"""
Planner LLM - Decides the next tool to call based on current evidence.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .provider import create_client_for
from .models import PlannerOutput, MissingEvidence, PlannerConfig

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a Code Search Strategist. Your goal is to gather enough code evidence to fully answer the user's query.

STRATEGIC RULES (EFFICIENCY FIRST):
1. **The "Rule of Content"**: If a file path is in 'Current Evidence' with 'Status: MISSING', and the Critic points it out as relevant, your VERY NEXT action MUST be `read_exact_file`. Do NOT perform more semantic searches until the core files are read.
2. **The "Graph Navigation"**: If `semantic_search` keeps hitting the wrong module, use `graph_list_files` with a `pattern` (e.g., "*Controller*") and a `cid` to see all relevant files in that community.
3. **The "Search Defense"**: If the Critic has REJECTED a domain (e.g., 'Stop searching in CID 73 Promo'), you MUST acknowledge this. In your next `semantic_search`, explicitly focus on different CIDs or paths. 
4. **No Redundancy**: Do not call the same tool with the same arguments if the previous result was empty or rejected.
5. **Completion**: If you have enough evidence and no missing elements are reported by the Critic, simply do not call any tools. This signals that the investigation is complete.

Your tool calls will be executed by Workers. Provide a clear rationale explaining why you are calling these specific tools and what rejected paths you are avoiding.
"""

TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Find initial entry points, APIs, or general concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms"},
                    "cid": {"type": "string", "description": "Optional Community ID to restrict the search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_list_files",
            "description": "List files in a directory or community via Knowledge Graph. Use this to find specific controllers/DTOs when semantic search is too broad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Optional path snippet"},
                    "cid": {"type": "string", "description": "Optional Community ID"},
                    "pattern": {"type": "string", "description": "Optional name pattern, e.g., '*Controller*'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_symbol_search",
            "description": "Find callers/callees of a specific class or method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "ClassName or MethodName"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_exact_file",
            "description": "Read the full body of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file. e.g., 'path/to/file.java'"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_search",
            "description": "Find existing execution flows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search for in process flows"}
                },
                "required": ["query"]
            }
        }
    }
]

class Planner:
    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.client, self.comp_cfg = create_client_for("planner")
        
        # Load global project context
        self.project_context = ""
        try:
            import os
            context_path = os.path.join(os.getcwd(), "PROJECT_CONTEXT.md")
            if os.path.exists(context_path):
                with open(context_path, "r", encoding="utf-8") as f:
                    self.project_context = f.read()
        except Exception as e:
            logger.warning(f"Planner could not load PROJECT_CONTEXT: {e}")

    def plan(
        self,
        query: str,
        evidence_summary: str,
        search_history: List[str],
        iteration: int,
        previous_missing: List[MissingEvidence] = None,
        sub_tasks: List[str] = None,
        tool_results: List[Dict[str, Any]] = None,
        critic_feedback: str = "",
        tool_messages: List[Dict[str, Any]] = None # 【新增：傳遞真實對話歷史】
    ) -> PlannerOutput:
        
        user_prompt = f"User Query: {query}\nIteration: {iteration}\n\nTarget Sub-tasks: {sub_tasks}\n\n"
        
        if critic_feedback:
            user_prompt += f"CRITICAL - Critic Feedback & Guidance:\n{critic_feedback}\n\n"
            
        user_prompt += f"Current Evidence Summary:\n{evidence_summary}\n\n"
        
        if previous_missing:
            critic_notes = [m.need for m in previous_missing]
            user_prompt += f"Missing Elements to Find:\n{json.dumps(critic_notes, indent=2)}\n\n"
            
        user_prompt += "What is your next tool call? Provide your reasoning, then call the appropriate tools. If no further action is needed, do not call any tools."

        system_prompt = PLANNER_PROMPT
        if self.project_context:
            system_prompt = f"--- GLOBAL PROJECT CONTEXT & NAVIGATION RULES ---\n{self.project_context}\n\n=======================================================\n\n{PLANNER_PROMPT}"

        # 組合歷史訊息
        messages = [{"role": "system", "content": system_prompt}]
        if tool_messages:
            messages.extend(tool_messages)
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.comp_cfg.model,
                messages=messages,
                tools=TOOLS_DEF,
                tool_choice="auto",
                temperature=self.config.temperature
            )
            msg = response.choices[0].message

            reasoning = getattr(msg, "reasoning", msg.content or "")
            if reasoning:
                logger.info(f"🤔 [Planner Reasoning]:\n{reasoning}")

            tool_calls = []
            should_stop = True
            raw_assistant_message = {"role": "assistant", "content": msg.content}
            
            if msg.tool_calls:
                should_stop = False
                raw_assistant_message["tool_calls"] = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        tool_calls.append({"tool": tc.function.name, "args": args, "id": tc.id})
                        raw_assistant_message["tool_calls"].append({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse tool arguments for {tc.function.name}: {e}")

            # 將這回合的助手訊息包裝進 tool_results 中，以便 WorkerGraph 收集
            tool_results_out = [{"type": "assistant_msg", "message": raw_assistant_message}]

            return PlannerOutput(
                next_queries=[], 
                missing_evidence=[],
                evidence_found=[],
                rationale=reasoning,
                should_stop=should_stop,
                tool_calls=tool_calls,
                tool_results=tool_results_out # 【新增回傳】
            )
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return PlannerOutput([], [], [], str(e), True, [])
