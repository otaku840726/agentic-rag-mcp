import json
import logging
from typing import Dict, Any

from src.agentic_rag_mcp.provider import create_client_for
from .state import AgenticState

logger = logging.getLogger(__name__)

MANAGER_PROMPT = """You are the Lead Software Architect and Manager. Your goal is to oversee the investigation of a codebase to answer the user's query perfectly.

YOU HAVE A DEDICATED WORKER.
You do NOT need to read code yourself. Your job is to assign highly specific, unambiguous "Work Orders" to your Worker, and review their reports.

YOUR AVAILABLE ACTIONS (Tools):
1. `semantic_search`: Use this ONLY for a quick "scouting" check to find which Community IDs (CIDs) are relevant before assigning a task.
2. `assign_task`: Dispatch a Work Order to your Worker.
3. `finish_investigation`: Output the final, structured answer for the user and end the process.

DELEGATION RULES (The "No Vague Tasks" Policy):
When calling `assign_task`, your `instruction` must be explicit and specify the EXACT format you want the report in.
- BAD: "Find merchantCode."
- GOOD: "Go to CID 288. Find the Controller handling Member Deposit. Read its code, and draw a Hierarchical Call Chain showing how it validates the merchantCode."

AUDIT RULES:
When the Worker submits a report (found in your `investigation_log`), review it strictly:
- Does it have hard code quotes?
- Does it form a complete logical chain?
- If the Worker reports a blocker (e.g. "File not found"), use your scouting tools or logic to provide a different CID or path in your next `assign_task`.
"""

MANAGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Scout the codebase to find relevant CIDs before assigning tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "High-level architectural keyword"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_task",
            "description": "Assign a strict, formatted work order to the Worker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "Highly specific instructions, including the required report format (e.g., 'Draw a Call Chain')."},
                    "target_cids": {"type": "array", "items": {"type": "string"}, "description": "CIDs the worker should focus on."}
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_investigation",
            "description": "End the investigation and provide the final answer to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string", "description": "The comprehensive, structured answer."}
                },
                "required": ["final_answer"]
            }
        }
    }
]

class ManagerAgent:
    def __init__(self, tool_executor):
        self.client, self.comp_cfg = create_client_for("analyst") # Using analyst model for high reasoning
        self.tool_executor = tool_executor

    def act(self, state: AgenticState) -> Dict[str, Any]:
        logger.info("👔 [Manager] Analyzing state and deciding next move...")
        
        system_prompt = MANAGER_PROMPT
        if state.get("project_context"):
            system_prompt = f"--- GLOBAL PROJECT CONTEXT ---\n{state['project_context']}\n\n====================\n{MANAGER_PROMPT}"

        user_content = f"USER QUERY: {state['query']}\n\n"
        
        if state.get("investigation_log"):
            user_content += "--- WORKER REPORTS (Investigation Log) ---\n"
            for idx, log in enumerate(state["investigation_log"]):
                user_content += f"\n[Report {idx+1}]\n{log}\n"
                
        if state.get("manager_thoughts"):
            user_content += "\n--- YOUR PREVIOUS THOUGHTS ---\n"
            user_content += "\n".join(state["manager_thoughts"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.client.chat.completions.create(
            model=self.comp_cfg.model,
            messages=messages,
            tools=MANAGER_TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        msg = response.choices[0].message
        
        # 只收集這回合產生的「新」筆記，交給 LangGraph 的 reducer (operator.add) 去合併
        new_thoughts_this_turn = []
        if hasattr(msg, 'reasoning') and msg.reasoning:
            logger.info(f"👔 [Manager Reasoning]:\n{msg.reasoning[:500]}...")
            new_thoughts_this_turn.append(msg.reasoning)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)
                
                if tool_name == "finish_investigation":
                    return {"is_finished": True, "final_answer": args["final_answer"], "manager_thoughts": new_thoughts_this_turn}
                
                elif tool_name == "assign_task":
                    logger.info(f"👔 [Manager] Assigning Task: {args['instruction'][:100]}...")
                    return {
                        "current_task": args["instruction"],
                        "target_cids": args.get("target_cids", []),
                        "manager_thoughts": new_thoughts_this_turn
                    }
                    
                elif tool_name == "semantic_search":
                    logger.info(f"👔 [Manager] Scouting with query: {args['query']}")
                    scout_result = self.tool_executor("semantic_search", args)
                    new_thoughts_this_turn.append(f"Scout Result for '{args['query']}': {str(scout_result)[:500]}")
                    return {"manager_thoughts": new_thoughts_this_turn}
                    
        return {"manager_thoughts": new_thoughts_this_turn}