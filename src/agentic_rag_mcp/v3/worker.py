import json
import logging
from typing import Dict, Any

from src.agentic_rag_mcp.provider import create_client_for
from .state import AgenticState

logger = logging.getLogger(__name__)

WORKER_PROMPT = """You are an Elite Investigation Worker. You are given a specific "Work Order" by your Manager.
Your ONLY goal is to fulfill that Work Order perfectly, reading actual code to find "Hard Proof".

YOUR AVAILABLE ACTIONS (Tools):
1. `semantic_search`: Search code by meaning (e.g. finding where a field is used).
2. `graph_list_files`: List files in a CID or directory to see what exists.
3. `read_exact_file`: READ THE FILE. This is your most important tool. You must read files to get Hard Proof.
4. `graph_symbol_search`: Trace callers/callees or find Advices/Interceptors.
5. `report_to_manager`: Submit your final findings back to the Manager.

RULES OF INVESTIGATION:
- If your manager tells you to find a Call Chain, you must use `read_exact_file` and `graph_symbol_search` to trace the flow.
- DO NOT report back to the manager until you have actually read the code and extracted quotes, UNLESS you are stuck.
- If you cannot find a file after trying, or if you hit a dead end, use `report_to_manager` with `has_blocker=true`.
"""

WORKER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search code semantics.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "cid": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_list_files",
            "description": "List files in a CID or path.",
            "parameters": {"type": "object", "properties": {"dir_path": {"type": "string"}, "cid": {"type": "string"}}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_exact_file",
            "description": "Read file content.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_symbol_search",
            "description": "Find callers/callees or related advices.",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_to_manager",
            "description": "Submit your investigation results or report a blocker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {"type": "string", "description": "Detailed notes with Code Quotes and Call Chains."},
                    "has_blocker": {"type": "boolean", "description": "True if you are stuck."}
                },
                "required": ["findings", "has_blocker"]
            }
        }
    }
]

class WorkerAgent:
    def __init__(self, tool_executor):
        self.client, self.comp_cfg = create_client_for("planner") # Fast model is usually fine for worker
        self.tool_executor = tool_executor

    def act(self, state: AgenticState) -> Dict[str, Any]:
        logger.info("👷 [Worker] Executing assigned task...")
        
        system_prompt = WORKER_PROMPT

        user_content = f"WORK ORDER FROM MANAGER:\n{state['current_task']}\n\nTARGET CIDS: {state.get('target_cids', [])}\n"

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_content})
        
        # 注入 Worker 自己的暫存對話歷史
        current_messages = state.get("current_worker_messages", [])
        
        # 防呆機制：如果 Worker 在同一個工單中呼叫太多次工具（陷入死循環），強迫終止並向主管回報
        if len(current_messages) > 12:
            logger.warning(f"👷 [Worker] Too many tool calls in one task! Forcing report to Manager.")
            new_log_entry = f"Task: {state['current_task']}\nFindings (Forced Stop):\nWorker reached internal limit (too many tool calls). It might be stuck in a loop."
            return {
                "investigation_log": [new_log_entry],
                "current_worker_messages": [], 
                "current_task": "" 
            }
            
        if current_messages:
            messages.extend(current_messages)

        response = self.client.chat.completions.create(
            model=self.comp_cfg.model,
            messages=messages,
            tools=WORKER_TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        msg = response.choices[0].message
        
        new_messages = state.get("current_worker_messages", [])
        
        # 為了保持對話歷史的連續性，需要把 LLM 的回覆也塞進去
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        
        if msg.tool_calls:
            assistant_msg["tool_calls"] = []
            for tc in msg.tool_calls:
                assistant_msg["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })
        new_messages.append(assistant_msg)

        if msg.tool_calls:
            # 必須處理所有 tool calls
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                if tool_name == "report_to_manager":
                    logger.info(f"👷 [Worker] Reporting back to Manager. Blocker: {args.get('has_blocker', False)}")
                    
                    # 只回傳這回合新增的 log 字串，因為 state.py 中 investigation_log 定義為 Annotated[List[str], operator.add]
                    new_log_entry = f"Task: {state['current_task']}\nFindings:\n{args.get('findings', '')}"
                    
                    return {
                        "investigation_log": [new_log_entry],
                        "current_worker_messages": [], # 任務結束，清空暫存記憶 (覆蓋模式)
                        "current_task": "" # 清空當前任務，交還控制權
                    }
                else:
                    logger.info(f"👷 [Worker] Using Tool: {tool_name} with args: {args}")
                    tool_result = self.tool_executor(tool_name, args)
                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": str(tool_result)[:4000] 
                    })
            # 所有的 tool_call 都執行並附加完畢後再 return
            return {"current_worker_messages": new_messages}
        
        # Fallback if no tool called: LLM might have output pure text instead of a tool call
        logger.warning(f"👷 [Worker] No tool called! Content: {msg.content[:200]}")
        new_log_entry = f"Task: {state['current_task']}\nFindings (Fallback):\n{msg.content}"
        return {
            "investigation_log": [new_log_entry],
            "current_worker_messages": [], 
            "current_task": "" 
        }