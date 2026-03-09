import json
import logging
from typing import Dict, Any

from .provider import create_client_for
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
- **NEVER OUTPUT PLAIN TEXT ANSWERS**: You are not answering the user directly. You are communicating with your Manager. If you find the answer, you MUST use the `report_to_manager` tool to submit your findings. Do not just type the answer in the chat.
- **CRITICAL ERROR HANDLING**: If `read_exact_file` or `semantic_search` returns an error (like "File not found") or empty results `[]`:
  1. If you used a `cid` to restrict the search, try the search again with an EMPTY `cid` to perform a global search.
  2. Do NOT try the exact same path/query again without changing parameters.
  3. Do NOT start randomly searching the root directory `/` or making up fake paths.
  4. If all else fails, IMMEDIATELY call `report_to_manager` with `has_blocker=true` and tell the Manager what failed so they can give you a new direction.
- **PARTIAL FINDINGS (Time Management)**: If you have called tools many times and traced deep into the code, but haven't found the *complete* answer yet, you can call `report_to_manager` with your **Partial Findings** and `has_blocker=false` or `true` (depending on if you are stuck). It is better to report partial progress than to search endlessly.
- **SURGICAL READING**: When reading a large file (like a Controller or Service), DO NOT invent a 'search' or 'grep' tool. You ONLY have the 5 tools listed above. If you need to find a specific method, FIRST use `list_file_symbols` with the exact file path to get a clean list of its methods and their `line_start`/`line_end`, THEN use `read_exact_file` with the `path`, `line_start`, and `line_end` parameters to read only that method.
- **COPY-PASTE MANDATE**: When using tools that require a `file_path` (like `list_file_symbols` or `read_exact_file`), you MUST copy the exact string from the `file_path` field returned by your search tools. DO NOT abbreviate it or guess it.
- **ENCAPSULATION AWARENESS**: In modern frameworks, specific fields (like `merchantCode`) are often encapsulated inside Request objects (DTOs). If you surgically read a method (e.g. a Controller endpoint) and don't see the exact field name, DO NOT panic and DO NOT read the entire file. Simply report back that the method accepts a specific DTO object, and suggest the Manager to investigate that DTO or the downstream Service.
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
            "name": "list_file_symbols",
            "description": "Get a clean list of all methods inside a File, including their line_start and line_end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The exact, full file_path string returned by search tools. DO NOT abbreviate."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_exact_file",
            "description": "Read file content. Specify line_start and line_end to read specific methods.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"}
                }, 
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_symbol_search",
            "description": "Find callers/callees or get a list of Methods (with start_line/end_line) for a Class.",
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
                    "findings": {"type": "string", "description": "Detailed notes with Code Quotes and Call Chains. Or Partial Findings if you ran out of time."},
                    "has_blocker": {"type": "boolean", "description": "True if you are stuck or need a new direction."}
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
        # 放寬至 100，允許深度探索，但保留最後的保險絲
        if len(current_messages) > 100:
            logger.warning(f"👷 [Worker] Too many tool calls in one task! Forcing report to Manager.")
            # 這裡我們無法輕易總結 current_messages，只能告訴 Manager 發生了超時
            new_log_entry = f"Task: {state['current_task']}\nFindings (Forced Stop): Worker executed too many tools (exceeded 50 tool calls). The worker was likely tracing a very deep chain or stuck. Please review the previous task and provide a narrower, more specific scope."
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
        
        # --- [DEBUG] 印出 Worker 的內心獨白 ---
        if msg.content:
            logger.info(f"\n🧠 [Worker] Thinking: {msg.content}")

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
                
                # 防呆：如果 LLM 發明了不存在的工具
                valid_tools = ["semantic_search", "graph_list_files", "read_exact_file", "graph_symbol_search", "list_class_methods", "report_to_manager"]
                if tool_name not in valid_tools:
                    logger.warning(f"👷 [Worker] Hallucinated tool '{tool_name}'. Forcing report_to_manager.")
                    new_log_entry = f"Task: {state['current_task']}\nFindings:\nWorker became stuck trying to invent non-existent tools (like '{tool_name}'). Please rephrase the task or provide a more specific instruction."
                    return {
                        "investigation_log": [new_log_entry],
                        "current_worker_messages": [],
                        "current_task": ""
                    }
                
                if tool_name == "report_to_manager":
                    logger.info(f"👷 [Worker] Reporting back to Manager. Blocker: {args.get('has_blocker', False)}")
                    logger.info(f"📝 [Worker] Report to Manager: {args.get('findings', '')[:300]}...")
                    
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
                    
                    # --- [DEBUG] 印出 Worker 看到的工具回傳內容 ---
                    str_result = str(tool_result)
                    logger.info(f"📥 [Worker] Tool '{tool_name}' returned: {str_result[:500]}...")
                    
                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": str_result[:4000] 
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