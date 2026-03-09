import json
import logging
from typing import Dict, Any

from .provider import create_client_for
from .state import AgenticState

logger = logging.getLogger(__name__)

MANAGER_PROMPT = """You are the Lead Software Architect and Manager. Your goal is to oversee the investigation of a codebase to answer the user's query perfectly.

YOU HAVE A DEDICATED WORKER.
You do NOT need to read code yourself. Your job is to assign highly specific, unambiguous "Work Orders" to your Worker, and review their reports.

YOUR AVAILABLE ACTIONS (Tools):
1. `community_search`: **(START HERE)** Use this to get a macro-view of the project. It returns the top relevant CIDs and their module names.
2. `semantic_search`: Scout the codebase to find relevant CIDs or code snippets before assigning a task.
3. `graph_list_files` / `graph_symbol_search` / `read_exact_file`: You have full access to these tools. Use them to verify your hypothesis before delegating heavy reading to the Worker.
4. `assign_task`: Dispatch a Work Order to your Worker.
5. `finish_investigation`: Output the final, structured answer for the user and end the process.

DELEGATION RULES (The "No Vague Tasks" Policy):
When calling `assign_task`, your `instruction` must be explicit and specify the EXACT format you want the report in.
- BAD: "Find merchantCode."
- GOOD: "Go to CID 288. Find the Controller handling Member Deposit. Read its code, and draw a Hierarchical Call Chain showing how it validates the merchantCode."
- **Path Purity**: When telling the worker to read a file, provide ONLY the clean file path (e.g. `paybnb-service/src/...`). Do NOT prepend it with CID or other text.
- **NO REPETITION (CRITICAL)**: Before assigning a task, READ the "WORKER REPORTS". If a file path or a specific class (e.g. `BrandCodeValidator.java`) has already been read and reported on by the Worker, DO NOT assign the Worker to read it again. Move on to the next logical step, or finish the investigation.

IMPORTANT DATA TYPES & SCOUTING RULES:
- **CID (Community ID)**: This is ALWAYS an Integer (e.g., 288, 0, 43). NEVER use a UUID string or folder name as a CID.
- **CID Scouting Strategy (Breadth-First)**: 
  1. Before diving into files, ALWAYS use `community_search` with your initial query to discover the top 5-6 relevant CIDs (Modules). 
  2. **YOU MUST EXHAUST ALL CIDs**: You must systematically inspect EVERY SINGLE CID returned by `community_search` (using `semantic_search` or `graph_list_files` with that specific `cid` parameter). Do NOT skip any of the top 6 CIDs just because the first one or two didn't have the answer. 
  3. Only after you have explicitly checked all the returned CIDs, if you still haven't found the answer, you can fallback to global search.
- **Fallback**: If a search returns empty `[]` within a specific CID, it means that module doesn't contain the keyword. Move on to the next CID on your list.
- **SURGICAL READING (CRITICAL)**: If you find a large file (like a Controller or Service) and want to inspect its code using `read_exact_file`, DO NOT read it blindly from line 1 to 400. Instead, FIRST use `list_file_symbols` with the exact file path to get a clean list of its methods and their `line_start` / `line_end`. THEN use `read_exact_file` with precise `line_start` and `line_end` parameters to read only the exact method you care about.
- **COPY-PASTE MANDATE**: When using tools that require a `file_path` (like `list_file_symbols` or `read_exact_file`), you MUST copy the exact string from the `file_path` field returned by your search tools (e.g. `paybnb-service/src/main/java/net/funpodium/paybnb/controller/v2/transaction/MemberDepositV2Controller.java`). DO NOT abbreviate it.
- **ENCAPSULATION AWARENESS**: In modern frameworks, specific fields (like `merchantCode`) are often encapsulated inside Request objects (DTOs). If you surgically read a Controller endpoint and don't see the exact field name, DO NOT read the entire Controller file. Instead, check the DTO object being passed to the method, and search for that DTO.
- **GLOBAL SEARCH MANDATE**: When calling tools, if you do NOT have a specific CID to target, you MUST omit the `cid` property entirely from your JSON payload. DO NOT send `"0"`, `""`, or `"root"`. Omit the key completely to trigger a global search.

AUDIT RULES & BLOCKER HANDLING:
When the Worker submits a report (found in your `investigation_log`), review it strictly.
- **CRITICAL**: If the Worker reports a blocker (e.g. "File not found", "Too many tool calls", "Stuck in a loop"), you MUST change your strategy. DO NOT assign the exact same task, path, or CID again. You must scout for an alternative route before your next assignment.
- **COMPLETION & EMPATHY (QUALITY ASSURANCE)**: You are not just a code finder; you are a Senior Architect mentoring a user. Before calling `finish_investigation`, you MUST review the "ANALYST'S INVESTIGATION PLAN". 
  1. Did you answer the core question?
  2. Did you provide the contextual/extra information the Analyst suggested (e.g. error codes, where to find the value, JSON examples)?
  3. Your final answer MUST be rich, deeply explanatory, and formatted beautifully (using Markdown, tables, or code blocks where appropriate). Always include "Tips/Warnings" for the user. Do not be lazy.
- **NO HALLUCINATION**: If your scouting tools return nothing, do NOT invent an answer. Keep searching with different keywords.

**ABSOLUTE MANDATE**: You are a programmable agent. You MUST ALWAYS call EXACTLY ONE tool in every single response. NEVER output plain text as your final action. If you are ready to give the final answer, you MUST call the `finish_investigation` tool. If you output plain text without calling a tool, the system will crash.
"""

MANAGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "community_search",
            "description": "Find logical modules/communities in the architecture. Returns top CIDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "High-level module name or business concept"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_file_symbols",
            "description": "Get a clean list of all methods/classes inside a File, including their line_start and line_end.",
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
            "name": "semantic_search",
            "description": "Scout the codebase to find relevant CIDs before assigning tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "High-level architectural keyword"},
                    "cid": {"type": "string", "description": "The exact Community ID integer. To search globally, OMIT this property entirely. DO NOT send '0' or empty strings."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_list_files",
            "description": "Scout files in a specific directory or CID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Optional path snippet"},
                    "cid": {"type": "string", "description": "The exact Community ID integer. To search globally, OMIT this property entirely."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_symbol_search",
            "description": "Trace callers/callees or find Advices/Interceptors quickly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_exact_file",
            "description": "Read file content directly.",
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
                    "final_answer": {"type": "string", "description": "The comprehensive, rich, and empathetic final answer in Markdown. Must address all points in the Analyst's Plan."}
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
        
        # 注入 Analyst 的意圖分析
        if state.get("intent_analysis"):
            user_content += f"--- ANALYST'S INVESTIGATION PLAN ---\n{state['intent_analysis']}\n\n"
        
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
        # 為了除錯，印出 Manager 的內心獨白
        if msg.content:
            logger.info(f"\n🧠 [Manager] Thinking: {msg.content}")

        new_thoughts_this_turn = []
        if hasattr(msg, 'reasoning') and msg.reasoning:
            logger.info(f"👔 [Manager Reasoning]:\n{msg.reasoning}")
            new_thoughts_this_turn.append(msg.reasoning)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                if tool_name == "finish_investigation":
                    logger.info(f"🏁 [Manager] Finished Investigation:\n{args.get('final_answer', '')}")
                    return {"is_finished": True, "final_answer": args.get("final_answer", ""), "manager_thoughts": new_thoughts_this_turn}
                
                elif tool_name == "assign_task":
                    logger.info(f"👔 [Manager] Assigning Task:\n{args.get('instruction', '')}")
                    return {
                        "current_task": args.get("instruction", ""),
                        "target_cids": args.get("target_cids", []),
                        "manager_thoughts": new_thoughts_this_turn
                    }
                    
                elif tool_name in ["community_search", "semantic_search", "graph_list_files", "graph_symbol_search", "list_file_symbols", "read_exact_file"]:
                    logger.info(f"👔 [Manager] Scouting with {tool_name}:\n{json.dumps(args, indent=2)}")
                    scout_result = self.tool_executor(tool_name, args)
                    
                    # --- [DEBUG] 印出 Manager 看到的工具回傳內容 ---
                    str_result = str(scout_result)
                    logger.info(f"📥 [Manager] Tool '{tool_name}' returned:\n{str_result}")
                    
                    new_thoughts_this_turn.append(f"Scout Result for '{tool_name}' ({args}):\n{str_result}")
                    return {"manager_thoughts": new_thoughts_this_turn}
                    
        # Fallback if no tool called (防呆：防止 Manager 忘記呼叫工具導致死循環)
        logger.warning(f"👔 [Manager] Failed to call a tool! Content:\n{msg.content}")
        new_thoughts_this_turn.append(f"SYSTEM WARNING: You did not call any tools! You MUST call a tool. If you know the answer, call 'finish_investigation'. Otherwise, call 'assign_task' or a scouting tool.")
        return {"manager_thoughts": new_thoughts_this_turn}