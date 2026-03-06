"""
Worker Graph - 獨立執行單個 SubTask，擁有隔離的上下文。
"""

import logging
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from ..models import WorkerState, EvidenceCard
from ..planner import Planner
from ..critic import Critic
from ..provider import create_client_for

logger = logging.getLogger(__name__)

class WorkerGraph:
    def __init__(self, search_tools_callback, budget_config):
        self.planner = Planner()
        self.critic = Critic()
        self.execute_tools = search_tools_callback
        self.max_iterations = budget_config.max_iterations
        self.investigator_client, self.inv_cfg = create_client_for("analyst")
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkerState)
        builder.add_node("planner", self._node_planner)
        builder.add_node("executor", self._node_executor)
        builder.add_node("investigator", self._node_investigator)
        builder.add_node("critic", self._node_critic)
        builder.add_node("reporter", self._node_reporter)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "investigator")
        builder.add_edge("investigator", "critic")
        builder.add_conditional_edges("critic", self._route_critic, {
            "continue": "planner",
            "done": "reporter"
        })
        builder.add_edge("reporter", END)
        return builder.compile()

    def _node_planner(self, state: WorkerState) -> Dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        logger.info(f"[Worker {state['task_id'][:4]}] Iteration {iteration}")
        
        current_evidence = state.get("local_evidence", [])
        rejected_ids = state.get("rejected_ids", [])
        if rejected_ids:
            current_evidence = [c for c in current_evidence if c.id not in rejected_ids and not any(c.id.startswith(rid) for rid in rejected_ids)]

        local_summary = self._summarize_evidence(current_evidence, state.get("local_findings", ""))
        
        context_parts = []
        if state.get("project_tree"):
            context_parts.append(f"--- Global Project Directory Tree ---\n{state['project_tree']}")
        if state.get("starting_knowledge") and state["starting_knowledge"] != "No evidence collected yet.":
            context_parts.append(f"--- Shared Knowledge (From Peers) ---\n{state['starting_knowledge']}")
            
        context_parts.append(f"--- Your Current Progress ---\n{state.get('local_findings', 'Beginning investigation...')}")
        context_parts.append(f"--- Your Evidence Assets ---\n{local_summary}")
        
        full_evidence_context = "\n\n".join(context_parts)

        tool_messages = state.get("tool_messages", [])

        out = self.planner.plan(
            query=state["query"],
            evidence_summary=full_evidence_context,
            search_history=state.get("search_history", []),
            iteration=iteration,
            previous_missing=state.get("missing_evidence", []),
            sub_tasks=[state["task_description"]],
            critic_feedback=state.get("critic_feedback", ""),
            tool_messages=tool_messages
        )
        
        if getattr(out, "tool_results", None):
            for res in out.tool_results:
                if res.get("type") == "assistant_msg":
                    tool_messages.append(res["message"])
        
        return {
            "iteration": iteration,
            "planner_tool_calls": out.tool_calls,
            "local_evidence": current_evidence,
            "should_stop": out.should_stop,
            "tool_messages": tool_messages
        }

    def _node_executor(self, state: WorkerState) -> Dict[str, Any]:
        tool_calls = state.get("planner_tool_calls", [])
        tool_messages = state.get("tool_messages", [])
        
        new_cards = self.execute_tools(tool_calls, state["iteration"], exclude_cids=state.get("exclude_cids"))
        
        if tool_calls:
            for tc in tool_calls:
                call_id = tc.get("id")
                if call_id:
                    snippets = [c.snippet for c in new_cards[:5]]
                    content = "\n".join(snippets) if snippets else "Tool executed, but no direct textual snippet generated or matched."
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tc["tool"],
                        "content": content[:2000]
                    })
        
        current_evidence = state.get("local_evidence", [])
        existing_ids = {c.id for c in current_evidence}
        for c in new_cards:
            if c.id not in existing_ids:
                current_evidence.append(c)
                
        return {
            "local_evidence": current_evidence,
            "current_turn_evidence": new_cards, # 【新增】儲存本輪新抓到的證據
            "planner_tool_calls": [],
            "tool_messages": tool_messages
        }

    def _node_investigator(self, state: WorkerState) -> Dict[str, Any]:
        """
        員工自省節點：強制構建階層式調用鏈路。
        【狀態分離優化】Investigator 僅分析當前輪次抓回來的新證據，避免陷入歷史垃圾的死循環。
        """
        new_evidence = state.get("current_turn_evidence", [])
        if not new_evidence:
            return {"local_findings": state.get("local_findings", "No new code read yet.")}

        # 僅過濾出本輪中帶有實際代碼內容的卡片
        valid_evidence = [c for c in new_evidence if len(c.chunk_text) > 100]
        
        # 依據搜尋/圖譜擴展的分數降冪排序，確保最核心的代碼在最前面
        valid_evidence.sort(key=lambda x: x.score_hybrid, reverse=True)
        active_evidence = valid_evidence[:10] # 【修復：從 3 提高到 10，讓 Investigator 能看到完整的拓樸鏈路】
        
        if not active_evidence:
            return {"local_findings": state.get("local_findings", "New evidence contains only paths, no content to analyze.")}

        evidence_dump = ""
        for i, c in enumerate(active_evidence):
            # 限制單份檔案長度，避免 10 份檔案撐爆 Token
            evidence_dump += f"\n[DOC ID: {c.id[:8]} | PATH: {c.path}]\n{c.chunk_text[:1500]}\n"

        prompt = f"""
        You are a Senior Engineer building a cumulative INVESTIGATION LOG.
        TASK: {state['task_description']}
        
        --- PREVIOUS LOG (Your Long-Term Memory) ---
        {state.get('local_findings', 'Empty.')}

        --- NEW DOCUMENTS TO ANALYZE (Your Short-Term Focus) ---
        {evidence_dump}

        MISSION:
        Update your log with FACTS found in the NEW documents.
        1. **Precision**: Use "Hard Proof" (direct code quotes) for valid facts.
        2. **Hierarchical Flow**: You MUST document the execution flow using arrows and indentation to show the call chain.
           Example:
           [Endpoint] MemberDepositV2Controller.createDeposit()
             └── [Validates] MemberCreateDepositRequest (DTO)
                 └── [Constraint] @BrandCode String merchantCode
             └── [Calls] MemberDepositV2Service.process()
        3. **Efficiency**: If a document is IRRELEVANT, simply record: "[REJECTED] {{DOC_ID}} - Reason". Do NOT summarize junk.
        4. **Evolution**: Correct old notes if the new code provides better clarity.

        FORMAT:
        - **Execution Call Chain**: (Hierarchical flow as shown above)
        - **Verified Facts**: (Additional facts + Code Quote)
        - **Rejected Paths**: (ID + Short Reason)
        - **Next Action**: (Specific target)
        """
        
        try:
            response = self.investigator_client.chat.completions.create(
                model=self.inv_cfg.model,
                messages=[{"role": "system", "content": "You are a surgical investigator. Build hierarchical call chains."}, {"role": "user", "content": prompt}],
                temperature=0.1
            )
            findings = response.choices[0].message.content
            logger.info(f"[Worker {state['task_id'][:4]}] Updated Log:\n{findings}")
            return {"local_findings": findings}
        except Exception as e:
            logger.error(f"Investigator failed: {e}")
            return {"local_findings": state.get("local_findings", "")}

    def _node_critic(self, state: WorkerState) -> Dict[str, Any]:
        structured_evidence = []
        for c in state.get("local_evidence", []):
            structured_evidence.append({
                "id": c.id, "path": c.path, "community_id": c.community_id,
                "symbol": c.symbol, "snippet": c.chunk_text[:300]
            })
        
        out = self.critic.evaluate(
            query=state["query"],
            structured_evidence=structured_evidence,
            findings=state.get("local_findings", ""),
            intent=state["task_description"],
            sub_tasks=[state["task_description"]]
        )
        
        from ..models import MissingEvidence
        missing = [MissingEvidence(need=m["need"], accept=m.get("accept", [])) for m in out.missing_elements]
        
        return {
            "should_stop": out.is_complete,
            "critic_feedback": out.critique,
            "missing_evidence": missing,
            "rejected_ids": out.rejected_ids
        }

    def _route_critic(self, state: WorkerState) -> str:
        if state.get("should_stop", False) or state["iteration"] >= self.max_iterations:
            return "done"
        return "continue"

    def _node_reporter(self, state: WorkerState) -> Dict[str, Any]:
        findings = state.get("local_findings", "No investigation findings.")
        report = f"### SubTask Report: {state['task_description']}\n"
        report += findings
        return {"final_report": report}
        
    def _summarize_evidence(self, cards: List[EvidenceCard], current_findings: str) -> str:
        if not cards: return "None."
        lines = []
        for c in cards:
            cid_short = c.id[:8]
            has_body = len(c.chunk_text) > 100
            if not has_body:
                status = "MISSING (Need to call read_exact_file)"
            elif cid_short in current_findings:
                status = "ANALYZED (Findings in log below)"
            else:
                status = "READ_BUT_NOT_YET_ANALYZED (Worker must summarize this)"
            
            lines.append(f"- ID: {cid_short} | {c.path} | {status}")
        return "\n".join(lines)

    def run(self, task_id: str, description: str, query: str, starting_knowledge: str = "", exclude_cids: List[int] = None, project_tree: str = "") -> Dict[str, Any]:
        initial_state = {
            "task_id": task_id,
            "task_description": description,
            "domain_constraint": "",
            "query": query,
            "iteration": 0,
            "search_history": [],
            "tool_messages": [],
            "planner_tool_calls": [],
            "tool_results": [],
            "local_evidence": [],
            "current_turn_evidence": [], # 【新增】
            "starting_knowledge": starting_knowledge,
            "project_tree": project_tree,
            "exclude_cids": exclude_cids or [],
            "local_findings": "",
            "missing_evidence": [],
            "critic_feedback": "",
            "rejected_ids": [],
            "should_stop": False,
            "final_report": ""
        }
        final_state = self.graph.invoke(initial_state)
        return final_state
