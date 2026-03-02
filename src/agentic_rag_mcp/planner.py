"""
Planner LLM - 生成下一輪查詢的短輸出結構化模型
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .provider import create_client_for
from .models import (
    PlannerOutput, QueryIntent, MissingEvidence,
    EvidenceCard, GraphState, AnalystOutput
)
from .utils import extract_json_from_response, strip_think_tags


PLANNER_SYSTEM_PROMPT = """你是代碼庫搜索 Planner。
你的任務是根據「黑板」(GraphState) 上的子任務 (sub_tasks)，決定要調用哪些工具。

**職責:**
1. 觀察「待辦清單 (sub_tasks)」與當前證據。
2. 使用提供的工具（Tool Calling API）來收集所需資訊（如 `semantic_search`, `graph_symbol_search`, `read_exact_file` 等）。
3. **強制要求**: 你必須在每一次回覆中，呼叫 `report_status` 工具，回報目前的決策理由 (rationale)、缺失的證據 (missing_evidence)，以及是否所有任務已完成且證據充足 (should_stop)。

請直接調用工具，不要產生多餘的文字說明。
"""

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "混合搜索，適合尋找模糊概念、邏輯。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的內容或概念"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_symbol_search",
            "description": "查找 AST 結構圖，精準尋找方法的調用者或實作。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "類名或方法名"},
                    "depth": {"type": "integer", "description": "探索深度", "default": 1}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_exact_file",
            "description": "閱讀特定檔案的完整內容或特定行數。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "檔案路徑"},
                    "lines": {"type": "string", "description": "行數範圍，例如 '10-50'，留空代表讀取全檔"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "探索資料夾結構，列出內容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "資料夾路徑"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_status",
            "description": "回報目前的計畫狀態與推論結果。這是一個強制的工具，每次回覆都必須調用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "should_stop": {"type": "boolean", "description": "是否所有任務都完成了且證據足夠"},
                    "rationale": {"type": "string", "description": "簡短的決策理由"},
                    "missing_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "need": {"type": "string", "description": "描述缺少什麼"},
                                "accept": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "接受條件列表"
                                },
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                            },
                            "required": ["need", "accept", "priority"]
                        },
                        "description": "列出目前仍然缺失的證據清單"
                    }
                },
                "required": ["should_stop", "rationale", "missing_evidence"]
            }
        }
    }
]


@dataclass
class PlannerConfig:
    """Planner 配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 4000  # thinking model 需要更多 tokens
    temperature: float = 0.1
    base_url: Optional[str] = None


class Planner:
    """Planner LLM"""

    def __init__(self, config: Optional[PlannerConfig] = None):
        if config:
            from .provider import create_client
            self.config = config
            self.client = create_client(config.provider)
        else:
            self.config = PlannerConfig()
            client, comp_cfg = create_client_for("planner")
            self.client = client
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model
            self.config.max_tokens = comp_cfg.max_tokens
            self.config.temperature = comp_cfg.temperature

    def plan(
        self,
        query: str,
        evidence_summary: str,
        search_history: List[str],
        iteration: int,
        previous_missing: Optional[List[MissingEvidence]] = None,
        analyst_output: Optional[AnalystOutput] = None,
        sub_tasks: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        logger: Any = None,
        search_id: str = "",
        usage_log: Optional[list] = None
    ) -> PlannerOutput:
        """
        生成下一輪計劃

        Args:
            query: 原始查詢
            evidence_summary: 當前證據摘要
            search_history: 已搜索的查詢
            iteration: 當前迭代次數
            previous_missing: 上一輪的 missing evidence
            analyst_output: Analyst 的全局分析結果
            sub_tasks: Analyst 拆解的子任務
            tool_results: 之前的工具執行結果

        Returns:
            PlannerOutput
        """
        # 構建用戶提示
        user_prompt = self._build_user_prompt(
            query, evidence_summary, search_history, iteration,
            previous_missing, analyst_output, sub_tasks, tool_results
        )

        # 調用 LLM
        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "tools": PLANNER_TOOLS,
            "tool_choice": "auto"
        }

        import time
        start_time = time.time()
        
        response = self.client.chat.completions.create(**kwargs)
        
        latency = (time.time() - start_time) * 1000

        # Log to trace if logger provided
        if logger and search_id:
            logger.log_llm_event(
                search_id=search_id,
                step=f"planner_iter_{iteration}",
                model=self.config.model,
                messages=kwargs["messages"],
                response=str(response.choices[0].message),
                latency_ms=latency
            )

        if usage_log is not None and hasattr(response, "usage") and response.usage:
            usage_log.append({
                "component": f"planner_iter_{iteration}",
                "model": self.config.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": round(latency),
            })

        return self._parse_tool_calls(response.choices[0].message)

    def _build_user_prompt(
        self,
        query: str,
        evidence_summary: str,
        search_history: List[str],
        iteration: int,
        previous_missing: Optional[List[MissingEvidence]],
        analyst_output: Optional[AnalystOutput] = None,
        sub_tasks: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """構建用戶提示"""
        parts = [
            f"**原始問題:** {query}",
            f"**當前迭代:** {iteration}",
            "",
            "**待辦清單 (sub_tasks):**",
            "\n".join([f"- {task}" for task in sub_tasks]) if sub_tasks else "目前沒有待辦事項",
            "",
            "**當前證據:**",
            evidence_summary if evidence_summary else "尚未收集到證據",
            "",
            "**工具執行結果 (這回合):**",
            str(tool_results) if tool_results else "尚無工具結果",
            "",
            "**已執行工具與搜索查詢:**",
            ", ".join(search_history) if search_history else "尚未進行搜索",
        ]

        # Analyst 全局分析結果
        if analyst_output and analyst_output.subject:
            parts.append("")
            parts.append("**Analyst 全局分析:**")
            parts.append(f"- 核心主題: {analyst_output.subject}")
            if analyst_output.actors:
                actor_strs = [f"{a['name']}({a['role']})" for a in analyst_output.actors]
                parts.append(f"- 角色: {', '.join(actor_strs)}")
            if analyst_output.covered:
                parts.append(f"- 已覆蓋維度: {'; '.join(analyst_output.covered)}")
            if analyst_output.gaps:
                parts.append(f"- **未覆蓋維度（必須生成查詢）:** {'; '.join(analyst_output.gaps)}")

        if previous_missing:
            parts.append("")
            parts.append("**上一輪識別的缺失證據:**")
            for m in previous_missing:
                parts.append(f"- {m.need} (priority: {m.priority})")
                parts.append(f"  accept: {', '.join(m.accept)}")

        parts.append("")
        parts.append("請根據 Analyst 分析和當前證據，輸出下一輪計劃（JSON 格式）。")

        return "\n".join(parts)

    def _parse_tool_calls(self, message: Any) -> PlannerOutput:
        """Parse LLM tool calls response"""
        out = PlannerOutput(
            next_queries=[],
            tool_calls=[],
            missing_evidence=[],
            evidence_found=[],
            rationale="No rationale provided.",
            should_stop=False
        )

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            # If the model didn't call any tools, fallback to content
            out.rationale = "Model generated content instead of tool calls."
            return out

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
            except Exception:
                continue

            name = call.function.name

            if name == "report_status":
                out.should_stop = args.get("should_stop", False)
                out.rationale = args.get("rationale", "")

                for m in args.get("missing_evidence", []):
                    out.missing_evidence.append(MissingEvidence(
                        need=m.get("need", ""),
                        accept=m.get("accept", []),
                        priority=m.get("priority", "medium")
                    ))
            elif name in ["semantic_search", "graph_symbol_search", "read_exact_file", "list_directory"]:
                out.tool_calls.append({"tool": name, "args": args})

        # Backward compatibility for existing code that checks `next_queries` inside the pipeline
        for tc in out.tool_calls:
            if tc["tool"] == "semantic_search":
                q = tc["args"].get("query", "")
                if q:
                    out.next_queries.append(QueryIntent(
                        query=q, purpose="Tool call wrapper", query_type="semantic", operator="hybrid"
                    ))

        return out

    def get_token_count(self, text: str) -> int:
        """估算 token 數量"""
        # 粗略估算: 1 token ≈ 4 字符
        return len(text) // 4


class LLMJudge:
    """
    LLM Judge - 用於 is_satisfied 的不確定情況
    極短輸出，只回答 true/false + 1句理由
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig(max_tokens=100)
        client, comp_cfg = create_client_for("judge")
        self.client = client
        if not config:
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model
            self.config.max_tokens = comp_cfg.max_tokens
            self.config.temperature = comp_cfg.temperature

    def judge(
        self,
        need: str,
        accept: List[str],
        covered: List[str],
        candidates_summary: str
    ) -> Dict[str, Any]:
        """
        判斷 missing evidence 是否被滿足

        Args:
            need: 需要什麼
            accept: 接受條件
            covered: 已覆蓋的條件
            candidates_summary: 候選卡片摘要

        Returns:
            {"satisfied": bool, "reason": str}
        """
        prompt = f"""判斷以下證據需求是否被滿足。

需求: {need}
接受條件: {', '.join(accept)}
已覆蓋: {', '.join(covered)}
候選證據:
{candidates_summary}

只回答 JSON: {{"satisfied": true/false, "reason": "一句話理由"}}"""

        kwargs = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
        }

        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        try:
            content = response.choices[0].message.content
            return extract_json_from_response(content)
        except Exception:
            return {"satisfied": False, "reason": "Failed to parse judge response"}
