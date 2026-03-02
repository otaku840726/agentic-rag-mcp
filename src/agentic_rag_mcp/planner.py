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

**可用工具:**
1. semantic_search:
   - 用法: `{"tool": "semantic_search", "args": {"query": "模糊概念"}}`
   - 說明: 混合搜索，適合尋找模糊概念、邏輯。
2. graph_symbol_search:
   - 用法: `{"tool": "graph_symbol_search", "args": {"symbol": "類名/方法名", "depth": 1}}`
   - 說明: 查找 AST 結構圖，精準尋找方法的調用者或實作。
3. read_exact_file:
   - 用法: `{"tool": "read_exact_file", "args": {"path": "檔案路徑", "lines": "10-50"}}`
   - 說明: 閱讀特定檔案的完整內容或特定行數。
4. list_directory:
   - 用法: `{"tool": "list_directory", "args": {"path": "資料夾路徑"}}`
   - 說明: 探索資料夾結構。

**職責:**
1. 觀察「待辦清單 (sub_tasks)」。
2. 決定這一步要調用哪個（或哪些）工具。
3. 判斷是否所有任務都完成了且證據足夠 (should_stop)。如果足夠，可以將 `should_stop` 設為 true。

**只輸出 JSON，不要任何解釋。**

**輸出格式:**
{
    "should_stop": false,
    "tool_calls": [
        {
            "tool": "semantic_search",
            "args": {"query": "密碼驗證邏輯"}
        }
    ],
    "missing_evidence": [
        {
            "need": "描述缺少什麼",
            "accept": ["接受條件1", "接受條件2"],
            "priority": "high|medium|low"
        }
    ],
    "rationale": "簡短的決策理由"
}
"""


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
        }

        # 非 local provider 才使用 response_format
        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        import time
        start_time = time.time()
        
        response = self.client.chat.completions.create(**kwargs)
        
        latency = (time.time() - start_time) * 1000

        # 解析回應
        content = response.choices[0].message.content

        if usage_log is not None and hasattr(response, "usage") and response.usage:
            usage_log.append({
                "component": f"planner_iter_{iteration}",
                "model": self.config.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": round(latency),
            })

        # Log to trace if logger provided
        if logger and search_id:
            logger.log_llm_event(
                search_id=search_id,
                step=f"planner_iter_{iteration}",
                model=self.config.model,
                messages=kwargs["messages"],
                response=content,
                latency_ms=latency
            )

        return self._parse_response(content)

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

    def _parse_response(self, content: str) -> PlannerOutput:
        """解析 LLM 回應"""
        try:
            data = extract_json_from_response(content)

            # 解析 missing_evidence
            missing_evidence = []
            for m in data.get("missing_evidence", []):
                missing_evidence.append(MissingEvidence(
                    need=m.get("need", ""),
                    accept=m.get("accept", []),
                    priority=m.get("priority", "medium")
                ))

            # 解析 next_queries (向後兼容，如果原本生成 query 的話)
            next_queries = []
            for q in data.get("next_queries", []):
                next_queries.append(QueryIntent(
                    query=q.get("query", ""),
                    purpose=q.get("purpose", ""),
                    query_type=q.get("query_type", "semantic"),
                    operator=q.get("operator", "semantic"),
                    filters=q.get("filters")
                ))

            # 解析 tool_calls
            tool_calls = data.get("tool_calls", [])

            return PlannerOutput(
                next_queries=next_queries,
                tool_calls=tool_calls,
                missing_evidence=missing_evidence,
                evidence_found=data.get("evidence_found", []),
                rationale=data.get("rationale", ""),
                should_stop=data.get("should_stop", False),
                symmetry_analysis=data.get("symmetry_analysis", "")
            )

        except json.JSONDecodeError as e:
            # 解析失敗時返回默認輸出
            return PlannerOutput(
                next_queries=[],
                tool_calls=[],
                missing_evidence=[],
                evidence_found=[],
                rationale=f"Failed to parse LLM response: {e}",
                should_stop=True
            )

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
