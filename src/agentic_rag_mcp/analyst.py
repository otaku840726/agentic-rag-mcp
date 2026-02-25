"""
Analyst LLM - 在 Planner 之前執行，負責「退後一步」看全局
職責：識別核心主題、角色、已覆蓋/未覆蓋的維度
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .provider import create_client_for, load_config
from .models import AnalystOutput
from .utils import extract_json_from_response


ANALYST_SYSTEM_PROMPT = """你是一個抽象分析者。
你的任務是對一組證據做「退後一步」的分析，不關心代碼細節，只關心結構性的全局視角。

**你要回答三個問題：**

1. **主題是什麼？**
   這些證據描述的動作，最終作用在什麼東西上？
   剝離所有傳遞方式、機制、流程名稱，只留下那個被作用的東西本身。
   它應該是一個任何人不需要看代碼也能理解的概念。

   **自檢：你寫出來的主題中，是否還包含「怎麼傳遞」或「怎麼操作」的部分？**
   如果是，繼續往下剝，直到只剩下「被傳遞的東西」或「被操作的對象」本身。
   例如：「存款通知」→ 「通知」是傳遞方式 → 剝掉 → 「存款信息」。
   例如：「訂單狀態變更事件」→ 「事件」是傳遞方式 → 剝掉 → 「訂單狀態」。

2. **涉及哪些角色？**
   誰生產它？誰消費它？誰中轉它？
   「角色」可以是人、系統、服務、模組 —— 任何獨立的參與者。
   **特別注意**：不要只列出證據中出現的角色。思考誰是這個主題的最終消費者（end consumer），
   即使證據中沒有直接提到他。

3. **已覆蓋了哪些維度？還缺哪些？**
   站到每一個角色的立場，問：他跟這個主題的所有互動方式，證據中覆蓋了幾種？
   如果某個角色有多種互動途徑，但證據只展示了其中一種，那就是缺失的維度。

   **特別注意最終消費者**：他獲取這個主題的所有途徑是什麼？
   證據可能只展示了其中一條途徑，其他途徑就是缺失的維度。

**只輸出 JSON，不要任何解釋。**

{
    "subject": "核心主題（一個概念，不是類名，不包含傳遞方式）",
    "actors": [
        {"name": "角色名稱", "role": "這個角色與主題的關係（生產者/消費者/中轉者等）"}
    ],
    "covered": ["已覆蓋的維度描述"],
    "gaps": ["未覆蓋的維度描述"],
    "reasoning": "1-2句推理過程"
}
"""


@dataclass
class AnalystConfig:
    """Analyst 配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.1
    base_url: Optional[str] = None


class Analyst:
    """Analyst LLM - 全局視角分析"""

    def __init__(self, config: Optional[AnalystConfig] = None):
        if config:
            from .provider import create_client
            self.config = config
            self.client = create_client(config.provider)
        else:
            self.config = AnalystConfig()
            client, comp_cfg = create_client_for("analyst")
            self.client = client
            self.config.provider = comp_cfg.provider
            self.config.model = comp_cfg.model
            self.config.max_tokens = comp_cfg.max_tokens
            self.config.temperature = comp_cfg.temperature

    def analyze(
        self,
        query: str,
        evidence_summary: str,
        iteration: int,
        logger: Any = None,
        search_id: str = ""
    ) -> AnalystOutput:
        """
        對當前證據做全局分析

        Args:
            query: 原始查詢
            evidence_summary: 當前證據摘要
            iteration: 當前迭代次數

        Returns:
            AnalystOutput
        """
        user_prompt = f"""**原始問題:** {query}
**當前迭代:** {iteration}

**當前證據:**
{evidence_summary if evidence_summary else "尚未收集到證據"}

請分析並輸出 JSON。"""

        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if self.config.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            start_time = time.time()
            response = self.client.chat.completions.create(**kwargs)
            latency = (time.time() - start_time) * 1000

            content = response.choices[0].message.content

            # Log
            if logger and search_id:
                logger.log_llm_event(
                    search_id=search_id,
                    step=f"analyst_iter_{iteration}" + (f"_retry{attempt}" if attempt > 0 else ""),
                    model=self.config.model,
                    messages=kwargs["messages"],
                    response=content,
                    latency_ms=latency
                )

            result = self._parse_response(content)
            if result.subject:  # 解析成功
                return result

            last_error = result.reasoning
            time.sleep(0.5)  # 短暫等待後重試

        # 所有重試都失敗
        return AnalystOutput(
            subject="",
            actors=[],
            covered=[],
            gaps=[],
            reasoning=last_error or "All retries failed"
        )

    def _parse_response(self, content: str) -> AnalystOutput:
        """解析 LLM 回應"""
        try:
            data = extract_json_from_response(content)

            actors = []
            for a in data.get("actors", []):
                if isinstance(a, dict):
                    actors.append({"name": a.get("name", ""), "role": a.get("role", "")})
                elif isinstance(a, str):
                    actors.append({"name": a, "role": ""})

            return AnalystOutput(
                subject=data.get("subject", ""),
                actors=actors,
                covered=data.get("covered", []),
                gaps=data.get("gaps", []),
                reasoning=data.get("reasoning", "")
            )

        except (json.JSONDecodeError, Exception) as e:
            return AnalystOutput(
                subject="",
                actors=[],
                covered=[],
                gaps=[],
                reasoning=f"Failed to parse analyst response: {e}"
            )

    def format_for_planner(self, output: AnalystOutput) -> str:
        """將 Analyst 結果格式化為 Planner 可讀的文字"""
        if not output.subject:
            return ""

        parts = [f"[Analyst] 核心主題: {output.subject}"]

        if output.actors:
            actor_strs = [f"{a['name']}({a['role']})" for a in output.actors]
            parts.append(f"[Analyst] 角色: {', '.join(actor_strs)}")

        if output.covered:
            parts.append(f"[Analyst] 已覆蓋: {'; '.join(output.covered)}")

        if output.gaps:
            parts.append(f"[Analyst] 未覆蓋: {'; '.join(output.gaps)}")

        return "\n".join(parts)


# ========== Ensemble Analyst ==========

ANALYST_PROMPT_DATA_FLOW = """你是一個數據流分析者（Data Flow Analyst）。
你的視角聚焦在 **數據的對稱性**：誰推送、誰拉取、誰生產、誰消費。

**你要回答三個問題：**

1. **主題是什麼？**
   這些證據描述的動作，最終作用在什麼東西上？
   剝離所有傳遞方式、機制、流程名稱，只留下那個被作用的東西本身。

2. **涉及哪些角色？**
   特別關注：數據的 **推送端(push)** 與 **拉取端(pull)** 是否都被覆蓋？
   如果有 Callback 推送，是否也有 API 查詢？如果有寫入，是否有讀取/展示？

3. **已覆蓋了哪些維度？還缺哪些？**
   **對稱性檢查**：
   - 推送 vs 查詢（如 Callback 推送 vs 商戶查詢 API）
   - 生產 vs 消費（如 Message Produce vs Message Consume）
   - 寫入 vs 讀取（如 DB 寫入 vs API/UI 展示）

**只輸出 JSON，不要任何解釋。**

{
    "subject": "核心主題",
    "actors": [{"name": "角色名稱", "role": "push/pull/produce/consume 等"}],
    "covered": ["已覆蓋的數據流維度"],
    "gaps": ["缺失的對稱數據流維度"],
    "reasoning": "1-2句推理過程"
}
"""

ANALYST_PROMPT_OPERATIONS = """你是一個運營缺口分析者（Operations Analyst）。
你的視角聚焦在 **運營完整性**：監控、手動介入、異常處理、日誌追踪。

**你要回答三個問題：**

1. **主題是什麼？**
   這些證據描述的動作，最終作用在什麼東西上？
   剝離所有傳遞方式、機制、流程名稱，只留下那個被作用的東西本身。

2. **涉及哪些角色？**
   特別關注：**運營人員**、**監控系統**、**告警機制** 是否被考慮在內？
   誰需要在異常時介入？誰需要查看狀態？

3. **已覆蓋了哪些維度？還缺哪些？**
   **運營檢查**：
   - 正常路徑 vs 異常路徑
   - 自動處理 vs 手動介入
   - 日誌/監控/告警是否覆蓋關鍵節點
   - 重試機制、超時處理、降級策略

**只輸出 JSON，不要任何解釋。**

{
    "subject": "核心主題",
    "actors": [{"name": "角色名稱", "role": "運營角色描述"}],
    "covered": ["已覆蓋的運營維度"],
    "gaps": ["缺失的運營維度"],
    "reasoning": "1-2句推理過程"
}
"""

ANALYST_PROMPT_CODE_QUALITY = """你是一個代碼質量分析者（Code Quality Analyst）。
你的視角聚焦在 **代碼健壯性**：類型安全、錯誤處理、邊界條件、一致性。

**你要回答三個問題：**

1. **主題是什麼？**
   這些證據描述的動作，最終作用在什麼東西上？
   剝離所有傳遞方式、機制、流程名稱，只留下那個被作用的東西本身。

2. **涉及哪些角色？**
   特別關注：不同服務間的 **數據契約(contract)** 是否一致？
   DTO 的欄位是否在所有使用處保持同步？

3. **已覆蓋了哪些維度？還缺哪些？**
   **質量檢查**：
   - Entity/DTO 欄位在不同層的映射是否完整
   - Null/空值處理
   - 跨服務數據一致性
   - 向後兼容性（新欄位是否 nullable）

**只輸出 JSON，不要任何解釋。**

{
    "subject": "核心主題",
    "actors": [{"name": "角色名稱", "role": "代碼層次角色"}],
    "covered": ["已覆蓋的質量維度"],
    "gaps": ["缺失的質量維度"],
    "reasoning": "1-2句推理過程"
}
"""

ENSEMBLE_PERSONAS = [
    ("data_flow", ANALYST_PROMPT_DATA_FLOW),
    ("operations", ANALYST_PROMPT_OPERATIONS),
    ("code_quality", ANALYST_PROMPT_CODE_QUALITY),
]

logger = logging.getLogger(__name__)


class EnsembleAnalyst:
    """Ensemble Analyst - 並行跑多個視角的 Analyst，合併結果"""

    def __init__(self, config: Optional[AnalystConfig] = None):
        self._base_config = config
        # Check ensemble enabled in config.yaml
        cfg = load_config()
        ensemble_cfg = cfg.get("ensemble", {})
        self._enabled = ensemble_cfg.get("enabled", True)
        # Build per-persona Analyst instances (share same model config)
        self._analysts: List[tuple[str, str, Analyst]] = []
        for name, prompt in ENSEMBLE_PERSONAS:
            analyst = Analyst(config)
            self._analysts.append((name, prompt, analyst))
        # Fallback single analyst
        self._fallback = Analyst(config)

    def analyze(
        self,
        query: str,
        evidence_summary: str,
        iteration: int,
        logger: Any = None,
        search_id: str = ""
    ) -> AnalystOutput:
        """並行跑 3 個 persona，合併結果"""
        if not self._enabled:
            return self._fallback.analyze(
                query=query,
                evidence_summary=evidence_summary,
                iteration=iteration,
                logger=logger,
                search_id=search_id
            )

        outputs: List[tuple[str, AnalystOutput]] = []

        def _run_persona(name: str, system_prompt: str, analyst: Analyst) -> tuple[str, AnalystOutput]:
            # Override system prompt for this call
            user_prompt = f"""**原始問題:** {query}
**當前迭代:** {iteration}

**當前證據:**
{evidence_summary if evidence_summary else "尚未收集到證據"}

請分析並輸出 JSON。"""

            kwargs = {
                "model": analyst.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": analyst.config.max_tokens,
                "temperature": analyst.config.temperature,
            }
            if analyst.config.provider != "local":
                kwargs["response_format"] = {"type": "json_object"}

            start_time = time.time()
            response = analyst.client.chat.completions.create(**kwargs)
            latency = (time.time() - start_time) * 1000

            content = response.choices[0].message.content

            if logger and search_id:
                logger.log_llm_event(
                    search_id=search_id,
                    step=f"analyst_ensemble_{name}_iter_{iteration}",
                    model=analyst.config.model,
                    messages=kwargs["messages"],
                    response=content,
                    latency_ms=latency
                )

            result = analyst._parse_response(content)
            return (name, result)

        # Parallel execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_persona, name, prompt, analyst): name
                for name, prompt, analyst in self._analysts
            }
            for future in as_completed(futures):
                try:
                    outputs.append(future.result())
                except Exception as e:
                    persona_name = futures[future]
                    logging.getLogger(__name__).warning(
                        f"Ensemble persona {persona_name} failed: {e}"
                    )

        if not outputs:
            # All personas failed, fallback
            return self._fallback.analyze(
                query=query,
                evidence_summary=evidence_summary,
                iteration=iteration,
                logger=logger,
                search_id=search_id
            )

        return self._merge_outputs(outputs)

    def _merge_outputs(self, outputs: List[tuple[str, AnalystOutput]]) -> AnalystOutput:
        """合併多個 persona 的輸出: union gaps, union covered"""
        # Pick subject from the first successful output
        subject = ""
        for _, out in outputs:
            if out.subject:
                subject = out.subject
                break

        # Union actors (deduplicate by name)
        seen_actors = set()
        actors = []
        for _, out in outputs:
            for a in out.actors:
                name = a.get("name", "")
                if name and name not in seen_actors:
                    seen_actors.add(name)
                    actors.append(a)

        # Union covered (deduplicate)
        covered = list(dict.fromkeys(
            c for _, out in outputs for c in out.covered
        ))

        # Union gaps (deduplicate)
        gaps = list(dict.fromkeys(
            g for _, out in outputs for g in out.gaps
        ))

        # Combine reasoning
        reasoning_parts = []
        for name, out in outputs:
            if out.reasoning:
                reasoning_parts.append(f"[{name}] {out.reasoning}")
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else ""

        return AnalystOutput(
            subject=subject,
            actors=actors,
            covered=covered,
            gaps=gaps,
            reasoning=reasoning
        )

    def format_for_planner(self, output: AnalystOutput) -> str:
        """與 Analyst 相同的格式化方法"""
        return self._fallback.format_for_planner(output)
