"""
Data models for Agentic RAG
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, TypedDict, Annotated
import operator
from enum import Enum


class SourceKind(str, Enum):
    FILE = "file"
    GRAPH = "graph"
    DOC = "doc"


@dataclass
class EvidenceCard:
    """單條證據卡片 (Chunk / Symbol)"""
    id: str
    path: str
    symbol: Optional[str]
    snippet: str
    chunk_text: str
    score_hybrid: float
    score_rerank: float
    tags: List[str] = field(default_factory=list)
    round_found: int = 1
    source_kind: SourceKind = SourceKind.FILE
    span: str = ""
    fingerprint: str = ""
    community_id: Optional[int] = None
    named_entities: List[str] = field(default_factory=list)


@dataclass
class QueryIntent:
    """LLM 識別的搜尋意圖"""
    query: str
    purpose: str
    query_type: str
    operator: str
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissingEvidence:
    """缺失的資訊片段"""
    need: str
    accept: List[str]


@dataclass
class PlannerOutput:
    """Planner LLM 的決策輸出"""
    next_queries: List[str]
    missing_evidence: List[MissingEvidence]
    evidence_found: List[str]
    rationale: str
    should_stop: bool
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalystOutput:
    """Analyst LLM 的分解輸出"""
    subject: str
    actors: List[str]
    covered: List[str]
    gaps: List[str]
    sub_tasks: List[str]
    reasoning: str
    intent: str = "Knowledge Discovery"


# ========== 基礎組件 (必須放在 SynthesizedResponse 之前) ==========

@dataclass
class FlowStep:
    """步驟鏈中的單步"""
    step: int = 0
    description: str = ""
    code_ref: str = ""


@dataclass
class DecisionPoint:
    """分支條件"""
    condition: str = ""
    true_branch: str = ""
    false_branch: str = ""
    code_ref: str = ""


@dataclass
class ConfigItem:
    """配置項 (順序必須符合 synthesizer.py: key, default_value, source, description)"""
    key: str = ""
    default_value: str = ""
    source: str = ""
    description: str = ""
    value: str = "" # synthesizer 雖然沒傳，但為了完整性保留在最後


@dataclass
class EvidenceRef:
    """最終答案中的證據引用"""
    card_id: str = ""
    path: str = ""
    span: str = ""
    quote: str = ""
    needs_expand: bool = False
    expand_reasons: List[str] = field(default_factory=list)


@dataclass
class SynthesizedResponse:
    """
    最終產出的結構化回答
    順序必須完全符合 synthesizer.py 呼叫：
    answer, flow, decision_points, config, evidence, search_history, iterations, total_evidence_found
    """
    answer: str = ""
    flow: List[FlowStep] = field(default_factory=list)
    decision_points: List[DecisionPoint] = field(default_factory=list)
    config: List[ConfigItem] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    search_history: List[str] = field(default_factory=list)
    iterations: int = 0
    total_evidence_found: int = 0
    
    # 舊有欄位保留(設為選填)以相容其他代碼
    key_files: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)


# ========== Worker State ==========
@dataclass
class SubTask:
    """由 Analyst 派發給 Worker 的子任務"""
    id: str
    description: str
    assigned_domain: str
    status: str = "pending"
    summary: str = ""
    evidence_collected: List[EvidenceCard] = field(default_factory=list)


class WorkerState(TypedDict):
    """員工(Worker)的獨立上下文視窗"""
    task_id: str
    task_description: str
    domain_constraint: str
    query: str
    iteration: int
    search_history: Annotated[List[str], operator.add]
    tool_messages: Annotated[List[Dict[str, Any]], operator.add]
    planner_tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    local_evidence: List[EvidenceCard]
    current_turn_evidence: List[EvidenceCard]  # 【新增】本回合最新抓取的證據
    starting_knowledge: str
    project_tree: str
    exclude_cids: List[int]
    local_findings: str
    missing_evidence: List[MissingEvidence]
    critic_feedback: str
    rejected_ids: List[str]
    should_stop: bool
    final_report: str


# ========== Search State ==========
class GraphState(TypedDict):
    """LangGraph 共享狀態"""
    query: str
    module_map: str
    project_tree: str                          # 【新增】全景地圖
    intent: str
    sub_tasks: List[SubTask]
    worker_reports: Annotated[List[str], operator.add]
    search_history: Annotated[List[str], operator.add]
    rejected_ids: List[str]
    final_response: Optional[SynthesizedResponse]


# ========== Config Models ==========
@dataclass
class AgenticSearchConfig:
    """搜尋配置"""
    max_iterations: int = 5
    top_n_search: int = 20
    total_token_budget: int = 60000
    search_id: str = ""


@dataclass
class SynthesizerConfig:
    """合成配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 4000
    temperature: float = 0.1


@dataclass
class PlannerConfig:
    """規劃器配置"""
    max_iterations: int = 5
    temperature: float = 0.1


# ========== Search Result (MCP Response) ==========
@dataclass
class SearchResult:
    """MCP 搜尋結果"""
    success: bool
    response: Optional[SynthesizedResponse] = None
    error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None
