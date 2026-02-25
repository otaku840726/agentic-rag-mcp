"""
Data models for Agentic RAG
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class SourceKind(str, Enum):
    CODE = "code"
    DOC = "doc"
    CONFIG = "config"
    SQL = "sql"
    JIRA = "jira"


class QueryOperator(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    EXACT = "exact"
    SYMBOL_REF = "symbol_ref"
    CALLSITE = "callsite"


class QueryType(str, Enum):
    KEYWORD = "keyword"
    SYMBOL = "symbol"
    CONFIG = "config"
    SEMANTIC = "semantic"


# ========== Evidence Card ==========
@dataclass
class EvidenceCard:
    """壓縮後的證據卡片"""
    id: str                                    # chunk id
    path: str                                  # 文件路徑
    symbol: Optional[str]                      # class/method/SP name
    snippet: str                               # 精簡片段 (用於顯示, < 200 chars)
    chunk_text: str                            # 完整內容 (用於 fingerprint)
    score_hybrid: float                        # hybrid search 分數
    score_rerank: float                        # rerank 分數
    tags: List[str]                            # [timeout, state-machine, robot...]
    round_found: int                           # 第幾輪找到的
    source_kind: str                           # "code" | "doc" | "config" | "sql" | "jira"
    span: str                                  # "L120-L170" 或 chunk offset
    fingerprint: str                           # 內容 hash (dedupe 用)
    named_entities: Dict[str, List[str]] = field(default_factory=dict)
    # {"config_keys": [], "enums": [], "constants": []}


# ========== Query Intent ==========
@dataclass
class QueryIntent:
    """單個查詢意圖"""
    query: str                                 # 查詢內容
    purpose: str                               # 目的 (入口/狀態機/config...)
    query_type: str                            # keyword | symbol | config | semantic
    operator: str                              # semantic | keyword | exact | symbol_ref | callsite
    filters: Optional[Dict[str, Any]] = None   # path/module/category 過濾


# ========== Missing Evidence ==========
@dataclass
class MissingEvidence:
    """結構化的缺失證據"""
    need: str                                  # "timeout config source"
    accept: List[str]                          # ["config key", "default value", "where read"]
    priority: str                              # "high" | "medium" | "low"


# ========== Analyst Output ==========
@dataclass
class AnalystOutput:
    """Analyst 的結構化輸出"""
    subject: str                               # 核心主題
    actors: List[Dict[str, str]]               # [{"name": "...", "role": "..."}]
    covered: List[str]                         # 已覆蓋的維度
    gaps: List[str]                            # 未覆蓋的維度
    reasoning: str = ""                        # 推理過程


# ========== Planner Output ==========
@dataclass
class PlannerOutput:
    """Planner 的結構化輸出"""
    next_queries: List[QueryIntent]            # 下一輪要查什麼
    missing_evidence: List[MissingEvidence]    # 還缺哪些關鍵證據
    evidence_found: List[str]                  # 這輪找到哪些 (evidence card ids)
    rationale: str                             # 決策理由 (簡短)
    should_stop: bool = False                  # Planner 認為是否應該停止
    symmetry_analysis: str = ""                # 對稱性思考結果（保留向後兼容）


# ========== Evidence Reference ==========
@dataclass
class EvidenceRef:
    """證據引用"""
    card_id: str
    path: str
    span: str
    quote: str                                 # 120-200 chars
    needs_expand: bool = False
    expand_reasons: List[str] = field(default_factory=list)


# ========== Flow Step ==========
@dataclass
class FlowStep:
    """步驟鏈中的單步"""
    step: int
    description: str
    code_ref: str                              # card_id + path + span


# ========== Decision Point ==========
@dataclass
class DecisionPoint:
    """分支條件"""
    condition: str
    true_branch: str
    false_branch: str
    code_ref: str


# ========== Config Item ==========
@dataclass
class ConfigItem:
    """配置項"""
    key: str
    default_value: Optional[str]
    source: str                                # path where defined
    description: str


# ========== Synthesized Response ==========
@dataclass
class SynthesizedResponse:
    """最終整合的回應"""
    answer: str                                # 結論
    flow: List[FlowStep]                       # 步驟鏈 A → B → C
    decision_points: List[DecisionPoint]       # 分支條件
    config: List[ConfigItem]                   # 配置來源與預設值
    evidence: List[EvidenceRef]                # 證據引用
    search_history: List[str]                  # 搜索歷史
    iterations: int                            # 迭代次數
    total_evidence_found: int                  # 總共找到的證據數


# ========== Search State ==========
@dataclass
class SearchState:
    """搜索狀態"""
    query: str                                 # 原始查詢
    iteration: int = 0                         # 當前迭代次數
    total_tokens: int = 0                      # 已消耗 token
    consecutive_no_new: int = 0                # 連續無新發現次數
    search_history: List[str] = field(default_factory=list)
    missing_evidence: List[MissingEvidence] = field(default_factory=list)
    fallback_triggered: bool = False           # 是否已觸發 fallback


# ========== Quality Gate ==========
@dataclass
class QualityGate:
    """質量門檻"""
    min_code_evidence: int = 2
    min_tag_diversity: int = 2
    require_call_edge: bool = True
    require_named_entity: bool = True


# ========== Budget ==========
@dataclass
class Budget:
    """預算配置"""
    max_iterations: int = 5
    max_tokens_per_round: int = 2000
    total_token_budget: int = 15000
    max_evidence_cards: int = 200
    working_set_size: int = 20


# ========== Search Result (MCP Response) ==========
@dataclass
class SearchResult:
    """MCP 搜索結果"""
    success: bool
    response: Optional[SynthesizedResponse] = None
    error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None
