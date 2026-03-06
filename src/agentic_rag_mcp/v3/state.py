from typing import TypedDict, Annotated, List, Dict, Any
import operator

class AgenticState(TypedDict):
    """
    V3 雙層架構的全局共享狀態。
    在 Manager 與 Worker 之間傳遞，消除了過往的資訊斷層。
    """
    query: str                       # 使用者原始問題
    project_context: str             # 專案地圖 (PROJECT_CONTEXT.md)
    
    # --- Manager 的狀態 ---
    manager_thoughts: Annotated[List[str], operator.add]     # Manager 的戰略思考筆記 (供自己參考)
    investigation_log: Annotated[List[str], operator.add]    # 所有 Worker 歷次提交的正式報告
    
    # --- 交接區 (Handoff Variables) ---
    current_task: str                # 當前進行中的工單指令
    target_cids: List[str]           # 當前工單限定的社群 ID (可為空)
    
    # --- Worker 的狀態 ---
    current_worker_messages: List[Dict[str, Any]] # Worker 執行工具的暫存對話 (每次 Worker 啟動前清空)
    
    # --- 終局狀態 ---
    final_answer: str                # 最終答案
    is_finished: bool                # 是否已結案