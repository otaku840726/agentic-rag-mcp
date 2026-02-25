
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class SearchTraceLogger:
    """
    Logs detailed trace of agentic search steps to a JSONL file.
    """
    def __init__(self, log_dir: str = ".agentic-rag-cache"):
        self.log_dir = Path(os.getcwd()) / log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "search_trace.jsonl"
        
    def _log(self, search_id: str, event: str, data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "search_id": search_id,
            "event": event,
            "data": data
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write search trace: {e}")

    def log_start(self, search_id: str, query: str, config: Dict[str, Any]):
        self._log(search_id, "start", {
            "query": query,
            "config": config
        })

    def log_iteration(self, search_id: str, iteration: int, plan: Dict[str, Any], results_summary: Dict[str, Any]):
        self._log(search_id, "iteration", {
            "iteration": iteration,
            "plan": plan,
            "results": results_summary
        })

    def log_end(self, search_id: str, success: bool, output: Any, error: Optional[str] = None):
        self._log(search_id, "end", {
            "success": success,
            "output": output,
            "error": error
        })

    def log_llm_event(self, search_id: str, step: str, model: str, messages: List[Dict[str, str]], response: str, latency_ms: float = 0):
        self._log(search_id, "llm_call", {
            "step": step,  # e.g., "planner", "synthesizer"
            "model": model,
            "messages": messages,
            "response": response,
            "latency_ms": latency_ms
        })
