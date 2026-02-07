"""
Provider — 統一的 LLM / Embedding client 工廠
讀取 config.yaml 的 providers 區塊，按 provider 名稱建立 OpenAI-compatible client
"""

import os
import re
import yaml
from importlib.resources import files
from typing import Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI


def _resolve_env(value: str) -> str:
    """解析 ${ENV_VAR} 或 ${ENV_VAR:-default} 佔位符"""
    if not isinstance(value, str):
        return value

    def _replace(m):
        expr = m.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            return os.getenv(var, default)
        return os.getenv(expr, "")

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _resolve_dict(d: Dict) -> Dict:
    """遞迴解析 dict 中所有 ${ENV} 佔位符"""
    resolved = {}
    for k, v in d.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_dict(v)
        elif isinstance(v, str):
            resolved[k] = _resolve_env(v)
        else:
            resolved[k] = v
    return resolved


@dataclass
class ComponentConfig:
    """單個 component (embedding/planner/synthesizer/judge) 的配置"""
    provider: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.1


# ---------------------------------------------------------------------------
# Singleton config cache
# ---------------------------------------------------------------------------
_config_cache: Optional[Dict] = None


def load_config() -> Dict[str, Any]:
    """讀取打包在 package 內的 config.yaml（解析所有 ${ENV} 佔位符）"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    try:
        config_text = files("agentic_rag_mcp").joinpath("config.yaml").read_text(encoding="utf-8")
        raw = yaml.safe_load(config_text) or {}
        _config_cache = _resolve_dict(raw)
    except Exception:
        _config_cache = {}
    return _config_cache


def get_component_config(component: str) -> ComponentConfig:
    """取得某個 component 的配置

    Args:
        component: "embedding" | "planner" | "synthesizer" | "judge"
    """
    cfg = load_config()
    comp = cfg.get(component, {})
    return ComponentConfig(
        provider=comp.get("provider", "openai"),
        model=comp.get("model", "gpt-4o-mini"),
        max_tokens=comp.get("max_tokens", 4000),
        temperature=comp.get("temperature", 0.1),
    )


def create_client(provider: str) -> OpenAI:
    """根據 provider 名稱建立 OpenAI-compatible client

    Args:
        provider: "openai" | "local" | "gemini" | ...

    providers 區塊範例:
        providers:
          openai:
            api_key: ${OPENAI_API_KEY}
          local:
            base_url: http://127.0.0.1:1234/v1
            api_key: "not-needed"
          gemini:
            base_url: https://generativelanguage.googleapis.com/v1beta/openai/
            api_key: ${GEMINI_API_KEY}
    """
    cfg = load_config()
    providers = cfg.get("providers", {})
    provider_cfg = providers.get(provider, {})

    api_key = provider_cfg.get("api_key", "")
    base_url = provider_cfg.get("base_url")

    # Fallback: 如果 config 裡沒有，嘗試常見 env vars
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
        elif provider == "local":
            api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def create_client_for(component: str) -> tuple[OpenAI, ComponentConfig]:
    """一步取得 component 的 client + config

    Returns:
        (client, component_config)
    """
    comp_cfg = get_component_config(component)
    client = create_client(comp_cfg.provider)
    return client, comp_cfg
