"""
Provider — 統一的 LLM / Embedding client 工廠
讀取 config.yaml 的 providers 區塊，按 provider 名稱建立 OpenAI-compatible client
"""

import os
import re
import yaml
from importlib.resources import files
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@dataclass
class ComponentConfig:
    provider: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.1
    identifier: str = "" # 【補上：索引器關鍵屬性】

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        config_path = files("agentic_rag_mcp").joinpath("config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()
            # 支援環境變數替換 ${VAR:-default}
            pattern = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')
            def replace_env(match):
                var_name = match.group(1)
                default_value = match.group(2) or ""
                return os.getenv(var_name, default_value)
            
            processed = pattern.sub(replace_env, raw)
            self._config = yaml.safe_load(processed)

    @property
    def config(self):
        return self._config

def load_config():
    return ConfigLoader().config

def get_neo4j_config() -> Dict[str, Any]:
    cfg = load_config().get("neo4j", {})
    if not cfg.get("enabled", False):
        raise ValueError("Neo4j is disabled in config.yaml")
    return cfg

def get_qdrant_config() -> Dict[str, Any]:
    return load_config().get("qdrant", {})

def get_component_config(name: str) -> ComponentConfig:
    cfg = load_config().get(name, {})
    return ComponentConfig(
        provider=cfg.get("provider", "openai"),
        model=cfg.get("model", "gpt-4o-mini"),
        max_tokens=int(cfg.get("max_tokens", 4000)),
        temperature=float(cfg.get("temperature", 0.1)),
        identifier=cfg.get("identifier", "") # 【補上：正確讀取】
    )

def get_section_config(section: str) -> Dict[str, Any]:
    return load_config().get(section, {})

def get_sparse_config() -> Dict[str, Any]:
    return load_config().get("sparse", {})

class ProviderFactory:
    def __init__(self):
        self.config = load_config()
        self.providers = {}
        for name, p_cfg in self.config.get("providers", {}).items():
            self.providers[name] = ProviderConfig(
                name=name,
                api_key=p_cfg.get("api_key"),
                base_url=p_cfg.get("base_url")
            )

    def create_client(self, provider_name: str):
        cfg = self.providers.get(provider_name)
        if not cfg:
            raise ValueError(f"Unknown provider: {provider_name}")

        from openai import OpenAI
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        
        # ──【底層 Token 防護罩】──
        original_create = client.chat.completions.create
        def safe_create(*args, **kwargs):
            if "max_tokens" in kwargs and kwargs["max_tokens"]:
                kwargs["max_tokens"] = min(kwargs["max_tokens"], 100000)
            return original_create(*args, **kwargs)
        client.chat.completions.create = safe_create
        
        return client

_factory = None

def create_client(provider_name: str):
    global _factory
    if _factory is None:
        _factory = ProviderFactory()
    return _factory.create_client(provider_name)

def create_client_for(component: str) -> Tuple[Any, ComponentConfig]:
    comp_cfg = get_component_config(component)
    client = create_client(comp_cfg.provider)
    return client, comp_cfg
