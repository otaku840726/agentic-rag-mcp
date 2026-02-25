"""
Utility functions for Agentic RAG
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any


# ========== LLM Response Parsing ==========
def strip_think_tags(content: str) -> str:
    """移除 LLM 回應中的 <think>...</think> 標籤"""
    if not content or '<think>' not in content:
        return content
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)


def extract_json_from_response(content: str) -> dict:
    """從 LLM 回應中提取並解析 JSON

    流程: strip think tags -> 定位 JSON -> parse
    Raises json.JSONDecodeError if no valid JSON found.
    """
    content = strip_think_tags(content).strip()

    if not content.startswith('{'):
        json_start = content.find('{')
        if json_start >= 0:
            json_end = content.rfind('}')
            if json_end > json_start:
                content = content[json_start:json_end + 1]

    return json.loads(content)


# ========== Content Normalization ==========
def normalize_content(content: str) -> str:
    """
    正規化內容用於去重
    - 先處理每行再合併
    - 去除行號、時間戳
    - 壓縮空白
    """
    if not content:
        return ""

    # Step 1: 逐行處理行號
    lines = content.strip().splitlines()
    lines = [re.sub(r'^\s*\d+[\s:|\-]+\s*', '', ln) for ln in lines]

    # Step 2: 合併後壓縮空白
    s = ' '.join(lines)
    s = re.sub(r'\s+', ' ', s)

    # Step 3: 去除時間戳 (含 ISO 格式)
    s = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?', '', s)
    s = re.sub(r'\d{4}-\d{2}-\d{2}', '', s)

    # Step 4: 只取前後 N 字 (避免中間差異導致不同 hash)
    if len(s) > 200:
        s = s[:100] + s[-100:]

    return s.strip()


def compute_fingerprint(chunk_text: str, chunk_id: str = "", path: str = "") -> str:
    """
    計算 fingerprint，處理空字串邊界情況
    """
    norm = normalize_content(chunk_text)

    # 保底：normalize 後為空時用原始內容或 id+path
    if not norm:
        fallback = chunk_text.strip()[:200]
        if not fallback:
            fallback = f"{chunk_id}:{path}"
        norm = fallback

    return hashlib.md5(norm.encode('utf-8')).hexdigest()[:16]


# ========== Named Entity Extraction ==========
def extract_named_entities(content: str) -> Dict[str, List[str]]:
    """從內容提取具名實體"""
    entities = {"config_keys": [], "enums": [], "constants": []}

    if not content:
        return entities

    # Config keys
    config_patterns = [
        r'([a-zA-Z0-9_.-]+\.(?:timeout|ttl|deadline|retry|interval|delay)[a-zA-Z0-9_.-]*)',
        r'@Value\s*\(\s*["\']?\$\{([^}]+)\}',
        r'@ConfigurationProperties\s*\(\s*prefix\s*=\s*["\']([^"\']+)',
        r'Configuration\[([^\]]+)\]',  # .NET IConfiguration
        r'appsettings[^"]*"([^"]+)"',
        r'GetValue[<\(][^>)]*[>)]?\s*\(\s*["\']([^"\']+)',  # .NET GetValue
    ]
    for pattern in config_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # 處理不同的匹配結果類型
            if isinstance(matches[0], str):
                entities["config_keys"].extend(matches)
            else:
                # 多個捕獲組，取第一個非空的
                for m in matches:
                    if isinstance(m, tuple):
                        for part in m:
                            if part:
                                entities["config_keys"].append(part)
                                break
                    else:
                        entities["config_keys"].append(m)

    # Enums
    enum_patterns = [
        r'enum\s+(\w+)',
        r'(\w+Status)\.\w+',
        r'(\w+State)\.\w+',
        r'(\w+Type)\.\w+',
    ]
    for pattern in enum_patterns:
        entities["enums"].extend(re.findall(pattern, content))

    # Constants (全大寫)
    constants = re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', content)
    # 過濾常見噪音
    noise = {'SELECT', 'UPDATE', 'INSERT', 'DELETE', 'FROM', 'WHERE', 'NULL', 'TRUE', 'FALSE',
             'AND', 'OR', 'NOT', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ORDER', 'GROUP',
             'HAVING', 'LIMIT', 'OFFSET', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX',
             'BEGIN', 'END', 'DECLARE', 'SET', 'EXEC', 'PROCEDURE', 'FUNCTION'}
    entities["constants"] = list(set(c for c in constants if c not in noise))

    # Dedupe
    entities["config_keys"] = list(set(entities["config_keys"]))
    entities["enums"] = list(set(entities["enums"]))

    return entities


# ========== Module Key Extraction ==========
def get_module_key(path: str) -> str:
    """
    取 module 層級 - 支援多種專案結構
    """
    if not path:
        return "unknown"

    parts = path.split('/')

    # Pattern 1: Java monorepo - modules/<name>/src/main/...
    for i, p in enumerate(parts):
        if p == 'modules' and i + 1 < len(parts):
            return f"modules/{parts[i+1]}"

    # Pattern 2: src/main 或 src/test - 往前回溯 1-2 層
    for i, p in enumerate(parts):
        if p == 'src' and i + 1 < len(parts) and parts[i+1] in ['main', 'test']:
            if i >= 1:
                return '/'.join(parts[max(0, i-1):i+1])

    # Pattern 3: 常見頂層目錄
    for i, p in enumerate(parts):
        if p in ['services', 'apps', 'packages', 'libs', 'service']:
            if i + 1 < len(parts):
                return f"{p}/{parts[i+1]}"

    # Pattern 4: .NET project directories (e.g., Project.Module)
    for i, p in enumerate(parts):
        if '.' in p and p[0].isupper() and not p.endswith(('.cs', '.py', '.js', '.ts')):
            return p

    # Fallback: 前 3 段
    return '/'.join(parts[:min(3, len(parts))])


# ========== Keyword Variants ==========
def build_keyword_variants(keyword: str) -> Dict[str, str]:
    """生成關鍵字變體"""
    if not keyword:
        return {"original": "", "lower": "", "upper": "", "camel": "", "pascal": ""}

    # 處理 snake_case
    words = keyword.replace('_', ' ').replace('-', ' ').split()

    return {
        "original": keyword,
        "lower": keyword.lower(),
        "upper": keyword.upper(),
        "camel": keyword[0].lower() + keyword[1:] if len(keyword) > 1 else keyword.lower(),
        "pascal": ''.join(w.capitalize() for w in words) if words else keyword,
        "snake": '_'.join(w.lower() for w in words) if words else keyword.lower(),
    }


# ========== Snippet Creation ==========
def create_snippet(content: str, max_length: int = 200) -> str:
    """創建摘要片段"""
    if not content:
        return ""

    # 清理內容
    s = content.strip()
    s = re.sub(r'\s+', ' ', s)

    if len(s) <= max_length:
        return s

    return s[:max_length - 3] + "..."


# ========== Tag Extraction ==========
def extract_tags(content: str, path: str) -> List[str]:
    """從內容和路徑提取標籤"""
    tags = set()

    content_lower = content.lower() if content else ""
    path_lower = path.lower() if path else ""

    # 從內容提取
    tag_patterns = {
        "timeout": r'timeout|ttl|deadline',
        "state-machine": r'state\s*machine|transition|enum.*status',
        "config": r'configuration|appsettings|\.config|\.yaml|\.json',
        "database": r'select|insert|update|delete|stored\s*proc|tbl\w+|sp\w+',
        "robot": r'robot|automation|playwright|selenium',
        "transaction": r'transaction|deposit|payout|transfer',
        "api": r'controller|endpoint|api|route',
        "callsite": r'\w+\.\w+\(|\w+::\w+',
        "invocation": r'invoke|call|execute|trigger',
        "error-handling": r'exception|error|catch|throw|try',
        "async": r'async|await|task|promise',
    }

    for tag, pattern in tag_patterns.items():
        if re.search(pattern, content_lower, re.IGNORECASE):
            tags.add(tag)

    # 從路徑提取
    if 'robot' in path_lower:
        tags.add('robot')
    if 'job' in path_lower:
        tags.add('job')
    if 'service' in path_lower:
        tags.add('service')
    if 'controller' in path_lower:
        tags.add('controller')
    if 'config' in path_lower or 'appsettings' in path_lower:
        tags.add('config')
    if '.sql' in path_lower or 'stored' in path_lower:
        tags.add('database')

    return list(tags)


# ========== Source Kind Detection ==========
def detect_source_kind(path: str) -> str:
    """根據路徑檢測來源類型"""
    if not path:
        return "code"

    path_lower = path.lower()

    if path_lower.endswith('.sql'):
        return "sql"
    if path_lower.endswith(('.md', '.txt', '.rst')):
        return "doc"
    if path_lower.endswith(('.yaml', '.yml', '.json', '.xml', '.config')):
        return "config"
    if 'appsettings' in path_lower or 'config' in path_lower:
        return "config"
    if 'jira' in path_lower or 'issue' in path_lower:
        return "jira"
    if 'knowledge' in path_lower or 'docs/' in path_lower:
        return "doc"

    return "code"


# ========== Accept Patterns ==========
ACCEPT_PATTERNS = {
    "config key": [r'\.timeout', r'\.ttl', r'config\[', r'@Value', r'Configuration\['],
    "default value": [r'default\s*[:=]', r'fallback', r'\?\?', r'\|\|'],
    "where read": [r'GetValue', r'Configuration\[', r'@ConfigurationProperties', r'IOptions'],
    "enum definition": [r'enum\s+\w+', r'public\s+enum'],
    "transition": [r'transition', r'ChangeState', r'UpdateStatus', r'SetStatus'],
    "entry point": [r'public\s+(async\s+)?Task', r'\[Http', r'@\w+Mapping', r'Main\('],
    "call edge": [r'\w+\.\w+\(', r'await\s+\w+', r'\.Invoke\(', r'\.Execute\('],
}


def check_accept_coverage(accept_items: List[str], content: str) -> List[str]:
    """檢查 accept 條件的覆蓋情況"""
    covered = []
    for item in accept_items:
        patterns = ACCEPT_PATTERNS.get(item, [re.escape(item)])
        if any(re.search(p, content, re.IGNORECASE) for p in patterns):
            covered.append(item)
    return covered


# ========== Causal Verbs ==========
CAUSAL_VERBS = ['update', 'change', 'transition', 'set', 'modify', 'trigger',
                'call', 'invoke', 'execute', 'retry', 'timeout', 'handle',
                'process', 'create', 'delete', 'insert', 'save']


def has_causal_verb(content: str) -> bool:
    """檢查是否包含因果動詞"""
    if not content:
        return False
    content_lower = content.lower()
    return any(v in content_lower for v in CAUSAL_VERBS)
