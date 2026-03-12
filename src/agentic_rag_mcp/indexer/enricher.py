"""
Symbol payload enricher — Pipeline A (Qdrant/tree-sitter) only.

Pipeline B (AuraDB / graph) enrichment is handled inside each analyzer:
  - Java: SpoonAnalyzer.java emits file_type, http_path, entry_point_type, etc.
  - PHP:  analyze.php (TODO: not yet migrated — routes_map fallback in core.py)
  - C#:   RoslynAnalyzer (TODO: not yet migrated)

For Pipeline A (Qdrant), only minimal enrichment is applied:
  - is_test  : detect test files / classes
  - is_deprecated: normalize across languages
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich_payload(payload: dict, content: str = "", routes_map: dict = None) -> dict:
    """
    Minimal enrichment for Pipeline A (Qdrant) payloads.
    Language-specific semantic enrichment belongs in each analyzer.
    """
    file_path = payload.get("file_path") or ""
    sym_name = payload.get("symbol_name") or payload.get("name") or ""
    language = (payload.get("language") or "").lower()

    payload.setdefault("is_test", _classify_is_test(file_path, sym_name, language))
    payload.setdefault("is_deprecated", bool(payload.get("is_deprecated", False)))

    return payload


# ---------------------------------------------------------------------------
# is_test
# ---------------------------------------------------------------------------

def _classify_is_test(file_path: str, sym_name: str, language: str) -> bool:
    fp = file_path.lower().replace("\\", "/")
    name = sym_name.lower().split(".")[-1]
    if "/src/test/" in fp or "/tests/" in fp or "/test/" in fp:
        return True
    if language == "csharp" and re.search(r"(test|unittest|spec)", fp):
        return True
    if name.endswith("test") or name.endswith("tests") or name.endswith("spec"):
        return True
    return False


# ---------------------------------------------------------------------------
# makes_http_call detection (kept for Pipeline A tree-sitter payloads)
# ---------------------------------------------------------------------------

_HTTP_CALL_PATTERNS = {
    "java":       re.compile(r'\b(RestTemplate|WebClient|HttpClient|FeignClient|OkHttp|restTemplate|webClient)\b'),
    "csharp":     re.compile(r'\b(HttpClient|_client|_httpClient|\.GetAsync|\.PostAsync|\.PutAsync|\.DeleteAsync)\b'),
    "php":        re.compile(r'\b(Http::|Guzzle|GuzzleHttp|curl_exec|file_get_contents)\b'),
    "javascript": re.compile(r'\b(axios|fetch\(|http\.get|http\.post|xhr\.open|XMLHttpRequest)\b'),
    "typescript": re.compile(r'\b(axios|fetch\(|http\.get|http\.post|xhr\.open|XMLHttpRequest)\b'),
}


def _detect_http_call(language: str, content: str) -> bool:
    pattern = _HTTP_CALL_PATTERNS.get(language)
    if pattern:
        return bool(pattern.search(content))
    return False
