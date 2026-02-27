"""
Query Builder - 多類型查詢生成
支援多種 operator 和模板庫
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .models import QueryIntent
from .utils import build_keyword_variants


# ========== Query Templates ==========
QUERY_TEMPLATES = {
    "entry_point": {
        "patterns": [
            '"{keyword}" handler',
            '"on{Pascal}" OR "handle{Pascal}"',
            '"{keyword}Handler" OR "{keyword}Service"',
        ],
        "annotations": [
            '@EventListener',
            '@Scheduled',
            '@KafkaListener',
            '@RabbitListener',
            '@PostMapping',
            '@GetMapping',
            '@HttpPost',
            '@HttpGet',
            '[HttpPost]',
            '[HttpGet]',
            '[Route]',
        ],
        "operator": "hybrid",
        "filters": {"category": ["source-code"]}
    },

    "timeout_config": {
        "patterns": [
            '(timeout OR ttl OR deadline OR interval) AND (config OR properties)',
            '"{keyword}" AND (timeout OR delay OR retry)',
            'TimeSpan OR TimeoutMs OR TimeoutSeconds',
        ],
        "operator": "keyword",
        "filters": {"category": ["configuration", "source-code"]}
    },

    "state_machine": {
        "patterns": [
            '(enum OR status OR state) AND "{keyword}"',
            '"{keyword}Status" OR "{keyword}State"',
            'transition AND "{keyword}"',
            'UpdateStatus OR ChangeState OR SetState',
        ],
        "operator": "hybrid",
        "filters": {}
    },

    "callsite": {
        "patterns": [
            '"{class}.{method}"',
            '"{class}#{method}"',
            '"{class}::{method}"',
            '"{method}("',
            'await {method}',
            '.{method}(',
        ],
        "operator": "exact",
        "filters": {"category": ["source-code"]}
    },

    "database": {
        "patterns": [
            'sp{Pascal} OR tbl{Pascal}',
            '"{keyword}" AND (SELECT OR UPDATE OR INSERT OR EXEC)',
            'FROM {keyword} OR JOIN {keyword}',
            'PROCEDURE.*{keyword}',
        ],
        "operator": "keyword",
        "filters": {"category": ["database-schema", "source-code"]}
    },

    "config_key": {
        "patterns": [
            '"{keyword}"',
            'Configuration["{keyword}"]',
            '@Value("${keyword}")',
            'appsettings.*"{keyword}"',
        ],
        "operator": "exact",
        "filters": {"category": ["configuration", "source-code"]}
    },

    "error_handling": {
        "patterns": [
            '(catch OR exception OR error) AND "{keyword}"',
            'try.*{keyword}.*catch',
            '"{keyword}Exception" OR "{keyword}Error"',
        ],
        "operator": "hybrid",
        "filters": {"category": ["source-code"]}
    },
}


class QueryBuilder:
    """查詢建構器"""

    def __init__(self, templates: Optional[Dict] = None):
        self.templates = templates or QUERY_TEMPLATES

    def build_from_intent(self, intent: QueryIntent) -> List[Dict[str, Any]]:
        """
        根據 QueryIntent 生成實際查詢
        Returns: List of {query, operator, filters}
        """
        queries = []
        variants = build_keyword_variants(intent.query)

        # 根據 operator 或 query_type 決定查詢方式
        is_graph_fallback = intent.query_type.startswith("graph_traverse")

        if intent.operator == "exact" or is_graph_fallback:
            # Graph fallback: treat it as exact symbol match in payload/content
            queries.append({
                "query": intent.query,
                "operator": "exact",
                "filters": intent.filters or {}
            })

        elif intent.operator == "callsite":
            # 生成多種 callsite 查詢
            for pattern in self.templates.get("callsite", {}).get("patterns", []):
                q = self._fill_pattern(pattern, variants, intent.query)
                if q:
                    queries.append({
                        "query": q,
                        "operator": "exact",
                        "filters": intent.filters or {"category": ["source-code"]}
                    })

        elif intent.operator == "symbol_ref":
            # 符號引用查詢
            queries.append({
                "query": f'"{intent.query}"',
                "operator": "exact",
                "filters": intent.filters or {}
            })
            # 加上變體
            if variants["pascal"] != intent.query:
                queries.append({
                    "query": f'"{variants["pascal"]}"',
                    "operator": "exact",
                    "filters": intent.filters or {}
                })

        elif intent.operator == "keyword":
            queries.append({
                "query": intent.query,
                "operator": "keyword",
                "filters": intent.filters or {}
            })

        else:  # semantic / hybrid
            queries.append({
                "query": intent.query,
                "operator": "hybrid",
                "filters": intent.filters or {}
            })

        return queries

    def build_from_template(
        self,
        template_name: str,
        keyword: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        根據模板名稱生成查詢
        """
        template = self.templates.get(template_name)
        if not template:
            return [{"query": keyword, "operator": "semantic", "filters": {}}]

        variants = build_keyword_variants(keyword)
        queries = []

        # 處理 patterns
        for pattern in template.get("patterns", []):
            q = self._fill_pattern(pattern, variants, keyword, **kwargs)
            if q:
                queries.append({
                    "query": q,
                    "operator": template.get("operator", "semantic"),
                    "filters": template.get("filters", {})
                })

        # 處理 annotations (入口點特有)
        for ann in template.get("annotations", []):
            queries.append({
                "query": f'{ann} AND "{keyword}"',
                "operator": "keyword",
                "filters": {"category": ["source-code"]}
            })

        return queries

    def _fill_pattern(
        self,
        pattern: str,
        variants: Dict[str, str],
        keyword: str,
        **kwargs
    ) -> Optional[str]:
        """填充模板中的佔位符"""
        try:
            result = pattern
            result = result.replace("{keyword}", variants.get("original", keyword))
            result = result.replace("{Keyword}", variants.get("camel", keyword))
            result = result.replace("{Pascal}", variants.get("pascal", keyword))
            result = result.replace("{KEYWORD}", variants.get("upper", keyword))
            result = result.replace("{snake}", variants.get("snake", keyword))

            # 額外參數
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value) if value else "")

            # 清理空的佔位符
            result = re.sub(r'\{[^}]+\}', '', result)
            result = re.sub(r'\s+', ' ', result).strip()

            # 清理無效查詢
            if not result or result in ['""', "''", '()', 'AND', 'OR']:
                return None

            return result
        except Exception:
            return None

    def build_fallback_queries(
        self,
        evidence_cards: List[Any],
        missing_evidence: List[Any]
    ) -> List[QueryIntent]:
        """
        生成 fallback 查詢 - 兩層策略：
        Layer A: 傳統鄰居查詢（callsite / config key）
        Layer B: Step-back — 從當前符號退後到 Controller/Route/Handler 入口
        """
        queries = []
        seen_queries = set()

        # ── Layer A: 傳統鄰居查詢 ──────────────────────────────

        # A1. 從 snippet 提取方法名，查 callsite
        method_pattern = r'\b(\w{4,})\s*\('
        _skip_keywords = {'if', 'while', 'for', 'switch', 'catch', 'return', 'throw',
                          'await', 'async', 'using', 'foreach', 'public', 'private'}
        for card in evidence_cards[:10]:
            methods = re.findall(method_pattern, card.snippet)
            for m in methods:
                if m.lower() in _skip_keywords:
                    continue
                if m not in seen_queries:
                    seen_queries.add(m)
                    queries.append(QueryIntent(
                        query=f'{m}(',
                        purpose=f"find callsites of {m}",
                        operator="exact",
                        query_type="callsite"
                    ))

        # A2. 從 named_entities 提取 config key
        for card in evidence_cards[:10]:
            for key in card.named_entities.get("config_keys", [])[:2]:
                if key not in seen_queries:
                    seen_queries.add(key)
                    queries.append(QueryIntent(
                        query=key,
                        purpose=f"find config usage of {key}",
                        operator="exact",
                        query_type="config"
                    ))

        # ── Layer B: Step-back — 退後到 API 入口層 ─────────────
        # 策略：提取現有證據中最高分的 symbol，搜索其對應的
        # Controller / Route / Handler / HttpPost / HttpGet / ApiController

        step_back_done = False
        if evidence_cards and not step_back_done:
            # 取最高 rerank 分數的卡片的 symbol 或文件路徑關鍵詞
            top_cards = sorted(evidence_cards, key=lambda c: c.score_rerank, reverse=True)[:3]
            for card in top_cards:
                # 提取最可能的業務關鍵詞（優先用 symbol，否則從路徑提取模塊名）
                anchor = card.symbol
                if not anchor and card.path:
                    # 從路徑取文件名去掉擴展名
                    parts = card.path.replace('\\', '/').split('/')
                    fname = parts[-1] if parts else ''
                    anchor = re.sub(r'\.(cs|py|java|ts|js)$', '', fname, flags=re.IGNORECASE)

                if anchor and f"stepback_{anchor}" not in seen_queries:
                    seen_queries.add(f"stepback_{anchor}")
                    # 搜索以此 anchor 為核心的 Controller / API 入口
                    queries.append(QueryIntent(
                        query=f"{anchor} Controller HttpPost HttpGet Route",
                        purpose=f"step-back: find API entry point for {anchor}",
                        operator="keyword",
                        query_type="keyword"
                    ))
                    # 同時嘗試語義搜索入口層
                    queries.append(QueryIntent(
                        query=f"API endpoint handler for {anchor} request",
                        purpose=f"step-back: semantic search for {anchor} entry",
                        operator="semantic",
                        query_type="semantic"
                    ))
                    step_back_done = True
                    break

        # A3. 根據 missing_evidence 的 need 類型做特化 fallback
        for m in missing_evidence:
            need_lower = m.need.lower()
            if 'timeout' in need_lower and 'timeout' not in seen_queries:
                seen_queries.add('timeout')
                queries.append(QueryIntent(
                    query="timeout config default value",
                    purpose="find timeout configuration",
                    operator="semantic",
                    query_type="config"
                ))
            if ('state' in need_lower or 'enum' in need_lower) and 'state' not in seen_queries:
                seen_queries.add('state')
                queries.append(QueryIntent(
                    query="enum Status transition update",
                    purpose="find state machine definition",
                    operator="semantic",
                    query_type="symbol"
                ))

        return queries[:7]  # 增加上限（原本 5 → 7）以容納 step-back 查詢
