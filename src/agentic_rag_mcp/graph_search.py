"""
GraphSearchEnhancer — Optional graph-based context expansion for agentic search.

When Neo4j is enabled, expands evidence cards by querying graph neighbors
of symbols found in search results. Focuses on structural relationships
(MEMBER_OF, INHERITS, IMPLEMENTS) to find related code files that vector
search might miss.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .indexer.graph_store import GraphStore
    from .hybrid_search import HybridSearch
    from .models import EvidenceCard

logger = logging.getLogger(__name__)

# Namespaces / prefixes that are too generic to be useful for graph expansion
_NOISE_PREFIXES = {
    "System", "Microsoft", "Newtonsoft", "AutoMapper", "Flurl",
    "NLog", "Casbin", "Serilog", "Swashbuckle", "Hangfire",
}

# Relationship types most useful for finding related code
_USEFUL_REL_TYPES = ["INHERITS", "IMPLEMENTS", "MEMBER_OF", "USES_TYPE", "CALLS", "CREATES"]


class GraphSearchEnhancer:
    """Optional graph-based context expansion for agentic search."""

    def __init__(self, graph_store: "GraphStore", hybrid_search: "HybridSearch"):
        self.graph = graph_store
        self.search = hybrid_search

    def expand_evidence(
        self,
        evidence_cards: List["EvidenceCard"],
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """利用物理鏈路與名稱相似性，提供全方位的拓樸資訊。"""
        existing_paths = {card.path for card in evidence_cards}
        symbol_names = self._extract_symbols(evidence_cards)
        target_communities = {c.community_id for c in evidence_cards if c.community_id is not None}

        if not symbol_names and not target_communities:
            return []

        project = getattr(self.graph, "default_project", "smilepay")
        neighbor_files: Dict[str, Dict[str, Any]] = {}

        # 1. [物理連通性擴展] 查詢 1-2 hop 內的鄰居
        if symbol_names:
            logger.info(f"Topology expansion for symbols...")
            linkage_query = """
            MATCH (start:Symbol)
            WHERE (start.name IN $names OR start.fqn IN $names) AND start.project = $project
            MATCH (start)-[r:CALLS|USES_TYPE|IMPLEMENTS|INHERITS*1..2]-(neighbor:Symbol)
            WHERE neighbor.file_path IS NOT NULL AND neighbor.project = $project
            RETURN DISTINCT neighbor.file_path as path, neighbor.name as name, type(r[0]) as rel_type, start.name as source
            LIMIT 30
            """
            try:
                res = self.graph.cypher_query(linkage_query, {"names": symbol_names[:15], "project": project})
                for r in res:
                    fp = r['path']
                    if fp and fp not in existing_paths:
                        desc = f"[Topology] '{r['name']}' is related to '{r['source']}' via {r['rel_type']}"
                        if fp not in neighbor_files:
                            neighbor_files[fp] = {"weight": 2.0, "reason": desc, "cid": None}
                        else:
                            neighbor_files[fp]["weight"] += 1.0
            except Exception as e:
                logger.warning(f"Linkage query failed: {e}")

        # 2. [名稱相似性擴展] 尋找全專案中名稱相似但版本/社群不同的符號 (解決 V1 vs V2)
        if symbol_names:
            logger.info(f"Sibling version expansion for symbols...")
            # 取 symbol 名稱的核心部分 (例如 MemberDepositV2Controller -> MemberDeposit)
            core_names = []
            import re
            for name in symbol_names[:5]:
                # 移除 V1, V2, Impl, Controller 等後綴
                core = re.sub(r'V\d+', '', name)
                core = core.replace('Controller', '').replace('Service', '').replace('Impl', '')
                if len(core) > 5:
                    core_names.append(core)
                    
            if core_names:
                sibling_query = """
                MATCH (s:Symbol {project: $project})
                WHERE any(core IN $cores WHERE s.name CONTAINS core) 
                  AND NOT s.name IN $names
                  AND s.file_path IS NOT NULL
                RETURN DISTINCT s.file_path as path, s.name as name, s.communityId as cid
                LIMIT 20
                """
                try:
                    res = self.graph.cypher_query(sibling_query, {
                        "project": project, 
                        "cores": core_names,
                        "names": symbol_names
                    })
                    for r in res:
                        fp = r['path']
                        if fp and fp not in existing_paths:
                            desc = f"[Similarity] '{r['name']}' shares core name pattern with current targets. (CID: {r.get('cid', 'Unknown')})"
                            if fp not in neighbor_files:
                                neighbor_files[fp] = {"weight": 1.5, "reason": desc, "cid": r.get('cid')}
                except Exception as e:
                    logger.warning(f"Sibling query failed: {e}")

        if not neighbor_files:
            return []

        # 排序並取 Top K
        sorted_files = sorted(neighbor_files.items(), key=lambda x: -x[1]["weight"])

        supplementary = []
        for file_path, data in sorted_files[:top_k]:
            try:
                results = self.search.search_by_file_path(file_path, limit=2)
                for r in results:
                    # 將拓樸原因注入到 snippet 或 content 頂部，讓 Worker 能直接看見
                    reason_header = f"--- GRAPH TOPOLOGY CONTEXT ---\n{data['reason']}\n------------------------------\n"
                    orig_content = r.get("content", "")
                    
                    supplementary.append({
                        "path": r.get("path", file_path),
                        "content": reason_header + orig_content,
                        "score": 1.5,
                        "score_hybrid": 1.5,
                        "payload": r.get("payload", {}),
                        "source": "graph_expansion",
                    })
            except Exception as e:
                logger.debug(f"Qdrant fetch failed for graph expansion {file_path}: {e}")

        logger.info(f"Graph expansion added {len(supplementary)} supplementary cards with topology context.")
        return supplementary

    def _extract_symbols(self, evidence_cards: List["EvidenceCard"]) -> List[str]:
        """Extract high-confidence symbol names from evidence cards.

        Prioritizes explicit symbol metadata over regex extraction to reduce noise.
        """
        symbols = []
        seen = set()

        for card in evidence_cards:
            # Priority 1: Explicit symbol field (from Qdrant payload)
            if card.symbol and card.symbol not in seen:
                if not self._is_noise(card.symbol):
                    symbols.append(card.symbol)
                    seen.add(card.symbol)

            # Priority 2: Named entities (already extracted by utils)
            if card.named_entities:
                entities = card.named_entities
                # named_entities can be dict {"config_keys": [...], ...} or list
                if isinstance(entities, dict):
                    for ent_list in entities.values():
                        if isinstance(ent_list, list):
                            for ent in ent_list:
                                if ent and ent not in seen and not self._is_noise(ent):
                                    symbols.append(ent)
                                    seen.add(ent)

        return symbols

    @staticmethod
    def _is_noise(name: str) -> bool:
        """Check if a symbol name is too generic to be useful for graph expansion."""
        if not name or len(name) < 4:
            return True
        # Check if it starts with a known noise prefix
        first_part = name.split('.')[0] if '.' in name else name
        return first_part in _NOISE_PREFIXES
