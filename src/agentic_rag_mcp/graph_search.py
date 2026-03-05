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
        top_k: int = 15, # 增加擴展數量
    ) -> List[Dict[str, Any]]:
        """利用物理鏈路與社群歸屬，自動拼湊完整業務鄰里。"""
        existing_paths = {card.path for card in evidence_cards}
        symbol_names = self._extract_symbols(evidence_cards)
        
        # 收集社群 ID
        target_communities = {c.community_id for c in evidence_cards if c.community_id is not None}

        if not symbol_names and not target_communities:
            return []

        project = getattr(self.graph, "default_project", "default")
        neighbor_files: Dict[str, float] = {}

        # 1. [物理連通性擴展] 查詢 2-hop 內的調用者與被調用者
        if symbol_names:
            logger.info(f"Deterministic linkage expansion for {len(symbol_names)} symbols...")
            linkage_query = """
            MATCH (start:Symbol)
            WHERE (start.name IN $names OR start.fqn IN $names) AND start.project = $project
            MATCH (start)-[:CALLS|USES_TYPE|IMPLEMENTS|INHERITS*1..2]-(neighbor:Symbol)
            WHERE neighbor.file_path IS NOT NULL AND neighbor.project = $project
            RETURN DISTINCT neighbor.file_path as path, COUNT(*) as weight
            LIMIT 50
            """
            res = self.graph.cypher_query(linkage_query, {"names": symbol_names[:15], "project": project})
            for r in res:
                fp = r['path']
                if fp not in existing_paths:
                    neighbor_files[fp] = neighbor_files.get(fp, 0) + float(r['weight'])

        # 2. [社群歸屬擴展] 查詢同社群的核心成員 (這是抓取 Advice 的關鍵)
        if target_communities:
            logger.info(f"Deterministic community expansion for CIDs: {list(target_communities)}")
            cid_list = [str(cid) for cid in target_communities] + list(target_communities)
            community_query = """
            MATCH (s:Symbol {project: $project})
            WHERE s.communityId IN $cids OR EXISTS { (s)-[:IN_COMMUNITY]->(c:Community) WHERE c.id IN $cids }
            RETURN DISTINCT s.file_path as path
            LIMIT 50
            """
            res = self.graph.cypher_query(community_query, {"cids": cid_list, "project": project})
            for r in res:
                fp = r['path']
                if fp and fp not in existing_paths:
                    neighbor_files[fp] = neighbor_files.get(fp, 0) + 2.0 # 社群成員權重更高

        if not neighbor_files:
            return []

        # Sort by weight and take top_k
        sorted_files = sorted(neighbor_files.items(), key=lambda x: -x[1])

        # Fetch content from Qdrant
        supplementary = []
        for file_path, _score in sorted_files[:top_k]:
            try:
                results = self.search.search_by_file_path(file_path, limit=2)
                for r in results:
                    supplementary.append({
                        "path": r.get("path", file_path),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                        "score_hybrid": 1.5, # 賦予高分確保進入工作集
                        "payload": r.get("payload", {}),
                        "source": "graph_expansion",
                    })
            except Exception as e:
                logger.debug(f"Qdrant fetch failed for graph expansion {file_path}: {e}")

        logger.info(f"Aggressive Graph expansion added {len(supplementary)} supplementary cards.")
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
