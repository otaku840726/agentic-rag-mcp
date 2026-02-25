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
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Expand evidence with graph neighbors.

        Strategy:
        1. Extract symbol names from evidence (prefer explicit symbol field over regex)
        2. Query graph for structural neighbors (inheritance, type usage)
        3. Collect neighbor file_paths that aren't already in evidence
        4. Fetch those files from Qdrant as supplementary results

        Focus on precision: only expand with structurally-related code,
        not every symbol mentioned in snippets.
        """
        existing_paths = {card.path for card in evidence_cards}
        symbol_names = self._extract_symbols(evidence_cards)

        if not symbol_names:
            return []

        # Query graph for neighbors — focus on structural relationships
        neighbor_files: Dict[str, float] = {}  # file_path -> relevance score
        for sym_name in symbol_names[:8]:
            try:
                result = self.graph.get_neighbors(
                    sym_name, depth=1,
                    relationship_types=_USEFUL_REL_TYPES,
                )
                for node in result.get("nodes", []):
                    fp = node.get("file_path")
                    kind = node.get("kind", "")
                    if not fp or fp in existing_paths:
                        continue
                    # Skip external/unresolved references
                    if kind == "external":
                        continue
                    # Accumulate score (more connections = more relevant)
                    neighbor_files[fp] = neighbor_files.get(fp, 0) + 1.0
            except Exception as e:
                logger.debug(f"Graph neighbor query failed for {sym_name}: {e}")

        if not neighbor_files:
            return []

        # Sort by relevance and take top_k
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
                        "score_hybrid": r.get("score_hybrid", 0.0),
                        "payload": r.get("payload", {}),
                        "source": "graph_expansion",
                    })
            except Exception as e:
                logger.debug(f"Qdrant fetch failed for graph neighbor {file_path}: {e}")

        logger.info(
            f"Graph expansion: {len(symbol_names)} symbols -> "
            f"{len(neighbor_files)} neighbor files -> "
            f"{len(supplementary)} supplementary results"
        )
        return supplementary[:top_k]

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
