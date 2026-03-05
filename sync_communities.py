
import os
import logging
from dotenv import load_dotenv
load_dotenv()

from src.agentic_rag_mcp.indexer.graph_store import GraphStore
from src.agentic_rag_mcp.provider import get_neo4j_config, load_config
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync_communities")

def sync_communities_to_qdrant(project_name: str = "smilepay"):
    # 1. 獲取配置
    neo_cfg = get_neo4j_config()
    full_cfg = load_config()
    qd_cfg = full_cfg.get("qdrant", {})
    
    gs = GraphStore(
        uri=neo_cfg["uri"],
        username=neo_cfg["username"],
        password=neo_cfg["password"],
        database=neo_cfg["database"]
    )
    
    qc = QdrantClient(
        url=qd_cfg["url"],
        api_key=qd_cfg["api_key"]
    )
    collection = qd_cfg["collection"]

    # 2. 從 Neo4j 讀取映射 (檔案路徑 -> 社群ID)
    logger.info(f"Fetching community mapping for project '{project_name}' from Neo4j...")
    query = """
    MATCH (s:Symbol {project: $project})
    WHERE s.communityId IS NOT NULL AND s.file_path IS NOT NULL
    RETURN DISTINCT s.file_path as path, s.communityId as cid
    """
    results = gs.cypher_query(query, {"project": project_name})
    
    path_to_cid = {r["path"]: int(r["cid"]) for r in results}
    logger.info(f"Found {len(path_to_cid)} files with community assignments.")

    if not path_to_cid:
        logger.warning("No community mapping found. Make sure community detection has run.")
        return

    # 3. 在 Qdrant 中按檔案更新 Payload
    logger.info(f"Updating Qdrant collection '{collection}' payloads...")
    
    success_count = 0
    for path, cid in path_to_cid.items():
        try:
            # 使用 set_payload 搭配 filter 進行批次更新
            qc.set_payload(
                collection_name=collection,
                payload={"community_id": cid},
                points=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_path",
                            match=rest.MatchValue(value=path)
                        )
                    ]
                ),
                wait=True
            )
            success_count += 1
            if success_count % 50 == 0:
                logger.info(f"Synced {success_count} files...")
        except Exception as e:
            logger.error(f"Failed to update payload for {path}: {e}")

    logger.info(f"Sync complete. Updated {success_count} files in Qdrant.")

if __name__ == "__main__":
    sync_communities_to_qdrant("smilepay")
