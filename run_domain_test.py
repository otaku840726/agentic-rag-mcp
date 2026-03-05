
import os
import json
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.agentic_rag_mcp.indexer.graph_store import GraphStore
from src.agentic_rag_mcp.provider import get_neo4j_config, create_client_for, get_component_config

def run_full_domain_analysis_test():
    """
    完整的業務領域分析測試：
    1. 掃描圖譜中的所有路徑與符號。
    2. 按目錄結構初步聚類。
    3. 針對每個聚類提取特徵關鍵字。
    4. 讓 LLM 識別這些聚類的業務領域名稱。
    """
    cfg = get_neo4j_config()
    gs = GraphStore(uri=cfg['uri'], username=cfg['username'], password=cfg['password'], database=cfg['database'])
    
    # --- 1. 獲取所有檔案路徑並按 Package 聚類 ---
    print("Step 1: Clustering files by package structure...")
    query = """
    MATCH (s:Symbol)
    WHERE s.file_path IS NOT NULL
    RETURN s.file_path as path, s.name as name, s.kind as kind
    """
    symbols = gs.cypher_query(query)
    
    clusters = defaultdict(list)
    for sym in symbols:
        # 取路徑的前 4 層作為一個業務聚類（Heuristic）
        parts = sym['path'].split('/')
        if len(parts) > 4:
            cluster_key = "/".join(parts[:5]) # 模組/src/main/java/package
            clusters[cluster_key].append(sym['name'])

    print(f"Detected {len(clusters)} potential business clusters.")

    # --- 2. 針對大型聚類提取特徵 ---
    # 我們只分析檔案數量較多的聚類
    significant_clusters = {k: v for k, v in clusters.items() if len(v) > 5}
    
    # --- 3. 調用 LLM 進行領域鑑定 ---
    print("\nStep 2: Identifying domains via LLM sampling...")
    llm_cfg = get_component_config("analyst")
    client, _ = create_client_for(llm_cfg.provider)
    
    domain_map = {}
    
    # 為了測試完整性，我們挑選 5 個最具代表性的聚類進行分析
    test_keys = sorted(significant_clusters.keys(), key=lambda x: len(significant_clusters[x]), reverse=True)[:10]
    
    for key in test_keys:
        sample_symbols = list(set(significant_clusters[key]))[:15]
        prompt = f"""
        根據以下檔案路徑和其中的類名，請判斷這屬於該金流系統的哪個「具體業務領域」？

        路徑: {key}
        代表性類名: {', '.join(sample_symbols)}

        請給出一個簡短的領域名稱（例如：Member_Deposit, Merchant_Security, BackOffice_Report 等）
        以及一項簡短的描述。

        輸出格式: JSON {{"domain": "...", "description": "..."}}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # 使用標準模型確保穩定
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            domain_map[key] = data
            print(f"Path: {key:<60} -> Domain: {data['domain']}")
        except Exception as e:
            print(f"Error analyzing {key}: {e}")

    # --- 4. 產出最終報告 ---
    print("\n" + "="*50)
    print("FINAL BUSINESS DOMAIN REPORT")
    print("="*50)
    for path, info in domain_map.items():
        print(f"Domain: {info['domain']}")
        print(f"Path:   {path}")
        print(f"Desc:   {info['description']}")
        print("-" * 30)

if __name__ == "__main__":
    run_full_domain_analysis_test()
