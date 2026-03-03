
# Agentic RAG MCP Server

Agentic RAG MCP 是一個為代碼庫量身打造的強大 AI 檢索與問答伺服器。它透過 Model Context Protocol (MCP) 無縫整合至你的開發環境（如 Cursor, Claude Desktop），並結合了最新的 Agentic Workflow 與知識圖譜技術，提供極致精準的代碼導航與修復建議。

## 🌟 核心架構 (Agentic Workflow)

系統底層採用 LangGraph 狀態機（State Graph）來協調多個專業節點，徹底解決了傳統 RAG 容易迷失上下文的問題：

* **感知節點 (Context Awareness)**：自動偵測專案的技術棧（如 Java/Spring, Python, C# 等）並初始化共享黑板 (Blackboard)。
* **分析大腦 (Analyst)**：解析用戶真實意圖，並將複雜問題拆解為具體的子任務 (Sub-tasks)。
* **決策與行動 (Planner & Executor)**：具備 Tool Calling 能力的代理，能根據任務清單主動決策要呼叫哪些工具（如語義搜尋、讀取檔案、查找圖譜等）來收集證據。
* **品質守門員 (Quality Gate)**：透過預算 (Budget) 與證據完整度評估，動態決定是要繼續深挖代碼，還是進入總結階段。
* **總結合成本 (Synthesizer)**：根據收集到的高價值上下文，產出精準的程式碼解析、執行流程圖或 Code Patch 修改建議。

## ✨ 主要功能特點

* **多維度混合檢索 (Hybrid Search)**：結合了 Dense Vector (支援 OpenAI, Voyage-code 等)、Sparse Vector (BM25 / Splade) 以及強大的 Cross-encoder / Voyage Reranker 進行重新排序，確保最相關的程式碼片段名列前茅。
* **AST 代碼知識圖譜 (Neo4j AuraDB)**：不只依賴字面意義，系統會透過 Roslyn (C#) 或 Spoon (Java) 將代碼解析為抽象語法樹，並將調用鏈 (CALLS)、繼承 (INHERITS)、實作 (IMPLEMENTS) 等關係存入圖資料庫。
* **進階圖形算法 (Graph Algorithms)**：內建 GDS (Graph Data Science) 支援，能夠執行 Leiden 社群偵測 (Community Detection) 來自動識別業務模組，並能追蹤從 Entry Point 開始的執行流 (Execution Flows)。
* **兩階段智慧索引 (Two-Phase Indexing)**：支援針對變更檔案的增量索引。第一階段利用專用 Analyzer 處理複雜專案結構，第二階段使用 Tree-sitter 與 Markdown 解析器處理其餘檔案，大幅提升建置效率。
* **動態工具箱 (Tools Box)**：Agent 可隨時呼叫 `read_exact_file` 展開完整檔案上下文，或使用 `list_directory` 探索專案結構，行為模式更貼近真人資深工程師。

## ⚙️ 環境需求與安裝

本專案使用 `uv` 進行 Python 套件管理，並透過 TypeScript 提供 MCP 橋接。

**基礎依賴：**
* Python 3.10+ (推薦使用 `uv`)
* Node.js (供 MCP Server 啟動使用)
* Qdrant (向量資料庫)
* Neo4j / AuraDB (圖資料庫，可選但強烈推薦)

**環境變數設定：**
請複製 `.env.example` 並建立 `.env` 檔案，填入必要的 API Keys：
```env
# 提供給 LLM 組件使用的 API Key (如 OpenAI, OpenRouter 或 本地模型)
OPENAI_API_KEY=your_openai_api_key

# 向量資料庫
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# 圖資料庫 (Neo4j)
NEO4J_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# 進階 Reranker (推薦使用 Voyage)
RERANKER_PROVIDER=voyage
RERANKER_MODEL=voyage-rerank-2
VOYAGE_API_KEY=your_voyage_api_key

```

## 🚀 快速開始

1. **安裝 Python 依賴：**
```bash
uv sync

```


2. **安裝 NPM 依賴與建置：**
```bash
npm install
npm run build

```


3. **進行代碼庫索引 (Indexing)：**
啟動你的 MCP Client，呼叫 `index_codebase` 工具，或使用測試腳本針對目標目錄進行掃描與圖形建立。
4. **開始對話：**
在你的開發環境中直接向 Assistant 提問，例如：「這段支付回調的邏輯中，如果驗證簽名失敗會發生什麼事？請幫我追蹤完整的調用鏈並提供修復建議。」

## 🛠️ 配置自定義 (config.yaml)

你可以透過修改 `src/agentic_rag_mcp/config.yaml` 來精細控制系統行為：

* 切換不同節點的 LLM Provider (`analyst`, `planner`, `synthesizer`)。
* 調整預算控制 `budget.max_iterations` 與 `budget.total_token_budget`。
* 設定 Quality Gate 的嚴格程度，例如要求必須包含 `call_edge` 才能停止搜尋。
