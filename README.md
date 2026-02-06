# Agentic RAG MCP Server

自主搜索代理 - 為代碼庫提供智能多跳搜索服務。

## 特性

- **Multi-hop Search**: 迭代搜索，自動追蹤邏輯關聯
- **Hybrid Search**: 結合語義搜索和關鍵字搜索
- **Cross-encoder Reranking**: 使用 cross-encoder 模型重排序
- **Evidence Store**: 兩層證據管理（全量池 + 多樣性工作集）
- **Quality Gate**: 質量門檻確保搜索結果完整性
- **MCP Integration**: 可作為 MCP Server 供 Claude Code 調用
- **Local LLM Support**: 支援本地 LLM（OpenAI 兼容 API）

## 安裝

### 方式 1: 使用 uvx (推薦)

```bash
# 直接運行
uvx agentic-rag-mcp

# 或安裝後運行
uv pip install agentic-rag-mcp
agentic-rag-mcp
```

### 方式 2: 使用 npx

```bash
npx agentic-rag-mcp
```

### 方式 3: 從源碼安裝

```bash
git clone https://github.com/your-org/agentic-rag-mcp.git
cd agentic-rag-mcp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## 配置

創建 `.env` 文件：

```bash
# Qdrant 向量數據庫
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=your-collection

# OpenAI API (用於 embeddings)
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 本地 LLM 配置 (可選)
USE_LOCAL_LLM=true
LOCAL_LLM_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=qwen/qwen3-4b-2507

# 搜索配置
MAX_ITERATIONS=5
TOKEN_BUDGET=15000
```

## 在 Claude Code 中使用

在 `.mcp.json` 中添加：

### 使用 uvx

```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "uvx",
      "args": ["agentic-rag-mcp"],
      "env": {
        "QDRANT_URL": "https://your-cluster.cloud.qdrant.io",
        "QDRANT_API_KEY": "your-api-key",
        "QDRANT_COLLECTION": "your-collection",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### 使用 npx

```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "npx",
      "args": ["agentic-rag-mcp"],
      "env": {
        "QDRANT_URL": "...",
        "QDRANT_API_KEY": "...",
        "QDRANT_COLLECTION": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

### 本地源碼

```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "python",
      "args": ["/path/to/agentic-rag-mcp/src/agentic_rag_mcp/mcp_server.py"],
      "env": {
        "QDRANT_URL": "...",
        "QDRANT_API_KEY": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

## 可用工具

### `agentic-search`

完整的自主搜索，迭代查詢直到找到完整答案：

```
適用於：
- 理解代碼流程和邏輯
- 跨服務追蹤相關代碼
- 追蹤配置和狀態機
- 回答 "X 是如何運作的？" 類型問題
```

參數：
- `query` (必填): 自然語言查詢
- `max_iterations` (可選): 最大迭代次數 (預設: 5)

### `quick-search`

快速單次搜索，適合簡單查詢：

```
適用於：
- 查找特定類或方法
- 查找配置值
- 快速代碼引用
```

參數：
- `query` (必填): 搜索查詢
- `operator` (可選): `hybrid`、`semantic`、`keyword` 或 `exact` (預設: `hybrid`)
- `top_k` (可選): 返回結果數量 (預設: 10)

### `index-files`

索引指定文件到 Qdrant 向量數據庫。讀取 → 分塊 → dense+sparse embedding → upsert。metadata 根據 config.yaml patterns 自動推斷。

```
適用於：
- 編輯文件後立即更新索引
- 索引新增的文件
```

參數：
- `file_paths` (必填): 相對於 codebase root 的文件路徑陣列
- `metadata` (可選): 額外 metadata，會合併到自動推斷的 metadata

### `index-status`

查看當前索引狀態，包括 Qdrant collection 統計和本地狀態。

```
返回：
- Qdrant stats: points_count, by_category 分佈
- 本地狀態: 已索引文件數, 上次索引時間
```

參數：無

### `index-by-pattern`

按 glob pattern 批量索引文件，自動排除 binary/build 目錄，支持增量索引。

```
範例：
- "knowledge/**/*.yaml"
- "docs/**/*.md"
- "src/**/*.cs"
```

參數：
- `pattern` (必填): glob pattern（相對於 codebase root）
- `metadata` (可選): 額外 metadata（例如 `{"category": "knowledge-base"}`）
- `force` (可選): 強制重新索引 (預設: false)

## 架構

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────────┐   ┌──────────────────────┐  │
│  │ Planner │──▶│ Query       │──▶│ Hybrid Search        │  │
│  │ (LLM)   │   │ Builder     │   │ (Qdrant + Keyword)   │  │
│  └─────────┘   └─────────────┘   └──────────────────────┘  │
│       │                                    │                 │
│       │                                    ▼                 │
│       │                          ┌──────────────────────┐   │
│       │                          │ Cross-encoder        │   │
│       │                          │ Reranker             │   │
│       │                          └──────────────────────┘   │
│       │                                    │                 │
│       ▼                                    ▼                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Evidence Store                        │   │
│  │  (Pool + Working Set with diversity sampling)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Synthesizer (LLM)                        │   │
│  │  (Generates structured answer with code refs)         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 模組說明

| 模組 | 說明 |
|------|------|
| `mcp_server.py` | MCP Server 入口 |
| `agentic_search.py` | 主循環控制 |
| `planner.py` | Planner LLM (生成查詢計劃) |
| `synthesizer.py` | Synthesizer LLM (整合最終回答) |
| `hybrid_search.py` | 混合搜索引擎 |
| `reranker.py` | Cross-encoder 重排序 |
| `evidence_store.py` | 兩層證據管理 |
| `query_builder.py` | 查詢模板和構建 |
| `budget.py` | 預算和質量門檻 |
| `models.py` | 資料結構定義 |
| `utils.py` | 工具函數 |
| `indexer/` | 索引服務子包 (index-files, index-status, index-source) |

## 配置選項

### Budget 配置

```python
max_iterations: 5           # 最大迭代次數
total_token_budget: 15000   # 總 token 預算
```

### Quality Gate

```python
min_code_evidence: 2        # 最少 code 類型證據
min_tag_diversity: 2        # 最少 tag 多樣性
require_call_edge: True     # 需要調用關係證據
require_named_entity: True  # 需要具名實體證據
```

## 開發

```bash
# 運行測試
pytest

# 格式化代碼
black src/
ruff check src/ --fix

# 構建包
python -m build

# 發布到 PyPI
twine upload dist/*

# 發布到 npm
npm publish
```

## 依賴

- Python 3.10+
- Qdrant Cloud / Self-hosted
- OpenAI API (embeddings)
- (Optional) Local LLM (OpenAI-compatible API)

## License

MIT License
