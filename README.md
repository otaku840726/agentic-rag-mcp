# Agentic RAG MCP Server

Agentic RAG search for any codebase — multi-hop retrieval with hybrid search, cross-encoder reranking, and iterative query planning.

## Features

- **Multi-hop Search** — iterative search that automatically traces logic across files
- **Hybrid Search** — dense (semantic) + sparse (BM25) with RRF fusion via Qdrant
- **Cross-encoder Reranking** — local cross-encoder model for precision re-ranking
- **Evidence Store** — two-tier evidence management (full pool + diversity working set)
- **Incremental Indexing** — hash-based change detection, only re-indexes modified files
- **Model Mismatch Detection** — auto-recreates collection when embedding model changes
- **Multi-Provider** — OpenAI, OpenRouter, Gemini, local LLM (Ollama/vLLM/LM Studio)
- **Per-Component Config** — mix providers freely (e.g. embedding=openrouter, planner=local)

## Quick Start

### 1. Add to `.mcp.json`

```json
{
  "mcpServers": {
    "agentic-rag": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/your-org/agentic-rag-mcp.git",
        "agentic-rag-mcp"
      ],
      "env": {
        "QDRANT_URL": "https://your-cluster.cloud.qdrant.io",
        "QDRANT_API_KEY": "your-qdrant-api-key",
        "QDRANT_COLLECTION": "your-collection-name",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### 2. Index your codebase

Use the `index-by-pattern` tool in Claude Code:

```
index-by-pattern("**/*.cs")
index-by-pattern("docs/**/*.md")
index-by-pattern("knowledge/**/*.yaml")
```

### 3. Search

```
agentic-search("How does the deposit flow work?")
quick-search("DepositService")
```

## Configuration

All configuration is done via **environment variables** in `.mcp.json`. The bundled `config.yaml` provides sensible defaults; env vars override them.

### Required

| Env Var | Description |
|---------|-------------|
| `QDRANT_URL` | Qdrant Cloud or self-hosted URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Collection name |

Plus at least one provider API key (see below).

### Providers

All providers use OpenAI-compatible API format. Only configure the ones you use.

| Provider | Env Vars |
|----------|----------|
| **openai** | `OPENAI_API_KEY` |
| **openrouter** | `OPENROUTER_API_KEY` |
| **gemini** | `GEMINI_API_KEY` |
| **local** (Ollama/vLLM/LM Studio) | `LOCAL_LLM_URL`, `LOCAL_LLM_API_KEY` (optional) |

### Component Overrides

Each component (embedding, planner, synthesizer, judge) can independently use any provider and model. Defaults: `openai` + `gpt-4o-mini`.

| Env Var | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openai` | Provider for embeddings |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_MAX_TOKENS` | `8191` | Max tokens per chunk (auto-split with tiktoken) |
| `PLANNER_PROVIDER` | `openai` | Provider for query planner |
| `PLANNER_MODEL` | `gpt-4o-mini` | Planner model |
| `SYNTHESIZER_PROVIDER` | `openai` | Provider for answer synthesizer |
| `SYNTHESIZER_MODEL` | `gpt-4o-mini` | Synthesizer model |
| `JUDGE_PROVIDER` | `openai` | Provider for satisfaction judge |
| `JUDGE_MODEL` | `gpt-4o-mini` | Judge model |

### Example: All Local

```json
"env": {
  "LOCAL_LLM_URL": "http://127.0.0.1:11434/v1",
  "EMBEDDING_PROVIDER": "local",
  "EMBEDDING_MODEL": "qwen3-embedding:4b",
  "EMBEDDING_MAX_TOKENS": "32768",
  "PLANNER_PROVIDER": "local",
  "PLANNER_MODEL": "qwen3:4b",
  "SYNTHESIZER_PROVIDER": "local",
  "SYNTHESIZER_MODEL": "qwen3:4b",
  "JUDGE_PROVIDER": "local",
  "JUDGE_MODEL": "qwen3:4b"
}
```

### Example: Mixed (OpenRouter embedding + local LLM)

```json
"env": {
  "OPENROUTER_API_KEY": "sk-or-v1-...",
  "LOCAL_LLM_URL": "http://127.0.0.1:11434/v1",
  "EMBEDDING_PROVIDER": "openrouter",
  "EMBEDDING_MODEL": "qwen/qwen3-embedding-8b",
  "PLANNER_PROVIDER": "local",
  "PLANNER_MODEL": "qwen3:4b",
  "SYNTHESIZER_PROVIDER": "local",
  "SYNTHESIZER_MODEL": "qwen3:4b",
  "JUDGE_PROVIDER": "local",
  "JUDGE_MODEL": "qwen3:4b"
}
```

## MCP Tools

### `agentic-search`

Multi-hop agentic search — iterates until the answer is complete.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `query` | Yes | — | Natural language query |
| `max_iterations` | No | 5 | Max search iterations |

### `quick-search`

Single-pass search for simple lookups.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `query` | Yes | — | Search query |
| `operator` | No | `hybrid` | `hybrid` / `semantic` / `keyword` / `exact` |
| `top_k` | No | 10 | Number of results |

### `index-files`

Index specific files. Auto-infers metadata from file paths.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `file_paths` | Yes | — | File paths relative to codebase root |
| `metadata` | No | — | Extra metadata to merge |

### `index-status`

Show current index status: Qdrant stats, embedding model, file counts.

### `index-by-pattern`

Batch index by glob pattern. Incremental — skips unchanged files.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `pattern` | Yes | — | Glob pattern (e.g. `**/*.cs`) |
| `metadata` | No | — | Extra metadata to merge |
| `force` | No | `false` | Force re-index unchanged files |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       MCP Server                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Planner  │───>│ Query Builder│───>│ Hybrid Search  │  │
│  │ (LLM)    │    └──────────────┘    │ (Qdrant)       │  │
│  └──────────┘                        └────────────────┘  │
│       │                                     │            │
│       │                              ┌──────────────┐    │
│       │                              │ Reranker     │    │
│       │                              │ (CrossEnc)   │    │
│       │                              └──────────────┘    │
│       │                                     │            │
│       v                                     v            │
│  ┌───────────────────────────────────────────────────┐   │
│  │              Evidence Store                        │   │
│  │         Pool + Diversity Working Set               │   │
│  └───────────────────────────────────────────────────┘   │
│                          │                                │
│                          v                                │
│  ┌───────────────────────────────────────────────────┐   │
│  │           Synthesizer (LLM)                        │   │
│  └───────────────────────────────────────────────────┘   │
│                                                           │
│  ┌───────────────────────────────────────────────────┐   │
│  │           Indexer (embed + upsert to Qdrant)       │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `mcp_server.py` | MCP server entry point, tool registration |
| `provider.py` | Unified LLM/embedding client factory |
| `agentic_search.py` | Main agentic loop controller |
| `planner.py` | Planner + Judge LLM |
| `synthesizer.py` | Answer synthesizer LLM |
| `hybrid_search.py` | Dense + sparse hybrid search engine |
| `reranker.py` | Cross-encoder reranking |
| `evidence_store.py` | Two-tier evidence management |
| `query_builder.py` | Query templates and builder |
| `budget.py` | Token budget and quality gate |
| `models.py` | Data structures |
| `indexer/` | Indexing subpackage (chunker, embedder, qdrant ops) |

## Embedding Model Change

When you change `EMBEDDING_MODEL`, the server automatically:
1. Detects model name or dimension mismatch
2. Recreates the Qdrant collection
3. Clears the index state
4. Returns a warning in the next index response

All files will need to be re-indexed after a model change.

## Development

```bash
git clone https://github.com/your-org/agentic-rag-mcp.git
cd agentic-rag-mcp
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

pytest
black src/
ruff check src/ --fix
```

## License

MIT