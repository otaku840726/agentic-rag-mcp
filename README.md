# Agentic RAG MCP Server

An MCP server that gives AI assistants **deep, structural understanding of large codebases** by combining hybrid vector search with an optional code knowledge graph.

- **Hybrid Search** — Dense (semantic) + sparse (BM25) fusion with RRF ranking and cross-encoder reranking
- **Agentic Multi-Hop** — Iterative planner → analyst → judge → synthesizer loop for complex, cross-service questions
- **Knowledge Graph** — Optional Neo4j/AuraDB storing AST-extracted symbols and call/inheritance/reference relationships
- **Graph-Enhanced RAG** — Automatically expands vector search results with structurally-adjacent code from the graph
- **AST-Aware Chunking** — Tree-sitter (all languages) + Roslyn (.NET) + Spoon (Java) extract symbol-level chunks with full metadata
- **Multi-Provider** — OpenAI, Gemini, Voyage AI, Vertex AI, OpenRouter, Ollama/vLLM — each component independently configurable

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Client                              │
│              (Claude Code / Claude Desktop / etc.)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │ MCP (stdio)
┌───────────────────────────────▼─────────────────────────────────┐
│                         MCP Server                              │
│                                                                 │
│   ┌─────────────────────┐       ┌──────────────────────────┐   │
│   │   agentic-search    │       │      quick-search        │   │
│   │   (multi-hop)       │       │   (single-pass, fast)    │   │
│   └──────────┬──────────┘       └─────────────┬────────────┘   │
│              │                                │                 │
│              └─────────────┬──────────────────┘                 │
│                            │                                    │
│              ┌─────────────▼──────────────┐                     │
│              │        HybridSearch        │                     │
│              │  dense + sparse BM25 + RRF │                     │
│              └─────────────┬──────────────┘                     │
│                            │                                    │
│              ┌─────────────▼──────────────┐                     │
│              │   CrossEncoder Reranker    │                     │
│              │  (ms-marco-MiniLM-L-6-v2)  │                     │
│              └─────────────┬──────────────┘                     │
│                            │                                    │
│              ┌─────────────▼──────────────┐                     │
│              │    GraphSearchEnhancer     │  (optional Neo4j)   │
│              │  neighbor expansion +      │                     │
│              │  call-edge evidence        │                     │
│              └─────────────┬──────────────┘                     │
│                            │                                    │
│   ┌────────────────────────▼──────────────────────────────┐    │
│   │              Agentic LLM Pipeline                     │    │
│   │   Planner → EnsembleAnalyst → Judge → Synthesizer     │    │
│   │           (iterates up to max_iterations)             │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                 │
└────────────────┬────────────────────────────┬───────────────────┘
                 │                            │
    ┌────────────▼────────────┐  ┌────────────▼────────────┐
    │         Qdrant          │  │          Neo4j           │
    │   dense + sparse        │  │  Symbol + File nodes     │
    │   vectors + payload     │  │  CALLS / INHERITS / ...  │
    └─────────────────────────┘  └─────────────────────────┘
```

### Indexing Pipelines

When you run `index-codebase`, two parallel pipelines execute:

```
index-codebase(directory)
        │
        ├── Pipeline A — Qdrant (ALL non-binary files)
        │       │
        │       ├─ Language-aware chunking
        │       │    ├─ .cs .java .py .ts .go ...  → Tree-sitter AST chunks
        │       │    ├─ .md .mdx                   → Markdown section chunks
        │       │    ├─ .yaml .yml .json            → YAML/JSON structural chunks
        │       │    └─ everything else             → plain text chunks
        │       │
        │       ├─ Hash-based change detection (only re-embeds changed files)
        │       ├─ Dense embedding  (your EMBEDDING_PROVIDER / EMBEDDING_MODEL)
        │       ├─ Sparse BM25      (Qdrant built-in or fastembed SPLADE)
        │       └─ Upsert → Qdrant collection
        │
        └── Pipeline B — AuraDB / Neo4j (project-level AST analysis)
                │
                ├─ Detect .sln / pom.xml files whose source files changed
                ├─ .NET solution   → Roslyn SemanticModel (via Docker)
                │    └─ CALLS, CREATES, OVERRIDES, REFERENCES (FQN-precise)
                ├─ Java project    → Spoon analyzer (via Docker)
                │    └─ MEMBER_OF, INHERITS, IMPLEMENTS, IMPORTS, USES_TYPE
                └─ Upsert Symbol + File nodes → Neo4j with project-scoped edges
```

Only changed or new files are reprocessed. Deleted files are cleaned from both databases. Re-running `index-codebase` is safe and idempotent.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `agentic-search` | Multi-hop search with iterative planning, ensemble analysis, quality judging, and final synthesis. Best for complex "how does X work?" questions |
| `quick-search` | Fast single-pass hybrid/semantic/keyword/exact search with lightweight reranking |
| `analyze-codebase` | AST analysis of a directory → JSON artifacts with symbols & relationships (step 1 of manual pipeline) |
| `index-codebase` | Incremental indexing into Qdrant + Neo4j with hash-based change detection (runs both pipelines) |
| `index-status` | Collection stats, indexed file count, embedding model info, last index time |
| `debug-env` | Dump MCP server environment and configuration for troubleshooting |
| `graph-neighbors` | Query symbol relationships: callers, callees, inheritance, implementations (requires Neo4j) |
| `graph-query` | Execute raw Cypher queries on the knowledge graph (requires Neo4j) |
| `graph-status` | Node/edge counts by type with per-project breakdown (requires Neo4j) |
| `graph-list-projects` | List all projects indexed in the graph database (requires Neo4j) |

> `graph-*` tools are only registered when `NEO4J_ENABLED=true`.

---

## Supported Providers

Every component (embedding, analyst, planner, synthesizer, judge) can independently use any provider.
All providers use the OpenAI-compatible API format.

| Provider | Key Variable(s) | Best For |
|----------|----------------|----------|
| `openai` | `OPENAI_API_KEY` | General purpose; GPT-4o default |
| `voyage` | `VOYAGE_API_KEY` | **Embedding** — `voyage-code-3` is state-of-the-art for code retrieval |
| `gemini` | `GEMINI_API_KEY` | Google AI Studio — high context window, cost-effective |
| `vertex` | `VERTEX_API_KEY`, `VERTEX_PROJECT_ID` | Google Cloud Vertex AI (enterprise IAM) |
| `openrouter` | `OPENROUTER_API_KEY` | Multi-model gateway — access 100+ models with one key |
| `local` | `LOCAL_LLM_URL`, `LOCAL_LLM_API_KEY` | Ollama / vLLM / LM Studio — fully offline |

### Embedding Provider Recommendations

| Provider + Model | Dimension | Context | Notes |
|-----------------|-----------|---------|-------|
| `voyage` / `voyage-code-3` | 1024 | 16 000 | Best-in-class code retrieval |
| `openai` / `text-embedding-3-large` | 3072 | 8 191 | Strong general purpose |
| `openai` / `text-embedding-3-small` | 1536 | 8 191 | Default; fast and cheap |
| `gemini` / `gemini-embedding-001` | 3072 | 2 048 | Strong multilingual |
| `local` / `nomic-embed-text` | 768 | 8 192 | Fully offline via Ollama |

---

## Installation

### As MCP Server (recommended)

Add to your MCP client config (Claude Code `.mcp.json`, Claude Desktop, etc.):

```json
{
  "mcpServers": {
    "agentic-rag": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/your-org/agentic-rag-mcp.git",
        "agentic-rag-mcp"
      ],
      "env": {
        "QDRANT_URL": "https://your-cluster.cloud.qdrant.io",
        "QDRANT_API_KEY": "your-qdrant-key",
        "QDRANT_COLLECTION": "my-codebase",
        "VOYAGE_API_KEY": "your-voyage-key",
        "EMBEDDING_PROVIDER": "voyage",
        "EMBEDDING_MODEL": "voyage-code-3",
        "EMBEDDING_IDENTIFIER": "voyage-code-3",
        "EMBEDDING_MAX_TOKENS": "16000",
        "EMBEDDING_BATCH_SIZE": "128",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Local Development

```bash
git clone <repo-url>
cd agentic-rag-mcp

# Using uv (recommended)
uv sync

# Or using pip
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Copy and configure your environment:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

---

## Configuration

All configuration is via environment variables. The full reference is in `.env.example`.

### Required

| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant Cloud or self-hosted URL (e.g. `https://xxx.cloud.qdrant.io`) |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Vector collection name (created automatically on first index) |

Plus at least one LLM provider key and one embedding provider key.

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openai` | Provider name (see Supported Providers) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `EMBEDDING_IDENTIFIER` | `openai-3-small` | Stable ID for the embedding space — changing this triggers full re-index |
| `EMBEDDING_MAX_TOKENS` | `512` | Max tokens per chunk sent to embedding API |
| `EMBEDDING_BATCH_SIZE` | `100` | Chunks per API batch call |

> **Note**: If you change `EMBEDDING_IDENTIFIER`, the server detects the mismatch and automatically recreates the Qdrant collection and re-indexes from scratch.

### LLM Components

Each component can use a different provider and model:

| Component | Provider Env | Model Env | Default Model |
|-----------|-------------|-----------|---------------|
| Analyst | `ANALYST_PROVIDER` | `ANALYST_MODEL` | `gpt-4o-mini` |
| Planner | `PLANNER_PROVIDER` | `PLANNER_MODEL` | `gpt-4o-mini` |
| Synthesizer | `SYNTHESIZER_PROVIDER` | `SYNTHESIZER_MODEL` | `gpt-4o-mini` |
| Judge | `JUDGE_PROVIDER` | `JUDGE_MODEL` | `gpt-4o-mini` |

Each also accepts `*_MAX_TOKENS` and `*_TEMPERATURE` overrides (e.g. `ANALYST_MAX_TOKENS=6000`).

### Agentic Search Budget

| Variable | Default | Description |
|----------|---------|-------------|
| `ENSEMBLE_ENABLED` | `true` | Run 3-persona parallel analyst ensemble (higher quality, more tokens) |

Budget and quality gates are configured in `config.yaml` (`budget` and `quality_gate` sections).

### Sparse Search

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARSE_MODE` | `qdrant-bm25` | `qdrant-bm25` (server-side, fast) \| `splade` (local model) \| `disabled` |
| `BM25_VOCAB_SIZE` | `30000` | BM25 vocabulary size |
| `SPLADE_MODEL` | `prithivida/Splade_PP_en_v1` | SPLADE model when mode=splade |

### Neo4j / Knowledge Graph (Optional)

| Variable | Description |
|----------|-------------|
| `NEO4J_ENABLED` | `true` to enable graph features (default: `false`) |
| `NEO4J_URI` | `bolt://localhost:7687` or `neo4j+s://xxx.databases.neo4j.io` |
| `NEO4J_USERNAME` | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Neo4j password |
| `NEO4J_DATABASE` | Database name (default: `neo4j`) |
| `GRAPH_PROJECT` | Logical project key — multiple codebases can share one DB |

Setup options:
1. **AuraDB Free** — https://console.neo4j.io → Create Free Instance → copy URI + password
2. **Docker** — `docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5`

### Index Filtering

The indexer uses an extension whitelist. Customize without code changes:

| Variable | Example | Description |
|----------|---------|-------------|
| `INDEX_EXTRA_EXTENSIONS` | `.tf,.hcl,.toml` | Add extensions to the whitelist |
| `INDEX_REMOVE_EXTENSIONS` | `.json,.xml` | Remove extensions from the whitelist |
| `INDEX_EXTRA_EXCLUDE_DIRS` | `tmp,logs,artifacts` | Additional directories to skip |
| `INDEX_EXTRA_EXCLUDE_FILES` | `.generated.cs,.auto.ts` | Additional filename suffix patterns to skip |

---

## Usage

### 1. Index Your Codebase

```
# Full incremental index (recommended — hash-based, safe to re-run)
index-codebase(directory="/path/to/your/codebase")

# Check indexing status
index-status()
```

The first run takes a few minutes depending on codebase size and embedding API speed. Subsequent runs only process changed files.

### 2. Search

```
# Simple lookup — find a class, method, or config value
quick-search(query="DepositService class definition")
quick-search(query="tblDeposit entity fields", operator="hybrid")

# Complex multi-hop analysis
agentic-search(query="How does the deposit callback flow work end-to-end?")
agentic-search(query="What services are affected if I add a field to tblDeposit?")
```

#### `agentic-search` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Search query in natural language |
| `max_iterations` | integer | `5` | Maximum search iterations |
| `llm_provider` | string | config default | Override LLM provider for all components (analyst / planner / synthesizer). See [Supported Providers](#supported-providers) |
| `llm_model` | string | config default | Override LLM model for all components |

`llm_provider` and `llm_model` can be used independently — specify one or both. When not set, `.env` / `config.yaml` defaults are used and the call reuses the shared singleton agent (zero overhead).

```
# Test with a different model, keep the same provider
agentic-search(query="...", llm_model="gpt-4o")

# Switch to a stronger model for a complex question
agentic-search(query="...", llm_provider="openrouter", llm_model="anthropic/claude-3.5-sonnet")

# Use local Ollama for fast offline testing
agentic-search(query="...", llm_provider="local", llm_model="qwen2.5-coder:32b")

# No override — uses .env defaults (singleton, no extra init cost)
agentic-search(query="...")
```

`quick-search` supports four operators:

| Operator | Description |
|----------|-------------|
| `hybrid` | Dense + sparse BM25 fusion via RRF (default, recommended) |
| `semantic` | Dense embedding similarity only |
| `keyword` | BM25 sparse only (exact term matching) |
| `exact` | String match in payload |

### 3. Graph Queries (Neo4j enabled)

```
# Who calls PublishMerchantCallBack?
graph-neighbors(symbol="PublishMerchantCallBack", relationship_types=["CALLS"])

# What does DepositService depend on?
graph-neighbors(symbol="DepositService", depth=2)

# Which files reference tblDeposit?
graph-query(query="""
  MATCH (s:Symbol)-[:REFERENCES]->(t:Symbol)
  WHERE t.name CONTAINS 'tblDeposit'
  RETURN DISTINCT s.file_path, s.name
  LIMIT 30
""")

# All implementations of IDepositService
graph-neighbors(symbol="IDepositService", relationship_types=["IMPLEMENTS"])

# Graph statistics
graph-status()
graph-list-projects()
```

---

## Knowledge Graph Schema

When Neo4j is enabled, the graph stores code structure extracted by Roslyn and Tree-sitter:

### Node Types

| Node | Properties | Source |
|------|-----------|--------|
| `Symbol` | `fqn`, `name`, `kind`, `file_path`, `namespace`, `source`, `project` | Roslyn + Tree-sitter |
| `File` | `path`, `service`, `layer`, `category`, `language`, `project` | Tree-sitter |

`kind` values: `class`, `method`, `interface`, `property`, `enum`, `field`, `constructor`

### Edge Types

| Type | Meaning | Source |
|------|---------|--------|
| `MEMBER_OF` | Method/property belongs to class | Tree-sitter |
| `INHERITS` | Class extends class | Tree-sitter |
| `IMPLEMENTS` | Class implements interface | Tree-sitter |
| `IMPORTS` | File imports namespace/type | Tree-sitter |
| `CALLS` | Method calls method | Roslyn (FQN-precise) |
| `USES_TYPE` | Property/parameter type reference | Tree-sitter |
| `REFERENCES` | General symbol reference | Roslyn |
| `CREATES` | Constructor invocation | Roslyn |
| `OVERRIDES` | Method overrides base method | Roslyn |
| `DEFINED_IN` | Symbol defined in file | Both |

> **DI limitation**: Controller → Service calls via dependency injection interfaces (e.g. `IDepositService`) appear as `CALLS` to the interface. Use `IMPLEMENTS` edges to trace to concrete implementations.

---

## Project Structure

```
src/agentic_rag_mcp/
├── mcp_server.py          # MCP tool registration and dispatch
├── agentic_search.py      # Multi-hop search orchestration (main loop)
├── hybrid_search.py       # Dense + sparse hybrid search + RRF fusion
├── graph_search.py        # Graph-enhanced evidence expansion (Neo4j)
├── reranker.py            # Cross-encoder reranking (ms-marco-MiniLM)
├── planner.py             # Query planning — generates sub-queries (LLM)
├── analyst.py             # Evidence analysis — EnsembleAnalyst (LLM)
├── synthesizer.py         # Final answer synthesis (LLM)
├── budget.py              # Stop-condition checker (token / iteration budget)
├── evidence_store.py      # Evidence card pool with LRU eviction
├── query_builder.py       # Query expansion and variation generation
├── search_logger.py       # Search trace logging
├── models.py              # Data models (EvidenceCard, SearchResult, etc.)
├── provider.py            # Multi-provider LLM/embedding client factory
├── config.yaml            # Default configuration (all overridable via env)
├── utils.py               # Shared utilities (fingerprint, NER, tagging)
│
└── indexer/
    ├── core.py                # IndexerService — orchestrates both pipelines
    ├── ast_chunker.py         # Tree-sitter AST analysis & semantic chunking
    ├── chunker.py             # File-level chunker dispatcher
    ├── markdown_analyzer.py   # Markdown section-aware chunking
    ├── yaml_analyzer.py       # YAML/JSON structural chunking
    ├── docker_analyzer.py     # Roslyn (.NET) and Spoon (Java) via Docker
    ├── two_phase_analysis.py  # Two-phase AST analysis orchestration
    ├── graph_store.py         # Neo4j client (read/write + Cypher helpers)
    ├── embedder.py            # Embedding generation with batch + cache
    ├── embedding_cache.py     # Disk-based embedding cache (avoids re-embedding)
    ├── qdrant_ops.py          # Qdrant upsert/search/delete operations
    ├── sparse_embedder.py     # BM25/SPLADE sparse vector generation
    ├── bm25_tokenizer.py      # BM25 tokenizer
    ├── project_detector.py    # Detect project type (.sln, pom.xml, etc.)
    └── analyzer.py            # Analyzer factory (Tree-sitter / Roslyn / Spoon)
```

---

## Dependencies

### Core

| Package | Purpose |
|---------|---------|
| `mcp` | Model Context Protocol SDK |
| `qdrant-client` | Vector database client |
| `openai` | LLM + embedding API client (OpenAI-compatible) |
| `sentence-transformers` | Cross-encoder reranking model |
| `fastembed` | Sparse SPLADE embeddings (optional) |
| `tree-sitter` + grammars | AST parsing for 15+ languages |
| `pydantic` | Data validation |
| `pyyaml` | Config file parsing |
| `python-dotenv` | `.env` file loading |
| `xxhash` | Fast file hash for change detection |

### Optional

| Package | Install | Purpose |
|---------|---------|---------|
| `neo4j` | `pip install agentic-rag-mcp[graph]` | Knowledge graph client |
| `google-auth` | `pip install google-auth` | Vertex AI ADC credentials |

### Supported Languages (Tree-sitter)

C#, Java, Python, TypeScript, JavaScript, Go, Kotlin, Rust, Swift, HTML, CSS, JSON, YAML, SQL, Bash

---

## Configuration Quick Reference

```bash
# Minimal setup (OpenAI embedding + LLM)
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=my-codebase
OPENAI_API_KEY=sk-...

# Voyage AI embedding (best for code)
VOYAGE_API_KEY=...
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-code-3
EMBEDDING_IDENTIFIER=voyage-code-3
EMBEDDING_MAX_TOKENS=16000
EMBEDDING_BATCH_SIZE=128

# Mix providers: Voyage embed + local LLM for analysis
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=not-needed
ANALYST_PROVIDER=local
ANALYST_MODEL=qwen2.5-coder:32b
PLANNER_PROVIDER=local
PLANNER_MODEL=qwen2.5-coder:32b
SYNTHESIZER_PROVIDER=local
SYNTHESIZER_MODEL=qwen2.5-coder:32b

# Enable knowledge graph
NEO4J_ENABLED=true
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
GRAPH_PROJECT=my-project
```

---

## License

MIT
