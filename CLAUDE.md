# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogniDoc is a Hybrid RAG (Vector + GraphRAG) document assistant that converts multi-format documents into a searchable knowledge base with intelligent query routing.

**Key Design Decisions:**
- No LangChain/LlamaIndex - direct Qdrant and Ollama integration for fine-grained control
- Multi-provider LLM support (Gemini default, Ollama, OpenAI, Anthropic)
- Parent-child chunk hierarchy for context-aware retrieval
- Custom semantic chunking with breakpoint detection
- Custom ReAct agent (~300 lines) instead of LangGraph for fine-grained control

## Commands

```bash
# Setup and installation (uses uv package manager)
make install          # Create venv and install dependencies
make sync             # Sync environment with lock file

# IMPORTANT: If path contains spaces, use UV_LINK_MODE=copy
UV_LINK_MODE=copy uv sync --all-extras
UV_LINK_MODE=copy uv pip install -e ".[all,dev]"

# Code quality (note: targets root *.py, not src/)
make format           # Format with black
make lint             # Run pylint
make refactor         # Format + lint

# For src/ code, run directly:
uv run black src/cognidoc/
uv run pylint src/cognidoc/

# Install dev dependencies (pytest, black, pylint, mypy)
uv sync --group dev

# Optional dependency groups (pyproject.toml):
# [ui]         - Gradio interface (gradio, plotly, pandas)
# [yolo]       - YOLO detection (torch, ultralytics)
# [ollama]     - Local Ollama inference
# [gemini]     - Google Gemini provider
# [openai]     - OpenAI provider
# [anthropic]  - Anthropic provider
# [cloud]      - All cloud providers
# [conversion] - Office document conversion
# [wizard]     - Interactive setup wizard (questionary)
# [all]        - Everything above
# [dev]        - Development tools
```

### First-Time Setup

```bash
# Interactive wizard (recommended) - guides through provider setup, API keys, ingestion
python -m cognidoc.setup
```

### Installing as a Package (in another project)

To use cognidoc in a separate project (not the development repo):

```bash
# Create a new project
mkdir my-doc-project && cd my-doc-project
python -m venv .venv && source .venv/bin/activate

# Install cognidoc with all dependencies
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Add documents and run
mkdir -p data/sources
cp /path/to/documents/* data/sources/
cognidoc ingest ./data/sources --llm gemini --embedding ollama
cognidoc serve --port 7860
```

**Note:** The `[all]` extra includes UI (gradio, plotly, pandas), YOLO, Ollama, cloud providers, and conversion tools.

### CLI Commands (`cognidoc`)

```bash
cognidoc init --schema --prompts   # Copy templates (non-interactive)
cognidoc ingest ./data/sources --llm gemini --embedding ollama
cognidoc query "What is X?"
cognidoc serve --port 7860 --share
cognidoc info                      # Show configuration
```

**Note:** `cognidoc init` only copies template files. For interactive setup, use `python -m cognidoc.setup`.

### Direct Module Execution

```bash
# Run ingestion pipeline directly (more options)
python -m cognidoc.run_ingestion_pipeline --vision-provider ollama

# Launch chat interface
python -m cognidoc.cognidoc_app
python -m cognidoc.cognidoc_app --no-rerank    # Faster, skip LLM reranking
python -m cognidoc.cognidoc_app --share        # Create public link
```

### Quick Test via Python API

```python
from cognidoc import CogniDoc
doc = CogniDoc(llm_provider='gemini', embedding_provider='ollama')
doc.ingest(skip_schema_wizard=True)  # Use existing schema
result = doc.query('Your question here')
print(result.answer)
```

### Pipeline Skip Flags

Resume from specific stage:
```bash
--skip-conversion     # Skip non-PDF to PDF conversion
--skip-pdf            # Skip PDF to image conversion
--skip-yolo           # Skip YOLO detection
--skip-extraction     # Skip text/table extraction
--skip-descriptions   # Skip image descriptions
--skip-chunking       # Skip semantic chunking
--skip-embeddings     # Skip embedding generation
--skip-indexing       # Skip vector index building
--skip-graph          # Skip knowledge graph building
--force-reembed       # Re-embed all (ignore cache)
```

### Performance Tuning Flags (M2/M3 Macs)

```bash
--yolo-batch-size 2       # YOLO batch size (default: 2, increase for more VRAM)
--no-yolo-batching        # Disable YOLO batching (sequential processing)
--entity-max-concurrent 4 # Concurrent LLM calls for entity extraction (default: auto, 2-8 based on CPU)
--no-async-extraction     # Disable async entity extraction (sequential)
```

## Architecture

### Ingestion Pipeline

```
Documents → PDF Conversion → Images (600 DPI) → YOLO Detection
                                                      ↓
                                    Text/Table/Image Extraction
                                                      ↓
                                            Semantic Chunking
                                       (Parent + Child hierarchy)
                                                      ↓
                        ┌─────────────────────────────┴─────────────────────────────┐
                        ↓                                                           ↓
               Vector Embeddings                                        Entity/Relationship
               (Qdrant + BM25)                                              Extraction
                        ↓                                                           ↓
                        └─────────────────────────────┬─────────────────────────────┘
                                                      ↓
                                            Hybrid Retriever
```

**Ingestion Report:** At pipeline completion, a comprehensive report is displayed showing:
- Documents processed, PDFs converted, pages generated
- YOLO detection stats (text/table/picture regions)
- Content extraction results
- Chunking & embeddings (child chunks, cache hits, parent chunks)
- GraphRAG stats (entities, relationships, nodes, edges, communities)
- Timing breakdown for each stage

### Query Processing Flow

```
                                    User Query
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │     Query Rewriter            │
                        │  (adds conversation context)  │
                        └───────────────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                ┌───────────────┐               ┌───────────────┐
                │   Classifier  │               │  Complexity   │
                │  (query type) │               │   Evaluator   │
                └───────────────┘               └───────────────┘
                        │                               │
                        ▼                               ▼
                   Query Type                    Complexity Score
              (factual/relational/               (0.0 - 1.0)
               exploratory/procedural)
                        │                               │
                        └───────────────┬───────────────┘
                                        │
                            ┌───────────┴───────────┐
                            │   score >= 0.55 ?     │
                            └───────────┬───────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │ NO                │                   │ YES
                    ▼                   ▼                   ▼
        ┌───────────────────┐  ┌───────────────┐  ┌───────────────────┐
        │    FAST PATH      │  │ ENHANCED PATH │  │    AGENT PATH     │
        │  (Standard RAG)   │  │ (score 0.35+) │  │   (ReAct Loop)    │
        └───────────────────┘  └───────────────┘  └───────────────────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌───────────────────────────────────────┐  ┌───────────────────┐
        │          Hybrid Retriever             │  │  THINK → ACT →    │
        │  (Vector weight + Graph weight based  │  │  OBSERVE → REFLECT│
        │   on query type)                      │  │  (max 7 steps)    │
        └───────────────────────────────────────┘  └───────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────┐               ┌───────────────────────────┐
        │  LLM Generation   │               │     9 Agent Tools         │
        │  (final answer)   │               │  (retrieve, synthesize,   │
        └───────────────────┘               │   compare, verify, etc.)  │
                    │                       └───────────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                                ┌───────────────┐
                                │    Response   │
                                │ (same language│
                                │  as query)    │
                                └───────────────┘
```

### Source Layout

Source code is in `src/cognidoc/` but installs as `cognidoc` package. When running modules:
- File path: `src/cognidoc/cognidoc_app.py`
- Module import: `from cognidoc import CogniDoc`
- CLI execution: `python -m cognidoc.cognidoc_app`

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/cognidoc/api.py` | Main CogniDoc class (public API) |
| `src/cognidoc/run_ingestion_pipeline.py` | Async pipeline orchestrator |
| `src/cognidoc/cognidoc_app.py` | Gradio chat with FastAPI static file serving |
| `src/cognidoc/hybrid_retriever.py` | Vector + Graph fusion with query orchestration and caching |
| `src/cognidoc/knowledge_graph.py` | NetworkX graph with Louvain community detection and batch embeddings |
| `src/cognidoc/query_orchestrator.py` | LLM-based query classification and routing |
| `src/cognidoc/complexity.py` | Query complexity evaluation for agentic routing |
| `src/cognidoc/agent.py` | ReAct agent with parallel reflection for complex queries |
| `src/cognidoc/agent_tools.py` | Tool implementations for agent (9 tools) |
| `src/cognidoc/helpers.py` | Query rewriting, parsing, conversation context |
| `src/cognidoc/schema_wizard.py` | Interactive/auto schema generation for GraphRAG |
| `src/cognidoc/constants.py` | Central config (paths, thresholds, model names) |
| `src/cognidoc/utils/llm_client.py` | Singleton LLM client (Gemini default) |
| `src/cognidoc/utils/llm_providers.py` | Multi-provider abstraction layer |
| `src/cognidoc/utils/rag_utils.py` | Document, VectorIndex, KeywordIndex classes |
| `src/cognidoc/utils/embedding_providers.py` | Embedding providers with async batch and connection pooling |
| `src/cognidoc/utils/tool_cache.py` | Persistent SQLite cache for tool results |
| `src/cognidoc/utils/metrics.py` | Performance metrics with SQLite storage |
| `src/cognidoc/create_embeddings.py` | Batched async embedding generation |
| `src/cognidoc/convert_pdf_to_image.py` | Parallel PDF to image conversion |
| `src/cognidoc/extract_objects_from_image.py` | YOLO detection with batch inference |
| `src/cognidoc/extract_entities.py` | Async entity/relationship extraction with adaptive concurrency |

### Query Routing

Query types determine vector/graph weight balance:

| Query Type | Vector Weight | Graph Weight | Use Case |
|------------|--------------|--------------|----------|
| **FACTUAL** | 70% | 30% | Simple fact lookup ("What is X?") |
| **RELATIONAL** | 20% | 80% | Entity relationships ("How does X relate to Y?") |
| **EXPLORATORY** | 0% | 100% | Broad topics ("Tell me about X") |
| **PROCEDURAL** | 80% | 20% | Step-by-step ("How to do X?") |

Skip logic: if weight < 15%, that retriever is skipped entirely.

### Retrieval Architecture Deep Dive

The retrieval system uses a **3-level fusion architecture**:

```
                              User Query
                                  │
                    ┌─────────────┴─────────────┐
                    │      Query Routing        │
                    │  (type, skip_vector,      │
                    │   skip_graph)             │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       │                       ▼
┌─────────────────────┐           │           ┌─────────────────────┐
│  VECTOR RETRIEVAL   │           │           │  GRAPH RETRIEVAL    │
│  (hybrid_retriever) │           │           │  (graph_retriever)  │
└─────────┬───────────┘           │           └─────────┬───────────┘
          │                       │                     │
    ┌─────┴─────┐                 │           ┌────────┴────────┐
    │           │                 │           │                 │
    ▼           ▼                 │           ▼                 ▼
┌───────┐  ┌───────┐              │    ┌───────────┐    ┌────────────┐
│ DENSE │  │SPARSE │              │    │  Entity   │    │ Community  │
│Vector │  │ BM25  │              │    │ Matching  │    │ Summaries  │
└───┬───┘  └───┬───┘              │    └─────┬─────┘    └─────┬──────┘
    │          │                  │          │                │
    └────┬─────┘                  │          └────────┬───────┘
         │                        │                   │
         ▼                        │                   ▼
┌─────────────────┐               │         ┌────────────────┐
│ LEVEL 1: HYBRID │               │         │ Graph Context  │
│ FUSION (RRF)    │               │         │ + Entities     │
│ α=0.6 default   │               │         │ + Relations    │
└────────┬────────┘               │         └───────┬────────┘
         │                        │                 │
         ▼                        │                 │
┌─────────────────┐               │                 │
│ Child → Parent  │               │                 │
│ Deduplication   │               │                 │
└────────┬────────┘               │                 │
         │                        │                 │
         ▼                        │                 │
┌─────────────────┐               │                 │
│ Reranking (LLM  │               │                 │
│ or CrossEncoder)│               │                 │
│ Top 5 parents   │               │                 │
└────────┬────────┘               │                 │
         │                        │                 │
         └─────────────┬──────────┴─────────────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │  LEVEL 3: FINAL     │
             │  FUSION (weighted)  │
             │  fuse_results()     │
             └─────────┬───────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │  Combined Context   │
             │  for LLM Generation │
             └─────────────────────┘
```

#### Three Levels of Fusion

| Level | Components | Method | Configuration |
|-------|-----------|--------|---------------|
| **1. Hybrid** | Dense (Vector) + Sparse (BM25) | RRF (Reciprocal Rank Fusion) | `HYBRID_DENSE_WEIGHT=0.6` (α) |
| **2. Parallel** | Vector Retrieval ∥ Graph Retrieval | Independent execution | Controlled by query routing |
| **3. Final** | Vector Results + Graph Results | Weighted fusion | Based on query type (see table above) |

#### Hybrid Search Formula (Level 1)

The RRF formula combines dense and sparse rankings:

```
Score = α × (1/(k + rank_dense)) + (1-α) × (1/(k + rank_sparse))

Where:
- α = HYBRID_DENSE_WEIGHT (default 0.6 = 60% dense, 40% sparse)
- k = 60 (RRF constant)
- rank_dense = position in vector search results
- rank_sparse = position in BM25 search results
```

#### Vector Retrieval Flow (Detailed)

```
Query → Dense Search (Qdrant) ──┐
                                ├── RRF Fusion → Children (top 10)
Query → BM25 Search ────────────┘
                                         │
                                         ▼
                              Child → Parent Mapping
                              (deduplicate by parent)
                                         │
                                         ▼
                              Reranking (LLM or CrossEncoder)
                              → Parents (top 5)
```

Key constants:
- `TOP_K_RETRIEVED_CHILDREN=10`: Initial retrieval count
- `TOP_K_RERANKED_PARENTS=5`: Final count after reranking
- `HYBRID_DENSE_WEIGHT=0.6`: Dense vs sparse balance

### Agentic RAG

Complex queries trigger the agent path (`complexity.py` evaluates this):
- **ANALYTICAL/COMPARATIVE** queries automatically use agent
- **Database meta-questions** (e.g., "combien de documents?") force agent path
- Score threshold: 0.55 triggers agent, 0.35 triggers enhanced retrieval

Agent tools (`agent_tools.py`):
- `retrieve_vector`, `retrieve_graph`, `lookup_entity`, `compare_entities`
- `database_stats(list_documents=True/False)` - returns unique source documents count (not chunks)
  - `total_documents`: unique source files (PDFs)
  - `total_chunks`: number of chunks in index
  - `document_names`: list of source names (when `list_documents=True`)
- `synthesize`, `verify_claim`, `ask_clarification`, `final_answer`

**Document listing patterns** (`complexity.py`): Queries like "liste les documents", "quels documents", "list all docs" trigger agent path via `DATABASE_META_PATTERNS`.

**Design choice**: Custom ReAct implementation (~300 lines) instead of LangGraph/LangChain for:
- Fine-grained control over the reasoning loop
- Easier debugging (no framework abstractions)
- Minimal dependencies
- Simple use case (single agent, 9 fixed tools)

**ReAct Loop:**

```
┌─────────────────────────────────────────────────────────────┐
│                         START                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  THINK: Analyze query, decide next action                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  final_answer?  │───YES───▶ Return Response
                    └─────────────────┘
                              │ NO
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ACT: Execute chosen tool (retrieve, compare, etc.)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OBSERVE: Process tool result                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  REFLECT: Do I have enough info? (max 7 steps)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  enough info?   │───YES───▶ THINK (final_answer)
                    └─────────────────┘
                              │ NO
                              ▼
                         Loop to THINK
```

Language rules are enforced in prompts to ensure responses match query language (French/English/Spanish/German).

### Performance Optimizations

**Ingestion Pipeline Parallelization:**

| Stage | Module | Optimization |
|-------|--------|--------------|
| PDF→Images | `convert_pdf_to_image.py` | `ProcessPoolExecutor` (4 workers) |
| YOLO Detection | `extract_objects_from_image.py` | Batch inference (batch_size=2) |
| Embeddings | `create_embeddings.py` | Batched async with `httpx.AsyncClient` |
| Entity Extraction | `extract_entities.py` | Async with adaptive semaphore (auto 2-8 based on CPU) |
| Cache | `utils/embedding_cache.py` | SQLite persistent cache |

```python
# PDF conversion - parallel CPU-bound
convert_pdf_to_image(pdf_dir, image_dir, max_workers=4, parallel=True)

# Embeddings - batched async I/O
create_embeddings(chunks_dir, embeddings_dir, batch_size=32, max_concurrent=4)
```

**M2/M3 16GB Guidelines:**
- `max_workers=4` for PDF conversion (avoids memory saturation)
- `max_concurrent=4` for embeddings (overlaps network I/O)
- `yolo_batch_size=2` for YOLO detection (batch inference)
- `entity_max_concurrent=auto` for async entity extraction (auto-detects 2-8 based on CPU cores)

**Query-Time Optimizations:**

| Optimization | Module | Impact |
|--------------|--------|--------|
| Parallel Retrieval | `hybrid_retriever.py` | Vector + Graph run concurrently via `ThreadPoolExecutor` |
| Retrieval Cache | `hybrid_retriever.py` | LRU cache (50 entries, 5min TTL) for identical queries |
| Lazy Graph Loading | `hybrid_retriever.py` | Graph loaded only on first graph query |
| BM25 Lazy Loading | `hybrid_retriever.py` | BM25 index loaded only on first hybrid search |
| Query Embedding Cache | `utils/rag_utils.py` | Avoids recomputing same query embedding |
| Qdrant Result Cache | `utils/rag_utils.py` | LRU cache (50 entries, 5min TTL) for vector searches |
| BM25 Tokenization Cache | `utils/advanced_rag.py` | LRU cache (1000 entries) for tokenized queries |
| HNSW Index | `utils/rag_utils.py` | Faster approximate vector search (m=16, ef=100) |
| HTTP Connection Pooling | `utils/embedding_providers.py` | Shared `httpx.AsyncClient` for Ollama embeddings |
| Reranking Cache | `utils/advanced_rag.py` | LRU cache for cross-encoder results (5min TTL) |
| Adaptive Reranker Batch | `utils/advanced_rag.py` | Batch size = min(configured, docs, cpu*2) |
| Startup Warm-up | `cognidoc_app.py` | Pre-loads LLM, embeddings, retriever, reranker |
| Streaming Progress | `cognidoc_app.py` | Shows "🔍 Searching (vector + graph)..." with mode |

**Agent Optimizations:**

| Optimization | Module | Impact |
|--------------|--------|--------|
| Parallel Reflection | `agent.py` | Reflection runs in background thread via `ThreadPoolExecutor` |
| Adaptive Concurrency | `extract_entities.py` | Auto-detects CPU cores (2-8 workers) |
| Batch Entity Embeddings | `knowledge_graph.py` | Uses `embed_async()` for 5-10x faster ingestion |
| Lazy Entity Embeddings | `knowledge_graph.py` | Computed on first semantic search, not during build |

```python
# Retrieval cache API
from cognidoc.hybrid_retriever import get_retrieval_cache_stats, clear_retrieval_cache

stats = get_retrieval_cache_stats()
# {'size': 5, 'hits': 12, 'misses': 8, 'hit_rate': 0.6, 'ttl_seconds': 300}

clear_retrieval_cache()  # Clear cache manually

# Reranking cache API
from cognidoc.utils.advanced_rag import get_reranking_cache_stats, clear_reranking_cache

stats = get_reranking_cache_stats()
# {'size': 10, 'hits': 5, 'misses': 3, 'hit_rate': 0.625, 'ttl_seconds': 300}

clear_reranking_cache()  # Clear cache manually

# Qdrant result cache API
from cognidoc.utils.rag_utils import _qdrant_result_cache

stats = _qdrant_result_cache.stats()
# {'size': 5, 'hits': 12, 'misses': 8, 'hit_rate': 0.6}

_qdrant_result_cache.clear()  # Clear cache manually

# BM25 tokenization cache (functools.lru_cache)
from cognidoc.utils.advanced_rag import _cached_tokenize

info = _cached_tokenize.cache_info()
# CacheInfo(hits=10, misses=5, maxsize=1000, currsize=15)

_cached_tokenize.cache_clear()  # Clear cache manually
```

**Tool Result Caching** (`utils/tool_cache.py`):
```python
class PersistentToolCache:
    TTL_CONFIG = {
        "database_stats": 1800,   # 30 min - rarely changes
        "retrieve_vector": 300,   # 5 min - search results
        "retrieve_graph": 300,    # 5 min - graph results
        "lookup_entity": 600,     # 10 min - entity data
        "compare_entities": 1800, # 30 min - stable comparisons
    }
```

**Streaming Progress** (`cognidoc_app.py`):
- Agent path shows emoji progress: 🤔 (thinking), ⚡ (acting), 👁️ (observing), 💭 (reflecting)
- Cache hits marked with `[cached]` indicator
- Real-time updates via Gradio yield

**Dark Mode** (`cognidoc_app.py`):
- Toggle button in header (🌙/☀️)
- Auto-detects system preference on first load
- Persists user choice in localStorage
- Comprehensive CSS variables for all UI elements

**Optimized Prompts** (`agent.py`):
- SYSTEM_PROMPT emphasizes efficiency: "Target 2-3 steps max"
- THINK_PROMPT encourages immediate action
- REFLECT_PROMPT: "Can you answer NOW?"

### Conversation Memory

The chatbot maintains context across messages via query rewriting (`helpers.py`):

```
User: "Combien de documents?"  →  Agent answers "5 documents"
User: "cite-les-moi"           →  Rewritten to "Cite-moi les 5 documents..."
```

Key functions:
- `rewrite_query_with_history()` - Incorporates conversation context into the query
- `parse_rewritten_query()` - Extracts bullet points from rewritten queries (handles `- ` and `* ` styles)

The agent receives the **rewritten query** (not raw user message) to understand references like "them", "list them", etc.

### REST API

The Gradio app exposes REST endpoints for external integration:

```bash
# Main endpoint - ask a question
POST http://localhost:7860/api/submit_handler
Content-Type: application/json
{"data": ["Your question?", [], true, true]}
# Parameters: [user_msg, history, rerank, use_graph]

# Other endpoints
POST /api/reset_conversation    # Clear conversation
POST /api/refresh_metrics       # Get performance stats
POST /api/export_csv            # Export metrics CSV
POST /api/export_json           # Export metrics JSON
```

See README.md "REST API" section for examples with curl and gradio_client.

## Configuration

### Path Resolution (`src/cognidoc/constants.py`)

CogniDoc uses the **current working directory** as project root:

| Variable | Default | Env Override | Purpose |
|----------|---------|--------------|---------|
| `PROJECT_DIR` | `cwd()` | `COGNIDOC_PROJECT_DIR` | Project root (models/, config/) |
| `DATA_DIR` | `PROJECT_DIR/data` | `COGNIDOC_DATA_DIR` | Data folder (sources/, pdfs/, etc.) |
| `PACKAGE_DIR` | Package install | - | Embedded resources (prompts) |
| `BASE_DIR` | = `PROJECT_DIR` | - | Backward compatibility |

**Important:** If `COGNIDOC_DATA_DIR` is set, it should point to the data folder directly (containing `sources/`, `pdfs/`, etc.), not the project root.

This allows CogniDoc to work correctly when installed as a package and used in a new project.

### Key Constants (`src/cognidoc/constants.py`)

All settings overridable via `.env`:
- `YOLO_CONFIDENCE_THRESHOLD`: 0.2
- `MAX_CHUNK_SIZE`: 512 tokens
- `SEMANTIC_CHUNK_BUFFER`: 5 sentences
- `TOP_K_RETRIEVED_CHILDREN`: 10
- `TOP_K_RERANKED_PARENTS`: 5
- `HYBRID_DENSE_WEIGHT`: 0.6 (dense vs BM25 balance)
- `ENABLE_CONTEXTUAL_COMPRESSION`: false (disabled by default, enable for high-noise domains)

### MODEL_SPECS (`src/cognidoc/constants.py`)

Official provider parameters for all supported models:

```python
MODEL_SPECS = {
    "gemini-2.5-flash": {"context_window": 1_048_576, "max_output_tokens": 65_536, ...},
    "gpt-4o": {"context_window": 128_000, "max_output_tokens": 16_384, ...},
    "claude-sonnet-4-20250514": {"context_window": 200_000, "max_output_tokens": 64_000, ...},
    "granite3.3:8b": {"context_window": 128_000, "max_output_tokens": 8_192, ...},
}

# Auto-load specs via LLMConfig.from_model()
config = LLMConfig.from_model("gemini-2.5-flash")  # Loads all specs automatically
```

### Dynamic MEMORY_WINDOW (`src/cognidoc/helpers.py`)

Conversation memory adapts to the LLM's context window:

```python
def get_memory_window() -> int:
    """Returns 50% of the model's context_window."""
    client = get_llm_client()
    return int(client.config.context_window * 0.5)  # Dynamic!
```

| Model | CONTEXT_WINDOW | MEMORY_WINDOW (50%) |
|-------|----------------|---------------------|
| Gemini 2.5 Flash | 1,048,576 (1M) | 524,288 |
| GPT-4o | 128,000 | 64,000 |
| Claude Sonnet 4 | 200,000 | 100,000 |
| Granite 3.3:8b | 128,000 | 64,000 |

### Other Config Files

- `config/graph_schema.yaml` - Entity types, relationship types, routing strategy
- `src/cognidoc/prompts/` - LLM prompt templates (markdown files)

## External Dependencies

- **Ollama** at `localhost:11434` - Local inference (embeddings, fallback LLM)
- **LibreOffice** - Required for Office document conversion (pptx, docx, xlsx)
- **Qdrant** - Embedded vector database (no separate server)

Required Ollama models:
```bash
ollama pull granite3.3:8b                   # LLM generation
ollama pull qwen3-embedding:0.6b            # Embeddings
ollama pull ibm/granite-docling:258m-bf16   # Document parsing
ollama pull qwen3-vl:8b-instruct            # Vision (optional)
```

### YOLO Model (optional)

YOLO detection requires a trained model file at `models/YOLOv11/yolov11x_best.pt`. This file is **not included** in the repository (~109 MB, gitignored).

**Behavior:**
- Model present → YOLO detection enabled (text/table/image regions)
- Model absent → Fallback to simple page-level extraction
- `--skip-yolo` or `use_yolo=False` → Explicitly disable YOLO

**To enable YOLO:** Provide a YOLOv11 model trained for document layout detection (DocLayNet/PubLayNet classes: text, table, picture, caption, etc.).

## Project Directories

| Directory | Content |
|-----------|---------|
| `models/YOLOv11/` | YOLO model file (`yolov11x_best.pt`, optional, ~109 MB) |
| `config/` | GraphRAG schema (`graph_schema.yaml`) |
| `data/sources/` | Input documents (any format, including subfolders) |
| `data/pdfs/` | Converted PDFs |
| `data/images/` | 600 DPI page images |
| `data/detections/` | YOLO-cropped regions |
| `data/processed/` | Extracted text/tables/descriptions |
| `data/chunks/` | Parent and child chunks |
| `data/indexes/` | Vector/keyword/graph indexes |
| `data/vector_store/` | Qdrant database |
| `data/cache/` | Embedding cache (SQLite) |

## Setup Wizards

### Interactive Setup Wizard (`python -m cognidoc.setup`)

Full guided setup for first-time users:
1. LLM provider selection (Gemini, OpenAI, Anthropic, Ollama)
2. API key validation
3. Embedding provider configuration
4. Document detection and ingestion
5. Web interface launch

### Schema Wizard (GraphRAG)

Runs **automatically during ingestion** when no `config/graph_schema.yaml` exists:

1. **Interactive mode** (requires `questionary`): Prompts for domain type, language, and whether to auto-generate
2. **Auto-generation**: Samples documents from `data/sources/` and uses LLM to identify entity/relationship types

Key options:
- `doc.ingest(skip_schema_wizard=True)` - Use existing schema without prompts
- `cognidoc init --schema` - Copy template schema (non-interactive)
- `config/graph_schema_generic.yaml` - Template for manual schema creation

## Tests

```bash
# Run all tests (note: make test is not configured, use pytest directly)
.venv/bin/python -m pytest tests/ -v

# Or with uv (if path has no spaces)
uv run pytest tests/ -v

# Run a single test file
pytest tests/test_agent.py -v

# Run a single test function
pytest tests/test_agent.py::test_agent_tool_parsing -v

# Run tests matching a pattern
pytest tests/ -v -k "complexity"

# Run E2E tests only (~30s)
pytest tests/test_00_e2e_pipeline.py -v

# Run full E2E with ingestion (~2-5 min)
pytest tests/test_00_e2e_pipeline.py -v --run-slow
```

| Module | Tests | Description |
|--------|-------|-------------|
| `test_00_e2e_pipeline.py` | 9 | E2E pipeline (runs first to avoid Qdrant lock) |
| `test_agent.py` | 27 | Agent ReAct loop |
| `test_agent_tools.py` | 33 | Tool implementations |
| `test_benchmark.py` | 10 | Precision/recall benchmark (vector vs GraphRAG) |
| `test_checkpoint.py` | 32 | Checkpoint/resume system for pipeline interruption |
| `test_complexity.py` | 25 | Query complexity evaluation |
| `test_e2e_language_and_count.py` | 24 | Language detection (FR/EN/ES/DE), document count |
| `test_helpers.py` | 34 | Token counting, chat history, query parsing, JSON |
| `test_providers.py` | 32 | LLM/Embedding providers |
| **Total** | **226** |

**Test Infrastructure:**
- `conftest.py` provides session-scoped `cognidoc_session` fixture to avoid Qdrant lock conflicts
- `--run-slow` flag enables slow E2E tests (registered via `pytest_addoption`)
- E2E tests are named `test_00_*` to run first alphabetically (Qdrant embedded only allows one client per storage folder)

**Benchmark tests with external data:**

The benchmark tests (`test_benchmark.py`) are designed to run against a real corpus. Use `COGNIDOC_DATA_DIR` to point to an external project's data:

```bash
# Run benchmarks with cognidoc-theologie-morale data (17,265 docs)
COGNIDOC_DATA_DIR="/path/to/cognidoc-theologie-morale/data" \
  uv run pytest tests/test_benchmark.py -v --run-slow

# Show benchmark summary only
COGNIDOC_DATA_DIR="/path/to/cognidoc-theologie-morale/data" \
  uv run pytest tests/test_benchmark.py::TestBenchmarkComparison::test_benchmark_summary -v --run-slow -s
```

**Note:** Benchmark tests query bioethics topics (PMA, avortement, bioéthique). Running against generic test fixtures will fail relevance tests.
