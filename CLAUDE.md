# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogniDoc is a Hybrid RAG (Vector + GraphRAG) document assistant that converts multi-format documents into a searchable knowledge base with intelligent query routing.

**Key Design Decisions:**
- No LangChain/LlamaIndex - direct Qdrant and Ollama integration for fine-grained control
- Multi-provider LLM support (Gemini default, Ollama, OpenAI, Anthropic)
- Parent-child chunk hierarchy for context-aware retrieval
- Custom semantic chunking with breakpoint detection

## Commands

```bash
# Setup and installation (uses uv package manager)
make install          # Create venv and install dependencies
make sync             # Sync environment with lock file

# IMPORTANT: If path contains spaces, use UV_LINK_MODE=copy
UV_LINK_MODE=copy uv sync --all-extras
UV_LINK_MODE=copy uv pip install -e ".[all,dev]"

# Code quality
make format           # Format with black
make lint             # Run pylint
make refactor         # Format + lint

# Quick test via Python API
python -c "
from cognidoc import CogniDoc
doc = CogniDoc(llm_provider='gemini', embedding_provider='ollama')
doc.ingest(skip_schema_wizard=True)  # Use existing schema
result = doc.query('Your question here')
print(result.answer)
"

# First-time setup (interactive wizard)
python -m src.setup

# Run ingestion pipeline directly
python -m src.run_ingestion_pipeline --vision-provider ollama

# Launch chat interface
python -m src.cognidoc_app
python -m src.cognidoc_app --no-rerank    # Faster, skip LLM reranking
python -m src.cognidoc_app --share        # Create public link
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
--entity-max-concurrent 4 # Concurrent LLM calls for entity extraction (default: 4)
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

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/cognidoc/api.py` | Main CogniDoc class (public API) |
| `src/cognidoc/run_ingestion_pipeline.py` | Async pipeline orchestrator |
| `src/cognidoc/cognidoc_app.py` | Gradio chat with FastAPI static file serving |
| `src/cognidoc/hybrid_retriever.py` | Vector + Graph fusion with query orchestration |
| `src/cognidoc/knowledge_graph.py` | NetworkX graph with Louvain community detection |
| `src/cognidoc/query_orchestrator.py` | LLM-based query classification and routing |
| `src/cognidoc/complexity.py` | Query complexity evaluation for agentic routing |
| `src/cognidoc/agent.py` | ReAct agent for complex multi-step queries |
| `src/cognidoc/agent_tools.py` | Tool implementations for agent (9 tools) |
| `src/cognidoc/helpers.py` | Query rewriting, parsing, conversation context |
| `src/cognidoc/schema_wizard.py` | Interactive/auto schema generation for GraphRAG |
| `src/cognidoc/constants.py` | Central config (paths, thresholds, model names) |
| `src/cognidoc/utils/llm_client.py` | Singleton LLM client (Gemini default) |
| `src/cognidoc/utils/llm_providers.py` | Multi-provider abstraction layer |
| `src/cognidoc/utils/rag_utils.py` | Document, VectorIndex, KeywordIndex classes |
| `src/cognidoc/utils/embedding_providers.py` | Embedding providers with async batch support |
| `src/cognidoc/utils/tool_cache.py` | Persistent SQLite cache for tool results |
| `src/cognidoc/utils/metrics.py` | Performance metrics with SQLite storage |
| `src/cognidoc/create_embeddings.py` | Batched async embedding generation |
| `src/cognidoc/convert_pdf_to_image.py` | Parallel PDF to image conversion |
| `src/cognidoc/extract_objects_from_image.py` | YOLO detection with batch inference |
| `src/cognidoc/extract_entities.py` | Async entity/relationship extraction |

### Query Routing

Query types determine vector/graph weight balance:
- **FACTUAL**: 70% vector, 30% graph
- **RELATIONAL**: 20% vector, 80% graph
- **EXPLORATORY**: 0% vector, 100% graph (skips vector)
- **PROCEDURAL**: 80% vector, 20% graph

Skip logic: if weight < 15%, that retriever is skipped entirely.

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
| Entity Extraction | `extract_entities.py` | Async with semaphore (max_concurrent=4) |
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
- `entity_max_concurrent=4` for async entity extraction

**Tool Result Caching** (`agent_tools.py`):
```python
class ToolCache:
    TTL_CONFIG = {
        "database_stats": 300,    # 5 min
        "retrieve_vector": 120,   # 2 min
        "retrieve_graph": 120,    # 2 min
        "lookup_entity": 300,     # 5 min
        "compare_entities": 180,  # 3 min
    }
```

**Streaming Progress** (`cognidoc_app.py`):
- Agent path shows emoji progress: 🤔 (thinking), ⚡ (acting), 👁️ (observing), 💭 (reflecting)
- Cache hits marked with `[cached]` indicator
- Real-time updates via Gradio yield

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

See README.md "API Integration" section for detailed examples with curl, requests, and gradio_client.

## Configuration

### Key Constants (`src/constants.py`)

All settings overridable via `.env`:
- `YOLO_CONFIDENCE_THRESHOLD`: 0.2
- `MAX_CHUNK_SIZE`: 512 tokens
- `SEMANTIC_CHUNK_BUFFER`: 5 sentences
- `TOP_K_RETRIEVED_CHILDREN`: 10
- `TOP_K_RERANKED_PARENTS`: 5
- `HYBRID_DENSE_WEIGHT`: 0.6 (dense vs BM25 balance)

### MODEL_SPECS (`src/constants.py`)

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

### Dynamic MEMORY_WINDOW (`src/helpers.py`)

Conversation memory adapts to the LLM's context window:

```python
def get_memory_window() -> int:
    """Returns 50% of the model's context_window."""
    client = get_llm_client()
    return int(client.config.context_window * 0.5)  # Dynamic!

# Examples:
# Gemini 2.5 Flash (1M context) → 524K memory window
# GPT-4o (128K context) → 64K memory window
# Claude Sonnet 4 (200K context) → 100K memory window
```

### Other Config Files

- `config/graph_schema.yaml` - Entity types, relationship types, routing strategy
- `src/prompts/` - LLM prompt templates (markdown files)

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

## Data Directories

| Directory | Content |
|-----------|---------|
| `data/sources/` | Input documents (any format, including subfolders) |
| `data/pdfs/` | Converted PDFs |
| `data/images/` | 600 DPI page images |
| `data/detections/` | YOLO-cropped regions |
| `data/processed/` | Extracted text/tables/descriptions |
| `data/chunks/` | Parent and child chunks |
| `data/indexes/` | Vector/keyword/graph indexes |
| `data/vector_store/` | Qdrant database |
| `data/cache/` | Embedding cache (SQLite) |

## Schema Wizard

The schema wizard runs automatically during ingestion when no `config/graph_schema.yaml` exists:

1. **Interactive mode** (requires `questionary`): Prompts for domain type, language, and whether to auto-generate
2. **Auto-generation**: Samples documents from `data/sources/` and uses LLM to identify entity/relationship types

Key options:
- `doc.ingest(skip_schema_wizard=True)` - Use existing schema without prompts
- `config/graph_schema_generic.yaml` - Template for manual schema creation

## Tests

```bash
# Run all tests (148 passed, 2 skipped)
.venv/bin/python -m pytest tests/ -v

# Or with uv (if path has no spaces)
uv run pytest tests/ -v

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
| `test_complexity.py` | 24 | Query complexity evaluation |
| `test_e2e_language_and_count.py` | 24 | Language detection (FR/EN/ES/DE), document count |
| `test_providers.py` | 33 | LLM/Embedding providers |
| **Total** | **150** (148 passed, 2 skipped) |

**Note:** E2E tests are named `test_00_*` to run first alphabetically. Qdrant embedded only allows one client per storage folder, so E2E tests must acquire the lock before other tests.
