# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogniDoc is a Hybrid RAG (Vector + GraphRAG) document assistant that converts multi-format documents into a searchable knowledge base with intelligent query routing.

**Key Design Decisions:**
- No LangChain/LlamaIndex - direct Qdrant and Ollama integration for fine-grained control
- Multi-provider LLM support (Gemini default, Ollama, OpenAI, Anthropic)
- Parent-child chunk hierarchy for context-aware retrieval
- Custom semantic chunking with breakpoint detection
- Custom ReAct agent (~300 lines in `agent.py`) instead of LangGraph for fine-grained control

## Commands

```bash
# Setup (uses uv package manager)
make install          # Create venv and install dependencies
make sync             # Sync environment with lock file
uv sync --group dev   # Install dev dependencies (pytest, black, pylint, mypy)

# IMPORTANT: If project path contains spaces
UV_LINK_MODE=copy uv sync --all-extras
UV_LINK_MODE=copy uv pip install -e ".[all,dev]"

# Code quality
make format                      # Format with black
make lint                        # Run pylint
make refactor                    # Format + lint
uv run black src/cognidoc/       # Format (direct)
uv run pylint src/cognidoc/      # Lint (direct)

# Run tests
uv run pytest tests/ -v                                    # All tests
uv run pytest tests/test_agent.py -v                       # Single file
uv run pytest tests/test_agent.py::test_agent_tool_parsing -v  # Single test
uv run pytest tests/ -v -k "complexity"                    # Pattern match
uv run pytest tests/test_00_e2e_pipeline.py -v --run-slow  # Full E2E (~2-5 min)

# CLI commands
cognidoc ingest ./data/sources --llm gemini --embedding ollama
cognidoc query "What is X?"
cognidoc serve --port 7860 --share
cognidoc info

# Direct module execution
python -m cognidoc.setup                  # Interactive wizard
python -m cognidoc.cognidoc_app           # Launch chat UI
python -m cognidoc.run_ingestion_pipeline # Ingestion with more options
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
--skip-resolution     # Skip entity resolution (semantic deduplication)
--force-reembed       # Re-embed all (ignore cache)
```

## Architecture

### Source Layout

Source code is in `src/cognidoc/` but installs as `cognidoc` package:
- File path: `src/cognidoc/cognidoc_app.py`
- Module import: `from cognidoc import CogniDoc`
- CLI execution: `python -m cognidoc.cognidoc_app`

### Key Modules

| Module | Purpose |
|--------|---------|
| `api.py` | Main CogniDoc class (public API) |
| `run_ingestion_pipeline.py` | Async pipeline orchestrator |
| `cognidoc_app.py` | Gradio chat with FastAPI static file serving |
| `hybrid_retriever.py` | Vector + Graph fusion with query orchestration and caching |
| `knowledge_graph.py` | NetworkX graph with Louvain community detection |
| `entity_resolution.py` | Semantic entity deduplication (4-phase: blocking, matching, clustering, merging) |
| `query_orchestrator.py` | LLM-based query classification and routing |
| `complexity.py` | Query complexity evaluation for agentic routing |
| `agent.py` | ReAct agent with parallel reflection for complex queries |
| `agent_tools.py` | Tool implementations for agent (9 tools) |
| `helpers.py` | Query rewriting, parsing, conversation context |
| `schema_wizard.py` | Interactive/auto schema generation for GraphRAG |
| `constants.py` | Central config (paths, thresholds, model names) |
| `utils/llm_client.py` | Singleton LLM client (Gemini default) |
| `utils/llm_providers.py` | Multi-provider abstraction layer |
| `utils/rag_utils.py` | Document, VectorIndex, KeywordIndex classes |
| `utils/embedding_providers.py` | Embedding providers with async batch and connection pooling |

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

### Query Processing

```
User Query → Query Rewriter (adds conversation context)
                   ↓
     ┌─────────────┴─────────────┐
     ↓                           ↓
Classifier                 Complexity
(query type)               Evaluator
     ↓                           ↓
     └─────────────┬─────────────┘
                   ↓
    score < 0.35: FAST PATH (simple RAG)
    0.35 - 0.55:  ENHANCED PATH
    score >= 0.55: AGENT PATH (ReAct loop)
```

### Query Routing

Query types determine vector/graph weight balance:

| Query Type | Vector | Graph | Use Case |
|------------|--------|-------|----------|
| FACTUAL | 70% | 30% | "What is X?" |
| RELATIONAL | 20% | 80% | "How does X relate to Y?" |
| EXPLORATORY | 0% | 100% | "Tell me about X" |
| PROCEDURAL | 80% | 20% | "How to do X?" |

Skip logic: if weight < 15%, that retriever is skipped entirely.

### 3-Level Fusion Retrieval

| Level | Components | Method |
|-------|------------|--------|
| 1. Hybrid | Dense + Sparse (BM25) | RRF fusion with α=0.6 default |
| 2. Parallel | Vector ∥ Graph | Independent execution via ThreadPoolExecutor |
| 3. Final | Vector + Graph results | Weighted fusion based on query type |

Key constants in `constants.py`:
- `TOP_K_RETRIEVED_CHILDREN=10`: Initial retrieval count
- `TOP_K_RERANKED_PARENTS=5`: Final count after reranking
- `HYBRID_DENSE_WEIGHT=0.6`: Dense vs sparse balance

### Agentic RAG

Complex queries (score >= 0.55) trigger the ReAct agent: `THINK → ACT → OBSERVE → REFLECT` (max 7 steps)

Agent tools (`agent_tools.py`):
- `retrieve_vector`, `retrieve_graph`, `lookup_entity`, `compare_entities`
- `database_stats(list_documents=True/False)` - returns unique source documents count
- `synthesize`, `verify_claim`, `ask_clarification`, `final_answer`

**Document listing patterns** (`complexity.py`): Queries like "liste les documents", "quels documents", "list all docs" trigger agent path via `DATABASE_META_PATTERNS`.

## Configuration

### Path Resolution (`constants.py`)

CogniDoc uses the **current working directory** as project root:

| Variable | Default | Env Override |
|----------|---------|--------------|
| `PROJECT_DIR` | `cwd()` | `COGNIDOC_PROJECT_DIR` |
| `DATA_DIR` | `PROJECT_DIR/data` | `COGNIDOC_DATA_DIR` |

**Important:** If `COGNIDOC_DATA_DIR` is set, it should point to the data folder directly (containing `sources/`, `pdfs/`, etc.), not the project root.

### Provider Selection (`utils/llm_providers.py`)

When switching providers at runtime (e.g., `DEFAULT_LLM_PROVIDER=ollama`), the code uses **provider-specific** env vars for model selection:

| Provider | LLM Model Env Var | Default Model |
|----------|-------------------|---------------|
| gemini | `GEMINI_LLM_MODEL` | gemini-3-flash-preview |
| ollama | `OLLAMA_LLM_MODEL` | granite3.3:8b |
| openai | `OPENAI_LLM_MODEL` | gpt-4o |
| anthropic | `ANTHROPIC_LLM_MODEL` | claude-sonnet-4-20250514 |

**Priority order:** Provider-specific env var → Built-in default for that provider

This ensures `DEFAULT_LLM_PROVIDER=ollama` uses `granite3.3:8b` even if `.env` has `DEFAULT_LLM_MODEL=gemini-3-flash-preview`.

### Models by Pipeline Stage

**Ingestion Pipeline:**

| Stage | Default Model | Provider |
|-------|---------------|----------|
| Document parsing | `ibm/granite-docling:258m-bf16` | Ollama |
| Text/table extraction | `gemini-3-pro-preview` (vision) | Gemini |
| Image descriptions | `gemini-3-pro-preview` (vision) | Gemini |
| Table descriptions | `gemini-3-pro-preview` | Gemini |
| Embeddings | `qwen3-embedding:4b-q8_0` | Ollama |
| Entity extraction | `gemini-3-pro-preview` | Gemini |
| Community summaries | `gemini-3-pro-preview` | Gemini |

**Query Pipeline:**

| Stage | Default Model | Provider |
|-------|---------------|----------|
| Query rewriting | `gemini-3-flash-preview` | Gemini |
| Query expansion | `gemini-3-flash-preview` | Gemini |
| Query classification | Rule-based (+ LLM fallback) | - |
| Reranking | `gemini-3-flash-preview` | Gemini |
| Generation | `gemini-3-flash-preview` | Gemini |

**Optional Features (disabled by default):**

| Feature | Default Model | Provider |
|---------|---------------|----------|
| Cross-encoder reranking | `dengcao/Qwen3-Reranker-0.6B:F16` | Ollama |
| Contextual compression | `gemini-3-flash-preview` | Gemini |

### Key Constants (overridable via `.env`)

- `YOLO_CONFIDENCE_THRESHOLD`: 0.2
- `MAX_CHUNK_SIZE`: 512 tokens
- `SEMANTIC_CHUNK_BUFFER`: 5 sentences
- `TOP_K_RETRIEVED_CHILDREN`: 10
- `TOP_K_RERANKED_PARENTS`: 5
- `HYBRID_DENSE_WEIGHT`: 0.6
- `OLLAMA_EMBED_MODEL`: qwen3-embedding:4b-q8_0
- `ENABLE_CONTEXTUAL_COMPRESSION`: false

### MODEL_SPECS (`constants.py`)

Official provider parameters are defined in `MODEL_SPECS` dict. Auto-load specs via:
```python
config = LLMConfig.from_model("gemini-3-flash-preview")  # Loads all specs automatically
```

### Dynamic MEMORY_WINDOW (`helpers.py`)

Conversation memory adapts to 50% of the model's context window:
```python
def get_memory_window() -> int:
    return int(client.config.context_window * 0.5)
```

## External Dependencies

- **Ollama** at `localhost:11434` - Local inference (embeddings, fallback LLM)
- **LibreOffice** - Required for Office document conversion (pptx, docx, xlsx)
- **Qdrant** - Embedded vector database (no separate server)

Required Ollama models:
```bash
ollama pull granite3.3:8b               # LLM generation
ollama pull qwen3-embedding:4b-q8_0         # Embeddings (default)
ollama pull ibm/granite-docling:258m-bf16   # Document parsing
ollama pull qwen3-vl:8b-instruct            # Vision (optional)
```

### YOLO Model (optional)

YOLO detection requires `models/YOLOv11/yolov11x_best.pt` (~109 MB, gitignored). Without it, falls back to simple page-level extraction.

## Project Directories

| Directory | Content |
|-----------|---------|
| `models/YOLOv11/` | YOLO model file (optional) |
| `config/` | GraphRAG schema (`graph_schema.yaml`) |
| `data/sources/` | Input documents (any format) |
| `data/pdfs/` | Converted PDFs |
| `data/images/` | 600 DPI page images |
| `data/detections/` | YOLO-cropped regions |
| `data/processed/` | Extracted text/tables/descriptions |
| `data/chunks/` | Parent and child chunks |
| `data/indexes/` | Vector/keyword/graph indexes |
| `data/vector_store/` | Qdrant database |
| `data/cache/` | Embedding cache (SQLite) |

## Tests

**Important:** Qdrant embedded only allows one client per storage folder. The app and tests cannot run simultaneously against the same data directory.

| Module | Tests | Description |
|--------|-------|-------------|
| `test_00_e2e_pipeline.py` | 9 | E2E pipeline (runs first to avoid Qdrant lock) |
| `test_agent.py` | 27 | Agent ReAct loop |
| `test_agent_tools.py` | 33 | Tool implementations |
| `test_benchmark.py` | 10 | Precision/recall benchmark |
| `test_checkpoint.py` | 32 | Checkpoint/resume system |
| `test_complexity.py` | 25 | Query complexity evaluation |
| `test_e2e_language_and_count.py` | 24 | Language detection (FR/EN/ES/DE) |
| `test_entity_resolution.py` | 34 | Entity resolution (blocking, matching, clustering, merging) |
| `test_helpers.py` | 34 | Token counting, chat history, query parsing |
| `test_optimizations.py` | 26 | Pipeline optimizations (concurrency, pooling) |
| `test_providers.py` | 32 | LLM/Embedding providers |

**Test Infrastructure:**
- `conftest.py` provides session-scoped `cognidoc_session` fixture to avoid Qdrant lock conflicts
- `--run-slow` flag enables slow E2E tests
- E2E tests named `test_00_*` run first alphabetically

**Benchmark with external data:**
```bash
COGNIDOC_DATA_DIR="/path/to/external/data" \
  uv run pytest tests/test_benchmark.py -v --run-slow
```

## Performance Notes

**M2/M3 16GB Guidelines:**
- `max_workers=4` for PDF conversion
- `max_concurrent=4` for embeddings
- `yolo_batch_size=2` for YOLO detection
- `entity_max_concurrent=auto` (2-8 based on CPU cores)

**Caching layers:**
- Retrieval cache: LRU (50 entries, 5min TTL) in `hybrid_retriever.py`
- Qdrant result cache: LRU (50 entries, 5min TTL) in `utils/rag_utils.py`
- BM25 tokenization cache: LRU (1000 entries) in `utils/advanced_rag.py`
- Tool result cache: SQLite with TTL (5-30 min by tool) in `utils/tool_cache.py`
- Embedding cache: SQLite persistent in `utils/embedding_cache.py`

## REST API

The Gradio app exposes REST endpoints at `http://localhost:7860`:

```bash
# Main endpoint
POST /api/submit_handler
{"data": ["Your question?", [], true, true]}
# Parameters: [user_msg, history, rerank, use_graph]

# Other endpoints
POST /api/reset_conversation
POST /api/refresh_metrics
POST /api/export_csv
POST /api/export_json
```
