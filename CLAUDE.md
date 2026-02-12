# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogniDoc is a Hybrid RAG (Vector + GraphRAG) document assistant that converts multi-format documents into a searchable knowledge base with intelligent query routing. Requires Python 3.10+. Built with Hatchling. Licensed under MIT.

**Key Design Decisions:**
- No LangChain/LlamaIndex - direct Qdrant and Ollama integration for fine-grained control
- Multi-provider LLM support (Gemini default, Ollama, OpenAI, Anthropic)
- Parent-child chunk hierarchy for context-aware retrieval
- Custom semantic chunking with breakpoint detection
- Custom ReAct agent (~300 lines in `agent.py`) instead of LangGraph for fine-grained control
- Incremental ingestion via manifest tracking (only new/modified documents are reprocessed)
- Auto-generated graph schema from corpus analysis (two-stage LLM pipeline)

## Commands

```bash
# Setup (uses uv package manager, Hatchling build backend)
make install          # Create venv and install dependencies
make sync             # Sync environment with lock file
make lock             # Lock dependencies
make refactor         # Format + lint (root-level *.py and code/*.py)
make help             # Display all available Makefile targets
uv sync --group dev   # Install dev dependencies (pytest, black, pylint, mypy)

# Optional dependency groups: ui, yolo, ollama, gemini, openai, anthropic, cloud, conversion, wizard, all, dev

# IMPORTANT: If project path contains spaces
UV_LINK_MODE=copy uv sync --all-extras
UV_LINK_MODE=copy uv pip install ".[all,dev]"   # Non-editable (NOT -e), editable breaks with spaces
# To launch the app after non-editable install, use .venv/bin/python (not uv run, which re-syncs editable)

# Code quality (black: line-length=100, target py310-py312)
# pylint disables: C0114, C0115, C0116 (missing docstrings), R0903 (too-few-public-methods)
# NOTE: Makefile targets are outdated — make format targets root-level *.py;
# make lint targets *.py and code/*.py (code/ doesn't exist). Neither covers src/cognidoc/.
# Use direct commands below for the main source code:
uv run black src/cognidoc/       # Format source code
uv run pylint src/cognidoc/      # Lint source code
uv run mypy src/cognidoc/        # Type check (ignore_missing_imports=true)
make container-lint              # Lint Dockerfile with hadolint

# Run tests
uv run pytest tests/ -v                                    # All tests
uv run pytest tests/test_agent.py -v                       # Single file
uv run pytest tests/test_agent.py::test_agent_tool_parsing -v  # Single test
uv run pytest tests/ -v -k "complexity"                    # Pattern match
uv run pytest tests/test_00_e2e_pipeline.py -v --run-slow  # Full E2E (~2-5 min)

# CLI commands
cognidoc schema-generate ./data/sources --language fr  # Auto-generate graph schema
cognidoc ingest ./data/sources --llm gemini --embedding ollama
cognidoc ingest ./data/sources --regenerate-schema     # Force schema regeneration
cognidoc query "What is X?"
cognidoc serve --port 7860 --share
cognidoc info
cognidoc migrate-graph                         # Migrate NetworkX graph to Kùzu

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
--full-reindex        # Force full re-ingestion (ignore incremental manifest)
--no-incremental      # Disable incremental detection
--regenerate-schema   # Force graph schema regeneration before ingestion
```

## CI Pipeline

CI runs on push/PR to `master` with three jobs:

- **lint**: black `--check`, pylint `--fail-under=7.0`, mypy on `src/cognidoc/`
- **test**: pytest across Python 3.10/3.11/3.12 with `-x` (fail-fast), excludes `test_00_e2e_pipeline.py` and `test_benchmark.py`
- **docker**: builds the Docker image (no runtime tests)

## Documentation

| File | Content |
|------|---------|
| `CLAUDE.md` | This file — instructions for Claude Code |
| `README.md` | User-facing setup and usage guide |
| `docs/QUICKSTART.md` | Step-by-step guide to deploy CogniDoc on a new project |
| `docs/ROADMAP.md` | Implementation plans (Phases 1-3: graph enrichment → Kùzu → Neo4j) |
| `docs/architecture/query_pipeline.md` | Deep dive into query internals |

## Architecture

> **Deep dive:** For detailed explanations of the query pipeline internals (complexity evaluation formula, LRU cache, vector/graph fusion, ReAct agent loop), see [`docs/architecture/query_pipeline.md`](docs/architecture/query_pipeline.md).

### Source Layout

Source code is in `src/cognidoc/` but installs as `cognidoc` package:
- File path: `src/cognidoc/cognidoc_app.py`
- Module import: `from cognidoc import CogniDoc`
- CLI entry point: `cognidoc` (defined in `pyproject.toml` → `cognidoc.cli:main`)
- Direct execution: `python -m cognidoc.cognidoc_app`
- Version: `__version__` in `src/cognidoc/__init__.py` and `version` in `pyproject.toml` (keep in sync)

### Key Modules

| Module | Purpose |
|--------|---------|
| `api.py` | Main CogniDoc class (public API) |
| `run_ingestion_pipeline.py` | Async pipeline orchestrator |
| `cognidoc_app.py` | Gradio chat UI, FastAPI middleware (CSS/JS injection), response formatting, graph viewer API |
| `hybrid_retriever.py` | Vector + Graph fusion with query orchestration and caching |
| `knowledge_graph.py` | Graph facade with pluggable backend (NetworkX/Kùzu) + Louvain communities |
| `graph_backend.py` | Abstract GraphBackend ABC (Strategy pattern) |
| `graph_backend_networkx.py` | NetworkX backend implementation (default) |
| `graph_backend_kuzu.py` | Kùzu backend implementation (optional, Cypher-native) |
| `entity_resolution.py` | Semantic entity deduplication (4-phase: blocking, matching, clustering, merging) |
| `query_orchestrator.py` | LLM-based query classification and routing |
| `complexity.py` | Query complexity evaluation for agentic routing |
| `agent.py` | ReAct agent with parallel reflection for complex queries |
| `agent_tools.py` | Tool implementations for agent (11 tools) |
| `helpers.py` | Query rewriting, parsing, conversation context |
| `schema_wizard.py` | Schema generation: interactive, template-based, or corpus-based (two-stage LLM) |
| `constants.py` | Central config (paths, thresholds, model names) |
| `checkpoint.py` | Resumable pipeline execution with atomic saves |
| `ingestion_manifest.py` | Incremental ingestion tracking (new/modified file detection via SHA-256) |
| `cli.py` | Command-line interface (ingest, query, serve, init, info, schema-generate, migrate-graph) |
| `graph_config.py` | GraphRAG schema loading and validation |
| `utils/llm_client.py` | Singleton LLM client (Gemini default) |
| `utils/llm_providers.py` | Multi-provider abstraction layer |
| `utils/rag_utils.py` | Document, VectorIndex, KeywordIndex classes |
| `utils/embedding_providers.py` | Embedding providers with async batch and connection pooling |
| `utils/metrics.py` | Performance metrics tracking (ingestion stats, query latency) |
| `utils/logger.py` | Structured logging utilities |
| `utils/tool_cache.py` | SQLite-backed persistent tool result caching with per-tool TTL |
| `utils/chat_history.py` | SQLite-backed persistent chat history (conversations, messages, export) |
| `utils/error_classifier.py` | Error classification for retry logic |
| `utils/advanced_rag.py` | BM25 sparse retrieval, cross-encoder reranking, lost-in-the-middle reordering, contextual compression |
| `utils/embedding_cache.py` | SQLite-backed embedding cache (SHA-256 content hashing to skip unchanged chunks) |
| `utils/async_utils.py` | Async concurrency helpers |
| `graph_retrieval.py` | Graph retrieval: entity/relationship traversal, community queries, path finding |
| `setup.py` | Interactive setup wizard (provider config, API key validation, document detection) |

Pipeline stage modules (map 1:1 to the ingestion diagram):
`convert_to_pdf.py`, `convert_pdf_to_image.py`, `extract_objects_from_image.py`,
`parse_image_with_text.py`, `parse_image_with_table.py`, `create_image_description.py`,
`chunk_text_data.py`, `chunk_table_data.py`, `create_embeddings.py`,
`build_indexes.py`, `extract_entities.py`

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

### Incremental Ingestion

The pipeline is **incremental by default**. An ingestion manifest (`data/indexes/ingestion_manifest.json`) tracks source files by path, size, and SHA-256 content hash.

**Behavior on re-ingestion:**

| Scenario | Pipeline behavior |
|----------|-------------------|
| New files added to `data/sources/` | Only new files processed through stages 1-10 |
| Existing file modified | Old intermediate files cleaned up, file reprocessed |
| No changes detected | Pipeline exits immediately ("Nothing to ingest") |
| `--full-reindex` flag | Full pipeline, manifest rebuilt |
| First ingestion (no manifest) | Full pipeline, manifest created at end |

**Stage-by-stage incremental behavior:**

| Stage | Incremental mode |
|-------|-----------------|
| 1-5 (Conversion → Descriptions) | `source_files` limited to new/modified only |
| 6-7 (Chunking) | `file_filter` limits to new stems |
| 8 (Embeddings) | `file_filter` + content-hash cache |
| 9 (Index building) | Rebuilt from ALL embeddings on disk (old + new) |
| 10 (Entity extraction) | `file_filter` limits to new chunk stems |
| 11 (Graph building) | Existing graph loaded via `KnowledgeGraph.load()`, new entities merged via `build_from_extraction_results()` |
| 12 (Communities/Resolution) | Re-run on full merged graph |

**Key implementation details:**
- `file_filter` parameter on `chunk_text_data()`, `chunk_table_data()`, `create_embeddings()`, `extract_from_chunks_dir()`, `extract_from_chunks_dir_async()`, `run_extraction_async()` — filters files by PDF stem prefix
- Index rebuilding always uses `recreate=True` because `Document.id` uses random UUIDs (upsert would create duplicates) and BM25 has no incremental mode
- Manifest saved only after successful pipeline completion (crash = re-run processes same files)
- Modified files: `_cleanup_intermediate_files(stem)` deletes old processed/chunks/embeddings before reprocessing
- Deleted files: `--prune` flag removes orphan intermediates, manifest entries, and knowledge graph entities/edges from deleted source files. Without `--prune`, use `--full-reindex`.

### Schema Auto-Generation

The graph schema (`config/graph_schema.yaml`) can be auto-generated from corpus analysis via `cognidoc schema-generate` or automatically at first `cognidoc ingest` when no schema exists.

**Pipeline:**
```
data/sources/ → PDF conversion → Sample ≤100 PDFs (distributed across subfolders)
                                        ↓
                    Extract text from distributed pages (beginning/middle/end)
                    Adaptive char budget per doc (fewer docs → more text each)
                                        ↓
                    Stage A: Batch analysis (auto-sized batches, parallel LLM calls)
                              → themes, entity types, relationship types per batch
                                        ↓
                    Stage B: Synthesis (single LLM call)
                              → deduplicated, unified YAML schema
                                        ↓
                                config/graph_schema.yaml
```

**Sampling strategy:**
- If `data/sources/` has subfolders: distribute `max_docs` equally across subfolders
- If flat structure: random sample up to `max_docs`
- Non-generic folder/file names are used as metadata signals for the LLM

**Adaptive page/character budget:**
- Total text budget: 300K chars across all sampled documents
- `chars_per_doc = clamp(300K / num_docs, min=3000, max=30000)`
- With 100 docs: 3K chars/doc (~3 pages). With 3 docs: 30K chars/doc (~20 pages)
- Pages sampled from 3 zones: beginning (40%), middle (30%), end (30%)
- Near-empty pages (< 100 chars) are skipped automatically
- `batch_size` auto-computed to keep ~40K chars per LLM call

**Design notes (hyperparameters):**
- `max_docs=100`: Pragmatic default. Topic modeling research shows 10-20% of a corpus suffices to identify main themes, with diminishing returns beyond that. 100 docs covers most use cases; configurable via `--max-docs`.
- Generic name detection filters out uninformative file/folder names (e.g., `doc_1`, `scan`, `untitled`, purely numeric).

**Fallback chain:**
1. Corpus-based two-stage pipeline (primary)
2. Legacy single-shot `generate_schema_from_documents()` (if <2 valid batches)
3. Template-based `generate_schema_from_answers("generic")` (if no text extracted)

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
- `aggregate_graph(operation, entity_type, attribute, attribute_value)` - COUNT/COUNT_BY/LIST/GROUP_BY/STATS on NetworkX
- `database_stats(list_documents=True/False)` - returns unique source documents count
- `exhaustive_search` - corpus-wide BM25 keyword search
- `synthesize`, `verify_claim`, `ask_clarification`, `final_answer`

**Document listing patterns** (`complexity.py`): Queries like "liste les documents", "quels documents", "list all docs" trigger agent path via `DATABASE_META_PATTERNS`.

**Aggregation patterns** (`complexity.py`): Queries like "combien de", "how many", "average", "list all" trigger agent path via `AGGREGATION_PATTERNS`.

## Configuration

### Path Resolution (`constants.py`)

CogniDoc uses the **current working directory** as project root:

| Variable | Default | Env Override |
|----------|---------|--------------|
| `PROJECT_DIR` | `cwd()` | `COGNIDOC_PROJECT_DIR` |
| `DATA_DIR` | `PROJECT_DIR/data` | `COGNIDOC_DATA_DIR` |
| `PACKAGE_DIR` | `Path(__file__).parent` | — (auto-resolved to installed package location) |

`PACKAGE_DIR` is used to locate embedded resources like prompt templates in `src/cognidoc/prompts/`.

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

**Note:** `constants.py` defines `GEMINI_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", ...)` — so the Python variable reads from `DEFAULT_LLM_MODEL`, while `llm_providers.py`'s runtime provider selection checks env var `GEMINI_LLM_MODEL`. Setting either will affect Gemini model selection, but through different code paths.

### Vision Model Configuration

Vision models are configured **separately** from LLM models (used for image/table extraction during ingestion):

| Env Var | Default |
|---------|---------|
| `DEFAULT_VISION_PROVIDER` | gemini |
| `DEFAULT_VISION_MODEL` | gemini-3-flash-preview |
| `VISION_TEMPERATURE` | 0.2 |
| `VISION_TOP_P` | 0.85 |
| `OLLAMA_VISION_MODEL` | qwen3-vl:8b-instruct |
| `OPENAI_VISION_MODEL` | gpt-4o |
| `ANTHROPIC_VISION_MODEL` | claude-sonnet-4-20250514 |

The ingestion pipeline can also use a separate LLM model via `INGESTION_LLM_MODEL` (for entity extraction, community summaries, table descriptions).

### Tools & Processing (non-LLM)

| Stage | Tool/Library | Details |
|-------|--------------|---------|
| Office → PDF | LibreOffice | DOCX, PPTX, XLSX, HTML conversion |
| PDF text extraction | PyMuPDF | Schema generation (first N pages) |
| PDF → Images | pdf2image | 600 DPI, PNG format |
| Layout Detection | YOLOv11x | `yolov11x_best.pt` (~109 MB, optional) |
| Vector Storage | Qdrant | Embedded mode (no server) |
| Sparse Index | BM25 | In-memory keyword index |
| Graph Storage | NetworkX (default) / Kùzu (optional) | Pluggable backend via `GRAPH_BACKEND` env var, Louvain community detection |

### LLM Models by Pipeline Stage

**Ingestion Pipeline:**

| Stage | Default Model | Provider |
|-------|---------------|----------|
| Schema generation (batch + synthesis) | `gemini-3-flash-preview` | Gemini |
| Document parsing | `ibm/granite-docling:258m-bf16` | Ollama |
| Text/table extraction | `gemini-3-flash-preview` (vision) | Gemini |
| Image descriptions | `gemini-3-flash-preview` (vision) | Gemini |
| Table descriptions | `gemini-3-flash-preview` | Gemini |
| Embeddings | `qwen3-embedding:4b-q8_0` | Ollama |
| Entity extraction | `gemini-3-flash-preview` | Gemini |
| Entity resolution | `gemini-3-flash-preview` | Gemini |
| Community summaries | `gemini-3-flash-preview` | Gemini |

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

See `.env.example` for the complete list of all configurable environment variables with inline documentation.

### Key Constants (overridable via `.env`)

- `YOLO_CONFIDENCE_THRESHOLD`: 0.2
- `MAX_CHUNK_SIZE`: 512 tokens
- `SEMANTIC_CHUNK_BUFFER`: 5 sentences
- `CHUNK_OVERLAP_PERCENTAGE`: 0.1
- `TOP_K_RETRIEVED_CHILDREN`: 10
- `TOP_K_RERANKED_PARENTS`: 5
- `TOP_K_REFS`: 5
- `HYBRID_DENSE_WEIGHT`: 0.6
- `BM25_K1`: 1.5, `BM25_B`: 0.75
- `OLLAMA_EMBED_MODEL`: qwen3-embedding:4b-q8_0
- `ENABLE_CONTEXTUAL_COMPRESSION`: false
- `CONTEXT_WINDOW`: 128000, `MEMORY_WINDOW`: 64000

### Graph Backend Configuration

- `GRAPH_BACKEND`: `networkx` (default) or `kuzu`
- `KUZU_DB_DIR`: Path to Kùzu database directory (default: `data/indexes/kuzu_db`)

### Entity Resolution Configuration

- `ENABLE_ENTITY_RESOLUTION`: true
- `ENTITY_RESOLUTION_SIMILARITY_THRESHOLD`: 0.75
- `ENTITY_RESOLUTION_LLM_CONFIDENCE`: 0.7
- `ENTITY_RESOLUTION_MAX_CONCURRENT`: 4
- `ENTITY_RESOLUTION_USE_LLM_DESCRIPTIONS`: true
- `ENTITY_RESOLUTION_CACHE_ENABLED`: true, `ENTITY_RESOLUTION_CACHE_TTL_HOURS`: 24

### Checkpoint Configuration

- `MAX_CONSECUTIVE_QUOTA_ERRORS`: 5 (auto-pause on API quota hits)
- `CHECKPOINT_SAVE_INTERVAL`: 10

### MODEL_SPECS (`constants.py`)

Official provider parameters are defined in `MODEL_SPECS` dict. Auto-load specs via:
```python
config = LLMConfig.from_model("gemini-3-flash-preview")  # Loads all specs automatically
```

### Dynamic MEMORY_WINDOW (`helpers.py`)

Conversation memory adapts to 50% of the model's context window (unless `MEMORY_WINDOW` env var is set explicitly):
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

YOLO detection requires `models/YOLOv11/yolov11x_best.pt` (~109 MB, gitignored). Without it, falls back to simple page-level extraction. Note: torch and torchvision versions must be compatible when installing the `yolo` extra.

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
| `src/cognidoc/prompts/` | LLM prompt templates (entity extraction, community summaries, etc.) |
| `src/cognidoc/static/` | Static web assets (`graph-viewer.html` — D3.js graph visualization) |

## Tests

**Important:** Qdrant embedded only allows one client per storage folder. The app and tests cannot run simultaneously against the same data directory.

| Module | Tests | Description |
|--------|-------|-------------|
| `test_00_e2e_pipeline.py` | 9 | E2E pipeline (runs first to avoid Qdrant lock) |
| `test_agent.py` | 60 | Agent ReAct loop |
| `test_agent_tools.py` | 57 | Tool implementations |
| `test_api.py` | 10 | CogniDoc public API, config validation |
| `test_create_embeddings.py` | 25 | Embedding generation (metadata, caching, file filter, stem dates) |
| `test_benchmark.py` | 12 | Precision/recall benchmark with reranking comparison |
| `test_build_indexes.py` | 10 | Vector/keyword index building |
| `test_checkpoint.py` | 32 | Checkpoint/resume system |
| `test_chunking.py` | 29 | Text/table chunking (parent/child, file filter, hard_split, table overlap) |
| `test_cli.py` | 35 | CLI commands and argument parsing |
| `test_complexity.py` | 60 | Query complexity evaluation |
| `test_conversion.py` | 29 | Document format conversion |
| `test_e2e_language_and_count.py` | 24 | Language detection (FR/EN/ES/DE) |
| `test_entity_resolution.py` | 34 | Entity resolution (blocking, matching, clustering, merging) |
| `test_extract_entities.py` | 28 | Entity/relationship extraction, JSON parsing, prompts, attributes |
| `test_graph_backend.py` | 51 | Graph backend ABC (NetworkX + Kùzu parametrized, CRUD, traversal, export) |
| `test_graph_config.py` | 30 | GraphRAG schema loading and validation |
| `test_graph_retrieval.py` | 16 | Graph retrieval cache, retriever, result dataclass |
| `test_helpers.py` | 34 | Token counting, chat history, query parsing |
| `test_hybrid_retriever.py` | 17 | Hybrid retrieval, cache serialization, context manager |
| `test_incremental_ingestion.py` | 28 | Incremental ingestion manifest and pipeline |
| `test_ingestion_manifest.py` | 22 | Ingestion manifest CRUD and file tracking |
| `test_knowledge_graph.py` | 37 | Knowledge graph CRUD, traversal, persistence, pruning |
| `test_optimizations.py` | 110 | Pipeline optimizations, caching, reranking parser |
| `test_pipeline_stages.py` | 22 | Individual pipeline stage unit tests |
| `test_providers.py` | 32 | LLM/Embedding providers |
| `test_query_orchestrator.py` | 31 | Query classification, routing, weight config |
| `test_schema_generation.py` | 85 | Corpus-based schema generation (sampling, distributed pages, LLM pipeline, fallbacks) |
| `test_chat_history.py` | 25 | Chat history CRUD, auto-title, export, concurrent access |
| `test_integration.py` | 26 | Integration tests: query pipeline, metrics, API endpoints, chat history, upload |

**Test Infrastructure:**
- `conftest.py` provides session-scoped `cognidoc_session` fixture to avoid Qdrant lock conflicts
- `conftest.py` provides `release_qdrant_lock` fixture for tests that need exclusive Qdrant access (e.g., full pipeline ingestion)
- `--run-slow` flag enables slow E2E and benchmark tests
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

# Graph viewer
GET  /graph-viewer              # Standalone D3.js graph visualization page
GET  /api/graph/data            # JSON: nodes, edges, communities, stats

# Other endpoints
POST /api/reset_conversation
POST /api/refresh_metrics
POST /api/export_csv
POST /api/export_json
```
