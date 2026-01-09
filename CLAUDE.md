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
- `database_stats(list_documents=True/False)` - stats and document listing
- `synthesize`, `verify_claim`, `ask_clarification`, `final_answer`

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

Language rules are enforced in prompts to ensure responses match query language (French/English).

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

## Configuration

### Key Constants (`src/constants.py`)

All settings overridable via `.env`:
- `YOLO_CONFIDENCE_THRESHOLD`: 0.2
- `MAX_CHUNK_SIZE`: 512 tokens
- `SEMANTIC_CHUNK_BUFFER`: 5 sentences
- `TOP_K_RETRIEVED_CHILDREN`: 10
- `TOP_K_RERANKED_PARENTS`: 5
- `HYBRID_DENSE_WEIGHT`: 0.6 (dense vs BM25 balance)

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
