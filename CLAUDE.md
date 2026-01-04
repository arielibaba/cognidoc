# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogniDoc is a document processing and retrieval pipeline combining **Vector RAG** and **GraphRAG**:
- Converts PDFs to images and detects objects (tables, text, pictures) using YOLO
- Parses content using Granite-DocLing model
- Extracts entities and relationships for knowledge graph
- Generates semantic chunks and embeddings
- Builds hybrid indexes (vector + knowledge graph) for intelligent retrieval
- Provides a Gradio chat interface with query routing and LLM reranking

## Build and Development Commands

```bash
# Package management (uses uv)
make install          # Install dependencies
make sync             # Sync environment with lock file

# Code quality
make format           # Format code with black
make lint             # Run pylint on source
make refactor         # Format and lint

# Interactive setup wizard (recommended for new users)
python -m src.setup

# Manual pipeline execution
python -m src.run_ingestion_pipeline --vision-provider ollama

# Pipeline skip options
--skip-pdf            # Skip PDF to image conversion
--skip-yolo           # Skip YOLO detection
--skip-extraction     # Skip text/table extraction
--skip-descriptions   # Skip image descriptions
--skip-chunking       # Skip semantic chunking
--skip-embeddings     # Skip embedding generation
--skip-indexing       # Skip vector index building
--skip-graph          # Skip knowledge graph building
--force-reembed       # Re-embed all content (ignore cache)

# Launch CogniDoc chat interface
python -m src.cognidoc_app
python -m src.cognidoc_app --no-rerank    # Faster, skip LLM reranking
python -m src.cognidoc_app --share        # Create public link
```

## Architecture

### Pipeline Stages

```
PDFs → Images (600 DPI) → YOLO Detection → Text/Table/Image Extraction
                                                    ↓
                                            Semantic Chunks
                                                    ↓
                              ┌─────────────────────┴─────────────────────┐
                              ↓                                           ↓
                     Vector Embeddings                         Entity/Relationship
                              ↓                                   Extraction
                     Qdrant Vector Store                              ↓
                              ↓                               Knowledge Graph
                              └─────────────────────┬─────────────────────┘
                                                    ↓
                                          Hybrid Retriever
                                                    ↓
                                          CogniDoc Chat UI
```

### Key Modules

- **`src/setup.py`**: Interactive setup wizard (provider config, model verification, pipeline execution)
- **`src/run_ingestion_pipeline.py`**: Main pipeline orchestrator (async)
- **`src/cognidoc_app.py`**: Gradio chat application with hybrid retrieval
- **`src/hybrid_retriever.py`**: Combines vector and graph retrieval with query routing
- **`src/knowledge_graph.py`**: NetworkX-based knowledge graph with community detection
- **`src/extract_entities.py`**: LLM-based entity/relationship extraction
- **`src/graph_config.py`**: GraphRAG configuration loader (from `config/graph_schema.yaml`)
- **`src/constants.py`**: All configuration values and path definitions
- **`src/utils/rag_utils.py`**: Custom Document, VectorIndex, KeywordIndex, and reranking utilities

### Hybrid Retrieval Flow

1. Query rewriting via LLM
2. Query analysis → classify type (factual, relational, comparative, exploratory, procedural)
3. Route to vector search and/or graph traversal based on query type
4. Result fusion with weighted scoring
5. LLM reranking (optional)
6. Streaming response generation

### Data Flow

| Stage | Input Directory | Output Directory |
|-------|----------------|------------------|
| PDF Conversion | `data/pdfs/` | `data/images/` |
| YOLO Detection | `data/images/` | `data/detections/` |
| Content Extraction | `data/detections/` | `data/processed/` |
| Chunking | `data/processed/` | `data/chunks/` |
| Embeddings | `data/chunks/` | `data/embeddings/` |
| Vector Indexing | `data/embeddings/` | `data/indexes/`, `data/vector_store/` |
| Knowledge Graph | `data/chunks/` | `data/indexes/knowledge_graph/` |

### External Services

Default providers are configurable via `.env` (defaults: Gemini for LLM/vision, Ollama for embeddings):
- **Ollama** at `http://localhost:11434`: embeddings (qwen3-embedding:0.6b), local LLM fallback
- **Vision providers**: gemini, ollama, openai, anthropic (configurable)

### GraphRAG Configuration

Edit `config/graph_schema.yaml` to customize:
- Entity types and their descriptions
- Relationship types between entities
- Routing strategy (hybrid, classifier, vector_only)
- Vector/graph weight balance per query type

## Configuration

Key settings in `src/constants.py` (overridable via `.env`):
- YOLO thresholds: confidence=0.2, IOU=0.8
- Chunking: max 512 tokens, buffer 5 tokens
- Retrieval: top-10 children → LLM rerank → top-5 parents
- Ollama timeout: 180 seconds

## Prompt Templates

All prompts are in `src/prompts/` as markdown files, covering:
- Image/text/table extraction
- Query rewriting and expansion
- Final answer generation
