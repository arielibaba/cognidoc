# CogniDoc

**Intelligent Document Assistant** powered by Hybrid RAG (Vector + GraphRAG).

A document processing and retrieval pipeline that combines **Vector RAG** and **GraphRAG** for intelligent document querying. Converts PDFs into a searchable knowledge base with a professional chat interface.

## Features

- **Hybrid RAG**: Combines vector similarity search with knowledge graph traversal
- **GraphRAG**: Automatic entity/relationship extraction with community detection
- **Multi-modal Document Processing**: Handles text, tables, and images from PDFs
- **YOLO Object Detection**: Automatically detects and extracts tables, pictures, and text regions
- **Semantic Chunking**: Uses embeddings to create semantically coherent text chunks
- **Hierarchical Retrieval**: Parent-child document structure for context-aware search
- **Query Routing**: Automatically routes queries to vector or graph retrieval based on query type
- **LLM Reranking**: Improves retrieval quality using LLM-based relevance scoring
- **Multi-Provider Support**: Ollama (local), Gemini, OpenAI, Anthropic
- **No LangChain/LlamaIndex**: Direct Qdrant and Ollama integration for fine-grained control

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) running locally
- Required Ollama models:

```bash
ollama pull granite3.3:8b          # LLM for generation and reranking
ollama pull qwen3-vl:8b-instruct   # Vision model for image descriptions
ollama pull qwen3-embedding:0.6b   # Embedding model
ollama pull ibm/granite-docling:258m-bf16  # Document parsing
```

## Installation

```bash
# Clone the repository
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc

# Install dependencies (requires uv package manager)
make install

# Or using pip
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

```bash
mkdir -p data/pdfs
# Copy your PDF files to data/pdfs/
```

### 2. Run the Ingestion Pipeline

```bash
python -m src.run_ingestion_pipeline --vision-provider ollama
```

### 3. Launch CogniDoc

```bash
python -m src.cognidoc_app
```

Access the chat interface at `http://localhost:7860`

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INGESTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PDFs ──► Images (600 DPI) ──► YOLO Detection ──► Content Extraction        │
│                                     │                    │                  │
│                              ┌──────┴──────┐      ┌──────┴──────┐          │
│                              │   Tables    │      │    Text     │          │
│                              │  Pictures   │      │   Regions   │          │
│                              └──────┬──────┘      └──────┬──────┘          │
│                                     │                    │                  │
│                                     └────────┬───────────┘                  │
│                                              ▼                              │
│                                     Semantic Chunking                       │
│                                     (Parent + Child)                        │
│                                              │                              │
│                    ┌─────────────────────────┼─────────────────────────┐    │
│                    ▼                         ▼                         ▼    │
│           Vector Embeddings          Entity Extraction         Table Summary│
│                    │                         │                         │    │
│                    ▼                         ▼                         │    │
│           Qdrant Vector Store        Knowledge Graph ◄─────────────────┘    │
│           (child_documents)          (NetworkX + Communities)               │
│                    │                         │                              │
└────────────────────┼─────────────────────────┼──────────────────────────────┘
                     │                         │
                     └────────────┬────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RETRIEVAL SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query ──► Query Rewriting ──► Query Analysis ──► Query Routing        │
│                                              │                              │
│                              ┌───────────────┼───────────────┐              │
│                              ▼               ▼               ▼              │
│                       Vector Search    Graph Traversal   Community          │
│                       (Similarity)     (Relationships)   (Global)           │
│                              │               │               │              │
│                              └───────────────┼───────────────┘              │
│                                              ▼                              │
│                                      Result Fusion                          │
│                                    (Weighted Scoring)                       │
│                                              │                              │
│                                              ▼                              │
│                                      LLM Reranking                          │
│                                              │                              │
│                                              ▼                              │
│                                   Streaming Response                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### Ingestion Pipeline (`src/run_ingestion_pipeline.py`)

Orchestrates the 10-stage document processing pipeline:

| Stage | Module | Description |
|-------|--------|-------------|
| 1. PDF Conversion | `convert_pdf_to_image.py` | Converts PDFs to 600 DPI images |
| 2. Object Detection | `extract_objects_from_image.py` | YOLOv11 detects text, tables, pictures |
| 3. Text Extraction | `parse_image_with_text.py` | Granite-DocLing extracts document text |
| 4. Table Extraction | `parse_image_with_table.py` | Converts tables to markdown format |
| 5. Image Description | `create_image_description.py` | Vision LLM describes images (async) |
| 6. Text Chunking | `chunk_text_data.py` | Semantic chunking with breakpoint detection |
| 7. Table Chunking | `chunk_table_data.py` | Chunks and summarizes tables |
| 8. Embeddings | `create_embeddings.py` | Generates vectors with SHA256 caching |
| 9. Indexing | `build_indexes.py` | Builds Qdrant vector + keyword indexes |
| 10. GraphRAG | `extract_entities.py` + `knowledge_graph.py` | Extracts entities, builds graph |

#### Semantic Chunker (`src/chunk_text_data.py`)

Custom implementation (no LangChain) that creates semantically coherent chunks:

1. **Sentence splitting**: Regex-based segmentation
2. **Grouping**: Overlapping windows with buffer context (5 sentences)
3. **Embedding**: Vector for each sentence group
4. **Breakpoint detection**: Cosine similarity drops below 95th percentile indicate topic boundaries
5. **Output**: Parent chunks (full context) + Child chunks (for vector search)

#### Knowledge Graph (`src/knowledge_graph.py`)

NetworkX-based graph with:

- **Entity deduplication**: Case-insensitive merging with description updates
- **Community detection**: Louvain algorithm for topic clustering
- **Community summaries**: LLM-generated 2-3 sentence descriptions
- **Multi-hop traversal**: Configurable depth (default: 3 hops)
- **Path finding**: Discovers connections between entities

#### Hybrid Retriever (`src/hybrid_retriever.py`)

Combines vector and graph retrieval with intelligent query routing:

```python
QueryType.FACTUAL      # "What is X?"        → vector=0.6, graph=0.4
QueryType.RELATIONAL   # "How is X related?" → vector=0.3, graph=0.7
QueryType.COMPARATIVE  # "Compare X and Y"   → vector=0.4, graph=0.6
QueryType.EXPLORATORY  # "List all..."       → vector=0.2, graph=0.8
QueryType.PROCEDURAL   # "How to do X?"      → vector=0.7, graph=0.3
```

#### Graph Retriever (`src/graph_retrieval.py`)

Three retrieval strategies:

1. **Entity-based**: Extract mentions → get neighbors → collect relationships
2. **Relationship-based**: Pattern matching → path finding → specific relationships
3. **Community-based**: Embedding similarity to community summaries (for global queries)

### Utilities (`src/utils/`)

| Module | Purpose |
|--------|---------|
| `rag_utils.py` | Document, VectorIndex, KeywordIndex, reranking functions |
| `llm_providers.py` | Multi-provider abstraction (Ollama, Gemini, OpenAI, Anthropic) |
| `embedding_cache.py` | SHA256-based SQLite cache for embeddings |
| `logger.py` | Structured logging with pipeline timing metrics |

### Configuration (`src/constants.py`)

Central configuration hub with environment variable overrides:

| Category | Settings |
|----------|----------|
| **Paths** | PDF_DIR, IMAGE_DIR, DETECTION_DIR, PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, INDEX_DIR |
| **YOLO** | Confidence: 0.2, IOU: 0.8 |
| **Chunking** | Max size: 512 tokens, Buffer: 5 sentences, Breakpoint: 95th percentile |
| **Retrieval** | Top-K children: 10, Top-K reranked: 5, Top-K refs: 3 |
| **Models** | Configurable per provider (Ollama, Gemini, OpenAI, Anthropic) |

## Project Structure

```
cognidoc/
├── config/
│   └── graph_schema.yaml          # GraphRAG entity/relationship configuration
├── data/
│   ├── pdfs/                      # Input: PDF files
│   ├── images/                    # Stage 1: Page images (600 DPI)
│   ├── detections/                # Stage 2: YOLO detection crops
│   ├── processed/                 # Stage 3-5: Extracted text/tables/descriptions
│   ├── chunks/                    # Stage 6-7: Parent + child chunks
│   ├── embeddings/                # Stage 8: Embedding vectors (JSON)
│   ├── indexes/                   # Stage 9-10: Vector/keyword/graph indexes
│   │   ├── child_documents/       # Vector index metadata
│   │   ├── parent_documents/      # Keyword index metadata
│   │   └── knowledge_graph/       # Graph persistence (gpickle + JSON)
│   ├── vector_store/              # Qdrant database files
│   └── cache/                     # Embedding cache (SQLite)
├── models/
│   └── YOLOv11/                   # YOLO model weights
├── src/
│   ├── run_ingestion_pipeline.py  # Main pipeline orchestrator
│   ├── cognidoc_app.py            # Gradio chat interface
│   ├── hybrid_retriever.py        # Vector + Graph fusion
│   ├── knowledge_graph.py         # NetworkX graph with communities
│   ├── extract_entities.py        # LLM entity/relationship extraction
│   ├── graph_retrieval.py         # Graph query strategies
│   ├── graph_config.py            # YAML config loader
│   ├── constants.py               # All configuration constants
│   ├── helpers.py                 # Query rewriting, reranking utilities
│   ├── prompts/                   # LLM prompt templates (markdown)
│   └── utils/                     # RAG utilities, logging, caching
└── experiments/                   # Jupyter notebooks
```

## Pipeline Options

```bash
python -m src.run_ingestion_pipeline [OPTIONS]

# Skip stages
--skip-pdf            # Skip PDF to image conversion
--skip-yolo           # Skip YOLO detection
--skip-extraction     # Skip text/table extraction
--skip-descriptions   # Skip image descriptions
--skip-chunking       # Skip semantic chunking
--skip-embeddings     # Skip embedding generation
--skip-indexing       # Skip vector index building
--skip-graph          # Skip knowledge graph building

# Configuration
--vision-provider     # ollama, gemini, openai, anthropic
--graph-config        # Path to custom graph schema
--force-reembed       # Re-embed all (ignore cache)
--no-cache            # Disable embedding cache
```

## CogniDoc App Options

```bash
python -m src.cognidoc_app [OPTIONS]

--no-rerank           # Disable LLM reranking (faster)
--port PORT           # Server port (default: 7860)
--share               # Create public shareable link
```

## GraphRAG Configuration

Edit `config/graph_schema.yaml` to customize extraction:

```yaml
domain:
  name: "your-domain"
  description: "Domain description guides LLM extraction"

entity_types:
  - name: "Concept"
    description: "Abstract ideas and principles"
    examples: ["machine learning", "data processing"]
    attributes: ["definition", "category"]

  - name: "Tool"
    description: "Software and technologies"
    examples: ["Python", "PostgreSQL"]

relationship_types:
  - name: "USES"
    description: "One entity uses another"
    source_types: ["Process", "Tool"]
    target_types: ["Tool", "Concept"]

extraction:
  confidence_threshold: 0.7
  max_entities_per_chunk: 15
  max_relationships_per_chunk: 20

graph_settings:
  entity_merge_threshold: 0.85
  community_resolution: 1.0
  max_traversal_depth: 3

routing:
  strategy: "hybrid"    # hybrid, classifier, vector_only
  vector_weight: 0.5
  graph_weight: 0.5
```

## Environment Variables

Create a `.env` file to override defaults:

```bash
# LLM Providers
DEFAULT_LLM_PROVIDER=gemini          # gemini, ollama, openai, anthropic
DEFAULT_VISION_PROVIDER=gemini

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=granite3.3:8b
OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b

# Generation
LLM_TEMPERATURE=0.7
CONTEXT_WINDOW=128000

# Retrieval
TOP_K_RETRIEVED_CHILDREN=10
TOP_K_RERANKED_PARENTS=5
ENABLE_RERANKING=true
```

## Development

```bash
make format    # Format code with black
make lint      # Run pylint
make refactor  # Format + lint
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `qdrant-client` | Vector database |
| `ollama` | Local LLM inference |
| `ultralytics` | YOLO object detection |
| `pdf2image` | PDF conversion |
| `gradio` | Web interface |
| `tiktoken` | Token counting |
| `networkx` | Knowledge graph |
| `python-louvain` | Community detection |
| `pyyaml` | Configuration parsing |

## License

MIT
