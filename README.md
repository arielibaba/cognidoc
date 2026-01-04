# Advanced Hybrid RAG

A document processing and retrieval pipeline that combines **Vector RAG** and **GraphRAG** for intelligent document querying. Converts PDFs into a searchable knowledge base with a chat interface.

## Features

- **Hybrid RAG**: Combines vector similarity search with knowledge graph traversal
- **GraphRAG**: Automatic entity/relationship extraction with community detection
- **Multi-modal Document Processing**: Handles text, tables, and images from PDFs
- **YOLO Object Detection**: Automatically detects and extracts tables, pictures, and text regions
- **Semantic Chunking**: Uses embeddings to create semantically coherent text chunks
- **Hierarchical Retrieval**: Parent-child document structure for context-aware search
- **Query Routing**: Automatically routes queries to vector or graph retrieval based on query type
- **LLM Reranking**: Improves retrieval quality using LLM-based relevance scoring
- **Local Inference**: All models run locally via Ollama

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) running locally
- Required Ollama models (run these commands):

```bash
ollama pull granite3.3:8b          # LLM for generation and reranking
ollama pull qwen3-vl:8b-instruct   # Vision model for image descriptions
ollama pull qwen3-embedding:0.6b   # Embedding model
ollama pull ibm/granite-docling:258m-bf16  # Document parsing
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-hybrid-rag.git
cd advanced-hybrid-rag

# Install dependencies (requires uv package manager)
make install

# Or using pip
pip install -e .
```

## Project Structure

```
advanced-hybrid-rag/
├── config/
│   └── graph_schema.yaml          # GraphRAG configuration
├── data/                          # All pipeline data
│   ├── pdfs/                      # Input: Place your PDF files here
│   ├── images/                    # Generated: Page images (600 DPI)
│   ├── detections/                # Generated: YOLO detection crops
│   ├── processed/                 # Generated: Extracted text/tables
│   ├── chunks/                    # Generated: Semantic chunks
│   ├── embeddings/                # Generated: Embedding vectors
│   ├── indexes/                   # Generated: Search indexes + knowledge graph
│   ├── vector_store/              # Generated: Qdrant vector database
│   └── cache/                     # Generated: Embedding cache
├── models/
│   └── YOLOv11/                   # YOLO model weights
├── src/                           # Source code
│   ├── prompts/                   # LLM prompt templates
│   └── utils/                     # Utility modules
└── experiments/                   # Jupyter notebooks
```

## Quick Start

### 1. Prepare Your Data

Create the required directories and add your PDFs:

```bash
mkdir -p data/pdfs
# Copy your PDF files to data/pdfs/
```

### 2. Configure GraphRAG (Optional)

Edit `config/graph_schema.yaml` to customize entity and relationship extraction for your domain:

```yaml
domain:
  name: "your-domain"
  description: "Description of your document domain"

entity_types:
  - name: "YourEntityType"
    description: "What this entity represents"
    examples: ["Example1", "Example2"]

relationship_types:
  - name: "YOUR_RELATIONSHIP"
    description: "What this relationship means"
    source_types: ["SourceEntityType"]
    target_types: ["TargetEntityType"]
```

### 3. Run the Ingestion Pipeline

Process your documents:

```bash
python -m src.run_ingestion_pipeline --vision-provider ollama
```

Pipeline options:
- `--skip-pdf`: Skip PDF to image conversion
- `--skip-yolo`: Skip YOLO detection
- `--skip-extraction`: Skip text/table extraction
- `--skip-descriptions`: Skip image descriptions
- `--skip-chunking`: Skip semantic chunking
- `--skip-embeddings`: Skip embedding generation
- `--skip-indexing`: Skip vector index building
- `--skip-graph`: Skip knowledge graph building
- `--force-reembed`: Re-embed all content (ignore cache)
- `--vision-provider`: Choose vision provider (ollama, gemini, openai, anthropic)
- `--graph-config`: Path to custom graph schema configuration

### 4. Launch the Chat Interface

Start the Gradio chat application:

```bash
python -m src.watchComplyChat_app
```

Chat options:
- `--no-rerank`: Disable LLM reranking for faster responses
- `--port`: Server port (default: 7860)
- `--share`: Create a public shareable link

## Pipeline Stages

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| PDF Conversion | Convert PDFs to images at 600 DPI | `data/pdfs/*.pdf` | `data/images/*.png` |
| YOLO Detection | Detect text, tables, pictures | `data/images/*.png` | `data/detections/` |
| Content Extraction | Extract text using DocLing | `data/detections/` | `data/processed/*.md` |
| Image Description | Describe images using vision LLM | `data/detections/` | `data/processed/*.txt` |
| Chunking | Semantic text chunking | `data/processed/` | `data/chunks/` |
| Embeddings | Generate vector embeddings | `data/chunks/` | `data/embeddings/` |
| Vector Indexing | Build vector search indexes | `data/embeddings/` | `data/indexes/`, `data/vector_store/` |
| Graph Extraction | Extract entities and relationships | `data/chunks/` | Extraction results |
| Graph Building | Build knowledge graph with communities | Extraction results | `data/indexes/knowledge_graph/` |

## GraphRAG Configuration

The `config/graph_schema.yaml` file controls entity and relationship extraction:

### Domain Configuration
```yaml
domain:
  name: "general"
  description: "General-purpose knowledge extraction"
```

### Entity Types
Define what types of entities to extract:
```yaml
entity_types:
  - name: "Concept"
    description: "Abstract ideas, principles, or methodologies"
    examples: ["machine learning", "data processing"]

  - name: "Tool"
    description: "Software, hardware, or instruments"
    examples: ["Python", "database", "API"]
```

### Relationship Types
Define how entities can be connected:
```yaml
relationship_types:
  - name: "USES"
    description: "One entity uses or requires another"
    source_types: ["Process", "Tool"]
    target_types: ["Tool", "Concept"]

  - name: "PRODUCES"
    description: "One entity produces or creates another"
    source_types: ["Process"]
    target_types: ["Concept", "Document"]
```

### Routing Configuration
Control when to use vector vs graph retrieval:
```yaml
routing:
  strategy: "hybrid"        # Options: hybrid, classifier, vector_only
  vector_weight: 0.5        # Weight for vector results
  graph_weight: 0.5         # Weight for graph results
```

## Query Routing

The system automatically classifies queries and adjusts retrieval strategy:

| Query Type | Example | Vector Weight | Graph Weight |
|------------|---------|---------------|--------------|
| Factual | "What is X?" | 0.6 | 0.4 |
| Relational | "How is X related to Y?" | 0.3 | 0.7 |
| Comparative | "Compare X and Y" | 0.4 | 0.6 |
| Exploratory | "List all concepts about X" | 0.2 | 0.8 |
| Procedural | "How to do X?" | 0.7 | 0.3 |

## Configuration

Key settings in `src/constants.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_CONFIDENCE_THRESHOLD` | 0.2 | YOLO detection confidence |
| `YOLO_IOU_THRESHOLD` | 0.8 | YOLO IOU threshold |
| `MAX_CHUNK_SIZE` | 512 | Maximum tokens per chunk |
| `BUFFER_SIZE` | 5 | Sentence buffer for chunking |
| `TOP_K_RETRIEVED_CHILDREN` | 10 | Number of chunks to retrieve |
| `TOP_K_RERANKED_PARENTS` | 5 | Number of results after reranking |
| `EMBED_MODEL` | qwen3-embedding:0.6b | Embedding model |

Environment variables can override defaults. Create a `.env` file:

```bash
OLLAMA_HOST=http://localhost:11434
DEFAULT_LLM_MODEL=granite3.3:8b
EMBED_MODEL=qwen3-embedding:0.6b
```

## Development

```bash
# Format code
make format

# Run linter
make lint

# Format and lint
make refactor
```

## Architecture

```
PDFs → Images (600 DPI) → YOLO Detection → Content Extraction
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
                                          Query Interface
```

**Hybrid Retrieval Flow:**
1. User query → Query rewriting (LLM)
2. Query analysis → Classify query type (factual, relational, etc.)
3. Parallel retrieval:
   - Vector search on child chunks (top-10)
   - Graph traversal for entity relationships
4. Result fusion with weighted scoring
5. LLM reranking (top-5 parents)
6. Context building + streaming response

## Dependencies

Core dependencies:
- `qdrant-client`: Vector database
- `ollama`: Local LLM inference
- `ultralytics`: YOLO object detection
- `pdf2image`: PDF conversion
- `gradio`: Web interface
- `tiktoken`: Token counting
- `networkx`: Knowledge graph storage and operations
- `pyyaml`: Configuration file parsing

## License

MIT
