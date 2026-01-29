# CogniDoc

[![CI](https://github.com/arielibaba/cognidoc/actions/workflows/ci.yml/badge.svg)](https://github.com/arielibaba/cognidoc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Intelligent Document Assistant** powered by Hybrid RAG (Vector + GraphRAG).

Transform any document collection into a searchable knowledge base with intelligent query routing, multi-step reasoning, and a professional chat interface.

> *Assistant documentaire intelligent propulsé par RAG Hybride (Vecteur + GraphRAG). Transformez n'importe quelle collection de documents en base de connaissances interrogeable.*

---

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid RAG** | Combines vector similarity search with knowledge graph traversal |
| **Agentic RAG** | Multi-step reasoning agent with 9 specialized tools |
| **Incremental Ingestion** | Only processes new/modified documents (SHA-256 manifest tracking) |
| **Auto Schema Generation** | Two-stage LLM pipeline generates GraphRAG schema from corpus analysis |
| **Entity Resolution** | Semantic deduplication of graph entities (4-phase pipeline) |
| **Multi-Format** | PDF, DOCX, PPTX, XLSX, HTML, Markdown, images |
| **Multi-Language** | Automatic FR/EN/ES/DE detection and response |
| **YOLO Detection** | Automatic table/image/text region detection (optional) |
| **Conversation Memory** | Context-aware follow-up questions |
| **Multi-Provider** | Gemini, OpenAI, Anthropic, Ollama |
| **Metrics Dashboard** | Performance tracking with Plotly charts and CSV/JSON export |

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [REST API](#rest-api)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick Start

```bash
# 1. Install
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# 2. Create project & add documents
mkdir my-project && cd my-project
mkdir -p data/sources
cp /path/to/your/documents/* data/sources/

# 3. Configure API key
echo "GOOGLE_API_KEY=your-key" > .env

# 4. Run
python -c "
from cognidoc import CogniDoc
doc = CogniDoc()
doc.ingest('./data/sources/')
doc.launch_ui(port=7860)
"
```

Open http://localhost:7860 - The schema wizard will guide you through first-time setup.

---

## Installation

### Requirements

| Requirement | Purpose | Required |
|-------------|---------|----------|
| Python 3.10+ | Runtime | Yes |
| API key (Gemini/OpenAI/Anthropic) | LLM inference | Yes (one) |
| [Ollama](https://ollama.ai/) | Local inference & embeddings | Recommended |
| [LibreOffice](https://www.libreoffice.org/) | Office document conversion | Optional |

### Install Options

```bash
# Full installation (recommended)
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Minimal (cloud-only, no YOLO)
pip install "cognidoc[ui,cloud] @ git+https://github.com/arielibaba/cognidoc.git"

# Development mode
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc
pip install -e ".[all,dev]"
```

### Optional Dependencies

| Group | Contents | Install |
|-------|----------|---------|
| `ui` | Gradio, Plotly, Pandas | `pip install "cognidoc[ui]"` |
| `yolo` | YOLO detection (PyTorch, Ultralytics) | `pip install "cognidoc[yolo]"` |
| `ollama` | Local Ollama inference | `pip install "cognidoc[ollama]"` |
| `cloud` | All cloud providers | `pip install "cognidoc[cloud]"` |
| `conversion` | Office document conversion | `pip install "cognidoc[conversion]"` |
| `all` | Everything above | `pip install "cognidoc[all]"` |
| `dev` | pytest, black, pylint, mypy | `pip install "cognidoc[dev]"` |

### Ollama Setup (Recommended)

```bash
# Install Ollama from https://ollama.ai/
# Then pull required models:
ollama pull granite3.3:8b               # LLM generation
ollama pull qwen3-embedding:4b-q8_0     # Embeddings (default)
ollama pull qwen3-vl:8b-instruct        # Vision (optional)
```

### YOLO Model (Optional)

For document layout detection (tables, images, text regions):

```bash
mkdir -p models/YOLOv11
# Place yolov11x_best.pt (~109 MB) trained on DocLayNet/PubLayNet
```

Without YOLO, the system uses simple page-level extraction.

---

## Configuration

### LLM Providers

| Provider | Model | Embeddings | API Key Env Variable |
|----------|-------|------------|----------------------|
| **Gemini** (default) | `gemini-3-flash-preview` | `text-embedding-004` | `GOOGLE_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku` | - | `ANTHROPIC_API_KEY` |
| **Ollama** | `granite3.3:8b` | `qwen3-embedding:4b-q8_0` | Local (no key) |

### Environment Variables (.env)

```bash
# API Keys (at least one required)
GOOGLE_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Provider settings
DEFAULT_LLM_PROVIDER=gemini          # gemini, openai, anthropic, ollama
DEFAULT_LLM_MODEL=gemini-3-flash-preview
LLM_TEMPERATURE=0.7

# Retrieval tuning
TOP_K_RETRIEVED_CHILDREN=10          # Initial retrieval count
TOP_K_RERANKED_PARENTS=5             # Final count after reranking
HYBRID_DENSE_WEIGHT=0.6              # Dense vs BM25 balance (0-1)

# YOLO detection
YOLO_CONFIDENCE_THRESHOLD=0.2
```

### Project Structure

```
your-project/
├── .env                      # API keys and configuration
├── config/
│   └── graph_schema.yaml     # GraphRAG schema (auto-generated)
├── models/
│   └── YOLOv11/
│       └── yolov11x_best.pt  # Optional (~109 MB)
└── data/
    ├── sources/              # Your documents (PDF, DOCX, etc.)
    ├── pdfs/                 # Converted PDFs
    ├── images/               # Page images (600 DPI)
    ├── chunks/               # Semantic chunks
    ├── indexes/              # Search indexes
    ├── vector_store/         # Qdrant database
    └── cache/                # SQLite caches
```

---

## Usage

### Python API

```python
from cognidoc import CogniDoc

# Initialize
doc = CogniDoc(
    llm_provider="gemini",        # gemini, ollama, openai, anthropic
    embedding_provider="ollama",  # ollama, gemini, openai
    use_yolo=True,                # Enable YOLO detection
    use_graph=True,               # Enable GraphRAG
)

# Ingest documents (first time only)
doc.ingest("./data/sources/")

# Query
result = doc.query("What are the main topics?")
print(result.answer)
print(result.sources)

# Launch web interface
doc.launch_ui(port=7860, share=True)
```

### Interactive Setup Wizard

For first-time users, the wizard guides through provider setup, API keys, and ingestion:

```bash
python -m cognidoc.setup
```

---

## Architecture

### Ingestion Pipeline

```
┌─────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────────┐
│  Documents  │ ─▶ │    PDF     │ ─▶ │   Images    │ ─▶ │     YOLO*      │
│ (any format)│    │ Conversion │    │  (600 DPI)  │    │   Detection    │
└─────────────┘    └────────────┘    └─────────────┘    └───────┬────────┘
                                                                │
                   ┌────────────────────────────────────────────┘
                   ▼
         ┌─────────────────┐    ┌─────────────────┐
         │  Text/Table/    │ ─▶ │    Semantic     │
         │ Image Extraction│    │    Chunking     │
         └─────────────────┘    │ (Parent/Child)  │
                                └────────┬────────┘
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              ▼                                                     ▼
    ┌─────────────────────┐                              ┌─────────────────────┐
    │  Vector Embeddings  │                              │  Entity/Relation    │
    │   (Qdrant + BM25)   │                              │     Extraction      │
    └──────────┬──────────┘                              └──────────┬──────────┘
               │                                                    │
               └────────────────────────┬───────────────────────────┘
                                        ▼
                              ┌─────────────────────┐
                              │   Hybrid Retriever  │
                              │  (Vector + Graph)   │
                              └─────────────────────┘
```

*YOLO is optional - falls back to page-level extraction if disabled or model not found.

### Incremental Ingestion

The pipeline is **incremental by default**. A manifest (`data/indexes/ingestion_manifest.json`) tracks files by path, size, and SHA-256 hash.

| Scenario | Behavior |
|----------|----------|
| New files added | Only new files are processed |
| Existing file modified | Old intermediates cleaned up, file reprocessed |
| No changes detected | Pipeline exits immediately |
| `--full-reindex` flag | Full pipeline, manifest rebuilt |

### Query Processing

```
                                User Query
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Query Rewriter    │
                         │ (conversation ctx)  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌────────────────┐              ┌────────────────┐
           │   Classifier   │              │   Complexity   │
           │  (query type)  │              │   Evaluator    │
           └───────┬────────┘              └───────┬────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │ score < 0.35     │ 0.35-0.55         │ score ≥ 0.55
               ▼                   ▼                   ▼
        ┌────────────┐      ┌────────────┐      ┌────────────┐
        │ FAST PATH  │      │  ENHANCED  │      │   AGENT    │
        │(simple RAG)│      │    PATH    │      │   PATH     │
        └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
              │                   │                   │
              └───────────────────┴───────────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │ LLM Generation │
                         │ (same language │
                         │   as query)    │
                         └────────────────┘
```

### Query Type Routing

Query classification determines how vector and graph retrieval are weighted:

| Query Type | Example | Vector | Graph |
|------------|---------|--------|-------|
| **FACTUAL** | "What is X?" | 70% | 30% |
| **RELATIONAL** | "How are A and B related?" | 20% | 80% |
| **EXPLORATORY** | "List all main topics" | 0% | 100% |
| **PROCEDURAL** | "How to configure X?" | 80% | 20% |

If a weight is < 15%, that retriever is skipped entirely.

### 3-Level Fusion Retrieval

```
                              User Query
                                  │
                    ┌─────────────┴─────────────┐
                    │      Query Routing        │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                                               ▼
┌─────────────────────┐                       ┌─────────────────────┐
│  VECTOR RETRIEVAL   │                       │  GRAPH RETRIEVAL    │
└─────────┬───────────┘                       └─────────┬───────────┘
          │                                             │
    ┌─────┴─────┐                             ┌────────┴────────┐
    ▼           ▼                             ▼                 ▼
┌───────┐  ┌───────┐                   ┌───────────┐    ┌────────────┐
│ DENSE │  │SPARSE │                   │  Entity   │    │ Community  │
│Vector │  │ BM25  │                   │ Matching  │    │ Summaries  │
└───┬───┘  └───┬───┘                   └─────┬─────┘    └─────┬──────┘
    │          │                             │                │
    └────┬─────┘                             └────────┬───────┘
         │                                            │
         ▼                                            ▼
┌─────────────────┐                         ┌────────────────┐
│ LEVEL 1: HYBRID │                         │ Graph Context  │
│ FUSION (RRF)    │                         │ + Entities     │
│ α=0.6 default   │                         └───────┬────────┘
└────────┬────────┘                                 │
         │                                          │
         ▼                                          │
┌─────────────────┐                                 │
│ Child → Parent  │                                 │
│ + Reranking     │                                 │
└────────┬────────┘                                 │
         │                                          │
         └─────────────────┬────────────────────────┘
                           ▼
                 ┌─────────────────────┐
                 │  LEVEL 3: FINAL     │
                 │  FUSION (weighted   │
                 │  by query type)     │
                 └─────────────────────┘
```

| Level | Components | Method |
|-------|------------|--------|
| **1. Hybrid** | Dense + Sparse (BM25) | RRF: `α × 1/(60+rank_dense) + (1-α) × 1/(60+rank_sparse)` |
| **2. Parallel** | Vector ∥ Graph | Independent execution via ThreadPoolExecutor |
| **3. Final** | Vector + Graph results | Weighted fusion based on query type |

### Agentic RAG

Complex queries (score ≥ 0.55) trigger a ReAct agent with the loop: `THINK → ACT → OBSERVE → REFLECT`

| Tool | Purpose |
|------|---------|
| `retrieve_vector` | Semantic document search |
| `retrieve_graph` | Knowledge graph traversal |
| `lookup_entity` | Get entity details |
| `compare_entities` | Compare multiple entities |
| `database_stats` | Document count and list |
| `synthesize` | Combine information |
| `verify_claim` | Fact-check against sources |
| `ask_clarification` | Request user clarification |
| `final_answer` | Provide final response |

**Agent triggers:** Analytical queries, meta-questions ("How many documents?"), comparative queries, ambiguous requests.

### Tools & Processing (non-LLM)

| Stage | Tool/Library | Details |
|-------|--------------|---------|
| Office → PDF | LibreOffice | DOCX, PPTX, XLSX, HTML conversion |
| PDF → Images | pdf2image | 600 DPI, PNG format |
| Layout Detection | YOLOv11x | `yolov11x_best.pt` (~109 MB, optional) |
| Vector Storage | Qdrant | Embedded mode (no server) |
| Sparse Index | BM25 | In-memory keyword index |
| Graph Storage | NetworkX | With Louvain community detection |

### LLM Models - Ingestion Pipeline

| Stage | Default Model | Provider |
|-------|---------------|----------|
| Document parsing | `ibm/granite-docling:258m-bf16` | Ollama |
| Text/table extraction | `gemini-3-flash-preview` (vision) | Gemini |
| Image descriptions | `gemini-3-flash-preview` (vision) | Gemini |
| Table descriptions | `gemini-3-flash-preview` | Gemini |
| Embeddings | `qwen3-embedding:4b-q8_0` | Ollama |
| Entity extraction | `gemini-3-flash-preview` | Gemini |
| Entity resolution | `gemini-3-flash-preview` | Gemini |
| Community summaries | `gemini-3-flash-preview` | Gemini |

### LLM Models - Query Pipeline

| Stage | Default Model | Provider |
|-------|---------------|----------|
| Query rewriting | `gemini-3-flash-preview` | Gemini |
| Query expansion | `gemini-3-flash-preview` | Gemini |
| Query classification | Rule-based (+ LLM fallback) | - |
| Reranking | `gemini-3-flash-preview` | Gemini |
| Generation | `gemini-3-flash-preview` | Gemini |

### Optional Features (disabled by default)

| Feature | Default Model | Provider |
|---------|---------------|----------|
| Cross-encoder reranking | `dengcao/Qwen3-Reranker-0.6B:F16` | Ollama |
| Contextual compression | `gemini-3-flash-preview` | Gemini |

---

## CLI Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `cognidoc ingest <path>` | Ingest documents into the knowledge base |
| `cognidoc query "<question>"` | Query the knowledge base |
| `cognidoc serve` | Launch the web interface |
| `cognidoc schema-generate [path]` | Auto-generate graph schema from corpus analysis |
| `cognidoc info` | Show current configuration |
| `cognidoc init` | Copy template files (schema, prompts) |

### Ingest Options

```bash
cognidoc ingest ./data/sources \
    --llm gemini \
    --embedding ollama \
    --skip-yolo \              # Skip YOLO detection
    --skip-graph \             # Skip GraphRAG building
    --skip-conversion \        # Skip PDF conversion
    --force-reembed            # Re-embed all (ignore cache)
```

**All skip flags:**

| Flag | Skips |
|------|-------|
| `--skip-conversion` | Non-PDF to PDF conversion |
| `--skip-pdf` | PDF to image conversion |
| `--skip-yolo` | YOLO detection |
| `--skip-extraction` | Text/table extraction |
| `--skip-descriptions` | Image descriptions |
| `--skip-chunking` | Semantic chunking |
| `--skip-embeddings` | Embedding generation |
| `--skip-indexing` | Vector index building |
| `--skip-graph` | Knowledge graph building |
| `--skip-resolution` | Entity resolution (semantic deduplication) |
| `--full-reindex` | Force full re-ingestion (ignore incremental manifest) |
| `--regenerate-schema` | Force graph schema regeneration before ingestion |

### Schema Generation

```bash
cognidoc schema-generate ./data/sources \
    --language fr \             # Schema language (default: en)
    --max-docs 100 \            # Max documents to sample (default: 100)
    --max-pages 3 \             # Max pages per document (default: 3)
    --regenerate                # Overwrite existing schema
```

Auto-generates `config/graph_schema.yaml` by analyzing the corpus: converts sources to PDF, samples up to 100 documents (distributed across subfolders), extracts text from first 3 pages, then runs a two-stage LLM pipeline (batch analysis → synthesis). Also auto-triggered during `cognidoc ingest` if no schema exists.

### Serve Options

```bash
cognidoc serve \
    --port 7860 \
    --share \                  # Create public Gradio link
    --no-rerank                # Skip LLM reranking (faster)
```

### Performance Tuning (M2/M3 Macs)

```bash
cognidoc ingest ./data/sources \
    --yolo-batch-size 2 \           # YOLO batch size (default: 2)
    --entity-max-concurrent 4 \     # Concurrent LLM calls (default: auto)
    --no-yolo-batching \            # Sequential YOLO processing
    --no-async-extraction           # Sequential entity extraction
```

---

## REST API

The Gradio app exposes REST endpoints at `http://localhost:7860`.

### Query Endpoint

```bash
curl -X POST http://localhost:7860/api/submit_handler \
    -H "Content-Type: application/json" \
    -d '{"data": ["What topics are covered?", [], true, true]}'
```

Parameters: `[user_msg, history, rerank, use_graph]`

### Python Client

```python
from gradio_client import Client

client = Client("http://localhost:7860")
result = client.predict(
    user_msg="How many documents?",
    history=[],
    rerank=True,
    use_graph=True,
    api_name="/submit_handler"
)
answer = result[0][-1]["content"][0]["text"]
```

### Other Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/reset_conversation` | Reset conversation history |
| `POST /api/refresh_metrics` | Get performance metrics |
| `POST /api/export_csv` | Export metrics as CSV |
| `POST /api/export_json` | Export metrics as JSON |

---

## Development

### Setup

```bash
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc
make install        # Create venv and install dependencies

# If path contains spaces:
UV_LINK_MODE=copy uv sync --all-extras
```

### Code Quality

```bash
make format         # Format with black
make lint           # Run pylint
make refactor       # Format + lint

# For src/ directory:
uv run black src/cognidoc/
uv run pylint src/cognidoc/
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_agent.py -v

# Single test function
pytest tests/test_agent.py::test_agent_tool_parsing -v

# Pattern matching
pytest tests/ -v -k "complexity"

# E2E tests only
pytest tests/test_00_e2e_pipeline.py -v

# Full E2E with ingestion (slow)
pytest tests/test_00_e2e_pipeline.py -v --run-slow
```

### Test Modules

| Module | Tests | Description |
|--------|-------|-------------|
| `test_00_e2e_pipeline.py` | 9 | E2E pipeline |
| `test_agent.py` | 27 | Agent ReAct loop |
| `test_agent_tools.py` | 33 | Tool implementations |
| `test_benchmark.py` | 10 | Precision/recall benchmark |
| `test_checkpoint.py` | 32 | Checkpoint/resume system |
| `test_complexity.py` | 25 | Query complexity evaluation |
| `test_e2e_language_and_count.py` | 24 | Language detection (FR/EN/ES/DE) |
| `test_entity_resolution.py` | 34 | Entity resolution (blocking, matching, clustering, merging) |
| `test_helpers.py` | 34 | Token counting, chat history, query parsing |
| `test_optimizations.py` | 38 | Pipeline optimizations, reranking parser |
| `test_providers.py` | 32 | LLM/Embedding providers |
| `test_schema_generation.py` | 75 | Corpus-based schema generation |

### Benchmark with External Data

```bash
COGNIDOC_DATA_DIR="/path/to/external/data" \
    pytest tests/test_benchmark.py -v --run-slow
```

---

## Performance

### Caching Strategy

| Cache | TTL | Purpose |
|-------|-----|---------|
| Retrieval results | 5 min | Avoid repeated searches |
| Qdrant results | 5 min | Vector search results |
| BM25 tokenization | ∞ (LRU 1000) | Tokenized queries |
| Tool results | 5-30 min | Agent tool outputs |
| Embeddings | ∞ | Computed embeddings |

### Optimizations

| Stage | Optimization |
|-------|--------------|
| PDF → Images | ProcessPoolExecutor (4 workers) |
| Embeddings | Batched async HTTP with connection pooling |
| Entity extraction | Adaptive concurrency (2-8 based on CPU) |
| Vector + Graph | Parallel retrieval via ThreadPoolExecutor |
| Reranking | Adaptive batch size |
| Agent reflection | Background thread execution |

### Ingestion Time Estimates

| Documents | Without GraphRAG | With GraphRAG |
|-----------|------------------|---------------|
| 5 pages | ~2 min | ~5 min |
| 50 pages | ~10 min | ~30 min |
| 500 pages | ~1h | ~3h |

---

## Roadmap

<!--
Add planned features here / Ajoutez les fonctionnalités prévues ici

Example format:
- [ ] Feature 1 - Description
- [ ] Feature 2 - Description
- [x] Completed feature - Description
-->

*Coming soon / À venir*

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with Qdrant, NetworkX, and multi-provider LLM support</sub>
</p>
