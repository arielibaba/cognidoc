# CogniDoc

**Intelligent Document Assistant** powered by Hybrid RAG (Vector + GraphRAG).

Transform any document collection into a searchable knowledge base with intelligent query routing, multi-step reasoning, and a professional chat interface.

**Key Features:**
- **Hybrid RAG** - Combines vector similarity search with knowledge graph traversal
- **Agentic RAG** - Multi-step reasoning agent with 9 specialized tools
- **Multi-Format** - PDF, DOCX, PPTX, XLSX, HTML, Markdown, images
- **Multi-Language** - Automatic FR/EN/ES/DE detection
- **YOLO Detection** - Automatic table/image/text region detection (optional)
- **Conversation Memory** - Context-aware follow-up questions

---

## Quick Start

```bash
# 1. Install
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# 2. Create project
mkdir my-project && cd my-project
mkdir -p data/sources

# 3. Add API key
echo "GOOGLE_API_KEY=your-key" > .env

# 4. Add documents and run
cp /path/to/documents/* data/sources/
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

### Via pip (recommended)

```bash
# Full installation (UI, YOLO, all providers)
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Minimal (cloud-only, no YOLO)
pip install "cognidoc[ui] @ git+https://github.com/arielibaba/cognidoc.git"
```

### Development mode

```bash
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc
pip install -e ".[all,dev]"

# If path contains spaces:
UV_LINK_MODE=copy uv sync --all-extras
```

### Requirements

| Requirement | Purpose | Required |
|-------------|---------|----------|
| Python 3.10+ | Runtime | Yes |
| API key (Gemini/OpenAI/Anthropic) | LLM inference | Yes (one) |
| [Ollama](https://ollama.ai/) | Local inference | Optional |
| [LibreOffice](https://www.libreoffice.org/) | Office conversion | Optional |

### YOLO Model (optional)

For document layout detection (tables, images, text regions):

```bash
mkdir -p models/YOLOv11
# Place yolov11x_best.pt (~109 MB) trained on DocLayNet/PubLayNet
```

Without YOLO, the system uses simple page-level extraction.

### Ollama Models (if using local inference)

```bash
ollama pull granite3.3:8b          # LLM
ollama pull qwen3-embedding:0.6b   # Embeddings
ollama pull qwen3-vl:8b-instruct   # Vision (optional)
```

---

## Configuration

### Providers

| Provider | LLM | Embeddings | API Key |
|----------|-----|------------|---------|
| **Gemini** | `gemini-2.5-flash` | `text-embedding-004` | `GOOGLE_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku` | - | `ANTHROPIC_API_KEY` |
| **Ollama** | `granite3.3:8b` | `qwen3-embedding:0.6b` | Local server |

### Environment Variables (.env)

```bash
# API Keys (at least one required)
GOOGLE_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# LLM settings
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.7

# Retrieval settings
TOP_K_RETRIEVED_CHILDREN=10
TOP_K_RERANKED_PARENTS=5
HYBRID_DENSE_WEIGHT=0.6

# YOLO
YOLO_CONFIDENCE_THRESHOLD=0.2
```

---

## Usage

### Python API

```python
from cognidoc import CogniDoc

# Initialize with providers
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

### CLI

```bash
# Interactive setup wizard (recommended for first-time)
python -m cognidoc.setup

# Individual commands
cognidoc ingest ./data/sources --llm gemini --embedding ollama
cognidoc query "Summarize the key findings"
cognidoc serve --port 7860 --share
cognidoc info  # Show current configuration

# Skip options
cognidoc ingest ./data/sources --skip-graph --skip-yolo
cognidoc serve --no-rerank  # Faster, skip LLM reranking
```

### REST API

CogniDoc exposes a REST API via Gradio at `http://localhost:7860`.

**Main endpoint:**

```bash
curl -X POST http://localhost:7860/api/submit_handler \
  -H "Content-Type: application/json" \
  -d '{"data": ["What topics are covered?", [], true, true]}'
```

Parameters: `[user_msg, history, rerank, use_graph]`

**Python example:**

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

**Other endpoints:**

| Endpoint | Description |
|----------|-------------|
| `/api/reset_conversation` | Reset conversation history |
| `/api/refresh_metrics` | Get performance metrics |
| `/api/export_csv` | Export metrics as CSV |

---

## Architecture

### Ingestion Pipeline

```
Documents → PDF Conversion → Images (600 DPI) → YOLO Detection*
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

*YOLO is optional - falls back to page-level extraction if disabled.

**Ingestion time estimates:**

| Documents | Without GraphRAG | With GraphRAG |
|-----------|------------------|---------------|
| 5 pages | ~2 min | ~5 min |
| 50 pages | ~10 min | ~30 min |
| 500 pages | ~1h | ~3h |

### Query Processing

```
User Query → Query Rewriter → Classifier + Complexity Evaluator
                                           │
                         ┌─────────────────┼─────────────────┐
                         ▼                 ▼                 ▼
                    FAST PATH        ENHANCED PATH      AGENT PATH
                   (score < 0.35)   (0.35-0.55)        (score ≥ 0.55)
                         │                 │                 │
                         ▼                 ▼                 ▼
                  Hybrid Retriever   Hybrid Retriever   ReAct Agent
                  (Vector + Graph)   (boosted weights)  (multi-step)
                         │                 │                 │
                         └─────────────────┴─────────────────┘
                                           ▼
                                    LLM Generation
                                           ▼
                                   Response (same language as query)
```

**Query routing weights:**

| Query Type | Example | Vector | Graph |
|------------|---------|--------|-------|
| FACTUAL | "What is X?" | 70% | 30% |
| RELATIONAL | "How are A and B related?" | 20% | 80% |
| EXPLORATORY | "List all main topics" | 0% | 100% |
| PROCEDURAL | "How to configure X?" | 80% | 20% |

### 3-Level Fusion Retrieval

```
                              User Query
                                  │
                    ┌─────────────┴─────────────┐
                    │      Query Routing        │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       │                       ▼
┌─────────────────────┐           │           ┌─────────────────────┐
│  VECTOR RETRIEVAL   │           │           │  GRAPH RETRIEVAL    │
└─────────┬───────────┘           │           └─────────┬───────────┘
          │                       │                     │
    ┌─────┴─────┐                 │           ┌────────┴────────┐
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
│ α=0.6 default   │               │         └───────┬────────┘
└────────┬────────┘               │                 │
         │                        │                 │
         ▼                        │                 │
┌─────────────────┐               │                 │
│ Child → Parent  │               │                 │
│ + Reranking     │               │                 │
└────────┬────────┘               │                 │
         │                        │                 │
         └─────────────┬──────────┴─────────────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │  LEVEL 3: FINAL     │
             │  FUSION (weighted   │
             │  by query type)     │
             └─────────────────────┘
```

| Level | Components | Method |
|-------|-----------|--------|
| **1. Hybrid** | Dense + Sparse (BM25) | RRF: `α × 1/(60+rank_dense) + (1-α) × 1/(60+rank_sparse)` |
| **2. Parallel** | Vector ∥ Graph | Independent execution |
| **3. Final** | Vector + Graph results | Weighted by query type |

### Agentic RAG

Complex queries trigger a ReAct agent (`THINK → ACT → OBSERVE → REFLECT → loop`):

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

**Agent triggers:** Analytical queries, meta-questions ("How many documents?"), ambiguous queries.

---

## Performance

### Caching

| Cache | TTL | Location |
|-------|-----|----------|
| Retrieval results | 5 min | `hybrid_retriever.py` |
| Qdrant results | 5 min | `rag_utils.py` |
| BM25 tokenization | ∞ | `advanced_rag.py` (LRU 1000) |
| Tool results | 5-30 min | `tool_cache.py` (SQLite) |
| Embeddings | ∞ | `data/cache/` (SQLite) |

### Optimizations

| Stage | Optimization |
|-------|--------------|
| PDF → Images | ProcessPoolExecutor (4 workers) |
| Embeddings | Batched async HTTP |
| Entity extraction | Adaptive concurrency (2-8 based on CPU) |
| Vector + Graph | Parallel retrieval via ThreadPoolExecutor |
| Reranking | Adaptive batch size |

### Real-time Progress

```
🤔 [Step 1/7] Analyzing query...
⚡ Calling retrieve_vector(query="...")
👁️ Result [cached]: Found 5 documents...
💭 Analysis: Sufficient information gathered
```

---

## Project Structure

```
your-project/
├── .env                    # API keys
├── config/
│   └── graph_schema.yaml   # GraphRAG schema (auto-generated)
├── models/
│   └── YOLOv11/
│       └── yolov11x_best.pt  # Optional (~109 MB)
└── data/
    ├── sources/            # Your documents (PDF, DOCX, etc.)
    ├── pdfs/               # Converted PDFs
    ├── images/             # Page images (600 DPI)
    ├── chunks/             # Semantic chunks
    ├── indexes/            # Search indexes
    ├── vector_store/       # Qdrant database
    └── cache/              # SQLite caches
```

---

## Development

```bash
make install   # Setup with uv
make format    # Format with black
make lint      # Run pylint

# Tests (151 total)
pytest tests/ -v
pytest tests/test_00_e2e_pipeline.py -v --run-slow  # Full E2E
```

### Pipeline Skip Flags (dev mode)

```bash
python -m cognidoc.run_ingestion_pipeline \
    --skip-conversion \
    --skip-pdf \
    --skip-yolo \
    --skip-extraction \
    --skip-chunking \
    --skip-embeddings \
    --skip-indexing \
    --skip-graph \
    --force-reembed
```

---

## License

MIT
