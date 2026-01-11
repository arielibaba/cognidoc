# CogniDoc

**Intelligent Document Assistant** powered by Hybrid RAG (Vector + GraphRAG).

Transform any document collection into a searchable knowledge base with intelligent query routing, multi-step reasoning, and a professional chat interface.

## Quick Start

### 1. Install

```bash
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"
```

### 2. Configure

Create a `.env` file with at least one API key:

```bash
GEMINI_API_KEY=your-key    # Recommended (free tier available)
# or OPENAI_API_KEY=your-key
# or ANTHROPIC_API_KEY=your-key
```

### 3. Run

```python
from cognidoc import CogniDoc

doc = CogniDoc()
doc.ingest("./documents/")  # Your PDF, DOCX, PPTX files
doc.launch_ui(port=7860)    # Open http://localhost:7860
```

That's it! The schema wizard will guide you through the first-time setup.

---

## Installation Options

```bash
# Full installation (recommended)
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Minimal (cloud-only, no YOLO detection)
pip install "cognidoc[ui] @ git+https://github.com/arielibaba/cognidoc.git"

# Development
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc && pip install -e ".[all,dev]"
```

---

## Provider Configuration

CogniDoc supports flexible provider mixing:

| Provider | LLM | Embeddings | Requires |
|----------|-----|------------|----------|
| **Gemini** | `gemini-2.5-flash` | `text-embedding-004` | `GEMINI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku` | - | `ANTHROPIC_API_KEY` |
| **Ollama** | `granite3.3:8b` | `qwen3-embedding:0.6b` | Local server |

### Configuration Examples

```python
# Full cloud (no local dependencies)
CogniDoc(llm_provider="gemini", embedding_provider="gemini")

# Full local (free, requires Ollama)
CogniDoc(llm_provider="ollama", embedding_provider="ollama")

# Hybrid (cloud LLM + local embeddings)
CogniDoc(llm_provider="gemini", embedding_provider="ollama")

# Skip YOLO detection (faster, simpler extraction)
CogniDoc(llm_provider="gemini", embedding_provider="gemini", use_yolo=False)
```

---

## Usage

### Python API

```python
from cognidoc import CogniDoc

# Initialize
doc = CogniDoc(
    llm_provider="gemini",
    embedding_provider="ollama",
)

# Ingest documents (first time only)
doc.ingest("./documents/")

# Query
result = doc.query("What are the main topics?")
print(result.answer)

# Launch web interface
doc.launch_ui(port=7860, share=True)
```

### CLI

```bash
# Initialize project (copy templates)
cognidoc init --schema --prompts

# Ingest documents
cognidoc ingest ./documents --llm gemini --embedding ollama

# Query
cognidoc query "Summarize the key findings"

# Launch web UI
cognidoc serve --port 7860 --share
```

---

## Features

- **Hybrid RAG**: Combines vector similarity search with knowledge graph traversal
- **Agentic RAG**: Multi-step reasoning agent with 9 specialized tools
- **Multi-Language**: Automatic French/English detection with consistent responses
- **Multi-Format**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, images
- **YOLO Detection**: Automatic table/image/text region detection (optional)
- **Conversation Memory**: Context-aware follow-up questions
- **Tool Caching**: TTL-based caching reduces latency for repeated queries
- **Real-time Progress**: Visual streaming of agent reasoning (🤔⚡👁️💭)
- **Clickable References**: Response citations link directly to source PDFs

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

### Query Processing

```
User Query → Query Rewriter → Classifier + Complexity Evaluator
                                           │
                         ┌─────────────────┼─────────────────┐
                         ▼                 ▼                 ▼
                    FAST PATH        ENHANCED PATH      AGENT PATH
                   (score < 0.35)   (0.35 ≤ score < 0.55)  (score ≥ 0.55)
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

### Query Routing Weights

| Query Type | Example | Vector | Graph |
|------------|---------|--------|-------|
| **FACTUAL** | "What is X?" | 70% | 30% |
| **RELATIONAL** | "How are A and B related?" | 20% | 80% |
| **EXPLORATORY** | "List all main topics" | 0% | 100% |
| **PROCEDURAL** | "How to configure X?" | 80% | 20% |

---

## Agentic RAG

Complex queries automatically trigger a ReAct agent with these tools:

| Tool | Purpose |
|------|---------|
| `retrieve_vector` | Semantic document search |
| `retrieve_graph` | Knowledge graph traversal |
| `lookup_entity` | Get entity details |
| `compare_entities` | Compare multiple entities |
| `database_stats` | Get document count and list (unique sources, not chunks) |
| `synthesize` | Combine information from multiple sources |
| `verify_claim` | Fact-check against sources |
| `ask_clarification` | Request user clarification |
| `final_answer` | Provide final response |

**Agent triggers:**
- Analytical/comparative queries
- Meta-questions ("How many documents?", "List all documents")
- Ambiguous queries needing clarification

**ReAct Loop:**
```
THINK → ACT → OBSERVE → REFLECT → (loop or final_answer)
```

---

## Conversation Memory

CogniDoc maintains context across messages:

```
User: "How many documents are in the database?"
Bot:  "The database contains 2 documents."

User: "List them"
Bot:  "The documents are: test_document, test_document 2"
```

The query rewriter automatically incorporates context, so "list them" becomes "List the 2 documents in the database."

---

## Performance

### Tool Result Caching

| Tool | TTL | Reason |
|------|-----|--------|
| `database_stats` | 5 min | Metadata rarely changes |
| `retrieve_vector` | 2 min | Same query, same results |
| `retrieve_graph` | 2 min | Graph traversal cached |
| `lookup_entity` | 5 min | Entity details stable |
| `compare_entities` | 3 min | Comparison cached |

### Ingestion Optimizations

| Stage | Optimization | Speedup |
|-------|--------------|---------|
| PDF → Images | ProcessPoolExecutor (4 workers) | ~2x |
| Embeddings | Batched async HTTP | ~5x |
| Cache | SQLite persistent | Instant (cached) |

### Real-time Progress

During agent execution:
```
🤔 [Step 1/7] Analyzing query...
⚡ Calling retrieve_vector(query="...")
👁️ Result [cached]: Found 5 documents...
💭 Analysis: Sufficient information gathered
```

---

## Schema Wizard

On first ingestion, an interactive wizard helps configure GraphRAG:

```
╭──────────────────────────────────────────────────────────────╮
│                   CogniDoc Schema Wizard                      │
╰──────────────────────────────────────────────────────────────╯

? What type of documents are you working with?
  ❯ Technical documentation
    Legal documents
    Medical/Scientific papers
    ...

? Auto-generate schema from document analysis?
  ❯ Yes (recommended)
    No (manual configuration)
```

**Options:**
- `doc.ingest("./docs/")` - Wizard runs automatically
- `doc.ingest("./docs/", skip_schema_wizard=True)` - Use existing schema
- Manual: Edit `config/graph_schema.yaml`

---

## Project Structure

```
your-project/
├── documents/              # Your source files
├── .env                    # API keys
└── data/                   # Created automatically
    ├── pdfs/               # Converted PDFs
    ├── images/             # Page images
    ├── chunks/             # Semantic chunks
    ├── indexes/            # Search indexes
    ├── vector_store/       # Qdrant database
    └── cache/              # SQLite caches
```

---

## Requirements

### Minimal (Cloud-only)

- Python 3.10+
- API key (Gemini, OpenAI, or Anthropic)

### Full Features

- [Ollama](https://ollama.ai/) for local inference
- [LibreOffice](https://www.libreoffice.org/) for Office conversion

### Ollama Models (if using local)

```bash
ollama pull granite3.3:8b          # LLM
ollama pull qwen3-embedding:0.6b   # Embeddings
ollama pull qwen3-vl:8b-instruct   # Vision (optional)
```

---

## Development

```bash
make install   # Setup with uv
make format    # Format with black
make lint      # Run pylint
make test      # Run tests (127 tests)
```

---

## License

MIT
