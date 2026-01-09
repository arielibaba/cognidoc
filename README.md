# CogniDoc

**Intelligent Document Assistant** powered by Hybrid RAG (Vector + GraphRAG).

A document processing and retrieval pipeline that combines **Vector RAG** and **GraphRAG** for intelligent document querying. Converts PDFs into a searchable knowledge base with a professional chat interface.

## Features

- **Hybrid RAG**: Combines vector similarity search with knowledge graph traversal
- **GraphRAG**: Automatic entity/relationship extraction with community detection
- **Agentic RAG**: Multi-step reasoning agent for complex queries with tool use
- **Multi-Language Support**: Automatic language detection (French/English) with consistent responses
- **Multi-Format Support**: PDF, PPTX, DOCX, XLSX, HTML, Markdown, images
- **Flexible Providers**: Mix and match LLM and embedding providers independently
- **YOLO Object Detection**: Automatically detects tables, pictures, text regions (optional)
- **Semantic Chunking**: Embedding-based coherent text chunks
- **Intelligent Query Routing**: LLM-based classification with smart skip logic
- **Clickable PDF References**: Response references link directly to source PDFs
- **No LangChain/LlamaIndex**: Direct Qdrant and provider integration

## Installation

```bash
# Basic installation (from GitHub)
pip install git+https://github.com/arielibaba/cognidoc.git

# With Gradio UI
pip install "cognidoc[ui] @ git+https://github.com/arielibaba/cognidoc.git"

# With YOLO detection
pip install "cognidoc[yolo] @ git+https://github.com/arielibaba/cognidoc.git"

# With local Ollama support
pip install "cognidoc[ollama] @ git+https://github.com/arielibaba/cognidoc.git"

# Full installation (all features)
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"
```

### Development Installation

```bash
git clone https://github.com/arielibaba/cognidoc.git
cd cognidoc
make install  # Uses uv package manager
# or
pip install -e ".[all,dev]"
```

## Quick Start

### Python API

```python
from cognidoc import CogniDoc

# Simple usage (Gemini LLM + Ollama embeddings)
doc = CogniDoc()
doc.ingest("./documents/")
result = doc.query("What is the main topic?")
print(result.answer)

# Full cloud mode (no local dependencies)
doc = CogniDoc(
    llm_provider="openai",
    embedding_provider="openai",
)

# Hybrid mode (mix providers)
doc = CogniDoc(
    llm_provider="gemini",
    embedding_provider="ollama",
    use_yolo=False,  # Skip YOLO, use simple extraction
)

# Launch web interface
doc.launch_ui(port=7860, share=True)
```

### CLI

```bash
# Initialize project (copy schema/prompts templates)
cognidoc init --schema --prompts

# Ingest documents
cognidoc ingest ./documents --llm gemini --embedding ollama

# Cloud-only mode
cognidoc ingest ./documents --llm openai --embedding openai

# Without YOLO (simpler extraction)
cognidoc ingest ./documents --no-yolo

# Query
cognidoc query "Summarize the key findings"

# Launch web UI
cognidoc serve --port 7860 --share
```

### Interactive Setup Wizard

For guided configuration:

```bash
python -m src.setup
```

## Provider Configuration

CogniDoc supports flexible provider mixing - use different providers for LLM and embeddings:

| Provider | LLM | Embeddings | Requires |
|----------|-----|------------|----------|
| **Gemini** | `gemini-2.0-flash` | `text-embedding-004` | `GEMINI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku` | - | `ANTHROPIC_API_KEY` |
| **Ollama** | `granite3.3:8b` | `qwen3-embedding:0.6b` | Local Ollama server |

### Example Configurations

```python
# Full local (free, requires Ollama)
CogniDoc(llm_provider="ollama", embedding_provider="ollama")

# Full cloud (no local deps, API costs)
CogniDoc(llm_provider="gemini", embedding_provider="openai")

# Hybrid (best of both)
CogniDoc(llm_provider="gemini", embedding_provider="ollama")
```

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

*YOLO is optional - falls back to simple page-level extraction if not installed.

### Query Routing

| Query Type | Example | Vector | Graph |
|------------|---------|--------|-------|
| **FACTUAL** | "What is X?" | 70% | 30% |
| **RELATIONAL** | "Relationship between A and B?" | 20% | 80% |
| **EXPLORATORY** | "List all main topics" | 0% | 100% |
| **PROCEDURAL** | "How to configure?" | 80% | 20% |

### Agentic RAG

For complex queries requiring multi-step reasoning, CogniDoc automatically activates an agent with specialized tools:

| Tool | Purpose |
|------|---------|
| `retrieve_vector` | Search documents by semantic similarity |
| `retrieve_graph` | Query knowledge graph for relationships |
| `lookup_entity` | Get detailed entity information |
| `compare_entities` | Compare multiple entities |
| `database_stats` | Get statistics about the knowledge base |
| `synthesize` | Combine information from multiple sources |
| `verify_claim` | Fact-check statements against sources |

The agent is triggered automatically for:
- Analytical queries requiring multi-step reasoning
- Comparative questions between entities
- Meta-questions about the database (e.g., "How many documents?")
- Ambiguous queries needing clarification

### Language Support

CogniDoc automatically detects the query language and responds in the same language:
- French queries receive French responses
- English queries receive English responses
- Clarification requests are also language-aware

## Configuration

### Environment Variables

```bash
# Provider selection
COGNIDOC_LLM_PROVIDER=gemini
COGNIDOC_EMBEDDING_PROVIDER=ollama
COGNIDOC_DATA_DIR=./data

# API Keys
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Ollama (if using local)
OLLAMA_HOST=http://localhost:11434
```

### GraphRAG Schema

Customize entity extraction in `config/graph_schema.yaml`:

```yaml
domain:
  name: "your-domain"
  description: "Domain context for LLM extraction"

entity_types:
  - name: "Concept"
    description: "Abstract ideas"
    examples: ["machine learning", "ethics"]

relationship_types:
  - name: "RELATED_TO"
    description: "General relationship"
```

### Schema Wizard

CogniDoc includes a **Schema Wizard** that helps you create an optimized GraphRAG schema for your documents. The wizard runs automatically during ingestion when no schema exists.

#### How It Works

When you run `doc.ingest()` without an existing schema, the wizard:

1. **Interactive Mode** (default): Asks questions about your domain to build a customized schema
2. **Auto-Generation Mode**: Analyzes sample documents and generates a schema using LLM

#### Interactive Mode

If `questionary` is installed (`pip install cognidoc[wizard]`), you'll get an interactive experience:

```
╭──────────────────────────────────────────────────────────────╮
│                   CogniDoc Schema Wizard                      │
│                                                              │
│  This wizard will help you create a GraphRAG schema for     │
│  your documents.                                             │
╰──────────────────────────────────────────────────────────────╯

? What type of documents are you working with?
  ❯ Technical documentation
    Legal documents
    Medical/Scientific papers
    Business/Corporate documents
    Educational materials
    Other (describe below)

? What language are your documents in? English

? Do you want to auto-generate the schema from document analysis?
  ❯ Yes - analyze my documents and generate schema automatically
    No - I'll provide entity types manually
```

If you choose auto-generation, the wizard samples your documents and uses LLM to identify:
- Relevant entity types (people, concepts, products, etc.)
- Relationship types between entities
- Domain-specific terminology

#### Non-Interactive Mode

Without `questionary`, or in automated pipelines, the wizard uses auto-generation:

```python
# Auto-generate schema from documents
doc = CogniDoc()
doc.ingest("./documents/")  # Wizard runs automatically

# Skip the wizard entirely
doc.ingest("./documents/", skip_schema_wizard=True)
```

#### Existing Schema Detection

If a schema already exists (`config/graph_schema.yaml`), you'll be prompted:

```
? A graph schema already exists at config/graph_schema.yaml. What would you like to do?
  ❯ Use existing schema
    Create new schema (will overwrite)
    Skip graph building for this run
```

#### Manual Schema Creation

You can also create a schema manually by copying and editing the template:

```bash
# Copy template
cp config/graph_schema_generic.yaml config/graph_schema.yaml

# Edit to match your domain
vim config/graph_schema.yaml
```

#### Example Generated Schema

For technical documentation, the wizard might generate:

```yaml
domain:
  name: Technical Documentation
  description: Software and technical documentation for developers
  language: en

entities:
  - name: Component
    description: Software components, modules, or services
    examples: [API, Database, Cache, Queue]
  - name: Technology
    description: Programming languages, frameworks, or tools
    examples: [Python, Docker, Kubernetes]
  - name: Concept
    description: Technical concepts or patterns
    examples: [microservices, REST API, authentication]

relationships:
  - name: USES
    description: One component uses another
    valid_source: [Component]
    valid_target: [Component, Technology]
  - name: IMPLEMENTS
    description: Component implements a concept
    valid_source: [Component]
    valid_target: [Concept]
```

### Prompts

All LLM prompts are in `prompts/` directory and can be customized.

## Requirements

### Minimal (Cloud-only)

- Python 3.10+
- API key for at least one provider (Gemini, OpenAI, or Anthropic)

### Full Features

- [Ollama](https://ollama.ai/) for local inference (optional)
- [LibreOffice](https://www.libreoffice.org/) for Office document conversion
- YOLO model for advanced document detection (optional)

### Ollama Models (if using local)

```bash
ollama pull granite3.3:8b          # LLM
ollama pull qwen3-embedding:0.6b   # Embeddings
ollama pull qwen3-vl:8b-instruct   # Vision (optional)
```

## Project Structure

```
cognidoc/
├── src/cognidoc/           # Main package
│   ├── api.py              # CogniDoc class
│   ├── cli.py              # Command-line interface
│   ├── app.py              # Gradio interface
│   ├── schema_wizard.py    # Interactive schema configuration
│   ├── pipeline/           # Ingestion pipeline
│   ├── retrieval/          # Hybrid retriever
│   └── providers/          # LLM/Embedding providers
├── config/
│   ├── graph_schema.yaml         # GraphRAG configuration (user)
│   └── graph_schema_generic.yaml # Generic schema template
├── prompts/                # Customizable prompts
└── data/                   # Document storage
    ├── sources/            # Input documents (any format)
    ├── pdfs/               # Converted PDFs
    ├── indexes/            # Vector/graph indexes
    └── vector_store/       # Qdrant database
```

## Development

```bash
make format    # Format with black
make lint      # Run pylint
make refactor  # Format + lint
make test      # Run tests
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `qdrant-client` | Vector database |
| `networkx` | Knowledge graph |
| `gradio` | Web interface (optional) |
| `ultralytics` | YOLO detection (optional) |
| `ollama` | Local inference (optional) |
| `questionary` | Interactive schema wizard (optional) |
| `google-genai` | Gemini API |
| `openai` | OpenAI API |
| `anthropic` | Claude API |

## License

MIT
