# CogniDoc Package Implementation Plan

## Objectif

Transformer CogniDoc en un package Python réutilisable installable depuis GitHub avec:
- API Python simple
- Interface Gradio
- CLI
- Providers flexibles (LLM ≠ Embedding provider)
- YOLO et Ollama optionnels

## Architecture Cible

```
cognidoc/
├── pyproject.toml              # Configuration du package
├── README.md                   # Documentation
├── CLAUDE.md                   # Instructions Claude Code
├── config/
│   └── graph_schema.yaml       # Schéma par défaut (copiable)
├── src/
│   └── cognidoc/               # Package principal (renommé de src/)
│       ├── __init__.py         # Exports publics + CogniDoc class
│       ├── __main__.py         # Point d'entrée CLI
│       ├── api.py              # Classe CogniDoc principale
│       ├── config.py           # Configuration centralisée
│       ├── cli.py              # Interface ligne de commande
│       ├── app.py              # Interface Gradio (renommé)
│       ├── pipeline/           # Pipeline d'ingestion
│       │   ├── __init__.py
│       │   ├── runner.py       # Orchestrateur principal
│       │   ├── converters.py   # Conversion documents
│       │   ├── extractors.py   # Extraction texte (YOLO optionnel)
│       │   └── chunkers.py     # Chunking sémantique
│       ├── retrieval/          # Système de récupération
│       │   ├── __init__.py
│       │   ├── hybrid.py       # Retriever hybride
│       │   ├── vector.py       # Index vectoriel
│       │   └── graph.py        # Knowledge graph
│       ├── providers/          # Providers LLM/Embedding
│       │   ├── __init__.py
│       │   ├── base.py         # Classes abstraites
│       │   ├── llm.py          # Providers LLM
│       │   ├── embeddings.py   # Providers Embeddings
│       │   └── vision.py       # Providers Vision
│       └── utils/              # Utilitaires
│           ├── __init__.py
│           ├── logger.py
│           └── helpers.py
└── prompts/                    # Prompts personnalisables
    ├── system.md
    ├── query_rewrite.md
    └── ...
```

## Phase 1: Restructuration du Package

### 1.1 Configuration du projet (`pyproject.toml`)

```toml
[project]
name = "cognidoc"
version = "0.1.0"
description = "Hybrid RAG document assistant with Vector + GraphRAG"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "qdrant-client>=1.7.0",
    "networkx>=3.0",
    "nltk>=3.8",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "httpx>=0.25",
    "tqdm>=4.65",
]

[project.optional-dependencies]
# Interface utilisateur
ui = ["gradio>=4.0"]
# YOLO pour détection avancée
yolo = ["ultralytics>=8.0", "pillow>=10.0"]
# Ollama local
ollama = ["ollama>=0.1"]
# Providers cloud
cloud = ["google-generativeai>=0.3", "openai>=1.0", "anthropic>=0.18"]
# Tout inclus
all = ["cognidoc[ui,yolo,ollama,cloud]"]
# Développement
dev = ["pytest>=7.0", "black>=23.0", "pylint>=3.0"]

[project.scripts]
cognidoc = "cognidoc.cli:main"

[project.urls]
Homepage = "https://github.com/arielibaba/cognidoc"
```

### 1.2 Classe principale `CogniDoc` (`api.py`)

```python
from cognidoc.config import CogniDocConfig
from cognidoc.pipeline import IngestionPipeline
from cognidoc.retrieval import HybridRetriever

class CogniDoc:
    """
    Main CogniDoc API class.

    Example:
        # Cloud-only mode (Gemini for LLM, OpenAI for embeddings)
        doc = CogniDoc(
            llm_provider="gemini",
            embedding_provider="openai",
            data_dir="./my_data"
        )

        # Local mode (Ollama for everything)
        doc = CogniDoc(
            llm_provider="ollama",
            embedding_provider="ollama",
        )

        # Hybrid mode (Gemini LLM + Ollama embeddings)
        doc = CogniDoc(
            llm_provider="gemini",
            embedding_provider="ollama",
        )
    """

    def __init__(
        self,
        # Provider configuration
        llm_provider: str = "gemini",           # gemini, openai, anthropic, ollama
        llm_model: str = None,                  # Auto-selected if None
        embedding_provider: str = "ollama",     # ollama, openai, gemini
        embedding_model: str = None,            # Auto-selected if None
        vision_provider: str = None,            # None = use llm_provider

        # Data paths
        data_dir: str = "./data",

        # Feature toggles
        use_yolo: bool = None,                  # None = auto-detect
        use_graph: bool = True,
        use_reranking: bool = True,

        # Advanced
        config: CogniDocConfig = None,          # Full config override
    ):
        ...

    # Ingestion
    def ingest(self, source: str | Path | list[str]) -> IngestionResult:
        """Ingest documents from path(s)."""
        ...

    def ingest_text(self, text: str, metadata: dict = None) -> IngestionResult:
        """Ingest raw text directly."""
        ...

    # Query
    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """Query the knowledge base."""
        ...

    def chat(self, message: str, history: list = None) -> ChatResponse:
        """Chat with conversation history."""
        ...

    # Export/Import
    def save(self, path: str):
        """Save indexes to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> "CogniDoc":
        """Load from saved indexes."""
        ...

    # UI
    def launch_ui(self, share: bool = False, port: int = 7860):
        """Launch Gradio interface."""
        ...
```

## Phase 2: Providers Flexibles

### 2.1 Architecture des Providers

```python
# providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key: str = None
    base_url: str = None

class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[Message]) -> str: ...

    @abstractmethod
    def stream(self, messages: list[Message]) -> Generator[str, None, None]: ...

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

### 2.2 Provider Registry

```python
# providers/__init__.py

# LLM Providers
LLM_PROVIDERS = {
    "gemini": ("google-generativeai", GeminiLLMProvider),
    "openai": ("openai", OpenAILLMProvider),
    "anthropic": ("anthropic", AnthropicLLMProvider),
    "ollama": ("ollama", OllamaLLMProvider),
}

# Embedding Providers
EMBEDDING_PROVIDERS = {
    "ollama": ("ollama", OllamaEmbeddingProvider),
    "openai": ("openai", OpenAIEmbeddingProvider),
    "gemini": ("google-generativeai", GeminiEmbeddingProvider),
}

def get_llm_provider(name: str, model: str = None) -> BaseLLMProvider:
    """Get LLM provider, checking dependencies."""
    package, cls = LLM_PROVIDERS[name]
    if not is_available(package):
        raise ImportError(f"Install {package}: pip install cognidoc[{name}]")
    return cls(model=model or DEFAULT_MODELS[name])

def get_embedding_provider(name: str, model: str = None) -> BaseEmbeddingProvider:
    """Get embedding provider, checking dependencies."""
    ...
```

### 2.3 Modèles par défaut

```python
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "ollama": "granite3.3:8b",
}

DEFAULT_EMBEDDING_MODELS = {
    "ollama": "qwen3-embedding:0.6b",
    "openai": "text-embedding-3-small",
    "gemini": "text-embedding-004",
}
```

## Phase 3: YOLO Optionnel

### 3.1 Détection avec fallback

```python
# pipeline/extractors.py

def detect_regions(image_path: str, use_yolo: bool = None) -> list[Region]:
    """
    Detect text/table/image regions in a page.

    Args:
        image_path: Path to page image
        use_yolo: Force YOLO on/off. None = auto-detect availability

    Returns:
        List of detected regions
    """
    if use_yolo is None:
        use_yolo = is_yolo_available()

    if use_yolo:
        return _detect_with_yolo(image_path)
    else:
        # Fallback: treat entire page as single text region
        return _detect_simple(image_path)

def _detect_simple(image_path: str) -> list[Region]:
    """Simple extraction without YOLO - uses OCR on full page."""
    return [Region(
        type="text",
        bbox=(0, 0, 1, 1),  # Full page
        confidence=1.0
    )]

def is_yolo_available() -> bool:
    """Check if YOLO is installed and model is available."""
    try:
        from ultralytics import YOLO
        # Check if model file exists
        return Path("models/yolo_trained.pt").exists()
    except ImportError:
        return False
```

## Phase 4: Ollama Optionnel

### 4.1 Mode cloud-only

```python
# providers/embeddings.py

class EmbeddingManager:
    """
    Manages embedding generation with provider fallback.

    Priority:
    1. Explicitly configured provider
    2. Ollama (if available and configured)
    3. Cloud provider (requires API key)
    """

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        fallback_provider: str = None,
    ):
        if provider:
            self._provider = get_embedding_provider(provider, model)
        elif is_ollama_available():
            self._provider = get_embedding_provider("ollama", model)
        elif os.getenv("OPENAI_API_KEY"):
            self._provider = get_embedding_provider("openai", model)
        elif os.getenv("GEMINI_API_KEY"):
            self._provider = get_embedding_provider("gemini", model)
        else:
            raise RuntimeError(
                "No embedding provider available. Either:\n"
                "- Install Ollama: pip install cognidoc[ollama]\n"
                "- Set OPENAI_API_KEY or GEMINI_API_KEY"
            )

def is_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False
```

## Phase 5: CLI

### 5.1 Interface ligne de commande

```python
# cli.py
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="CogniDoc - Hybrid RAG Document Assistant"
    )
    subparsers = parser.add_subparsers(dest="command")

    # cognidoc ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("source", help="Path to documents")
    ingest_parser.add_argument("--llm", default="gemini", help="LLM provider")
    ingest_parser.add_argument("--embedding", default="ollama", help="Embedding provider")
    ingest_parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO")
    ingest_parser.add_argument("--no-graph", action="store_true", help="Disable GraphRAG")

    # cognidoc query
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--data-dir", default="./data", help="Data directory")

    # cognidoc serve
    serve_parser = subparsers.add_parser("serve", help="Launch Gradio UI")
    serve_parser.add_argument("--port", type=int, default=7860)
    serve_parser.add_argument("--share", action="store_true")
    serve_parser.add_argument("--no-rerank", action="store_true")

    # cognidoc init
    init_parser = subparsers.add_parser("init", help="Initialize project")
    init_parser.add_argument("--schema", action="store_true", help="Copy schema template")
    init_parser.add_argument("--prompts", action="store_true", help="Copy prompts")

    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "query":
        _run_query(args)
    elif args.command == "serve":
        _run_serve(args)
    elif args.command == "init":
        _run_init(args)
    else:
        parser.print_help()
```

### 5.2 Exemples d'utilisation CLI

```bash
# Initialiser un projet
cognidoc init --schema --prompts

# Ingérer des documents (Gemini LLM + Ollama embeddings)
cognidoc ingest ./documents --llm gemini --embedding ollama

# Ingérer sans YOLO (fallback simple)
cognidoc ingest ./documents --no-yolo

# Mode full cloud (OpenAI pour tout)
cognidoc ingest ./documents --llm openai --embedding openai

# Lancer l'interface
cognidoc serve --port 7860

# Requête simple
cognidoc query "Quelle est la position de l'Église sur l'euthanasie?"
```

## Phase 6: Configuration Simplifiée

### 6.1 Smart Defaults

```python
# config.py

@dataclass
class CogniDocConfig:
    """
    Configuration with smart defaults.

    Most users only need to specify providers.
    Everything else has sensible defaults.
    """
    # Providers (required)
    llm_provider: str = "gemini"
    embedding_provider: str = "ollama"

    # Models (auto-selected based on provider)
    llm_model: str = None
    embedding_model: str = None
    vision_model: str = None

    # Paths (relative to data_dir)
    data_dir: str = "./data"

    # Feature toggles
    use_yolo: bool = None           # None = auto-detect
    use_graph: bool = True
    use_reranking: bool = True

    # Retrieval (expert tuning)
    top_k: int = 10
    rerank_top_k: int = 5
    dense_weight: float = 0.6       # Dense vs BM25

    # Chunking (expert tuning)
    max_chunk_size: int = 512
    chunk_overlap: int = 50

    @classmethod
    def from_yaml(cls, path: str) -> "CogniDocConfig":
        """Load config from YAML file."""
        ...

    @classmethod
    def from_env(cls) -> "CogniDocConfig":
        """Load config from environment variables."""
        return cls(
            llm_provider=os.getenv("COGNIDOC_LLM_PROVIDER", "gemini"),
            embedding_provider=os.getenv("COGNIDOC_EMBEDDING_PROVIDER", "ollama"),
            data_dir=os.getenv("COGNIDOC_DATA_DIR", "./data"),
            use_yolo=os.getenv("COGNIDOC_USE_YOLO", "").lower() == "true" or None,
        )
```

## Timeline d'Implémentation

### Étape 1: Structure de base
- [ ] Réorganiser `src/` → `src/cognidoc/`
- [ ] Créer `pyproject.toml`
- [ ] Créer `__init__.py` avec exports
- [ ] Créer classe `CogniDoc` basique

### Étape 2: Providers flexibles
- [ ] Séparer LLM et Embedding providers
- [ ] Implémenter provider registry
- [ ] Ajouter détection automatique des dépendances
- [ ] Tests de combinaisons (Gemini+Ollama, etc.)

### Étape 3: YOLO optionnel
- [ ] Créer fallback simple pour extraction
- [ ] Ajouter détection automatique YOLO
- [ ] Tester pipeline sans YOLO

### Étape 4: CLI
- [ ] Implémenter commandes CLI
- [ ] Ajouter `cognidoc init`
- [ ] Tester toutes les commandes

### Étape 5: Documentation
- [ ] Mettre à jour README
- [ ] Ajouter exemples d'utilisation
- [ ] Documenter configuration

### Étape 6: Tests et polish
- [ ] Tests unitaires providers
- [ ] Tests intégration pipeline
- [ ] Cleanup et refactoring

## Exemples d'Utilisation Finale

### Python API

```python
from cognidoc import CogniDoc

# Mode simple (Gemini + Ollama local)
doc = CogniDoc()
doc.ingest("./mes_documents/")
result = doc.query("Résumez le document principal")
print(result.answer)

# Mode full cloud
doc = CogniDoc(
    llm_provider="openai",
    embedding_provider="openai",
)

# Mode hybride personnalisé
doc = CogniDoc(
    llm_provider="gemini",
    llm_model="gemini-1.5-pro",
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
    use_yolo=False,
    use_graph=True,
)

# Avec configuration YAML
doc = CogniDoc.from_config("./cognidoc.yaml")

# Lancer l'UI
doc.launch_ui(share=True)
```

### CLI

```bash
# Installation
pip install git+https://github.com/arielibaba/cognidoc.git

# Avec YOLO
pip install "cognidoc[yolo] @ git+https://github.com/arielibaba/cognidoc.git"

# Tout inclus
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Utilisation
cognidoc init
cognidoc ingest ./docs --llm gemini --embedding ollama
cognidoc serve --port 7860
```
