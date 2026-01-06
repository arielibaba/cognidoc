"""
CogniDoc - Main API class for the Hybrid RAG document assistant.

Provides a simple interface for:
- Document ingestion
- Querying the knowledge base
- Launching the Gradio UI
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from .utils.logger import logger
from .utils.llm_providers import (
    LLMProvider,
    LLMConfig,
    create_llm_provider,
    BaseLLMProvider,
)
from .utils.llm_client import set_llm_provider, reset_llm_client
from .utils.embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    create_embedding_provider,
    set_embedding_provider,
    reset_embedding_provider,
    is_ollama_available,
    DEFAULT_EMBEDDING_MODELS,
)


# Default LLM models per provider
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "ollama": "granite3.3:8b",
}


@dataclass
class CogniDocConfig:
    """
    Configuration for CogniDoc with smart defaults.

    Most users only need to specify providers.
    """
    # Providers
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None
    embedding_provider: str = "ollama"
    embedding_model: Optional[str] = None
    vision_provider: Optional[str] = None  # Uses llm_provider if not set

    # Paths
    data_dir: str = "./data"

    # Feature toggles
    use_yolo: Optional[bool] = None  # None = auto-detect
    use_graph: bool = True
    use_reranking: bool = True

    # Retrieval settings
    top_k: int = 10
    rerank_top_k: int = 5
    dense_weight: float = 0.6  # Dense vs BM25 balance

    # Chunking settings
    max_chunk_size: int = 512
    chunk_overlap: int = 50

    def __post_init__(self):
        """Apply smart defaults based on provider selection."""
        # Default LLM model
        if self.llm_model is None:
            self.llm_model = DEFAULT_LLM_MODELS.get(self.llm_provider, "gemini-2.0-flash")

        # Default embedding model
        if self.embedding_model is None:
            try:
                provider = EmbeddingProvider(self.embedding_provider)
                self.embedding_model = DEFAULT_EMBEDDING_MODELS.get(
                    provider, "qwen3-embedding:0.6b"
                )
            except ValueError:
                self.embedding_model = "qwen3-embedding:0.6b"

        # Vision uses LLM provider by default
        if self.vision_provider is None:
            self.vision_provider = self.llm_provider


@dataclass
class QueryResult:
    """Result from a query operation."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    query_type: Optional[str] = None
    retrieval_stats: Optional[Dict[str, Any]] = None


@dataclass
class IngestionResult:
    """Result from an ingestion operation."""
    documents_processed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    errors: List[str] = field(default_factory=list)


class CogniDoc:
    """
    Main CogniDoc API class.

    Provides a simple interface for document ingestion and retrieval
    using Hybrid RAG (Vector + GraphRAG).

    Example:
        # Simple usage (Gemini LLM + Ollama embeddings)
        doc = CogniDoc()
        doc.ingest("./documents/")
        result = doc.query("What is the main topic?")
        print(result.answer)

        # Full cloud mode
        doc = CogniDoc(
            llm_provider="openai",
            embedding_provider="openai",
        )

        # Hybrid mode (mix providers)
        doc = CogniDoc(
            llm_provider="gemini",
            embedding_provider="ollama",
            use_yolo=False,
        )

        # Launch web interface
        doc.launch_ui(port=7860, share=True)
    """

    def __init__(
        self,
        # Provider configuration
        llm_provider: str = "gemini",
        llm_model: Optional[str] = None,
        embedding_provider: str = "ollama",
        embedding_model: Optional[str] = None,
        vision_provider: Optional[str] = None,
        # Data paths
        data_dir: str = "./data",
        # Feature toggles
        use_yolo: Optional[bool] = None,
        use_graph: bool = True,
        use_reranking: bool = True,
        # Advanced
        config: Optional[CogniDocConfig] = None,
    ):
        """
        Initialize CogniDoc.

        Args:
            llm_provider: LLM provider ("gemini", "openai", "anthropic", "ollama")
            llm_model: LLM model name (auto-selected if None)
            embedding_provider: Embedding provider ("ollama", "openai", "gemini")
            embedding_model: Embedding model name (auto-selected if None)
            vision_provider: Vision provider (uses llm_provider if None)
            data_dir: Directory for data storage
            use_yolo: Use YOLO for document detection (None = auto-detect)
            use_graph: Use GraphRAG for retrieval
            use_reranking: Use LLM reranking
            config: Full configuration object (overrides other parameters)
        """
        if config is not None:
            self.config = config
        else:
            self.config = CogniDocConfig(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                vision_provider=vision_provider,
                data_dir=data_dir,
                use_yolo=use_yolo,
                use_graph=use_graph,
                use_reranking=use_reranking,
            )

        self._setup_providers()
        self._retriever = None
        self._initialized = False

    def _setup_providers(self):
        """Configure LLM and embedding providers."""
        # Set LLM provider
        logger.info(f"Setting up LLM provider: {self.config.llm_provider}/{self.config.llm_model}")
        set_llm_provider(self.config.llm_provider, self.config.llm_model)

        # Set embedding provider
        logger.info(f"Setting up embedding provider: {self.config.embedding_provider}/{self.config.embedding_model}")
        set_embedding_provider(self.config.embedding_provider, self.config.embedding_model)

    def _check_yolo_available(self) -> bool:
        """Check if YOLO is available for document detection."""
        from .extract_objects_from_image import is_yolo_model_available
        return is_yolo_model_available()

    def _get_use_yolo(self) -> bool:
        """Determine whether to use YOLO."""
        if self.config.use_yolo is not None:
            return self.config.use_yolo
        return self._check_yolo_available()

    def ingest(
        self,
        source: Union[str, Path, List[str]],
        skip_conversion: bool = False,
        skip_yolo: bool = False,
        skip_graph: bool = False,
    ) -> IngestionResult:
        """
        Ingest documents from path(s).

        Args:
            source: Path to document(s) or directory
            skip_conversion: Skip non-PDF to PDF conversion
            skip_yolo: Skip YOLO detection (use simple extraction)
            skip_graph: Skip GraphRAG processing

        Returns:
            IngestionResult with statistics
        """
        from .run_ingestion_pipeline import run_pipeline

        # Convert source to list of paths
        if isinstance(source, (str, Path)):
            source = [str(source)]
        sources = [str(s) for s in source]

        # Determine YOLO usage
        use_yolo = self._get_use_yolo() and not skip_yolo

        logger.info(f"Starting ingestion of {len(sources)} source(s)")
        logger.info(f"YOLO: {use_yolo}, Graph: {self.config.use_graph and not skip_graph}")

        # Run pipeline
        # TODO: Integrate with run_ingestion_pipeline properly
        # For now, return a placeholder result
        return IngestionResult(
            documents_processed=0,
            chunks_created=0,
            entities_extracted=0,
            relationships_extracted=0,
            errors=["Full pipeline integration pending"],
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """
        Query the knowledge base.

        Args:
            question: Question to ask
            top_k: Number of results to retrieve

        Returns:
            QueryResult with answer and sources
        """
        from .hybrid_retriever import HybridRetriever

        if self._retriever is None:
            self._retriever = HybridRetriever(
                use_reranking=self.config.use_reranking,
            )

        top_k = top_k or self.config.top_k

        # Execute retrieval
        result = self._retriever.retrieve(question, top_k=top_k)

        # Format sources
        sources = []
        for ref in result.references[:5]:
            sources.append({
                "text": ref.text[:500] + "..." if len(ref.text) > 500 else ref.text,
                "source": ref.metadata.get("source", "Unknown"),
                "page": ref.metadata.get("page_number", None),
            })

        return QueryResult(
            answer=result.generated_answer or "",
            sources=sources,
            query_type=result.query_type.value if result.query_type else None,
            retrieval_stats={
                "vector_results": result.vector_results_count,
                "graph_results": result.graph_results_count,
            },
        )

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Chat with the knowledge base.

        Args:
            message: User message
            history: Conversation history
            stream: Whether to stream the response

        Returns:
            Response string or generator (if streaming)
        """
        # For now, just use query
        result = self.query(message)
        if stream:
            def generate():
                yield result.answer
            return generate()
        return result.answer

    def launch_ui(
        self,
        port: int = 7860,
        share: bool = False,
        no_rerank: bool = False,
    ):
        """
        Launch the Gradio web interface.

        Args:
            port: Port to run the server on
            share: Create a public shareable link
            no_rerank: Disable LLM reranking
        """
        try:
            from .cognidoc_app import create_app
        except ImportError:
            raise ImportError(
                "Gradio not installed. Install with: pip install cognidoc[ui]"
            )

        # Update reranking setting
        if no_rerank:
            self.config.use_reranking = False

        logger.info(f"Launching CogniDoc UI on port {port}")
        app = create_app(use_reranking=self.config.use_reranking)
        app.launch(server_port=port, share=share)

    def save(self, path: str):
        """
        Save indexes to disk.

        Args:
            path: Directory to save to
        """
        # TODO: Implement save functionality
        logger.info(f"Saving CogniDoc state to {path}")
        raise NotImplementedError("Save functionality not yet implemented")

    @classmethod
    def load(cls, path: str, **kwargs) -> "CogniDoc":
        """
        Load from saved indexes.

        Args:
            path: Directory to load from
            **kwargs: Additional configuration options

        Returns:
            CogniDoc instance
        """
        # TODO: Implement load functionality
        logger.info(f"Loading CogniDoc state from {path}")
        raise NotImplementedError("Load functionality not yet implemented")

    @classmethod
    def from_config(cls, config_path: str) -> "CogniDoc":
        """
        Create CogniDoc from a YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            CogniDoc instance
        """
        import yaml

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        config = CogniDocConfig(**config_data)
        return cls(config=config)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        return {
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "use_yolo": self._get_use_yolo(),
            "use_graph": self.config.use_graph,
            "use_reranking": self.config.use_reranking,
            "data_dir": self.config.data_dir,
        }


__all__ = [
    "CogniDoc",
    "CogniDocConfig",
    "QueryResult",
    "IngestionResult",
]
