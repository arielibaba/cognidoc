"""
CogniDoc - Main API class for the Hybrid RAG document assistant.

Provides a simple interface for:
- Document ingestion
- Querying the knowledge base
- Launching the Gradio UI
"""

import asyncio
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
    "gemini": "gemini-3-flash-preview",
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
            self.llm_model = DEFAULT_LLM_MODELS.get(self.llm_provider, "gemini-3-flash-preview")

        # Default embedding model
        if self.embedding_model is None:
            try:
                provider = EmbeddingProvider(self.embedding_provider)
                self.embedding_model = DEFAULT_EMBEDDING_MODELS.get(
                    provider, "qwen3-embedding:4b-q8_0"
                )
            except ValueError:
                self.embedding_model = "qwen3-embedding:4b-q8_0"

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
        logger.info(
            f"Setting up embedding provider: {self.config.embedding_provider}/{self.config.embedding_model}"
        )
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

    def _handle_schema_wizard(self, regenerate_schema: bool = False) -> Optional[str]:
        """
        Handle schema wizard flow for graph configuration.

        Args:
            regenerate_schema: Force regeneration even if schema exists

        Returns:
            Path to schema file, or "__skip_graph__" to skip graph, or None
        """
        from .schema_wizard import (
            is_wizard_available,
            check_existing_schema,
            prompt_schema_choice,
            run_interactive_wizard,
            run_non_interactive_wizard,
            generate_schema_from_corpus_sync,
            save_schema,
        )
        from .constants import SOURCES_DIR, PDF_DIR

        # Force regeneration if requested
        if regenerate_schema:
            logger.info("Regenerating graph schema from corpus analysis...")
            try:
                schema = generate_schema_from_corpus_sync(
                    sources_dir=str(SOURCES_DIR),
                    pdf_dir=str(PDF_DIR),
                )
                save_schema(schema)
                return check_existing_schema()
            except Exception as e:
                logger.error(f"Schema regeneration failed: {e}")
                # Fall through to normal flow

        # Check for existing schema
        existing_schema = check_existing_schema()

        if existing_schema:
            # Schema exists - ask user what to do
            if is_wizard_available():
                choice = prompt_schema_choice(existing_schema)
                if choice == "use_existing":
                    logger.info(f"Using existing schema: {existing_schema}")
                    return existing_schema
                elif choice == "skip":
                    logger.info("Skipping graph extraction")
                    return "__skip_graph__"
                # choice == "create_new" -> continue to wizard
            else:
                # No interactive mode - use existing
                logger.info(f"Using existing schema: {existing_schema}")
                return existing_schema

        # No schema exists - auto-generate from corpus
        logger.info("No graph schema found. Auto-generating from corpus analysis...")

        # No schema or user wants new one - run wizard
        if is_wizard_available():
            schema = run_interactive_wizard(SOURCES_DIR)
            if schema:
                return check_existing_schema()  # Return path to saved schema
            else:
                # User cancelled - use non-interactive fallback
                logger.info("Wizard cancelled, generating schema from corpus...")
                run_non_interactive_wizard(SOURCES_DIR, use_auto=True)
                return check_existing_schema()
        else:
            # No interactive mode - generate automatically from corpus
            logger.info("Generating schema automatically from corpus...")
            run_non_interactive_wizard(SOURCES_DIR, use_auto=True)
            return check_existing_schema()

    def ingest(
        self,
        source: Union[str, Path, List[str]] = None,
        skip_conversion: bool = False,
        skip_pdf: bool = False,
        skip_yolo: bool = False,
        skip_extraction: bool = False,
        skip_descriptions: bool = False,
        skip_chunking: bool = False,
        skip_embeddings: bool = False,
        skip_indexing: bool = False,
        skip_graph: bool = False,
        force_reembed: bool = False,
        use_cache: bool = True,
        graph_config_path: Optional[str] = None,
        skip_schema_wizard: bool = False,
        full_reindex: bool = False,
        regenerate_schema: bool = False,
        prune: bool = False,
    ) -> IngestionResult:
        """
        Ingest documents from path(s).

        The pipeline processes documents in the data/sources directory by default.
        If source is specified, files will be copied to data/sources first.

        On first ingestion with graph enabled, a schema wizard will guide you through
        creating a custom graph schema. Use skip_schema_wizard=True to skip this.

        Args:
            source: Path to document(s) or directory (optional, uses data/sources if None)
            skip_conversion: Skip non-PDF to PDF conversion
            skip_pdf: Skip PDF to image conversion
            skip_yolo: Skip YOLO detection (use simple extraction)
            skip_extraction: Skip text/table extraction
            skip_descriptions: Skip image description generation
            skip_chunking: Skip semantic chunking
            skip_embeddings: Skip embedding generation
            skip_indexing: Skip vector index building
            skip_graph: Skip GraphRAG processing
            force_reembed: Force re-embedding even for cached content
            use_cache: Use embedding cache
            graph_config_path: Path to custom graph configuration
            skip_schema_wizard: Skip the interactive schema wizard
            regenerate_schema: Force schema regeneration even if one exists

        Returns:
            IngestionResult with statistics
        """
        from .run_ingestion_pipeline import run_ingestion_pipeline_async
        from .constants import SOURCES_DIR

        # Handle graph schema configuration
        actual_skip_graph = skip_graph or not self.config.use_graph

        if not actual_skip_graph and graph_config_path is None and not skip_schema_wizard:
            graph_config_path = self._handle_schema_wizard(regenerate_schema=regenerate_schema)
            if graph_config_path == "__skip_graph__":
                actual_skip_graph = True
                graph_config_path = None

        # Collect source files to process (if specific sources provided)
        source_files = None
        if source is not None:
            if isinstance(source, (str, Path)):
                source = [str(source)]
            sources = [str(s) for s in source]
            source_files = []

            for src_path in sources:
                src = Path(src_path).resolve()
                if src.is_file():
                    source_files.append(str(src))
                    logger.info(f"Will process file: {src}")
                elif src.is_dir():
                    # Add all files from directory
                    for file in src.rglob("*"):
                        if file.is_file():
                            source_files.append(str(file))
                    logger.info(f"Will process {len(source_files)} file(s) from directory: {src}")

            if not source_files:
                logger.warning(f"No files found in specified source(s): {sources}")
                source_files = None

        # Determine YOLO usage
        use_yolo = self._get_use_yolo() and not skip_yolo
        actual_skip_yolo = not use_yolo

        logger.info(f"Starting ingestion pipeline")
        logger.info(f"YOLO: {use_yolo}, Graph: {not actual_skip_graph}")

        # Run async pipeline
        try:
            stats = asyncio.run(
                run_ingestion_pipeline_async(
                    vision_provider=self.config.vision_provider,
                    extraction_provider=self.config.llm_provider,
                    skip_conversion=skip_conversion,
                    skip_pdf=skip_pdf,
                    skip_yolo=actual_skip_yolo,
                    skip_extraction=skip_extraction,
                    skip_descriptions=skip_descriptions,
                    skip_chunking=skip_chunking,
                    skip_embeddings=skip_embeddings,
                    force_reembed=force_reembed,
                    use_cache=use_cache,
                    skip_indexing=skip_indexing,
                    skip_graph=actual_skip_graph,
                    graph_config_path=graph_config_path,
                    source_files=source_files,
                    full_reindex=full_reindex,
                    prune=prune,
                )
            )

            # Extract stats from pipeline result
            entities = stats.get("graph_extraction", {}).get("entities_extracted", 0)
            relationships = stats.get("graph_extraction", {}).get("relationships_extracted", 0)
            chunks = stats.get("graph_extraction", {}).get("chunks_processed", 0)
            doc_stats = stats.get("document_conversion", {})
            docs = (
                doc_stats.get("pdfs_copied", 0)
                + doc_stats.get("images_copied", 0)
                + doc_stats.get("converted", 0)
            )

            return IngestionResult(
                documents_processed=docs,
                chunks_created=chunks,
                entities_extracted=entities,
                relationships_extracted=relationships,
                errors=[],
            )
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            return IngestionResult(
                documents_processed=0,
                chunks_created=0,
                entities_extracted=0,
                relationships_extracted=0,
                errors=[str(e)],
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
            self._retriever = HybridRetriever()

        top_k = top_k or self.config.top_k

        # Execute retrieval
        result = self._retriever.retrieve(
            question,
            top_k=top_k,
            use_reranking=self.config.use_reranking,
        )

        # Format sources from vector results
        sources = []
        for node_with_score in result.vector_results[:5]:
            doc = node_with_score.node
            text = doc.text
            sources.append(
                {
                    "text": text[:500] + "..." if len(text) > 500 else text,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page_number", None),
                }
            )

        # Get query type from query analysis
        query_type = None
        if result.query_analysis and result.query_analysis.query_type:
            query_type = result.query_analysis.query_type.value

        return QueryResult(
            answer=result.fused_context or "",
            sources=sources,
            query_type=query_type,
            retrieval_stats={
                "vector_results": len(result.vector_results),
                "graph_results": len(result.graph_results.entities) if result.graph_results else 0,
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
            from .cognidoc_app import create_gradio_app, create_fastapi_app
        except ImportError as e:
            raise ImportError(
                f"UI dependencies not installed: {e}. Install with: pip install cognidoc[ui]"
            ) from e

        # Update reranking setting
        if no_rerank:
            self.config.use_reranking = False

        logger.info(f"Launching CogniDoc UI on port {port}")
        demo = create_gradio_app(default_reranking=self.config.use_reranking)

        if share:
            # Share mode: use Gradio's built-in tunnel (no FastAPI wrapper)
            demo.launch(server_port=port, share=True)
        else:
            # Normal mode: wrap in FastAPI with PDF serving endpoint
            import uvicorn

            app = create_fastapi_app(demo)
            uvicorn.run(app, host="0.0.0.0", port=port)

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
