"""
Utility modules for the Advanced Hybrid RAG system.

Exports:
- logger: Structured logging with timing metrics
- llm_providers: Multi-provider LLM abstraction
- embedding_providers: Multi-provider embedding abstraction
- embedding_cache: Content-based embedding caching
- rag_utils: Document, Vector/Keyword indexes, reranking
"""

from .logger import (
    logger,
    timer,
    timed,
    PipelineTimer,
    retrieval_metrics,
)

from .llm_providers import (
    LLMProvider,
    LLMConfig,
    BaseLLMProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    create_llm_provider,
    get_default_generation_provider,
    get_default_vision_provider,
)

from .embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    BaseEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    GeminiEmbeddingProvider,
    create_embedding_provider,
    get_embedding_provider,
    set_embedding_provider,
    reset_embedding_provider,
    is_ollama_available,
    is_provider_available,
    get_available_embedding_providers,
    DEFAULT_EMBEDDING_MODELS,
)

from .embedding_cache import (
    EmbeddingCache,
    get_embedding_cache,
)

from .rag_utils import (
    Document,
    NodeWithScore,
    VectorIndex,
    KeywordIndex,
    get_embedding,
    get_embedding_dimension,
    rerank_documents,
    stream_chat,
)

__all__ = [
    # Logger
    "logger",
    "timer",
    "timed",
    "PipelineTimer",
    "retrieval_metrics",
    # LLM Providers
    "LLMProvider",
    "LLMConfig",
    "BaseLLMProvider",
    "GeminiProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_llm_provider",
    "get_default_generation_provider",
    "get_default_vision_provider",
    # Embedding Providers
    "EmbeddingProvider",
    "EmbeddingConfig",
    "BaseEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "create_embedding_provider",
    "get_embedding_provider",
    "set_embedding_provider",
    "reset_embedding_provider",
    "is_ollama_available",
    "is_provider_available",
    "get_available_embedding_providers",
    "DEFAULT_EMBEDDING_MODELS",
    # Embedding Cache
    "EmbeddingCache",
    "get_embedding_cache",
    # RAG Utilities
    "Document",
    "NodeWithScore",
    "VectorIndex",
    "KeywordIndex",
    "get_embedding",
    "get_embedding_dimension",
    "rerank_documents",
    "stream_chat",
]
