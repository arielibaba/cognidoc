"""
Multi-provider Embedding abstraction layer.

Supports:
- Ollama (local, default)
- OpenAI (text-embedding-3-small, text-embedding-ada-002)
- Google Gemini (text-embedding-004)

Each provider implements a common interface for generating embeddings.
"""

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

from .logger import logger, timer

# Load environment variables
load_dotenv()


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


# Default models per provider
DEFAULT_EMBEDDING_MODELS = {
    EmbeddingProvider.OLLAMA: "qwen3-embedding:0.6b",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.GEMINI: "text-embedding-004",
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: EmbeddingProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    batch_size: int = 100
    timeout: float = 60.0


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._dimension: Optional[int] = None

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @property
    def dimension(self) -> int:
        """Get embedding dimension (computed lazily)."""
        if self._dimension is None:
            test_embedding = self.embed_single("test")
            self._dimension = len(test_embedding)
        return self._dimension


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama local embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            import ollama
            self.client = ollama.Client(
                host=config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                timeout=config.timeout,
            )
            self._available = True
        except ImportError:
            self._available = False
            raise ImportError(
                "Ollama package not installed. Install with: pip install cognidoc[ollama]"
            )

    def embed_single(self, text: str) -> List[float]:
        response = self.client.embeddings(model=self.config.model, prompt=text)
        return response["embedding"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (Ollama processes one at a time)."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_single(text))
        return embeddings


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or config")

            self.client = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install cognidoc[openai]"
            )

    def embed_single(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.config.model,
            input=text,
        )
        return response.data[0].embedding

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (OpenAI supports batching)."""
        if not texts:
            return []

        # OpenAI supports batching up to ~8000 tokens total
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai

            api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment or config")

            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Install with: pip install cognidoc[gemini]"
            )

    def embed_single(self, text: str) -> List[float]:
        result = self.genai.embed_content(
            model=f"models/{self.config.model}",
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        # Gemini supports batch embedding
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            for text in batch:
                embedding = self.embed_single(text)
                all_embeddings.append(embedding)

        return all_embeddings


# Provider registry
EMBEDDING_PROVIDERS = {
    EmbeddingProvider.OLLAMA: OllamaEmbeddingProvider,
    EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider,
    EmbeddingProvider.GEMINI: GeminiEmbeddingProvider,
}


def create_embedding_provider(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """Create an embedding provider based on configuration."""
    provider_class = EMBEDDING_PROVIDERS.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unknown embedding provider: {config.provider}")

    logger.info(f"Creating {config.provider.value} embedding provider with model {config.model}")
    return provider_class(config)


def get_default_embedding_provider() -> BaseEmbeddingProvider:
    """Get the default embedding provider based on environment."""
    provider_name = os.getenv("COGNIDOC_EMBEDDING_PROVIDER", "ollama").lower()
    model = os.getenv("COGNIDOC_EMBEDDING_MODEL")

    try:
        provider = EmbeddingProvider(provider_name)
    except ValueError:
        logger.warning(f"Unknown provider {provider_name}, falling back to ollama")
        provider = EmbeddingProvider.OLLAMA

    if model is None:
        model = DEFAULT_EMBEDDING_MODELS[provider]

    return create_embedding_provider(EmbeddingConfig(
        provider=provider,
        model=model,
    ))


def is_ollama_available() -> bool:
    """Check if Ollama is installed and server is running."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def is_provider_available(provider: EmbeddingProvider) -> bool:
    """Check if a provider is available (package installed + credentials)."""
    if provider == EmbeddingProvider.OLLAMA:
        try:
            import ollama
            return is_ollama_available()
        except ImportError:
            return False

    elif provider == EmbeddingProvider.OPENAI:
        try:
            import openai
            return bool(os.getenv("OPENAI_API_KEY"))
        except ImportError:
            return False

    elif provider == EmbeddingProvider.GEMINI:
        try:
            import google.generativeai
            return bool(os.getenv("GOOGLE_API_KEY"))
        except ImportError:
            return False

    return False


def get_available_embedding_providers() -> List[EmbeddingProvider]:
    """Get list of available embedding providers."""
    available = []
    for provider in EmbeddingProvider:
        if is_provider_available(provider):
            available.append(provider)
    return available


# Global embedding provider instance (lazy-loaded singleton)
_embedding_provider: Optional[BaseEmbeddingProvider] = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Get the global embedding provider instance."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = get_default_embedding_provider()
    return _embedding_provider


def set_embedding_provider(provider: str, model: str = None) -> BaseEmbeddingProvider:
    """Set the global embedding provider."""
    global _embedding_provider

    try:
        provider_enum = EmbeddingProvider(provider.lower())
    except ValueError:
        raise ValueError(f"Unknown provider: {provider}. Available: {[p.value for p in EmbeddingProvider]}")

    if model is None:
        model = DEFAULT_EMBEDDING_MODELS[provider_enum]

    _embedding_provider = create_embedding_provider(EmbeddingConfig(
        provider=provider_enum,
        model=model,
    ))
    logger.info(f"Embedding provider set to: {provider}/{model}")
    return _embedding_provider


def reset_embedding_provider() -> None:
    """Reset the global embedding provider."""
    global _embedding_provider
    _embedding_provider = None


__all__ = [
    "EmbeddingProvider",
    "EmbeddingConfig",
    "BaseEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "create_embedding_provider",
    "get_default_embedding_provider",
    "get_embedding_provider",
    "set_embedding_provider",
    "reset_embedding_provider",
    "is_ollama_available",
    "is_provider_available",
    "get_available_embedding_providers",
    "DEFAULT_EMBEDDING_MODELS",
]
