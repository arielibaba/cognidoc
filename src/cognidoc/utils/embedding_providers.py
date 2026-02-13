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
    EmbeddingProvider.OLLAMA: os.getenv("EMBED_MODEL", "qwen3-embedding:4b-q8_0"),
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

    def embed_query(self, query: str, task: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a query with task instruction.

        For Qwen3-Embedding, uses format: "Instruct: {task}\nQuery:{query}"
        For other providers, delegates to embed_single (no instruction support).

        Args:
            query: The query text to embed
            task: Task instruction (optional, uses default if not specified)

        Returns:
            Embedding vector as list of floats
        """
        # Default implementation: just use embed_single
        return self.embed_single(query)

    @property
    def dimension(self) -> int:
        """Get embedding dimension (computed lazily)."""
        if self._dimension is None:
            test_embedding = self.embed_single("test")
            self._dimension = len(test_embedding)
        return self._dimension


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama local embedding provider with async batch support and connection pooling."""

    # Shared httpx client for connection reuse (class-level singleton)
    _shared_async_client = None
    _shared_async_client_loop = None  # Track which event loop owns the client

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._host = config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._timeout = config.timeout
        try:
            import ollama

            self.client = ollama.Client(
                host=self._host,
                timeout=self._timeout,
            )
            self._available = True
        except ImportError:
            self._available = False
            raise ImportError(
                "Ollama package not installed. Install with: pip install cognidoc[ollama]"
            )

    @classmethod
    def _get_async_client(cls, timeout: float):
        """Get shared async client with connection pooling.

        Recreates the client if the event loop it was created on has changed
        (e.g. when run_coroutine spawns a thread with asyncio.run).
        """
        import asyncio
        import atexit
        import httpx

        # Detect stale client: if the event loop changed, the old client is unusable
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if cls._shared_async_client is not None and current_loop != cls._shared_async_client_loop:
            # Client was created on a different (possibly closed) loop — discard it
            try:
                if not cls._shared_async_client.is_closed:
                    # Best-effort close; ignore errors on dead loops
                    import contextlib

                    with contextlib.suppress(Exception):
                        asyncio.get_event_loop().run_until_complete(
                            cls._shared_async_client.aclose()
                        )
            except Exception:
                pass
            cls._shared_async_client = None
            cls._shared_async_client_loop = None

        if cls._shared_async_client is None:
            # Create client with connection pooling (keeps connections alive)
            cls._shared_async_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
            cls._shared_async_client_loop = current_loop

            # Register cleanup so the client is closed on interpreter shutdown
            def _close_client():
                client = cls._shared_async_client
                if client is not None:
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(client.aclose())

            atexit.register(_close_client)
        return cls._shared_async_client

    def embed_single(self, text: str) -> List[float]:
        response = self.client.embeddings(model=self.config.model, prompt=text)
        if "embedding" not in response:
            raise ValueError(f"Ollama response missing 'embedding' key: {list(response.keys())}")
        result: List[float] = response["embedding"]
        return result

    def embed_query(self, query: str, task: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a query with task instruction.

        For Qwen3-Embedding, uses format: "Instruct: {task}\\nQuery:{query}"
        This improves retrieval accuracy by ~1-5% compared to plain query embedding.

        Args:
            query: The query text to embed
            task: Task instruction (uses QWEN_EMBEDDING_TASK from constants if not specified)

        Returns:
            Embedding vector as list of floats
        """
        # Only apply task instruction for Qwen3-Embedding models
        if "qwen" in self.config.model.lower() and "embedding" in self.config.model.lower():
            from cognidoc.constants import QWEN_EMBEDDING_TASK

            if task is None:
                task = QWEN_EMBEDDING_TASK
            # Format: "Instruct: {task}\nQuery:{query}"
            formatted_query = f"Instruct: {task}\nQuery:{query}"
            return self.embed_single(formatted_query)
        else:
            # For other models, just embed the query directly
            return self.embed_single(query)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (Ollama processes one at a time)."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_single(text))
        return embeddings

    async def embed_async(self, texts: List[str], max_concurrent: int = 4) -> List[List[float]]:
        """
        Embed multiple texts with concurrent async requests.

        Uses asyncio to overlap network I/O while Ollama processes sequentially.
        This reduces total time by hiding network latency.
        Uses connection pooling for better performance.

        Args:
            texts: List of texts to embed
            max_concurrent: Max concurrent requests (default 4, good for local)

        Returns:
            List of embeddings in same order as input texts
        """
        import asyncio

        if not texts:
            return []

        # Use shared client with connection pooling
        client = self._get_async_client(self._timeout)
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(texts)

        async def embed_one(idx: int, text: str):
            async with semaphore:
                try:
                    response = await client.post(
                        f"{self._host}/api/embeddings",
                        json={"model": self.config.model, "prompt": text},
                    )
                    response.raise_for_status()
                    results[idx] = response.json()["embedding"]
                except Exception as e:
                    logger.error(f"Embedding error for text[{idx}] ({len(text)} chars): {e}")

        await asyncio.gather(*[embed_one(i, t) for i, t in enumerate(texts)])
        return [r for r in results if r is not None]  # type: ignore[misc]


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
        result: List[float] = response.data[0].embedding
        return result

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (OpenAI supports batching)."""
        if not texts:
            return []

        # OpenAI supports batching up to ~8000 tokens total
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider using the new google-genai SDK."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            from google import genai

            api_key = config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or config"
                )

            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            )

    def embed_single(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.config.model,
            contents=text,
        )
        values: List[float] = response.embeddings[0].values
        return values

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        # Embed texts in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
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
    return provider_class(config)  # type: ignore[abstract]


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

    return create_embedding_provider(
        EmbeddingConfig(
            provider=provider,
            model=model,
        )
    )


def is_ollama_available() -> bool:
    """Check if Ollama is installed and server is running."""
    try:
        import httpx
        from ..constants import OLLAMA_URL

        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return resp.status_code == 200
    except (ImportError, OSError, ConnectionError, ValueError):
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
            from google import genai

            return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
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


def set_embedding_provider(provider: str, model: Optional[str] = None) -> BaseEmbeddingProvider:
    """Set the global embedding provider."""
    global _embedding_provider

    try:
        provider_enum = EmbeddingProvider(provider.lower())
    except ValueError:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {[p.value for p in EmbeddingProvider]}"
        )

    if model is None:
        model = DEFAULT_EMBEDDING_MODELS[provider_enum]

    _embedding_provider = create_embedding_provider(
        EmbeddingConfig(
            provider=provider_enum,
            model=model,
        )
    )
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
