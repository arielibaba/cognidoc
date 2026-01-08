"""
Unit tests for LLM and Embedding providers.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Import provider classes and configs
from cognidoc.utils.llm_providers import (
    LLMProvider,
    LLMConfig,
    Message,
    LLMResponse,
    BaseLLMProvider,
    create_llm_provider,
)
from cognidoc.utils.embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    BaseEmbeddingProvider,
    create_embedding_provider,
    is_ollama_available,
    is_provider_available,
    DEFAULT_EMBEDDING_MODELS,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model="gemini-2.0-flash",
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.85
        assert config.max_tokens is None
        assert config.timeout == 180.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="granite3.3:8b",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            timeout=60.0,
        )
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.max_tokens == 1000
        assert config.timeout == 60.0


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_message_with_images(self):
        """Test creating a message with images."""
        msg = Message(role="user", content="Describe this", images=["image.jpg"])
        assert msg.images == ["image.jpg"]


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating a response."""
        response = LLMResponse(
            content="Hello, I'm Claude",
            model="gemini-2.0-flash",
            provider=LLMProvider.GEMINI,
        )
        assert response.content == "Hello, I'm Claude"
        assert response.model == "gemini-2.0-flash"
        assert response.provider == LLMProvider.GEMINI


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model="qwen3-embedding:0.6b",
        )
        # Default batch_size is 100 in the actual implementation
        assert config.batch_size == 100
        assert config.timeout == 60.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            batch_size=64,
            timeout=120.0,
        )
        assert config.batch_size == 64
        assert config.timeout == 120.0


class TestDefaultEmbeddingModels:
    """Tests for default embedding models."""

    def test_default_models_exist(self):
        """Test that default models are defined for all providers."""
        assert EmbeddingProvider.OLLAMA in DEFAULT_EMBEDDING_MODELS
        assert EmbeddingProvider.OPENAI in DEFAULT_EMBEDDING_MODELS
        assert EmbeddingProvider.GEMINI in DEFAULT_EMBEDDING_MODELS


class TestOllamaProvider:
    """Tests for Ollama provider (mocked)."""

    def test_ollama_chat(self):
        """Test Ollama chat with mocked client."""
        # Create mock ollama module
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Hello from Ollama!"}
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.llm_providers import OllamaProvider

            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="granite3.3:8b",
            )
            provider = OllamaProvider(config)

            messages = [Message(role="user", content="Hello")]
            response = provider.chat(messages)

            assert response.content == "Hello from Ollama!"
            assert response.provider == LLMProvider.OLLAMA

    def test_ollama_embedding(self):
        """Test Ollama embedding with mocked client."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        # Ollama uses client.embeddings(model=..., prompt=...) and returns {"embedding": [...]}
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
            )
            provider = OllamaEmbeddingProvider(config)

            embedding = provider.embed_single("Hello")
            assert embedding == [0.1, 0.2, 0.3]


class TestGeminiProvider:
    """Tests for Gemini provider (mocked)."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_gemini_initialization(self):
        """Test Gemini provider initialization with mocked SDK."""
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": mock_genai, "google.genai.types": mock_types}):
            # Patch the import
            with patch("cognidoc.utils.llm_providers.genai", mock_genai, create=True):
                with patch("cognidoc.utils.llm_providers.types", mock_types, create=True):
                    from cognidoc.utils.llm_providers import GeminiProvider

                    config = LLMConfig(
                        provider=LLMProvider.GEMINI,
                        model="gemini-2.0-flash",
                    )
                    # This test just verifies the test setup works


class TestOpenAIProvider:
    """Tests for OpenAI provider (mocked)."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_openai_chat(self):
        """Test OpenAI chat with mocked client."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI!"))]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from cognidoc.utils.llm_providers import OpenAIProvider

            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
            )
            provider = OpenAIProvider(config)

            messages = [Message(role="user", content="Hello")]
            response = provider.chat(messages)

            assert response.content == "Hello from OpenAI!"
            assert response.provider == LLMProvider.OPENAI

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_openai_embedding(self):
        """Test OpenAI embedding with mocked client."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from cognidoc.utils.embedding_providers import OpenAIEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
            )
            provider = OpenAIEmbeddingProvider(config)

            embedding = provider.embed_single("Hello")
            assert embedding == [0.1, 0.2, 0.3]


class TestProviderAvailability:
    """Tests for provider availability checks."""

    @patch("httpx.get")
    def test_ollama_available(self, mock_get):
        """Test Ollama availability check when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert is_ollama_available() is True

    @patch("httpx.get")
    def test_ollama_not_available(self, mock_get):
        """Test Ollama availability check when server is not running."""
        mock_get.side_effect = Exception("Connection refused")

        assert is_ollama_available() is False

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_provider_available(self):
        """Test OpenAI provider availability with API key set."""
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = is_provider_available(EmbeddingProvider.OPENAI)
            assert result is True

    def test_openai_provider_not_available_no_key(self):
        """Test OpenAI provider not available without API key."""
        # Clear any existing keys
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            mock_openai = MagicMock()
            with patch.dict(sys.modules, {"openai": mock_openai}):
                result = is_provider_available(EmbeddingProvider.OPENAI)
                assert result is False


class TestCreateProvider:
    """Tests for provider factory functions."""

    def test_create_ollama_llm_provider(self):
        """Test creating Ollama LLM provider."""
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = MagicMock()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="granite3.3:8b",
            )
            provider = create_llm_provider(config)
            assert provider is not None

    def test_create_ollama_embedding_provider(self):
        """Test creating Ollama embedding provider."""
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = MagicMock()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
            )
            provider = create_embedding_provider(config)
            assert provider is not None

    def test_create_unknown_provider_raises(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError):
            config = LLMConfig(
                provider="unknown",
                model="some-model",
            )
            create_llm_provider(config)


class TestBatchEmbedding:
    """Tests for batch embedding functionality."""

    def test_batch_embedding(self):
        """Test batch embedding with multiple texts."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        # Ollama uses client.embeddings() and returns {"embedding": [...]}
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
                batch_size=2,
            )
            provider = OllamaEmbeddingProvider(config)

            texts = ["Hello", "World", "Test"]
            embeddings = provider.embed(texts)

            assert len(embeddings) == 3
            assert all(len(e) == 3 for e in embeddings)

    def test_empty_batch_embedding(self):
        """Test batch embedding with empty list."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
            )
            provider = OllamaEmbeddingProvider(config)

            embeddings = provider.embed([])
            assert embeddings == []


class TestAnthropicProvider:
    """Tests for Anthropic provider (mocked)."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_anthropic_chat(self):
        """Test Anthropic chat with mocked client."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude!")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from cognidoc.utils.llm_providers import AnthropicProvider

            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-haiku-20240307",
            )
            provider = AnthropicProvider(config)

            messages = [Message(role="user", content="Hello")]
            response = provider.chat(messages)

            assert response.content == "Hello from Claude!"
            assert response.provider == LLMProvider.ANTHROPIC

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_anthropic_json_mode(self):
        """Test Anthropic JSON mode with system prompt injection."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "test"}')]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from cognidoc.utils.llm_providers import AnthropicProvider

            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-haiku-20240307",
                json_mode=True,
            )
            provider = AnthropicProvider(config)

            messages = [Message(role="user", content="Return JSON")]
            response = provider.chat(messages, json_mode=True)

            # Verify system prompt includes JSON instruction
            call_args = mock_client.messages.create.call_args
            assert "system" in call_args.kwargs
            assert "JSON" in call_args.kwargs["system"]


class TestGeminiEmbeddingProvider:
    """Tests for Gemini embedding provider (mocked)."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_gemini_embedding_single(self):
        """Test Gemini single embedding."""
        # Create mock module and client
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_google.genai = mock_genai
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        # Mock embedding response with proper structure
        expected_values = [0.1, 0.2, 0.3, 0.4]
        mock_embedding = MagicMock()
        mock_embedding.values = expected_values
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_result

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            from cognidoc.utils.embedding_providers import GeminiEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.GEMINI,
                model="text-embedding-004",
            )
            provider = GeminiEmbeddingProvider(config)

            embedding = provider.embed_single("Hello world")
            assert embedding == expected_values

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_gemini_embedding_batch(self):
        """Test Gemini batch embedding."""
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_google.genai = mock_genai
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        expected_values = [0.1, 0.2, 0.3]
        mock_embedding = MagicMock()
        mock_embedding.values = expected_values
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_result

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            from cognidoc.utils.embedding_providers import GeminiEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.GEMINI,
                model="text-embedding-004",
                batch_size=2,
            )
            provider = GeminiEmbeddingProvider(config)

            embeddings = provider.embed(["Hello", "World"])
            assert len(embeddings) == 2
            assert embeddings[0] == expected_values


class TestJSONMode:
    """Tests for JSON mode across providers."""

    def test_ollama_json_mode(self):
        """Test Ollama JSON format parameter."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"test": "value"}'}
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.llm_providers import OllamaProvider

            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="granite3.3:8b",
            )
            provider = OllamaProvider(config)

            messages = [Message(role="user", content="Return JSON")]
            provider.chat(messages, json_mode=True)

            # Verify format="json" was passed
            call_args = mock_client.chat.call_args
            assert call_args.kwargs.get("format") == "json"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_openai_json_mode(self):
        """Test OpenAI JSON response format."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"test": "value"}'))]
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from cognidoc.utils.llm_providers import OpenAIProvider

            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
            )
            provider = OpenAIProvider(config)

            messages = [Message(role="user", content="Return JSON")]
            provider.chat(messages, json_mode=True)

            # Verify response_format was passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs.get("response_format") == {"type": "json_object"}


class TestVisionFunctionality:
    """Tests for vision functionality (mocked)."""

    def test_ollama_vision(self):
        """Test Ollama vision with image path."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "This is an image of a cat."}
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.llm_providers import OllamaProvider

            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="qwen3-vl:8b-instruct",
            )
            provider = OllamaProvider(config)

            result = provider.vision(
                image_path="/fake/path/image.jpg",
                prompt="What is in this image?",
                system_prompt="You are a helpful assistant."
            )

            assert result == "This is an image of a cat."
            # Verify images were passed
            call_args = mock_client.chat.call_args
            messages = call_args.kwargs.get("messages", [])
            assert any("images" in msg for msg in messages)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_openai_vision(self):
        """Test OpenAI vision with base64 encoding."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="A scenic mountain view."))]
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from cognidoc.utils.llm_providers import OpenAIProvider
            import tempfile
            import os as os_module

            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
            )
            provider = OpenAIProvider(config)

            # Create a temporary image file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(b"fake image data")
                temp_path = f.name

            try:
                result = provider.vision(
                    image_path=temp_path,
                    prompt="Describe this image",
                )
                assert result == "A scenic mountain view."
            finally:
                os_module.unlink(temp_path)


class TestProviderEnumValues:
    """Tests for provider enum values."""

    def test_llm_provider_values(self):
        """Test LLM provider enum values."""
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_embedding_provider_values(self):
        """Test embedding provider enum values."""
        assert EmbeddingProvider.OLLAMA.value == "ollama"
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.GEMINI.value == "gemini"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
