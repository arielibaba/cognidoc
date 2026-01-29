"""
Multi-provider LLM abstraction layer.

Supports:
- Google Gemini (default)
- Ollama (local)
- OpenAI
- Anthropic

Each provider implements a common interface for:
- Text generation (chat)
- Vision (image description)
- Embeddings
"""

import os
import base64
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional, Union
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .logger import logger, timer

# Load environment variables
load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM providers.

    Can be created manually or using from_model() to auto-load specs.
    """

    provider: LLMProvider
    model: str
    temperature: float = 0.7
    top_p: float = 0.85
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None  # Max input tokens
    timeout: float = 180.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    json_mode: bool = False  # Force JSON output (supported by Gemini, OpenAI)
    supports_vision: bool = False
    supports_streaming: bool = True
    extra_params: dict = field(default_factory=dict)

    @classmethod
    def from_model(
        cls,
        model: str,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        json_mode: bool = False,
    ) -> "LLMConfig":
        """
        Create LLMConfig from model name, auto-loading specs from MODEL_SPECS.

        Args:
            model: Model name (e.g., "gemini-2.5-flash", "gpt-4o")
            provider: Override provider (auto-detected from specs if not provided)
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Override max output tokens
            timeout: Request timeout
            api_key: API key for the provider
            base_url: Custom base URL (for Ollama or proxies)
            json_mode: Force JSON output

        Returns:
            LLMConfig with model specs applied
        """
        from ..constants import get_model_specs

        specs = get_model_specs(model)

        # Determine provider
        if provider:
            llm_provider = LLMProvider(provider.lower())
        elif "provider" in specs:
            llm_provider = LLMProvider(specs["provider"])
        else:
            # Guess from model name
            if "gemini" in model.lower():
                llm_provider = LLMProvider.GEMINI
            elif "gpt" in model.lower():
                llm_provider = LLMProvider.OPENAI
            elif "claude" in model.lower():
                llm_provider = LLMProvider.ANTHROPIC
            else:
                llm_provider = LLMProvider.OLLAMA

        return cls(
            provider=llm_provider,
            model=model,
            temperature=(
                temperature if temperature is not None else specs.get("default_temperature", 0.7)
            ),
            top_p=top_p if top_p is not None else specs.get("default_top_p", 0.9),
            max_tokens=max_tokens if max_tokens is not None else specs.get("max_output_tokens"),
            context_window=specs.get("context_window"),
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            json_mode=json_mode,
            supports_vision=specs.get("supports_vision", False),
            supports_streaming=specs.get("supports_streaming", True),
        )


@dataclass
class Message:
    """Chat message."""

    role: str  # "system", "user", "assistant"
    content: str
    images: Optional[list[str]] = None  # List of image paths or base64


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    provider: LLMProvider
    usage: Optional[dict] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def chat(self, messages: list[Message], json_mode: bool = False) -> LLMResponse:
        """Send messages and get response. json_mode forces JSON output if supported."""
        pass

    @abstractmethod
    def stream_chat(self, messages: list[Message]) -> Generator[str, None, None]:
        """Stream chat response."""
        pass

    @abstractmethod
    async def achat(self, messages: list[Message]) -> LLMResponse:
        """Async chat."""
        pass

    @abstractmethod
    def vision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Describe an image."""
        pass

    @abstractmethod
    async def avision(
        self, image_path: str, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Async image description."""
        pass

    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type from image path."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider using the new google-genai SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from google import genai
        from google.genai import types

        api_key = config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or config")

        self.client = genai.Client(api_key=api_key)
        self.types = types

        # Store generation config for later use
        self.generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_tokens if config.max_tokens else None,
        )

    def _convert_messages(self, messages: list[Message]) -> tuple[list, str | None]:
        """Convert messages to Gemini format."""
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append(
                    self.types.Content(
                        role=role, parts=[self.types.Part.from_text(text=msg.content)]
                    )
                )

        return gemini_messages, system_instruction

    def chat(self, messages: list[Message], json_mode: bool = False) -> LLMResponse:
        with timer(f"Gemini chat ({self.config.model})"):
            gemini_messages, system_instruction = self._convert_messages(messages)

            # Build generation config
            config_kwargs = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            if self.config.max_tokens:
                config_kwargs["max_output_tokens"] = self.config.max_tokens
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            # Enable JSON mode if requested
            if json_mode or self.config.json_mode:
                config_kwargs["response_mime_type"] = "application/json"

            generation_config = self.types.GenerateContentConfig(**config_kwargs)

            # Build contents from messages
            if gemini_messages:
                contents = gemini_messages
            else:
                contents = [
                    self.types.Content(role="user", parts=[self.types.Part.from_text(text="")])
                ]

            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            )

            return LLMResponse(
                content=response.text,
                model=self.config.model,
                provider=LLMProvider.GEMINI,
                usage=(
                    {"candidates": len(response.candidates)}
                    if hasattr(response, "candidates")
                    else None
                ),
            )

    def stream_chat(self, messages: list[Message]) -> Generator[str, None, None]:
        gemini_messages, system_instruction = self._convert_messages(messages)

        config_kwargs = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        generation_config = self.types.GenerateContentConfig(**config_kwargs)

        if gemini_messages:
            contents = gemini_messages
        else:
            contents = [self.types.Content(role="user", parts=[self.types.Part.from_text(text="")])]

        for chunk in self.client.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=generation_config,
        ):
            if chunk.text:
                yield chunk.text

    async def achat(self, messages: list[Message]) -> LLMResponse:
        # Gemini SDK is sync, run in thread
        return await asyncio.to_thread(self.chat, messages)

    def vision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        with timer(f"Gemini vision ({self.config.model})"):
            import mimetypes

            # Read image as bytes
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Determine mime type
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = "image/jpeg"

            config_kwargs = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            generation_config = self.types.GenerateContentConfig(**config_kwargs)

            # Create content with image bytes and prompt (new SDK format)
            contents = [
                self.types.Content(
                    role="user",
                    parts=[
                        self.types.Part.from_text(text=prompt),
                        self.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                )
            ]

            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            )
            return response.text

    async def avision(
        self, image_path: str, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        return await asyncio.to_thread(self.vision, image_path, prompt, system_prompt)


class OllamaProvider(BaseLLMProvider):
    """Ollama local provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import ollama

        self.client = ollama.Client(
            host=config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            timeout=config.timeout,
        )
        self.ollama = ollama

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Ollama format."""
        ollama_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.images:
                m["images"] = msg.images
            ollama_messages.append(m)
        return ollama_messages

    def chat(self, messages: list[Message], json_mode: bool = False) -> LLMResponse:
        with timer(f"Ollama chat ({self.config.model})"):
            ollama_messages = self._convert_messages(messages)
            kwargs = {
                "model": self.config.model,
                "messages": ollama_messages,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
            }
            # Enable JSON format if requested
            if json_mode or self.config.json_mode:
                kwargs["format"] = "json"

            response = self.client.chat(**kwargs)
            return LLMResponse(
                content=response["message"]["content"],
                model=self.config.model,
                provider=LLMProvider.OLLAMA,
            )

    def stream_chat(self, messages: list[Message]) -> Generator[str, None, None]:
        ollama_messages = self._convert_messages(messages)
        stream = self.client.chat(
            model=self.config.model,
            messages=ollama_messages,
            options={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
            stream=True,
        )
        for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    async def achat(self, messages: list[Message]) -> LLMResponse:
        return await asyncio.to_thread(self.chat, messages)

    def vision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        with timer(f"Ollama vision ({self.config.model})"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            )

            response = self.client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
            )
            return response["message"]["content"]

    async def avision(
        self, image_path: str, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        return await asyncio.to_thread(self.vision, image_path, prompt, system_prompt)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or config")

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        for msg in messages:
            if msg.images:
                content = [{"type": "text", "text": msg.content}]
                for img_path in msg.images:
                    base64_img = self._encode_image_base64(img_path)
                    mime_type = self._get_image_mime_type(img_path)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                        }
                    )
                openai_messages.append({"role": msg.role, "content": content})
            else:
                openai_messages.append({"role": msg.role, "content": msg.content})
        return openai_messages

    def chat(self, messages: list[Message], json_mode: bool = False) -> LLMResponse:
        with timer(f"OpenAI chat ({self.config.model})"):
            openai_messages = self._convert_messages(messages)
            kwargs = {
                "model": self.config.model,
                "messages": openai_messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }
            # Enable JSON format if requested
            if json_mode or self.config.json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.config.model,
                provider=LLMProvider.OPENAI,
                usage=response.usage.model_dump() if response.usage else None,
            )

    def stream_chat(self, messages: list[Message]) -> Generator[str, None, None]:
        openai_messages = self._convert_messages(messages)
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def achat(self, messages: list[Message]) -> LLMResponse:
        return await asyncio.to_thread(self.chat, messages)

    def vision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt, images=[image_path]))
        response = self.chat(messages)
        return response.content

    async def avision(
        self, image_path: str, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        return await asyncio.to_thread(self.vision, image_path, prompt, system_prompt)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import anthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or config")

        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=config.timeout,
        )

    def _convert_messages(self, messages: list[Message]) -> tuple[Optional[str], list[dict]]:
        """Convert messages to Anthropic format."""
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                if msg.images:
                    content = []
                    for img_path in msg.images:
                        base64_img = self._encode_image_base64(img_path)
                        mime_type = self._get_image_mime_type(img_path)
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_img,
                                },
                            }
                        )
                    content.append({"type": "text", "text": msg.content})
                    anthropic_messages.append({"role": msg.role, "content": content})
                else:
                    anthropic_messages.append({"role": msg.role, "content": msg.content})

        return system_prompt, anthropic_messages

    def chat(self, messages: list[Message], json_mode: bool = False) -> LLMResponse:
        with timer(f"Anthropic chat ({self.config.model})"):
            system_prompt, anthropic_messages = self._convert_messages(messages)

            # Anthropic doesn't have native JSON mode, but we can prepend a system instruction
            if json_mode or self.config.json_mode:
                json_instruction = (
                    "You MUST respond with ONLY valid JSON. No text before or after the JSON."
                )
                if system_prompt:
                    system_prompt = f"{json_instruction}\n\n{system_prompt}"
                else:
                    system_prompt = json_instruction

            kwargs = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens or 4096,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
            return LLMResponse(
                content=response.content[0].text,
                model=self.config.model,
                provider=LLMProvider.ANTHROPIC,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

    def stream_chat(self, messages: list[Message]) -> Generator[str, None, None]:
        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    async def achat(self, messages: list[Message]) -> LLMResponse:
        return await asyncio.to_thread(self.chat, messages)

    def vision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt, images=[image_path]))
        response = self.chat(messages)
        return response.content

    async def avision(
        self, image_path: str, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        return await asyncio.to_thread(self.vision, image_path, prompt, system_prompt)


# Factory function
def create_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """Create an LLM provider based on configuration."""
    providers = {
        LLMProvider.GEMINI: GeminiProvider,
        LLMProvider.OLLAMA: OllamaProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
    }

    provider_class = providers.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unknown provider: {config.provider}")

    logger.info(f"Creating {config.provider.value} provider with model {config.model}")
    return provider_class(config)


# Default models per provider (fallback if env vars not set)
_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "granite3.3:8b",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}

_DEFAULT_VISION_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3-vl:8b-instruct",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}

# Environment variable names for provider-specific models
_MODEL_ENV_VARS = {
    "gemini": "GEMINI_LLM_MODEL",
    "ollama": "OLLAMA_LLM_MODEL",
    "openai": "OPENAI_LLM_MODEL",
    "anthropic": "ANTHROPIC_LLM_MODEL",
}

_VISION_MODEL_ENV_VARS = {
    "gemini": "GEMINI_VISION_MODEL",
    "ollama": "OLLAMA_VISION_MODEL",
    "openai": "OPENAI_VISION_MODEL",
    "anthropic": "ANTHROPIC_VISION_MODEL",
}


def _get_model_for_provider(provider: str, vision: bool = False) -> str:
    """
    Get the model name for a given provider.

    Priority order:
    1. Provider-specific env var (e.g., OLLAMA_LLM_MODEL)
    2. Built-in default for that provider

    This ensures that when switching providers at runtime (e.g., DEFAULT_LLM_PROVIDER=ollama),
    the correct model for that provider is used, even if DEFAULT_LLM_MODEL is set to a
    different provider's model in .env.
    """
    env_vars = _VISION_MODEL_ENV_VARS if vision else _MODEL_ENV_VARS
    defaults = _DEFAULT_VISION_MODELS if vision else _DEFAULT_MODELS

    # Check provider-specific env var first
    env_var = env_vars.get(provider)
    if env_var:
        model = os.getenv(env_var)
        if model:
            return model

    # Fall back to built-in default
    return defaults.get(provider, defaults["gemini"])


# Convenience functions for common configurations
def get_default_generation_provider() -> BaseLLMProvider:
    """Get the default generation provider with auto-loaded model specs."""
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
    # Use provider-specific model (from env var or built-in default)
    model = _get_model_for_provider(provider)

    # Use from_model to auto-load specs, but allow env overrides
    config = LLMConfig.from_model(
        model=model,
        provider=provider,
        temperature=float(os.getenv("LLM_TEMPERATURE")) if os.getenv("LLM_TEMPERATURE") else None,
        top_p=float(os.getenv("LLM_TOP_P")) if os.getenv("LLM_TOP_P") else None,
    )
    return create_llm_provider(config)


def get_default_vision_provider() -> BaseLLMProvider:
    """Get the default vision provider with auto-loaded model specs."""
    provider = os.getenv("DEFAULT_VISION_PROVIDER", "gemini").lower()
    # Use provider-specific model (from env var or built-in default)
    model = _get_model_for_provider(provider, vision=True)

    # Use from_model to auto-load specs, with lower temperature for vision
    config = LLMConfig.from_model(
        model=model,
        provider=provider,
        temperature=float(os.getenv("VISION_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("VISION_TOP_P")) if os.getenv("VISION_TOP_P") else None,
    )
    return create_llm_provider(config)


__all__ = [
    "LLMProvider",
    "LLMConfig",
    "Message",
    "LLMResponse",
    "BaseLLMProvider",
    "GeminiProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_llm_provider",
    "get_default_generation_provider",
    "get_default_vision_provider",
]
