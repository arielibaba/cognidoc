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
    """Configuration for LLM providers."""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    top_p: float = 0.85
    max_tokens: Optional[int] = None
    timeout: float = 180.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: dict = field(default_factory=dict)


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
    def chat(self, messages: list[Message]) -> LLMResponse:
        """Send messages and get response."""
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
    async def avision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
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
    """Google Gemini provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import google.generativeai as genai

        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or config")

        genai.configure(api_key=api_key)
        self.genai = genai

        generation_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.max_tokens:
            generation_config["max_output_tokens"] = config.max_tokens

        self.model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=generation_config,
        )
        self.vision_model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=generation_config,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Gemini format."""
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [msg.content]
                })

        return gemini_messages, system_instruction

    def chat(self, messages: list[Message]) -> LLMResponse:
        with timer(f"Gemini chat ({self.config.model})"):
            gemini_messages, system_instruction = self._convert_messages(messages)

            # Create chat with system instruction if present
            if system_instruction:
                model = self.genai.GenerativeModel(
                    model_name=self.config.model,
                    system_instruction=system_instruction,
                    generation_config={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                    }
                )
            else:
                model = self.model

            chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
            response = chat.send_message(gemini_messages[-1]["parts"][0] if gemini_messages else "")

            return LLMResponse(
                content=response.text,
                model=self.config.model,
                provider=LLMProvider.GEMINI,
                usage={"candidates": len(response.candidates)} if hasattr(response, "candidates") else None,
            )

    def stream_chat(self, messages: list[Message]) -> Generator[str, None, None]:
        gemini_messages, system_instruction = self._convert_messages(messages)

        if system_instruction:
            model = self.genai.GenerativeModel(
                model_name=self.config.model,
                system_instruction=system_instruction,
            )
        else:
            model = self.model

        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        response = chat.send_message(
            gemini_messages[-1]["parts"][0] if gemini_messages else "",
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def achat(self, messages: list[Message]) -> LLMResponse:
        # Gemini SDK is sync, run in thread
        return await asyncio.to_thread(self.chat, messages)

    def vision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        with timer(f"Gemini vision ({self.config.model})"):
            from PIL import Image
            img = Image.open(image_path)

            if system_prompt:
                model = self.genai.GenerativeModel(
                    model_name=self.config.model,
                    system_instruction=system_prompt,
                )
            else:
                model = self.vision_model

            response = model.generate_content([prompt, img])
            return response.text

    async def avision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
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

    def chat(self, messages: list[Message]) -> LLMResponse:
        with timer(f"Ollama chat ({self.config.model})"):
            ollama_messages = self._convert_messages(messages)
            response = self.client.chat(
                model=self.config.model,
                messages=ollama_messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
            )
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
            messages.append({
                "role": "user",
                "content": prompt,
                "images": [image_path],
            })

            response = self.client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
            )
            return response["message"]["content"]

    async def avision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
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
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_img}"
                        }
                    })
                openai_messages.append({"role": msg.role, "content": content})
            else:
                openai_messages.append({"role": msg.role, "content": msg.content})
        return openai_messages

    def chat(self, messages: list[Message]) -> LLMResponse:
        with timer(f"OpenAI chat ({self.config.model})"):
            openai_messages = self._convert_messages(messages)
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
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

    async def avision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
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
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_img,
                            }
                        })
                    content.append({"type": "text", "text": msg.content})
                    anthropic_messages.append({"role": msg.role, "content": content})
                else:
                    anthropic_messages.append({"role": msg.role, "content": msg.content})

        return system_prompt, anthropic_messages

    def chat(self, messages: list[Message]) -> LLMResponse:
        with timer(f"Anthropic chat ({self.config.model})"):
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

            response = self.client.messages.create(**kwargs)
            return LLMResponse(
                content=response.content[0].text,
                model=self.config.model,
                provider=LLMProvider.ANTHROPIC,
                usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
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

    async def avision(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
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


# Convenience functions for common configurations
def get_default_generation_provider() -> BaseLLMProvider:
    """Get the default generation provider (Gemini)."""
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
    model = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")

    return create_llm_provider(LLMConfig(
        provider=LLMProvider(provider),
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        top_p=float(os.getenv("LLM_TOP_P", "0.85")),
    ))


def get_default_vision_provider() -> BaseLLMProvider:
    """Get the default vision provider (Gemini)."""
    provider = os.getenv("DEFAULT_VISION_PROVIDER", "gemini").lower()
    model = os.getenv("DEFAULT_VISION_MODEL", "gemini-2.0-flash")

    return create_llm_provider(LLMConfig(
        provider=LLMProvider(provider),
        model=model,
        temperature=float(os.getenv("VISION_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("VISION_TOP_P", "0.85")),
    ))


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
