"""
Unified LLM client for CogniDoc.

Provides a singleton LLM client that uses the configured provider (Gemini by default).
Includes utilities for parallel LLM calls and streaming.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple

from .llm_providers import (
    LLMConfig,
    LLMProvider,
    Message,
    BaseLLMProvider,
    create_llm_provider,
    get_default_generation_provider,
)
from .logger import logger

# Global LLM client (lazy-loaded singleton)
_llm_client: Optional[BaseLLMProvider] = None
# Thread pool for async LLM calls - sized to match max entity extraction concurrency (8)
# The semaphore in extract_entities.py controls actual concurrency, this just provides threads
_executor = ThreadPoolExecutor(max_workers=8)


def get_llm_client() -> BaseLLMProvider:
    """
    Get the global LLM client (Gemini by default).

    Returns:
        BaseLLMProvider instance configured from environment
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = get_default_generation_provider()
        logger.info(
            f"LLM client initialized: {_llm_client.config.provider.value}/{_llm_client.config.model}"
        )
    return _llm_client


def reset_llm_client() -> None:
    """Reset the global LLM client (useful for testing or provider switching)."""
    global _llm_client
    _llm_client = None
    logger.info("LLM client reset")


def set_llm_provider(provider: str, model: str = None) -> BaseLLMProvider:
    """
    Set a specific LLM provider with auto-loaded model specs.

    Args:
        provider: Provider name ("gemini", "ollama", "openai", "anthropic")
        model: Model name (optional, uses default for provider)

    Returns:
        Configured BaseLLMProvider with model specs applied
    """
    global _llm_client

    default_models = {
        "gemini": "gemini-2.5-flash",
        "ollama": "granite3.3:8b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
    }

    model = model or default_models.get(provider, "gemini-2.5-flash")

    # Use from_model to auto-load specs
    _llm_client = create_llm_provider(
        LLMConfig.from_model(
            model=model,
            provider=provider,
        )
    )
    logger.info(f"LLM client set to: {provider}/{model}")
    return _llm_client


def llm_chat(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
    """
    Send a chat request to the LLM.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        temperature: Override default temperature
        json_mode: Force JSON output (supported by Gemini, OpenAI, Ollama)

    Returns:
        Response text from LLM
    """
    client = get_llm_client()

    # Convert dict messages to Message objects
    msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]

    # Override temperature if specified
    if temperature is not None:
        original_temp = client.config.temperature
        client.config.temperature = temperature
        try:
            response = client.chat(msg_objects, json_mode=json_mode)
        finally:
            client.config.temperature = original_temp
    else:
        response = client.chat(msg_objects, json_mode=json_mode)

    return response.content


def llm_stream(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
) -> Generator[str, None, None]:
    """
    Stream a chat response from the LLM.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        temperature: Override default temperature

    Yields:
        Response text chunks (accumulated)
    """
    client = get_llm_client()
    msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]

    # Override temperature if specified
    if temperature is not None:
        original_temp = client.config.temperature
        client.config.temperature = temperature

    accumulated = ""
    try:
        for chunk in client.stream_chat(msg_objects):
            accumulated += chunk
            yield accumulated
    finally:
        if temperature is not None:
            client.config.temperature = original_temp


async def llm_chat_async(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
    """
    Async version of llm_chat for parallel execution.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        temperature: Override default temperature
        json_mode: Force JSON output (supported by Gemini, OpenAI, Ollama)

    Returns:
        Response text from LLM
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: llm_chat(messages, temperature, json_mode))


# Ingestion-specific LLM client (uses INGESTION_LLM_MODEL for higher quality)
_ingestion_llm_client: Optional[BaseLLMProvider] = None


def get_ingestion_llm_client() -> BaseLLMProvider:
    """
    Get the ingestion-specific LLM client (uses INGESTION_LLM_MODEL).

    This client uses gemini-3-pro-preview by default for higher quality
    during ingestion operations like entity extraction and resolution.

    Returns:
        BaseLLMProvider instance configured for ingestion
    """
    global _ingestion_llm_client
    if _ingestion_llm_client is None:
        from ..constants import INGESTION_LLM_MODEL

        _ingestion_llm_client = create_llm_provider(
            LLMConfig.from_model(
                model=INGESTION_LLM_MODEL,
                provider="gemini",
            )
        )
        logger.info(f"Ingestion LLM client initialized: gemini/{INGESTION_LLM_MODEL}")
    return _ingestion_llm_client


def llm_chat_ingestion(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
    """
    Send a chat request using the ingestion LLM (higher quality model).

    Uses INGESTION_LLM_MODEL (gemini-3-pro-preview by default) for
    ingestion operations like entity extraction and resolution.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        temperature: Override default temperature
        json_mode: Force JSON output

    Returns:
        Response text from LLM
    """
    client = get_ingestion_llm_client()
    msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]

    if temperature is not None:
        original_temp = client.config.temperature
        client.config.temperature = temperature
        try:
            response = client.chat(msg_objects, json_mode=json_mode)
        finally:
            client.config.temperature = original_temp
    else:
        response = client.chat(msg_objects, json_mode=json_mode)

    return response.content


async def llm_chat_async_ingestion(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
    """
    Async version of llm_chat_ingestion for parallel execution.

    Uses INGESTION_LLM_MODEL (gemini-3-pro-preview by default) for
    ingestion operations like entity extraction and resolution.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        temperature: Override default temperature
        json_mode: Force JSON output

    Returns:
        Response text from LLM
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, lambda: llm_chat_ingestion(messages, temperature, json_mode)
    )


async def run_parallel_llm_calls(
    calls: List[Tuple[str, List[Dict[str, str]], Optional[float]]],
) -> Dict[str, str]:
    """
    Run multiple LLM calls in parallel.

    Args:
        calls: List of (name, messages, temperature) tuples

    Returns:
        Dict mapping name to response text

    Example:
        results = await run_parallel_llm_calls([
            ("rewrite", rewrite_messages, 0.3),
            ("classify", classify_messages, 0.1),
        ])
        rewritten = results["rewrite"]
        classification = results["classify"]
    """

    async def call_with_name(
        name: str, messages: List[Dict[str, str]], temp: Optional[float]
    ) -> Tuple[str, str]:
        result = await llm_chat_async(messages, temp)
        return (name, result)

    tasks = [call_with_name(name, msgs, temp) for name, msgs, temp in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Parallel LLM call failed: {result}")
            continue
        name, response = result
        output[name] = response

    return output


def run_parallel_sync(
    calls: List[Tuple[str, List[Dict[str, str]], Optional[float]]],
) -> Dict[str, str]:
    """
    Synchronous wrapper for parallel LLM calls.

    Args:
        calls: List of (name, messages, temperature) tuples

    Returns:
        Dict mapping name to response text
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, use thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_parallel_llm_calls(calls))
                return future.result()
        else:
            return loop.run_until_complete(run_parallel_llm_calls(calls))
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(run_parallel_llm_calls(calls))


def get_provider_info() -> Dict[str, Any]:
    """Get information about the current LLM provider including model specs."""
    client = get_llm_client()
    return {
        "provider": client.config.provider.value,
        "model": client.config.model,
        "temperature": client.config.temperature,
        "top_p": client.config.top_p,
        "max_tokens": client.config.max_tokens,
        "context_window": client.config.context_window,
        "supports_vision": client.config.supports_vision,
        "supports_streaming": client.config.supports_streaming,
    }


__all__ = [
    "get_llm_client",
    "reset_llm_client",
    "set_llm_provider",
    "llm_chat",
    "llm_stream",
    "llm_chat_async",
    "run_parallel_llm_calls",
    "run_parallel_sync",
    "get_provider_info",
]
