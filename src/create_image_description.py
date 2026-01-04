"""
Image description generation using vision LLMs.

Supports multiple providers:
- Gemini (default)
- Ollama (qwen3-vl)
- OpenAI (gpt-4o)
- Anthropic (claude)

Features:
- Async processing with bounded concurrency
- Automatic retry on failure
- Table extraction from descriptions
"""

import asyncio
from pathlib import Path
from typing import Optional

from .helpers import extract_markdown_tables, is_relevant_image
from .constants import (
    IMAGE_PIXEL_THRESHOLD,
    IMAGE_PIXEL_VARIANCE_THRESHOLD,
    DEFAULT_VISION_PROVIDER,
)
from .utils.logger import logger, timer, PipelineTimer
from .utils.llm_providers import (
    LLMProvider,
    LLMConfig,
    create_llm_provider,
    get_default_vision_provider,
)


async def _describe_single_image_with_provider(
    provider,
    image_path: Path,
    system_prompt: str,
    description_prompt: str,
    max_retries: int = 3,
) -> str:
    """
    Describe an image using the configured provider with retry logic.

    Args:
        provider: LLM provider instance
        image_path: Path to the image
        system_prompt: System prompt for the model
        description_prompt: User prompt for description
        max_retries: Number of retry attempts

    Returns:
        Image description text
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            description = await provider.avision(
                image_path=str(image_path),
                prompt=description_prompt,
                system_prompt=system_prompt,
            )
            return description
        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {image_path.name}: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"All attempts failed for {image_path.name}: {last_error}")
    raise last_error


async def _describe_single_image_ollama(
    client,
    model: str,
    system_prompt: str,
    description_prompt: str,
    model_options: dict,
    image_path: Path,
) -> str:
    """
    Legacy Ollama-specific image description (for backward compatibility).

    Args:
        client: Ollama client
        model: Model name
        system_prompt: System prompt
        description_prompt: User prompt
        model_options: Model generation options
        image_path: Path to image

    Returns:
        Description text
    """
    def sync_chat() -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": description_prompt,
                "images": [str(image_path)]
            }
        ]
        resp = client.chat(
            model=model,
            messages=messages,
            options=model_options
        )
        return resp["message"]["content"]

    description = await asyncio.to_thread(sync_chat)
    return description


async def create_image_descriptions_async(
    image_dir: str,
    output_dir: str,
    system_prompt: str,
    description_prompt: str,
    provider: str = None,
    model: str = None,
    temperature: float = 0.2,
    top_p: float = 0.85,
    max_concurrency: int = 5,
    max_retries: int = 3,
    # Legacy parameters for backward compatibility
    ollama_client=None,
    model_options: dict = None,
) -> dict:
    """
    Describe all relevant images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Directory for output descriptions
        system_prompt: System prompt for the model
        description_prompt: User prompt for descriptions
        provider: LLM provider name (gemini, ollama, openai, anthropic)
        model: Model name (defaults to provider's default)
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_concurrency: Maximum concurrent requests
        max_retries: Number of retry attempts per image
        ollama_client: Legacy Ollama client (for backward compatibility)
        model_options: Legacy model options (for backward compatibility)

    Returns:
        Statistics dictionary
    """
    sem = asyncio.Semaphore(max_concurrency)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize timer
    pipeline_timer = PipelineTimer("image_description").start()

    stats = {
        "total_images": 0,
        "described": 0,
        "skipped_irrelevant": 0,
        "tables_extracted": 0,
        "errors": 0,
    }

    # Determine provider
    use_new_provider = ollama_client is None

    if use_new_provider:
        # Use new multi-provider abstraction
        provider_name = provider or DEFAULT_VISION_PROVIDER
        try:
            llm_provider = get_default_vision_provider()
            logger.info(f"Using vision provider: {provider_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vision provider: {e}")
            raise
    else:
        # Legacy Ollama path
        logger.info("Using legacy Ollama client for vision")

    async def _worker(img_path: Path):
        async with sem:
            try:
                logger.info(f"Describing {img_path.name}")

                if use_new_provider:
                    desc = await _describe_single_image_with_provider(
                        provider=llm_provider,
                        image_path=img_path,
                        system_prompt=system_prompt,
                        description_prompt=description_prompt,
                        max_retries=max_retries,
                    )
                else:
                    desc = await _describe_single_image_ollama(
                        client=ollama_client,
                        model=model,
                        system_prompt=system_prompt,
                        description_prompt=description_prompt,
                        model_options=model_options or {},
                        image_path=img_path,
                    )

                # Write description
                desc_file = output_dir / f"{img_path.stem}_description.txt"
                desc_file.write_text(desc, encoding="utf-8")
                stats["described"] += 1

                # Extract and write tables
                tables = extract_markdown_tables(desc)
                for idx, table_md in enumerate(tables, start=1):
                    tbl_file = output_dir / f"{img_path.stem}_table_{idx}.md"
                    tbl_file.write_text(table_md, encoding="utf-8")
                    stats["tables_extracted"] += 1

                logger.info(f"Completed {img_path.name} ({len(tables)} tables)")

            except Exception as e:
                logger.error(f"Failed {img_path.name}: {e}")
                stats["errors"] += 1

    # Find and filter images
    pipeline_timer.stage("finding_images")
    images = list(image_dir.glob("*_Picture_*.jpg"))
    stats["total_images"] = len(images)

    # Filter by relevance
    relevant_images = []
    for img in images:
        if is_relevant_image(img, IMAGE_PIXEL_THRESHOLD, IMAGE_PIXEL_VARIANCE_THRESHOLD):
            relevant_images.append(img)
        else:
            stats["skipped_irrelevant"] += 1

    if not relevant_images:
        logger.warning(f"No relevant images found in {image_dir}")
        pipeline_timer.end()
        return stats

    logger.info(
        f"Found {len(relevant_images)} relevant images "
        f"({stats['skipped_irrelevant']} skipped as irrelevant)"
    )

    # Process images
    pipeline_timer.stage("describing_images")
    await asyncio.gather(*[_worker(img) for img in relevant_images])

    # End timing
    pipeline_timer.end()

    logger.info(f"""
    Image Description Complete:
    - Total images: {stats['total_images']}
    - Described: {stats['described']}
    - Skipped (irrelevant): {stats['skipped_irrelevant']}
    - Tables extracted: {stats['tables_extracted']}
    - Errors: {stats['errors']}
    - Output: {output_dir}
    """)

    return stats
