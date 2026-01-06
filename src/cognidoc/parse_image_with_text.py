"""
Parse images containing text using vision LLM providers.

Extracts text content from document images and exports as Markdown.
Supports multiple providers: Gemini (default), Ollama, OpenAI, Anthropic.
"""

import re
from pathlib import Path
from typing import Optional

from .utils.logger import logger, timer
from .utils.llm_providers import (
    LLMProvider,
    LLMConfig,
    create_llm_provider,
    get_default_vision_provider,
)
from .constants import (
    SYSTEM_PROMPT_TEXT_EXTRACT,
    USER_PROMPT_TEXT_EXTRACT,
    DOCLING_MODEL,
    OLLAMA_VISION_MODEL,
    OPENAI_VISION_MODEL,
    ANTHROPIC_VISION_MODEL,
    GEMINI_VISION_MODEL,
)
from .helpers import load_prompt


def clean_extracted_text(text: str) -> str:
    """
    Clean and format extracted text from vision model.

    Args:
        text: Raw text from vision model

    Returns:
        Cleaned markdown text
    """
    # Remove common artifacts from model responses
    text = text.strip()

    # Remove markdown code block wrappers if present
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Remove extraction markers if present
    markers = [
        "## START OF EXTRACTED TEXT",
        "## END OF EXTRACTED TEXT",
        "START OF EXTRACTED TEXT",
        "END OF EXTRACTED TEXT",
    ]
    for marker in markers:
        text = text.replace(marker, "")

    # Clean up excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = text.strip()

    return text


def parse_image_with_text_func(
    image_path: Path,
    output_dir: Path,
    provider: str = "gemini",
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> bool:
    """
    Parse a single image containing text using vision LLM.

    Args:
        image_path: Path to the image file
        output_dir: Directory for output files
        provider: LLM provider ("gemini", "ollama", "openai", "anthropic")
        model: Model name (uses provider default if not specified)
        system_prompt: System prompt for extraction
        user_prompt: User prompt for extraction

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Parsing text from: {image_path.name} using {provider}")

    # Load prompts if not provided
    if system_prompt is None:
        system_prompt = load_prompt(SYSTEM_PROMPT_TEXT_EXTRACT)
    if user_prompt is None:
        user_prompt = load_prompt(USER_PROMPT_TEXT_EXTRACT)
        # Remove the {text} placeholder since we're extracting, not filling in
        user_prompt = user_prompt.replace("{text}", "").strip()

    # Determine model based on provider
    if model is None:
        if provider == "gemini":
            model = GEMINI_VISION_MODEL
        elif provider == "ollama":
            model = OLLAMA_VISION_MODEL
        elif provider == "openai":
            model = OPENAI_VISION_MODEL
        elif provider == "anthropic":
            model = ANTHROPIC_VISION_MODEL
        else:
            model = GEMINI_VISION_MODEL

    # Create provider
    try:
        config = LLMConfig(
            provider=LLMProvider(provider),
            model=model,
            temperature=0.1,  # Low temperature for accurate extraction
            top_p=0.85,
        )
        llm = create_llm_provider(config)
    except Exception as e:
        logger.error(f"Failed to create {provider} provider: {e}")
        return False

    # Extract text using vision
    with timer(f"extract text from {image_path.name}"):
        try:
            output = llm.vision(
                image_path=str(image_path),
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Vision extraction failed for {image_path.name}: {e}")
            return False

    # Clean the extracted text
    markdown_content = clean_extracted_text(output)

    if not markdown_content:
        logger.warning(f"No text extracted from {image_path.name}")
        return False

    # Save as Markdown
    doc_path = output_dir / f"{image_path.stem}.md"
    try:
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Text content saved to: {doc_path.name} ({len(markdown_content)} chars)")
        return True
    except Exception as e:
        logger.error(f"Failed to save {doc_path.name}: {e}")
        return False


def parse_image_with_text(
    image_dir: str,
    output_dir: str = None,
    provider: str = "gemini",
    model: str = None,
    system_prompt: str = None,
    user_prompt: str = None,
) -> dict:
    """
    Process all text images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Output directory for extracted text
        provider: LLM provider ("gemini", "ollama", "openai", "anthropic")
        model: Model name (uses provider default if not specified)
        system_prompt: System prompt for extraction
        user_prompt: User prompt for extraction

    Returns:
        Statistics dictionary
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "errors": 0, "skipped": 0}

    logger.info(f"Processing text images in: {image_dir}")
    logger.info(f"Using provider: {provider}, model: {model or 'default'}")

    # Find text images (YOLO-detected text regions)
    text_images = list(image_dir.glob("*_Text.jpg"))
    logger.info(f"Found {len(text_images)} text images to process")

    for image_path in text_images:
        # Check if already processed
        output_file = output_dir / f"{image_path.stem}.md"
        if output_file.exists():
            logger.debug(f"Skipping already processed: {image_path.name}")
            stats["skipped"] += 1
            continue

        try:
            success = parse_image_with_text_func(
                image_path=image_path,
                output_dir=output_dir,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if success:
                stats["processed"] += 1
            else:
                stats["errors"] += 1
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            stats["errors"] += 1

    logger.info(
        f"Text extraction complete: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats
