"""
Parse images containing tables using vision LLM providers.

Extracts table content from document images and exports as Markdown tables.
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
)
from .constants import (
    OLLAMA_VISION_MODEL,
    OPENAI_VISION_MODEL,
    ANTHROPIC_VISION_MODEL,
    GEMINI_VISION_MODEL,
)


# Default prompts for table extraction
DEFAULT_TABLE_SYSTEM_PROMPT = """You are an expert at extracting tables from document images.
Your task is to accurately extract table content and format it as a proper Markdown table.
Preserve the exact content, structure, and data from the original table."""

DEFAULT_TABLE_USER_PROMPT = """Extract the table from this image and format it as a Markdown table.

Requirements:
1. Preserve all cell content exactly as shown
2. Maintain the correct number of rows and columns
3. Use proper Markdown table syntax with | separators
4. Include a header row separator (|---|---|...)
5. Handle merged cells by repeating content if needed
6. If there are multiple tables, extract all of them separated by blank lines

Output ONLY the Markdown table(s), no explanations or additional text."""


def clean_table_output(text: str) -> str:
    """
    Clean and validate extracted table markdown.

    Args:
        text: Raw output from vision model

    Returns:
        Cleaned markdown table
    """
    text = text.strip()

    # Remove markdown code block wrappers if present
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def parse_image_with_table_func(
    image_path: Path,
    output_dir: Path,
    provider: str = "gemini",
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> bool:
    """
    Parse a single image containing a table using vision LLM.

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
    logger.info(f"Parsing table from: {image_path.name} using {provider}")

    # Use default prompts if not provided
    if system_prompt is None:
        system_prompt = DEFAULT_TABLE_SYSTEM_PROMPT
    if user_prompt is None:
        user_prompt = DEFAULT_TABLE_USER_PROMPT

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
            max_tokens=8192,  # Tables can be large
        )
        llm = create_llm_provider(config)
    except Exception as e:
        logger.error(f"Failed to create {provider} provider: {e}")
        return False

    # Extract table using vision
    with timer(f"extract table from {image_path.name}"):
        try:
            output = llm.vision(
                image_path=str(image_path),
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Vision extraction failed for {image_path.name}: {e}")
            return False

    # Clean the extracted table
    markdown_content = clean_table_output(output)

    if not markdown_content:
        logger.warning(f"No table extracted from {image_path.name}")
        return False

    # Save as Markdown
    doc_path = output_dir / f"{image_path.stem}.md"
    try:
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Table content saved to: {doc_path.name} ({len(markdown_content)} chars)")
        return True
    except Exception as e:
        logger.error(f"Failed to save {doc_path.name}: {e}")
        return False


def parse_image_with_table(
    image_dir: str,
    output_dir: str = None,
    provider: str = "gemini",
    model: str = None,
    system_prompt: str = None,
    user_prompt: str = None,
    image_filter: list = None,
) -> dict:
    """
    Process all table images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Output directory for extracted tables
        provider: LLM provider ("gemini", "ollama", "openai", "anthropic")
        model: Model name (uses provider default if not specified)
        system_prompt: System prompt for extraction
        user_prompt: User prompt for extraction
        image_filter: Optional list of PDF stems to filter images by.
                     Detections are named {pdf_stem}_page_{n}_Table_{idx}.jpg.

    Returns:
        Statistics dictionary
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "errors": 0, "skipped": 0}

    logger.info(f"Processing table images in: {image_dir}")
    logger.info(f"Using provider: {provider}, model: {model or 'default'}")

    # Find table images (YOLO-detected table regions)
    table_images = list(image_dir.glob("*_Table_*.jpg"))

    # Filter by PDF stems if provided
    if image_filter:
        original_count = len(table_images)
        table_images = [
            p for p in table_images
            if any(p.stem.startswith(f"{stem}_page_") for stem in image_filter)
        ]
        logger.info(f"Filtered to {len(table_images)} table images (from {original_count}) matching filter")
    logger.info(f"Found {len(table_images)} table images to process")

    for image_path in table_images:
        # Check if already processed
        output_file = output_dir / f"{image_path.stem}.md"
        if output_file.exists():
            logger.debug(f"Skipping already processed: {image_path.name}")
            stats["skipped"] += 1
            continue

        try:
            success = parse_image_with_table_func(
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
        f"Table extraction complete: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats
