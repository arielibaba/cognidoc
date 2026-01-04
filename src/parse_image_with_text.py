"""
Parse images containing text using granite-docling via Ollama.

Extracts text content from document images and exports as Markdown.
Uses DocTags format for structured document parsing.
"""

import re
from pathlib import Path
from typing import List, Tuple

import ollama
from PIL import Image

from .utils.logger import logger, timer
from .constants import DOCLING_MODEL


def extract_text_from_doctags(doctags: str) -> List[Tuple[str, str]]:
    """
    Extract text content from DocTags format.

    Args:
        doctags: Raw DocTags string from granite-docling

    Returns:
        List of (tag_type, content) tuples
    """
    elements = []

    # Pattern to match DocTags elements with content
    # Matches: <tag><loc_...>...<loc_...>content</tag>
    patterns = [
        (r"<section_header_level_(\d+)>(?:<loc_\d+>)*([^<]+)</section_header_level_\d+>", "header"),
        (r"<text>(?:<loc_\d+>)*([^<]+)</text>", "text"),
        (r"<paragraph>(?:<loc_\d+>)*([^<]+)</paragraph>", "paragraph"),
        (r"<caption>(?:<loc_\d+>)*([^<]+)</caption>", "caption"),
        (r"<list_item>(?:<loc_\d+>)*([^<]+)</list_item>", "list_item"),
        (r"<title>(?:<loc_\d+>)*([^<]+)</title>", "title"),
    ]

    for pattern, tag_type in patterns:
        matches = re.findall(pattern, doctags)
        for match in matches:
            if isinstance(match, tuple):
                # For headers with level
                content = match[-1].strip()
            else:
                content = match.strip()
            if content:
                elements.append((tag_type, content))

    return elements


def doctags_to_markdown(doctags: str) -> str:
    """
    Convert DocTags to Markdown format.

    Args:
        doctags: Raw DocTags string

    Returns:
        Markdown formatted string
    """
    elements = extract_text_from_doctags(doctags)

    if not elements:
        # Fallback: extract any text between tags
        text_content = re.sub(r"<[^>]+>", " ", doctags)
        text_content = re.sub(r"\s+", " ", text_content).strip()
        return text_content

    lines = []
    for tag_type, content in elements:
        if tag_type == "header":
            lines.append(f"## {content}\n")
        elif tag_type == "title":
            lines.append(f"# {content}\n")
        elif tag_type == "list_item":
            lines.append(f"- {content}")
        elif tag_type == "caption":
            lines.append(f"*{content}*\n")
        else:
            lines.append(content)

    return "\n\n".join(lines)


def parse_image_with_text_func(
    image_path: Path,
    client: ollama.Client,
    model: str,
    prompt: str,
    output_dir: Path,
) -> None:
    """
    Parse a single image containing text.

    Args:
        image_path: Path to the image file
        client: Ollama client instance
        model: Model name for Ollama
        prompt: Extraction prompt
        output_dir: Directory for output files
    """
    logger.info(f"Parsing text from: {image_path.name}")

    # Call Ollama with vision
    with timer(f"extract text from {image_path.name}"):
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
            options={"num_predict": 4096},
        )

    output = response["message"]["content"]

    # Convert DocTags to Markdown
    markdown_content = doctags_to_markdown(output)

    # Save as Markdown
    doc_path = output_dir / f"{image_path.stem}.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Text content saved to: {doc_path.name} ({len(markdown_content)} chars)")


def parse_image_with_text(
    image_dir: str,
    model: str = None,
    prompt: str = "Extract all text from this document image.",
    output_dir: str = None,
) -> dict:
    """
    Process all text images in a directory.

    Args:
        image_dir: Directory containing images
        model: Ollama model name (defaults to DOCLING_MODEL)
        prompt: Extraction prompt
        output_dir: Output directory for extracted text

    Returns:
        Statistics dictionary
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model or DOCLING_MODEL
    client = ollama.Client()

    stats = {"processed": 0, "errors": 0}

    logger.info(f"Processing text images in: {image_dir}")
    logger.info(f"Using model: {model}")

    for image_path in image_dir.glob("*_Text.jpg"):
        try:
            parse_image_with_text_func(
                image_path=image_path,
                client=client,
                model=model,
                prompt=prompt,
                output_dir=output_dir,
            )
            stats["processed"] += 1
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            stats["errors"] += 1

    logger.info(f"Text extraction complete: {stats['processed']} processed, {stats['errors']} errors")
    return stats
