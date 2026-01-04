"""
Parse images containing tables using granite-docling via Ollama.

Extracts table content from document images and exports as Markdown.
Uses OTSL (Open Table Semantic Language) format for table parsing.
"""

import re
from pathlib import Path
from typing import List

import ollama

from .utils.logger import logger, timer
from .constants import DOCLING_MODEL


def parse_otsl_to_rows(otsl: str) -> List[List[str]]:
    """
    Parse OTSL format to table rows.

    OTSL format:
    - <otsl>...</otsl> - table container
    - <fcel>content - filled cell
    - <ecel> - empty cell
    - <nl> - new row

    Args:
        otsl: Raw OTSL string from granite-docling

    Returns:
        List of rows, each row is a list of cell values
    """
    # Extract content between <otsl> tags
    match = re.search(r"<otsl>.*?(<fcel>.*?)(?:</otsl>|$)", otsl, re.DOTALL)
    if not match:
        return []

    content = match.group(1)

    # Remove location tags
    content = re.sub(r"<loc_\d+>", "", content)

    # Split by newline markers
    row_strings = re.split(r"<nl>", content)

    rows = []
    for row_str in row_strings:
        if not row_str.strip():
            continue

        cells = []
        # Split by cell markers
        parts = re.split(r"(<fcel>|<ecel>)", row_str)

        current_cell = None
        for part in parts:
            if part == "<fcel>":
                if current_cell is not None:
                    cells.append(current_cell.strip())
                current_cell = ""
            elif part == "<ecel>":
                if current_cell is not None:
                    cells.append(current_cell.strip())
                cells.append("")
                current_cell = None
            elif current_cell is not None:
                current_cell += part

        if current_cell is not None:
            cells.append(current_cell.strip())

        if cells:
            rows.append(cells)

    return rows


def rows_to_markdown_table(rows: List[List[str]]) -> str:
    """
    Convert table rows to Markdown table format.

    Args:
        rows: List of rows, each row is a list of cell values

    Returns:
        Markdown formatted table string
    """
    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]

    # Calculate column widths
    col_widths = []
    for col_idx in range(max_cols):
        width = max(len(str(row[col_idx])) for row in normalized)
        col_widths.append(max(width, 3))  # Minimum width of 3

    lines = []

    # Header row
    if normalized:
        header = "| " + " | ".join(
            str(cell).ljust(col_widths[i])
            for i, cell in enumerate(normalized[0])
        ) + " |"
        lines.append(header)

        # Separator
        separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
        lines.append(separator)

        # Data rows
        for row in normalized[1:]:
            data_row = "| " + " | ".join(
                str(cell).ljust(col_widths[i])
                for i, cell in enumerate(row)
            ) + " |"
            lines.append(data_row)

    return "\n".join(lines)


def doctags_to_markdown(doctags: str) -> str:
    """
    Convert DocTags/OTSL to Markdown format.

    Handles both table (OTSL) and text (DocTags) content.

    Args:
        doctags: Raw output from granite-docling

    Returns:
        Markdown formatted string
    """
    # Check if it's a table (OTSL format)
    if "<otsl>" in doctags or "<fcel>" in doctags:
        rows = parse_otsl_to_rows(doctags)
        if rows:
            return rows_to_markdown_table(rows)

    # Fallback: extract any text content
    text_content = re.sub(r"<[^>]+>", " ", doctags)
    text_content = re.sub(r"\s+", " ", text_content).strip()
    return text_content


def parse_image_with_table_func(
    image_path: Path,
    client: ollama.Client,
    model: str,
    prompt: str,
    output_dir: Path,
) -> None:
    """
    Parse a single image containing a table.

    Args:
        image_path: Path to the image file
        client: Ollama client instance
        model: Model name for Ollama
        prompt: Extraction prompt
        output_dir: Directory for output files
    """
    logger.info(f"Parsing table from: {image_path.name}")

    # Call Ollama with vision
    with timer(f"extract table from {image_path.name}"):
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
            options={"num_predict": 8192},  # Tables can be large
        )

    output = response["message"]["content"]

    # Convert to Markdown
    markdown_content = doctags_to_markdown(output)

    # Save as Markdown
    doc_path = output_dir / f"{image_path.stem}.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Table content saved to: {doc_path.name} ({len(markdown_content)} chars)")


def parse_image_with_table(
    image_dir: str,
    model: str = None,
    prompt: str = "Extract the table from this document image.",
    output_dir: str = None,
) -> dict:
    """
    Process all table images in a directory.

    Args:
        image_dir: Directory containing images
        model: Ollama model name (defaults to DOCLING_MODEL)
        prompt: Extraction prompt
        output_dir: Output directory for extracted tables

    Returns:
        Statistics dictionary
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model or DOCLING_MODEL
    client = ollama.Client()

    stats = {"processed": 0, "errors": 0}

    logger.info(f"Processing table images in: {image_dir}")
    logger.info(f"Using model: {model}")

    for image_path in image_dir.glob("*_Table_*.jpg"):
        try:
            parse_image_with_table_func(
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

    logger.info(f"Table extraction complete: {stats['processed']} processed, {stats['errors']} errors")
    return stats
