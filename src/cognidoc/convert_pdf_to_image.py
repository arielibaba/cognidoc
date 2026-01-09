"""
PDF to image conversion with parallel processing support.

Converts PDF pages to high-resolution PNG images using poppler.
Supports parallel processing for faster conversion of multiple PDFs.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .utils.logger import logger


# Path separator for encoding relative paths in filenames
PATH_SEPARATOR = "__"

# Default DPI for image conversion (600 = high quality for OCR)
DEFAULT_DPI = 600

# Default number of parallel workers (good for M2 with 16GB)
DEFAULT_MAX_WORKERS = 4


@dataclass
class ConversionResult:
    """Result of a single PDF conversion."""
    pdf_path: str
    success: bool
    pages_converted: int
    error: Optional[str] = None


def get_relative_path_prefix(pdf_path: Path, base_dir: Path) -> str:
    """
    Get the relative path prefix for a PDF file.

    For a PDF at data/pdfs/projet_A/doc.pdf, returns "projet_A__doc"
    For a PDF at data/pdfs/doc.pdf, returns "doc"

    Args:
        pdf_path: Path to the PDF file
        base_dir: Base directory (e.g., data/pdfs)

    Returns:
        Encoded filename prefix with path information
    """
    try:
        relative = pdf_path.relative_to(base_dir)
        # Remove the .pdf extension and encode path separators
        relative_stem = relative.with_suffix("")
        parts = relative_stem.parts
        return PATH_SEPARATOR.join(parts)
    except ValueError:
        # Fallback if pdf_path is not relative to base_dir
        return pdf_path.stem


def _convert_single_pdf(args: Tuple[str, str, str, int]) -> ConversionResult:
    """
    Convert a single PDF to images (worker function for parallel processing).

    Args:
        args: Tuple of (pdf_path, image_dir, prefix, dpi)

    Returns:
        ConversionResult with success status and page count
    """
    from pdf2image import convert_from_path

    pdf_path_str, image_dir_str, prefix, dpi = args
    pdf_path = Path(pdf_path_str)
    image_dir = Path(image_dir_str)

    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        pages_converted = 0

        # Save each page
        for i, image in enumerate(images, start=1):
            output_path = image_dir / f"{prefix}_page_{i}.png"
            image.save(output_path, 'PNG')
            pages_converted += 1

        return ConversionResult(
            pdf_path=pdf_path_str,
            success=True,
            pages_converted=pages_converted,
        )

    except Exception as e:
        return ConversionResult(
            pdf_path=pdf_path_str,
            success=False,
            pages_converted=0,
            error=str(e),
        )


def convert_pdf_to_image(
    pdf_dir,
    image_dir,
    dpi: int = DEFAULT_DPI,
    max_workers: int = DEFAULT_MAX_WORKERS,
    parallel: bool = True,
) -> Dict[str, int]:
    """
    Converts each page of every PDF file in the specified directory into separate image files.

    Supports parallel processing for faster conversion of multiple PDFs.
    PDFs in subdirectories have their paths encoded in the output filename.
    Example: data/pdfs/projet_A/doc.pdf -> projet_A__doc_page_1.png

    Args:
        pdf_dir: Directory containing PDF files (searched recursively)
        image_dir: Output directory for images
        dpi: Resolution for image conversion (default 600)
        max_workers: Number of parallel workers (default 4, good for M2 16GB)
        parallel: Whether to use parallel processing (default True)

    Returns:
        Statistics dictionary with counts
    """
    # Ensure paths are Path objects
    pdf_dir = Path(pdf_dir)
    image_dir = Path(image_dir)

    # Check if the pdf_dir exists
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        logger.error(f"The specified pdf_dir '{pdf_dir}' does not exist or is not a directory.")
        return {"total": 0, "success": 0, "failed": 0, "pages": 0}

    # Create image_dir if it doesn't exist
    image_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    logger.info(f"Scanning for PDF files in: {pdf_dir}")
    pdf_files = list(pdf_dir.rglob('*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        return {"total": 0, "success": 0, "failed": 0, "pages": 0}

    # Prepare conversion tasks
    tasks = []
    for pdf_path in pdf_files:
        prefix = get_relative_path_prefix(pdf_path, pdf_dir)
        tasks.append((str(pdf_path), str(image_dir), prefix, dpi))

    stats = {
        "total": len(pdf_files),
        "success": 0,
        "failed": 0,
        "pages": 0,
    }

    if parallel and len(pdf_files) > 1:
        # Parallel processing
        logger.info(f"Converting PDFs in parallel ({max_workers} workers, {dpi} DPI)...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_convert_single_pdf, task): task[0] for task in tasks}

            with tqdm(total=len(futures), desc="Converting PDFs", unit="pdf") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result.success:
                        stats["success"] += 1
                        stats["pages"] += result.pages_converted
                        logger.debug(f"Converted {result.pdf_path}: {result.pages_converted} pages")
                    else:
                        stats["failed"] += 1
                        logger.error(f"Failed to convert {result.pdf_path}: {result.error}")

                    pbar.update(1)
    else:
        # Sequential processing
        logger.info(f"Converting PDFs sequentially ({dpi} DPI)...")

        with tqdm(total=len(tasks), desc="Converting PDFs", unit="pdf") as pbar:
            for task in tasks:
                result = _convert_single_pdf(task)

                if result.success:
                    stats["success"] += 1
                    stats["pages"] += result.pages_converted
                    logger.info(f"Converted {result.pdf_path}: {result.pages_converted} pages")
                else:
                    stats["failed"] += 1
                    logger.error(f"Failed to convert {result.pdf_path}: {result.error}")

                pbar.update(1)

    # Log summary
    logger.info(f"""
    PDF to Image Conversion Complete:
    - Total PDFs: {stats['total']}
    - Successfully converted: {stats['success']}
    - Failed: {stats['failed']}
    - Total pages: {stats['pages']}
    - Output directory: {image_dir}
    """)

    return stats


def convert_pdf_to_image_sequential(pdf_dir, image_dir, dpi: int = DEFAULT_DPI) -> Dict[str, int]:
    """
    Sequential PDF to image conversion (legacy mode).

    Use convert_pdf_to_image() with parallel=True for faster processing.
    """
    return convert_pdf_to_image(pdf_dir, image_dir, dpi=dpi, parallel=False)
