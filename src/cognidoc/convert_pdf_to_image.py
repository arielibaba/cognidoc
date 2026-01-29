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

# Default number of parallel workers
# Reduced to 2 for memory safety - each worker can use 1-2GB for large PDFs
# Increase to 4 only if you have 32GB+ RAM
DEFAULT_MAX_WORKERS = 2

# Default batch size for page conversion
# Batching reduces convert_from_path() initialization overhead (20-30% speedup)
# Each page at 600 DPI uses ~50-100MB, so batch_size=5 uses ~250-500MB per worker
DEFAULT_PAGE_BATCH_SIZE = 5


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


def _get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF without loading all pages into memory."""
    from pdf2image.pdf2image import pdfinfo_from_path

    try:
        info = pdfinfo_from_path(str(pdf_path))
        return info.get("Pages", 0)
    except Exception:
        # Fallback: try loading first page to check if PDF is valid
        return -1  # Unknown, will process page by page until error


def _convert_single_pdf(args: Tuple[str, str, str, int, int]) -> ConversionResult:
    """
    Convert a single PDF to images (worker function for parallel processing).

    Memory-optimized: converts pages in batches to balance speed and memory usage.
    Batching reduces convert_from_path() initialization overhead (20-30% speedup).

    Args:
        args: Tuple of (pdf_path, image_dir, prefix, dpi, page_batch_size)

    Returns:
        ConversionResult with success status and page count
    """
    from pdf2image import convert_from_path

    pdf_path_str, image_dir_str, prefix, dpi, page_batch_size = args
    pdf_path = Path(pdf_path_str)
    image_dir = Path(image_dir_str)

    try:
        # Get page count first (memory efficient)
        page_count = _get_pdf_page_count(pdf_path)

        if page_count == -1:
            # Fallback: try to convert and see how many pages we get
            # This is less memory efficient but handles edge cases
            images = convert_from_path(pdf_path, dpi=dpi)
            for i, image in enumerate(images, start=1):
                output_path = image_dir / f"{prefix}_page_{i}.png"
                image.save(output_path, "PNG")
            return ConversionResult(
                pdf_path=pdf_path_str,
                success=True,
                pages_converted=len(images),
            )

        # Convert pages in batches to reduce initialization overhead
        # While still maintaining reasonable memory usage
        pages_converted = 0
        for batch_start in range(1, page_count + 1, page_batch_size):
            batch_end = min(batch_start + page_batch_size - 1, page_count)

            try:
                # Convert batch of pages in single call (reduces overhead)
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=batch_start,
                    last_page=batch_end,
                )

                # Save each page in the batch
                for i, image in enumerate(images):
                    page_num = batch_start + i
                    output_path = image_dir / f"{prefix}_page_{page_num}.png"
                    image.save(output_path, "PNG")
                    pages_converted += 1

                # Explicitly free memory after each batch
                del images

            except MemoryError:
                # If batch conversion fails due to memory, fall back to single-page mode
                # for remaining pages
                for page_num in range(batch_start, batch_end + 1):
                    images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        first_page=page_num,
                        last_page=page_num,
                    )
                    if images:
                        output_path = image_dir / f"{prefix}_page_{page_num}.png"
                        images[0].save(output_path, "PNG")
                        pages_converted += 1
                        del images

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
    pdf_filter: Optional[List[str]] = None,
    page_batch_size: int = DEFAULT_PAGE_BATCH_SIZE,
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
        max_workers: Number of parallel workers (default 2, good for M2 16GB)
        parallel: Whether to use parallel processing (default True)
        pdf_filter: Optional list of PDF file stems (without extension) to process.
                    If provided, only PDFs with matching stems will be converted.
        page_batch_size: Number of pages to convert per batch (default 5).
                        Batching reduces convert_from_path() initialization overhead.
                        Lower values use less memory, higher values are faster.

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
    pdf_files = list(pdf_dir.rglob("*.pdf"))

    # Filter by specific PDF names if provided
    if pdf_filter:
        pdf_filter_set = set(pdf_filter)
        original_count = len(pdf_files)
        pdf_files = [p for p in pdf_files if p.stem in pdf_filter_set]
        logger.info(
            f"Filtered to {len(pdf_files)} PDF files (from {original_count}) matching filter"
        )
    else:
        logger.info(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        return {"total": 0, "success": 0, "failed": 0, "pages": 0}

    # Prepare conversion tasks
    tasks = []
    for pdf_path in pdf_files:
        prefix = get_relative_path_prefix(pdf_path, pdf_dir)
        tasks.append((str(pdf_path), str(image_dir), prefix, dpi, page_batch_size))

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
    logger.info(
        f"""
    PDF to Image Conversion Complete:
    - Total PDFs: {stats['total']}
    - Successfully converted: {stats['success']}
    - Failed: {stats['failed']}
    - Total pages: {stats['pages']}
    - Output directory: {image_dir}
    """
    )

    return stats


def convert_pdf_to_image_sequential(pdf_dir, image_dir, dpi: int = DEFAULT_DPI) -> Dict[str, int]:
    """
    Sequential PDF to image conversion (legacy mode).

    Use convert_pdf_to_image() with parallel=True for faster processing.
    """
    return convert_pdf_to_image(pdf_dir, image_dir, dpi=dpi, parallel=False)
