from pathlib import Path
from pdf2image import convert_from_path

from .utils.logger import logger


# Path separator for encoding relative paths in filenames
PATH_SEPARATOR = "__"


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


def convert_pdf_to_image(pdf_dir, image_dir):
    """
    Converts each page of every PDF file in the specified directory (and subdirectories) into separate image files.

    PDFs in subdirectories have their paths encoded in the output filename.
    Example: data/pdfs/projet_A/doc.pdf -> projet_A__doc_page_1.png
    """
    # Ensure pdf_dir and image_dir are Path objects
    pdf_dir = Path(pdf_dir)
    image_dir = Path(image_dir)

    # Set parameter to control image resolution
    dpi = 600

    # Check if the pdf_dir exists
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        logger.error(f"The specified pdf_dir '{pdf_dir}' does not exist or is not a directory.")
        return

    # Create image_dir if it doesn't exist
    image_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all PDF files recursively
    logger.info(f"Processing documents in folder (recursively): {pdf_dir} ...")
    pdf_files = list(pdf_dir.rglob('*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        # Get the encoded prefix for this PDF
        prefix = get_relative_path_prefix(pdf_path, pdf_dir)
        logger.info(f"Converting document: {pdf_path} (prefix: {prefix})")

        # Convert the PDF to images
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            logger.error(f"Failed to convert '{pdf_path}': {e}")
            continue

        # Save each image
        for i, image in enumerate(images, start=1):
            # Construct the output image file path with encoded prefix
            output_path = image_dir / f"{prefix}_page_{i}.png"
            try:
                image.save(output_path, 'PNG')
                logger.info(f"Page {i} saved as: {output_path.name}")
            except Exception as e:
                logger.error(f"Failed to save page {i} of '{pdf_path}': {e}")

