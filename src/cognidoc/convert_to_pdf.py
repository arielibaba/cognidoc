"""
Process source documents for the ingestion pipeline.

Supports multiple document formats:
- Office: ppt, pptx, doc, docx, xls, xlsx, odt, odp, ods, rtf → converted to PDF
- Web: html, htm → converted to PDF
- Text: txt, md (Markdown) → converted to PDF
- Images: jpg, jpeg, png, tiff, bmp → copied directly to images folder (no PDF conversion)
- PDF: copied directly to pdfs folder

Uses LibreOffice for Office documents (cross-platform) and Python libraries for others.
"""

import os
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

from .utils.logger import logger

# Supported file extensions by conversion method
OFFICE_EXTENSIONS = {
    ".ppt",
    ".pptx",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".odt",
    ".odp",
    ".ods",
    ".rtf",
}
HTML_EXTENSIONS = {".html", ".htm"}
TEXT_EXTENSIONS = {".txt"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

# Extensions that need PDF conversion (documents)
PDF_CONVERTIBLE_EXTENSIONS = (
    OFFICE_EXTENSIONS | HTML_EXTENSIONS | TEXT_EXTENSIONS | MARKDOWN_EXTENSIONS
)

# All supported extensions
ALL_SUPPORTED_EXTENSIONS = PDF_CONVERTIBLE_EXTENSIONS | IMAGE_EXTENSIONS


def find_libreoffice() -> Optional[str]:
    """
    Find LibreOffice executable path based on the operating system.

    Returns:
        Path to LibreOffice executable or None if not found
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "/opt/homebrew/bin/soffice",
            "/usr/local/bin/soffice",
        ]
    elif system == "Linux":
        paths = [
            "/usr/bin/soffice",
            "/usr/bin/libreoffice",
            "/usr/local/bin/soffice",
            "/snap/bin/libreoffice",
        ]
    elif system == "Windows":
        paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    else:
        paths = []

    # Check each path
    for path in paths:
        if os.path.isfile(path):
            return path

    # Try to find via 'which' or 'where' command
    try:
        cmd = "where" if system == "Windows" else "which"
        result = subprocess.run([cmd, "soffice"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception as e:
        logger.debug(f"Could not find soffice via system command: {e}")

    return None


def convert_with_libreoffice(
    input_path: Path, output_dir: Path, libreoffice_path: str
) -> Optional[Path]:
    """
    Convert a document to PDF using LibreOffice.

    Args:
        input_path: Path to the input document
        output_dir: Directory to save the PDF
        libreoffice_path: Path to LibreOffice executable

    Returns:
        Path to the converted PDF or None if conversion failed
    """
    try:
        # Run LibreOffice in headless mode
        cmd = [
            libreoffice_path,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(input_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120  # 2 minute timeout
        )

        if result.returncode != 0:
            logger.warning(f"LibreOffice conversion failed for {input_path.name}: {result.stderr}")
            return None

        # Expected output filename
        output_pdf = output_dir / f"{input_path.stem}.pdf"

        if output_pdf.exists():
            return output_pdf
        else:
            logger.warning(f"PDF not found after conversion: {output_pdf}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"LibreOffice conversion timed out for {input_path.name}")
        return None
    except Exception as e:
        logger.error(f"LibreOffice conversion error for {input_path.name}: {e}")
        return None


def convert_html_to_pdf(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Convert HTML to PDF using weasyprint.

    Args:
        input_path: Path to the HTML file
        output_path: Path for the output PDF

    Returns:
        Path to the converted PDF or None if conversion failed
    """
    try:
        from weasyprint import HTML

        HTML(filename=str(input_path)).write_pdf(str(output_path))

        if output_path.exists():
            return output_path
        return None

    except ImportError:
        logger.warning("weasyprint not installed. Trying alternative method...")
        # Fallback: try LibreOffice for HTML
        return None
    except Exception as e:
        logger.error(f"HTML to PDF conversion failed for {input_path.name}: {e}")
        return None


def convert_markdown_to_pdf(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Convert Markdown to PDF.

    Args:
        input_path: Path to the Markdown file
        output_path: Path for the output PDF

    Returns:
        Path to the converted PDF or None if conversion failed
    """
    try:
        import markdown
        from weasyprint import HTML

        # Read markdown content
        with open(input_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to HTML
        html_content = markdown.markdown(
            md_content, extensions=["tables", "fenced_code", "codehilite"]
        )

        # Wrap in basic HTML structure with styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                       line-height: 1.6; padding: 40px; max-width: 800px; margin: 0 auto; }}
                code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 16px; border-radius: 6px; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f4f4f4; }}
            </style>
        </head>
        <body>{html_content}</body>
        </html>
        """

        # Convert to PDF
        HTML(string=full_html).write_pdf(str(output_path))

        if output_path.exists():
            return output_path
        return None

    except ImportError as e:
        logger.warning(f"Required library not installed for Markdown conversion: {e}")
        return None
    except Exception as e:
        logger.error(f"Markdown to PDF conversion failed for {input_path.name}: {e}")
        return None


def convert_text_to_pdf(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Convert plain text to PDF.

    Args:
        input_path: Path to the text file
        output_path: Path for the output PDF

    Returns:
        Path to the converted PDF or None if conversion failed
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        # Read text content
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            text_content = f.read()

        # Create PDF
        c = canvas.Canvas(str(output_path), pagesize=letter)
        width, height = letter

        # Set font
        c.setFont("Courier", 10)

        # Write text line by line
        y = height - inch
        lines = text_content.split("\n")

        for line in lines:
            if y < inch:
                c.showPage()
                c.setFont("Courier", 10)
                y = height - inch

            # Truncate long lines
            if len(line) > 100:
                line = line[:100] + "..."

            c.drawString(inch, y, line)
            y -= 12

        c.save()

        if output_path.exists():
            return output_path
        return None

    except ImportError:
        logger.warning("reportlab not installed. Cannot convert text to PDF.")
        return None
    except Exception as e:
        logger.error(f"Text to PDF conversion failed for {input_path.name}: {e}")
        return None


def convert_image_to_pdf(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Convert an image to PDF.

    Args:
        input_path: Path to the image file
        output_path: Path for the output PDF

    Returns:
        Path to the converted PDF or None if conversion failed
    """
    try:
        from PIL import Image

        # Open image with context manager to avoid file descriptor leaks
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if img.mode in ("RGBA", "LA", "P"):
                # Create white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Save as PDF
            img.save(str(output_path), "PDF", resolution=100.0)

        if output_path.exists():
            return output_path
        return None

    except ImportError:
        logger.warning("Pillow not installed. Cannot convert image to PDF.")
        return None
    except Exception as e:
        logger.error(f"Image to PDF conversion failed for {input_path.name}: {e}")
        return None


def convert_document_to_pdf(
    input_path: Path, output_dir: Path, libreoffice_path: Optional[str] = None
) -> Optional[Path]:
    """
    Convert a document to PDF based on its file type.

    Args:
        input_path: Path to the input document
        output_dir: Directory to save the PDF
        libreoffice_path: Path to LibreOffice executable (optional)

    Returns:
        Path to the converted PDF or None if conversion failed
    """
    ext = input_path.suffix.lower()
    output_path = output_dir / f"{input_path.stem}.pdf"

    # Office documents - use LibreOffice
    if ext in OFFICE_EXTENSIONS:
        if not libreoffice_path:
            logger.error(f"LibreOffice not found. Cannot convert {input_path.name}")
            return None
        return convert_with_libreoffice(input_path, output_dir, libreoffice_path)

    # HTML files
    elif ext in HTML_EXTENSIONS:
        result = convert_html_to_pdf(input_path, output_path)
        if result is None and libreoffice_path:
            # Fallback to LibreOffice
            return convert_with_libreoffice(input_path, output_dir, libreoffice_path)
        return result

    # Markdown files
    elif ext in MARKDOWN_EXTENSIONS:
        return convert_markdown_to_pdf(input_path, output_path)

    # Text files
    elif ext in TEXT_EXTENSIONS:
        return convert_text_to_pdf(input_path, output_path)

    # Image files
    elif ext in IMAGE_EXTENSIONS:
        return convert_image_to_pdf(input_path, output_path)

    else:
        logger.warning(f"Unsupported file format: {ext}")
        return None


def process_source_documents(
    sources_dir: str,
    pdf_output_dir: str,
    image_output_dir: str = None,
    source_files: List[str] = None,
) -> Dict[str, Any]:
    """
    Process documents from sources directory.

    This function:
    1. If source_files is provided, processes only those specific files
    2. Otherwise, scans sources_dir recursively for all documents
    3. Copies PDFs directly to pdf_output_dir
    4. Copies images directly to image_output_dir (skips PDF conversion for efficiency)
    5. Converts documents (Office, HTML, text) to PDF and saves in pdf_output_dir
    6. Keeps original files in sources_dir (archive)
    7. Skips processing if output file already exists

    Args:
        sources_dir: Directory containing source documents (input)
        pdf_output_dir: Directory to save PDFs (output)
        image_output_dir: Directory to save images directly (optional, uses pdf_output_dir if None)
        source_files: Optional list of specific file paths to process (limits processing to these files only)

    Returns:
        Statistics dictionary with processing results
    """
    sources_path = Path(sources_dir)
    pdf_output_path = Path(pdf_output_dir)
    image_output_path = Path(image_output_dir) if image_output_dir else None

    # Create directories if they don't exist
    sources_path.mkdir(parents=True, exist_ok=True)
    pdf_output_path.mkdir(parents=True, exist_ok=True)
    if image_output_path:
        image_output_path.mkdir(parents=True, exist_ok=True)

    # Find LibreOffice
    libreoffice_path = find_libreoffice()
    if libreoffice_path:
        logger.info(f"Found LibreOffice at: {libreoffice_path}")
    else:
        logger.warning(
            "LibreOffice not found. Office document conversion will not be available. "
            "Install LibreOffice for full format support."
        )

    # Statistics
    stats = {
        "total_files": 0,
        "pdfs_copied": 0,
        "images_copied": 0,
        "converted": 0,
        "skipped_existing": 0,
        "unsupported": 0,
        "failed": 0,
        "by_format": {},
        "errors": [],
        "processed_pdf_stems": [],  # Track PDF file stems for pipeline filtering
    }

    # Get files to process: either specific files or scan recursively
    if source_files:
        all_files = [Path(f) for f in source_files if Path(f).is_file()]
        logger.info(f"Processing {len(all_files)} specific file(s)")
    else:
        all_files = list(sources_path.rglob("*"))
        logger.info(f"Scanning {sources_path} recursively, found {len(all_files)} items")

    for file_path in all_files:
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        stats["total_files"] += 1

        # Track format statistics
        if ext not in stats["by_format"]:
            stats["by_format"][ext] = {"processed": 0, "failed": 0, "skipped": 0}

        # Calculate relative path to preserve directory structure
        try:
            relative_path = file_path.relative_to(sources_path)
            relative_dir = relative_path.parent
        except ValueError:
            relative_dir = Path("")

        # Handle PDFs - copy directly to pdf_output_dir
        if ext == ".pdf":
            target_output_dir = pdf_output_path / relative_dir
            target_output_dir.mkdir(parents=True, exist_ok=True)
            target_pdf = target_output_dir / file_path.name
            # Track this PDF for later pipeline stages
            stats["processed_pdf_stems"].append(file_path.stem)
            if target_pdf.exists():
                logger.debug(f"PDF already exists in output, skipping: {file_path.name}")
                stats["skipped_existing"] += 1
                stats["by_format"][ext]["skipped"] += 1
            else:
                shutil.copy2(str(file_path), str(target_pdf))
                stats["pdfs_copied"] += 1
                stats["by_format"][ext]["processed"] += 1
                logger.info(f"Copied PDF: {file_path.name}")
            continue

        # Handle images - copy directly to image_output_dir (skip PDF conversion)
        if ext in IMAGE_EXTENSIONS:
            if image_output_path:
                # Copy image directly to images folder (optimization: skip PDF roundtrip)
                # Use a naming convention compatible with the pipeline: filename_page_1.ext
                target_image = image_output_path / f"{file_path.stem}_page_1{ext}"
                if target_image.exists():
                    logger.debug(f"Image already exists in output, skipping: {file_path.name}")
                    stats["skipped_existing"] += 1
                    stats["by_format"][ext]["skipped"] += 1
                else:
                    shutil.copy2(str(file_path), str(target_image))
                    stats["images_copied"] += 1
                    stats["by_format"][ext]["processed"] += 1
                    logger.info(f"Copied image directly: {file_path.name} -> {target_image.name}")
            else:
                # Fallback: convert to PDF if no image_output_dir specified
                target_output_dir = pdf_output_path / relative_dir
                target_output_dir.mkdir(parents=True, exist_ok=True)
                target_pdf = target_output_dir / f"{file_path.stem}.pdf"
                if target_pdf.exists():
                    logger.debug(f"Converted PDF already exists, skipping: {file_path.name}")
                    stats["skipped_existing"] += 1
                    stats["by_format"][ext]["skipped"] += 1
                else:
                    result = convert_document_to_pdf(file_path, target_output_dir, libreoffice_path)
                    if result:
                        stats["converted"] += 1
                        stats["by_format"][ext]["processed"] += 1
                        logger.info(f"Converted image to PDF: {file_path.name}")
                    else:
                        stats["failed"] += 1
                        stats["by_format"][ext]["failed"] += 1
                        stats["errors"].append(f"Failed to convert: {file_path.name}")
            continue

        # Check if format is supported for PDF conversion
        if ext not in PDF_CONVERTIBLE_EXTENSIONS:
            logger.debug(f"Skipping unsupported format: {file_path.name}")
            stats["unsupported"] += 1
            continue

        # Create output directory for PDF conversion
        target_output_dir = pdf_output_path / relative_dir
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if converted PDF already exists
        target_pdf = target_output_dir / f"{file_path.stem}.pdf"
        if target_pdf.exists():
            logger.debug(f"Converted PDF already exists, skipping: {file_path.name}")
            stats["skipped_existing"] += 1
            stats["by_format"][ext]["skipped"] += 1
            continue

        # Convert the document to PDF
        logger.info(f"Converting: {file_path}")
        result = convert_document_to_pdf(file_path, target_output_dir, libreoffice_path)

        if result:
            stats["converted"] += 1
            stats["by_format"][ext]["processed"] += 1
            # Track the converted PDF stem for later pipeline stages
            stats["processed_pdf_stems"].append(file_path.stem)
            logger.info(f"Successfully converted: {file_path.name} -> {result.name}")
        else:
            stats["failed"] += 1
            stats["by_format"][ext]["failed"] += 1
            stats["errors"].append(f"Failed to convert: {file_path.name}")
            logger.error(f"Failed to convert: {file_path.name}")

    # Log summary
    logger.info(
        f"""
    ============================================
    Source Documents Processing Summary
    ============================================
    Total files scanned:    {stats['total_files']}
    PDFs copied:            {stats['pdfs_copied']}
    Images copied:          {stats['images_copied']}
    Documents converted:    {stats['converted']}
    Already processed:      {stats['skipped_existing']}
    Unsupported formats:    {stats['unsupported']}
    Failed conversions:     {stats['failed']}

    By format: {stats['by_format']}
    ============================================
    """
    )

    return stats
