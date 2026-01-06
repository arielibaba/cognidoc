"""
Convert non-PDF documents to PDF format.

Supports multiple document formats:
- Office: ppt, pptx, doc, docx, xls, xlsx, odt, odp, ods, rtf
- Web: html, htm
- Text: txt, md (Markdown)
- Images: jpg, jpeg, png, tiff, bmp

Uses LibreOffice for Office documents (cross-platform) and Python libraries for others.
"""

import os
import shutil
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .utils.logger import logger

# Supported file extensions by conversion method
OFFICE_EXTENSIONS = {'.ppt', '.pptx', '.doc', '.docx', '.xls', '.xlsx', '.odt', '.odp', '.ods', '.rtf'}
HTML_EXTENSIONS = {'.html', '.htm'}
TEXT_EXTENSIONS = {'.txt'}
MARKDOWN_EXTENSIONS = {'.md', '.markdown'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

ALL_SUPPORTED_EXTENSIONS = (
    OFFICE_EXTENSIONS | HTML_EXTENSIONS | TEXT_EXTENSIONS |
    MARKDOWN_EXTENSIONS | IMAGE_EXTENSIONS
)


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
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    return None


def convert_with_libreoffice(
    input_path: Path,
    output_dir: Path,
    libreoffice_path: str
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
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(input_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
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
        with open(input_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite']
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
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            text_content = f.read()

        # Create PDF
        c = canvas.Canvas(str(output_path), pagesize=letter)
        width, height = letter

        # Set font
        c.setFont("Courier", 10)

        # Write text line by line
        y = height - inch
        lines = text_content.split('\n')

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

        # Open image
        img = Image.open(input_path)

        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as PDF
        img.save(str(output_path), 'PDF', resolution=100.0)

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
    input_path: Path,
    output_dir: Path,
    libreoffice_path: Optional[str] = None
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


def convert_non_pdfs_to_pdf(
    input_dir: str,
    pdf_output_dir: str,
    non_pdf_archive_dir: str,
) -> Dict[str, any]:
    """
    Convert all non-PDF documents in a directory (and subdirectories) to PDF format.

    This function:
    1. Scans input_dir recursively for non-PDF documents
    2. Converts them to PDF and saves in the same relative location under pdf_output_dir
    3. Moves original files to the same relative location under non_pdf_archive_dir
    4. Skips conversion if PDF with same name already exists

    Args:
        input_dir: Directory containing documents to convert
        pdf_output_dir: Directory to save converted PDFs
        non_pdf_archive_dir: Directory to move original non-PDF files

    Returns:
        Statistics dictionary with conversion results
    """
    input_path = Path(input_dir)
    pdf_output_path = Path(pdf_output_dir)
    archive_path = Path(non_pdf_archive_dir)

    # Create directories if they don't exist
    pdf_output_path.mkdir(parents=True, exist_ok=True)
    archive_path.mkdir(parents=True, exist_ok=True)

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
        "converted": 0,
        "skipped_existing": 0,
        "skipped_pdf": 0,
        "failed": 0,
        "moved_to_archive": 0,
        "by_format": {},
        "errors": [],
    }

    # Scan for files recursively
    all_files = list(input_path.rglob("*"))
    logger.info(f"Scanning {input_path} recursively, found {len(all_files)} items")

    for file_path in all_files:
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        stats["total_files"] += 1

        # Skip PDFs - they're already in the right format
        if ext == '.pdf':
            stats["skipped_pdf"] += 1
            continue

        # Check if format is supported
        if ext not in ALL_SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping unsupported format: {file_path.name}")
            continue

        # Track format statistics
        if ext not in stats["by_format"]:
            stats["by_format"][ext] = {"converted": 0, "failed": 0, "skipped": 0}

        # Calculate relative path to preserve directory structure
        try:
            relative_path = file_path.relative_to(input_path)
            relative_dir = relative_path.parent
        except ValueError:
            relative_dir = Path("")

        # Create output directory preserving structure
        target_output_dir = pdf_output_path / relative_dir
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if PDF already exists
        target_pdf = target_output_dir / f"{file_path.stem}.pdf"
        if target_pdf.exists():
            logger.info(f"PDF already exists, skipping conversion: {file_path.name}")
            stats["skipped_existing"] += 1
            stats["by_format"][ext]["skipped"] += 1

            # Still move the original to archive (preserving structure)
            archive_target_dir = archive_path / relative_dir
            archive_target_dir.mkdir(parents=True, exist_ok=True)
            archive_dest = archive_target_dir / file_path.name
            if not archive_dest.exists():
                shutil.move(str(file_path), str(archive_dest))
                stats["moved_to_archive"] += 1
                logger.info(f"Moved to archive: {file_path.name}")
            continue

        # Convert the document
        logger.info(f"Converting: {file_path}")
        result = convert_document_to_pdf(file_path, target_output_dir, libreoffice_path)

        if result:
            stats["converted"] += 1
            stats["by_format"][ext]["converted"] += 1
            logger.info(f"Successfully converted: {file_path.name} -> {result.name}")

            # Move original to archive (preserving structure)
            archive_target_dir = archive_path / relative_dir
            archive_target_dir.mkdir(parents=True, exist_ok=True)
            archive_dest = archive_target_dir / file_path.name
            if archive_dest.exists():
                # Add suffix if file exists in archive
                counter = 1
                while archive_dest.exists():
                    archive_dest = archive_target_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1

            shutil.move(str(file_path), str(archive_dest))
            stats["moved_to_archive"] += 1
            logger.info(f"Moved to archive: {file_path.name}")
        else:
            stats["failed"] += 1
            stats["by_format"][ext]["failed"] += 1
            stats["errors"].append(f"Failed to convert: {file_path.name}")
            logger.error(f"Failed to convert: {file_path.name}")

    # Log summary
    logger.info(f"""
    ============================================
    Document Conversion Summary
    ============================================
    Total files scanned:    {stats['total_files']}
    PDFs (skipped):         {stats['skipped_pdf']}
    Already converted:      {stats['skipped_existing']}
    Newly converted:        {stats['converted']}
    Failed conversions:     {stats['failed']}
    Moved to archive:       {stats['moved_to_archive']}

    By format: {stats['by_format']}
    ============================================
    """)

    return stats
