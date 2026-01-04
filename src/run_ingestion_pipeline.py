"""
Main ingestion pipeline for the Advanced Hybrid RAG system.

This pipeline:
1. Converts PDFs to images (600 DPI)
2. Detects objects (text, tables, pictures) using YOLO
3. Extracts content using SmolDocling
4. Describes images using vision LLM
5. Chunks text semantically
6. Generates embeddings with caching
7. Builds vector indexes

Features:
- Multi-provider vision support (Gemini, Ollama, OpenAI, Anthropic)
- Structured logging with timing metrics
- Error handling with retries
- Embedding caching
"""

import os
import asyncio
import argparse
from pathlib import Path

from .constants import (
    PDF_DIR,
    IMAGE_DIR,
    DETECTION_DIR,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    DOCLING_MODEL,
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBED_MODEL,
    MAX_CHUNK_SIZE,
    BUFFER_SIZE,
    BREAKPOINT_THRESHOLD_TYPE,
    BREAKPOINT_THRESHOLD_AMOUNT,
    SENTENCE_SPLIT_REGEX,
    SUMMARIZE_TABLE_PROMPT,
    EMBEDDINGS_DIR,
    LLM,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    SYSTEM_PROMPT_IMAGE_DESC,
    USER_PROMPT_IMAGE_DESC,
    DEFAULT_VISION_PROVIDER,
)

from .utils.logger import logger, PipelineTimer
from .helpers import clear_pytorch_cache, load_prompt
from .convert_pdf_to_image import convert_pdf_to_image
from .extract_objects_from_image import extract_objects_from_image
from .parse_image_with_text import parse_image_with_text
from .parse_image_with_table import parse_image_with_table
from .create_image_description import create_image_descriptions_async
from .chunk_text_data import chunk_text_data
from .chunk_table_data import chunk_table_data
from .create_embeddings import create_embeddings

import ollama


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Hybrid RAG Ingestion Pipeline"
    )
    parser.add_argument(
        "--vision-provider",
        type=str,
        default=None,
        choices=["gemini", "ollama", "openai", "anthropic"],
        help="Vision provider for image description (default: from .env)"
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF to image conversion"
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO detection"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip text/table extraction"
    )
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Skip image description generation"
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Force re-embedding even for cached content"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache"
    )
    return parser.parse_args()


async def run_ingestion_pipeline_async(
    vision_provider: str = None,
    skip_pdf: bool = False,
    skip_yolo: bool = False,
    skip_extraction: bool = False,
    skip_descriptions: bool = False,
    skip_chunking: bool = False,
    skip_embeddings: bool = False,
    force_reembed: bool = False,
    use_cache: bool = True,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        vision_provider: Vision provider for image descriptions
        skip_pdf: Skip PDF conversion
        skip_yolo: Skip YOLO detection
        skip_extraction: Skip content extraction
        skip_descriptions: Skip image descriptions
        skip_chunking: Skip chunking
        skip_embeddings: Skip embeddings
        force_reembed: Force re-embedding
        use_cache: Use embedding cache

    Returns:
        Pipeline statistics
    """
    # Initialize pipeline timer
    pipeline_timer = PipelineTimer("ingestion_pipeline").start()

    stats = {
        "pdf_conversion": {},
        "yolo_detection": {},
        "text_extraction": {},
        "table_extraction": {},
        "image_description": {},
        "text_chunking": {},
        "table_chunking": {},
        "embeddings": {},
    }

    # 1. Clear GPU memory
    pipeline_timer.stage("clear_cache")
    clear_pytorch_cache()

    # 2. Convert PDFs to images
    if not skip_pdf:
        pipeline_timer.stage("pdf_conversion")
        try:
            logger.info("Converting PDFs to images...")
            convert_pdf_to_image(PDF_DIR, IMAGE_DIR)
            logger.info("PDF conversion completed")
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    else:
        logger.info("Skipping PDF conversion")

    # 3. YOLO detection
    if not skip_yolo:
        pipeline_timer.stage("yolo_detection")
        try:
            logger.info("Running YOLO detection...")
            yolo_stats = extract_objects_from_image(
                IMAGE_DIR,
                DETECTION_DIR,
                YOLO_MODEL_PATH,
                YOLO_CONFIDENCE_THRESHOLD,
                YOLO_IOU_THRESHOLD,
                high_quality=True,
                enable_fallback=True,
            )
            stats["yolo_detection"] = yolo_stats
            logger.info("YOLO detection completed")
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            raise
    else:
        logger.info("Skipping YOLO detection")

    # 4. Extract text and tables
    if not skip_extraction:
        pipeline_timer.stage("content_extraction")
        try:
            logger.info("Extracting text from images...")
            parse_image_with_text(
                image_dir=DETECTION_DIR,
                model=DOCLING_MODEL,
                prompt="Extract all text content from this document page.",
                output_dir=PROCESSED_DIR,
            )

            logger.info("Extracting tables from images...")
            parse_image_with_table(
                image_dir=DETECTION_DIR,
                model=DOCLING_MODEL,
                prompt="Extract the table from this document image.",
                output_dir=PROCESSED_DIR,
            )
            logger.info("Content extraction completed")
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            raise
    else:
        logger.info("Skipping content extraction")

    # 5. Generate image descriptions
    if not skip_descriptions:
        pipeline_timer.stage("image_description")
        try:
            logger.info("Generating image descriptions...")
            system_p = load_prompt(SYSTEM_PROMPT_IMAGE_DESC)
            user_p = load_prompt(USER_PROMPT_IMAGE_DESC)

            desc_stats = await create_image_descriptions_async(
                image_dir=DETECTION_DIR,
                output_dir=PROCESSED_DIR,
                system_prompt=system_p,
                description_prompt=user_p,
                provider=vision_provider,
                max_concurrency=5,
                max_retries=3,
            )
            stats["image_description"] = desc_stats
            logger.info("Image description completed")
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            # Continue with pipeline even if descriptions fail
            logger.warning("Continuing pipeline without image descriptions")
    else:
        logger.info("Skipping image descriptions")

    # 6. Chunk text data
    if not skip_chunking:
        pipeline_timer.stage("text_chunking")
        try:
            logger.info("Chunking text data...")
            chunk_text_data(
                PROCESSED_DIR,
                EMBED_MODEL,
                MAX_CHUNK_SIZE,
                None,
                CHUNKS_DIR,
                BUFFER_SIZE,
                BREAKPOINT_THRESHOLD_TYPE,
                BREAKPOINT_THRESHOLD_AMOUNT,
                SENTENCE_SPLIT_REGEX,
                verbose=True
            )
            logger.info("Text chunking completed")
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            raise

        # 7. Chunk tables
        pipeline_timer.stage("table_chunking")
        try:
            logger.info("Chunking table data...")
            client = ollama.Client()
            with open(SUMMARIZE_TABLE_PROMPT, encoding="utf-8") as f:
                table_prompt = f.read()

            chunk_table_data(
                table_prompt,
                PROCESSED_DIR,
                None,
                MAX_CHUNK_SIZE,
                int(0.25 * MAX_CHUNK_SIZE),
                client,
                LLM,
                {"temperature": TEMPERATURE_GENERATION, "top_p": TOP_P_GENERATION},
                CHUNKS_DIR
            )
            logger.info("Table chunking completed")
        except Exception as e:
            logger.error(f"Table chunking failed: {e}")
            raise
    else:
        logger.info("Skipping chunking")

    # 8. Generate embeddings
    if not skip_embeddings:
        pipeline_timer.stage("embedding_generation")
        try:
            logger.info("Creating embeddings...")
            embed_stats = create_embeddings(
                CHUNKS_DIR,
                EMBEDDINGS_DIR,
                EMBED_MODEL,
                use_cache=use_cache,
                force_reembed=force_reembed,
            )
            stats["embeddings"] = embed_stats
            logger.info("Embedding generation completed")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    else:
        logger.info("Skipping embeddings")

    # End pipeline
    pipeline_summary = pipeline_timer.end()

    logger.info("""
    ============================================
    INGESTION PIPELINE COMPLETED SUCCESSFULLY
    ============================================
    """)

    return stats


def main():
    """Main entry point."""
    args = parse_args()

    asyncio.run(run_ingestion_pipeline_async(
        vision_provider=args.vision_provider,
        skip_pdf=args.skip_pdf,
        skip_yolo=args.skip_yolo,
        skip_extraction=args.skip_extraction,
        skip_descriptions=args.skip_descriptions,
        skip_chunking=args.skip_chunking,
        skip_embeddings=args.skip_embeddings,
        force_reembed=args.force_reembed,
        use_cache=not args.no_cache,
    ))


if __name__ == "__main__":
    main()
