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
8. Extracts entities and relationships (GraphRAG)
9. Builds knowledge graph with community detection

Features:
- Multi-provider vision support (Gemini, Ollama, OpenAI, Anthropic)
- Structured logging with timing metrics
- Error handling with retries
- Embedding caching
- Hybrid RAG (Vector + Graph)
"""

import os
import asyncio
import argparse
from pathlib import Path

from .constants import (
    SOURCES_DIR,
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
    SEMANTIC_CHUNK_BUFFER,
    SEMANTIC_BREAKPOINT_METHOD,
    SEMANTIC_BREAKPOINT_VALUE,
    SENTENCE_SPLIT_REGEX,
    TABLE_SUMMARY_PROMPT_PATH,
    EMBEDDINGS_DIR,
    OLLAMA_LLM_MODEL,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    SYSTEM_PROMPT_IMAGE_DESC,
    USER_PROMPT_IMAGE_DESC,
    DEFAULT_VISION_PROVIDER,
)

from .utils.logger import logger, PipelineTimer
from .helpers import clear_pytorch_cache, load_prompt
from .convert_to_pdf import process_source_documents
from .convert_pdf_to_image import convert_pdf_to_image
from .extract_objects_from_image import extract_objects_from_image
from .parse_image_with_text import parse_image_with_text
from .parse_image_with_table import parse_image_with_table
from .create_image_description import create_image_descriptions_async
from .chunk_text_data import chunk_text_data
from .chunk_table_data import chunk_table_data
from .create_embeddings import create_embeddings
from .build_indexes import build_indexes
from .extract_entities import extract_from_chunks_dir, run_extraction_async, save_extraction_results
from .knowledge_graph import build_knowledge_graph
from .graph_config import get_graph_config

import ollama


def format_ingestion_report(stats: dict, timing: dict) -> str:
    """
    Format a comprehensive ingestion report as a table.

    Args:
        stats: Pipeline statistics dictionary
        timing: Timing information from PipelineTimer

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("                    INGESTION REPORT")
    lines.append("=" * 70)

    # ========== DOCUMENTS SECTION ==========
    doc_stats = stats.get("document_conversion", {})
    pdf_stats = stats.get("pdf_conversion", {})

    lines.append("")
    lines.append("┌" + "─" * 68 + "┐")
    lines.append("│" + " DOCUMENTS ".center(68) + "│")
    lines.append("├" + "─" * 34 + "┬" + "─" * 33 + "┤")

    total_files = doc_stats.get("total_files", 0)
    pdfs_copied = doc_stats.get("pdfs_copied", 0)
    converted = doc_stats.get("converted", 0)
    images_copied = doc_stats.get("images_copied", 0)
    skipped = doc_stats.get("skipped_existing", 0)
    failed = doc_stats.get("failed", 0)

    lines.append(f"│ {'Files processed':<32} │ {total_files:>31} │")
    lines.append(f"│ {'  PDFs copied':<32} │ {pdfs_copied:>31} │")
    lines.append(f"│ {'  Documents converted':<32} │ {converted:>31} │")
    lines.append(f"│ {'  Images copied':<32} │ {images_copied:>31} │")
    lines.append(f"│ {'  Already processed (skipped)':<32} │ {skipped:>31} │")
    lines.append(f"│ {'  Failed':<32} │ {failed:>31} │")

    # PDF to image conversion
    total_pdfs = pdf_stats.get("total", 0)
    total_pages = pdf_stats.get("pages", 0)
    if total_pdfs > 0 or total_pages > 0:
        lines.append("├" + "─" * 34 + "┼" + "─" * 33 + "┤")
        lines.append(f"│ {'PDFs converted to images':<32} │ {total_pdfs:>31} │")
        lines.append(f"│ {'Total pages generated':<32} │ {total_pages:>31} │")

    lines.append("└" + "─" * 34 + "┴" + "─" * 33 + "┘")

    # ========== YOLO DETECTION SECTION ==========
    yolo_stats = stats.get("yolo_detection", {})
    if yolo_stats:
        lines.append("")
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│" + " YOLO DETECTION ".center(68) + "│")
        lines.append("├" + "─" * 34 + "┬" + "─" * 33 + "┤")

        images_processed = yolo_stats.get("images", 0)
        text_regions = yolo_stats.get("text", 0)
        table_regions = yolo_stats.get("tables", 0)
        picture_regions = yolo_stats.get("pictures", 0)
        fallbacks = yolo_stats.get("fallbacks", 0)
        errors = yolo_stats.get("errors", 0)

        lines.append(f"│ {'Images processed':<32} │ {images_processed:>31} │")
        lines.append(f"│ {'Text regions detected':<32} │ {text_regions:>31} │")
        lines.append(f"│ {'Table regions detected':<32} │ {table_regions:>31} │")
        lines.append(f"│ {'Picture regions detected':<32} │ {picture_regions:>31} │")
        if fallbacks > 0:
            lines.append(f"│ {'Fallbacks (full page)':<32} │ {fallbacks:>31} │")
        if errors > 0:
            lines.append(f"│ {'Errors':<32} │ {errors:>31} │")

        lines.append("└" + "─" * 34 + "┴" + "─" * 33 + "┘")

    # ========== CONTENT EXTRACTION SECTION ==========
    text_stats = stats.get("text_extraction", {})
    table_stats = stats.get("table_extraction", {})
    desc_stats = stats.get("image_description", {})

    has_extraction = any([text_stats, table_stats, desc_stats])
    if has_extraction:
        lines.append("")
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│" + " CONTENT EXTRACTION ".center(68) + "│")
        lines.append("├" + "─" * 34 + "┬" + "─" * 33 + "┤")

        text_processed = text_stats.get("processed", 0)
        text_skipped = text_stats.get("skipped", 0)
        text_errors = text_stats.get("errors", 0)

        table_processed = table_stats.get("processed", 0)
        table_skipped = table_stats.get("skipped", 0)
        table_errors = table_stats.get("errors", 0)

        desc_total = desc_stats.get("total_images", 0)
        desc_described = desc_stats.get("described", 0)
        desc_skipped = desc_stats.get("skipped_irrelevant", 0)

        lines.append(f"│ {'Text regions extracted':<32} │ {text_processed:>31} │")
        if text_skipped > 0:
            lines.append(f"│ {'  (skipped existing)':<32} │ {text_skipped:>31} │")
        lines.append(f"│ {'Tables extracted':<32} │ {table_processed:>31} │")
        if table_skipped > 0:
            lines.append(f"│ {'  (skipped existing)':<32} │ {table_skipped:>31} │")
        if desc_total > 0:
            lines.append(f"│ {'Images described':<32} │ {desc_described:>31} │")
            if desc_skipped > 0:
                lines.append(f"│ {'  (skipped irrelevant)':<32} │ {desc_skipped:>31} │")

        total_errors = text_errors + table_errors + desc_stats.get("errors", 0)
        if total_errors > 0:
            lines.append(f"│ {'Errors':<32} │ {total_errors:>31} │")

        lines.append("└" + "─" * 34 + "┴" + "─" * 33 + "┘")

    # ========== CHUNKING & EMBEDDINGS SECTION ==========
    embed_stats = stats.get("embeddings", {})
    if embed_stats:
        lines.append("")
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│" + " CHUNKING & EMBEDDINGS ".center(68) + "│")
        lines.append("├" + "─" * 34 + "┬" + "─" * 33 + "┤")

        # Keys from create_embeddings: total_files, cached, to_embed, embedded, skipped_parent, skipped_short, errors
        total_files = embed_stats.get("total_files", 0)
        from_cache = embed_stats.get("cached", 0)
        to_embed = embed_stats.get("to_embed", 0)
        newly_embedded = embed_stats.get("embedded", 0)
        skipped_parent = embed_stats.get("skipped_parent", 0)
        total_chunks = from_cache + to_embed  # Total child chunks processed

        lines.append(f"│ {'Child chunks processed':<32} │ {total_chunks:>31} │")
        lines.append(f"│ {'  From cache':<32} │ {from_cache:>31} │")
        lines.append(f"│ {'  Newly embedded':<32} │ {newly_embedded:>31} │")
        if skipped_parent > 0:
            lines.append(f"│ {'Parent chunks (not embedded)':<32} │ {skipped_parent:>31} │")

        lines.append("└" + "─" * 34 + "┴" + "─" * 33 + "┘")

    # ========== GRAPHRAG SECTION ==========
    graph_extract = stats.get("graph_extraction", {})
    graph_build = stats.get("graph_building", {})

    has_graph = graph_extract or (graph_build and graph_build.get("status") != "failed")
    if has_graph:
        lines.append("")
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│" + " KNOWLEDGE GRAPH (GraphRAG) ".center(68) + "│")
        lines.append("├" + "─" * 34 + "┬" + "─" * 33 + "┤")

        chunks_processed = graph_extract.get("chunks_processed", 0)
        entities_extracted = graph_extract.get("entities_extracted", 0)
        relationships_extracted = graph_extract.get("relationships_extracted", 0)

        total_nodes = graph_build.get("total_nodes", 0)
        total_edges = graph_build.get("total_edges", 0)
        total_communities = graph_build.get("total_communities", 0)

        lines.append(f"│ {'Chunks processed':<32} │ {chunks_processed:>31} │")
        lines.append(f"│ {'Entities extracted':<32} │ {entities_extracted:>31} │")
        lines.append(f"│ {'Relationships extracted':<32} │ {relationships_extracted:>31} │")
        lines.append("├" + "─" * 34 + "┼" + "─" * 33 + "┤")
        lines.append(f"│ {'Graph nodes (after merge)':<32} │ {total_nodes:>31} │")
        lines.append(f"│ {'Graph edges':<32} │ {total_edges:>31} │")
        lines.append(f"│ {'Communities detected':<32} │ {total_communities:>31} │")

        # Node types breakdown
        node_types = graph_build.get("node_types", {})
        if node_types:
            lines.append("├" + "─" * 34 + "┼" + "─" * 33 + "┤")
            lines.append(f"│ {'Entity types:':<32} │ {'':<31} │")
            for node_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
                lines.append(f"│ {'  ' + node_type:<32} │ {count:>31} │")

        lines.append("└" + "─" * 34 + "┴" + "─" * 33 + "┘")

    # ========== TIMING SECTION ==========
    stages = timing.get("stages", {})
    total_time = timing.get("total_seconds", 0)

    if stages:
        lines.append("")
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│" + " TIMING ".center(68) + "│")
        lines.append("├" + "─" * 34 + "┬" + "─" * 33 + "┤")

        # Map stage names to readable labels
        stage_labels = {
            "clear_cache": "Clear cache",
            "document_conversion": "Document conversion",
            "pdf_conversion": "PDF to images",
            "yolo_detection": "YOLO detection",
            "content_extraction": "Content extraction",
            "image_description": "Image descriptions",
            "text_chunking": "Text chunking",
            "table_chunking": "Table chunking",
            "embedding_generation": "Embedding generation",
            "index_building": "Index building",
            "graph_extraction": "Entity extraction",
            "graph_building": "Graph building",
        }

        for stage, duration in stages.items():
            if duration > 0.01:  # Only show stages that took more than 0.01s
                label = stage_labels.get(stage, stage)
                duration_str = f"{duration:.2f}s"
                lines.append(f"│ {label:<32} │ {duration_str:>31} │")

        lines.append("├" + "─" * 34 + "┼" + "─" * 33 + "┤")

        # Format total time nicely
        if total_time >= 60:
            minutes = int(total_time // 60)
            seconds = total_time % 60
            total_str = f"{minutes}m {seconds:.1f}s ({total_time:.2f}s)"
        else:
            total_str = f"{total_time:.2f}s"

        lines.append(f"│ {'TOTAL':<32} │ {total_str:>31} │")
        lines.append("└" + "─" * 34 + "┴" + "─" * 33 + "┘")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


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
        "--extraction-provider",
        type=str,
        default="gemini",
        choices=["gemini", "ollama", "openai", "anthropic"],
        help="Provider for text/table extraction from images (default: gemini)"
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip non-PDF to PDF conversion"
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
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip vector index building"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip knowledge graph building (GraphRAG)"
    )
    parser.add_argument(
        "--graph-config",
        type=str,
        default=None,
        help="Path to custom graph schema configuration"
    )
    # Performance optimization arguments
    parser.add_argument(
        "--yolo-batch-size",
        type=int,
        default=2,
        help="YOLO batch size for detection (default: 2, recommended 2-3 for M2/M3 16GB)"
    )
    parser.add_argument(
        "--no-yolo-batching",
        action="store_true",
        help="Disable YOLO batch processing (use sequential mode)"
    )
    parser.add_argument(
        "--entity-max-concurrent",
        type=int,
        default=4,
        help="Max concurrent entity extractions (default: 4, recommended 4-6 for Gemini API)"
    )
    parser.add_argument(
        "--no-async-extraction",
        action="store_true",
        help="Disable async entity extraction (use sequential mode)"
    )
    return parser.parse_args()


async def run_ingestion_pipeline_async(
    vision_provider: str = None,
    extraction_provider: str = "gemini",
    skip_conversion: bool = False,
    skip_pdf: bool = False,
    skip_yolo: bool = False,
    skip_extraction: bool = False,
    skip_descriptions: bool = False,
    skip_chunking: bool = False,
    skip_embeddings: bool = False,
    force_reembed: bool = False,
    use_cache: bool = True,
    skip_indexing: bool = False,
    skip_graph: bool = False,
    graph_config_path: str = None,
    yolo_batch_size: int = 2,
    use_yolo_batching: bool = True,
    entity_max_concurrent: int = 4,
    use_async_extraction: bool = True,
    source_files: list = None,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        vision_provider: Vision provider for image descriptions
        extraction_provider: Provider for text/table extraction (default: gemini)
        skip_conversion: Skip non-PDF to PDF conversion
        skip_pdf: Skip PDF conversion
        skip_yolo: Skip YOLO detection
        skip_extraction: Skip content extraction
        skip_descriptions: Skip image descriptions
        skip_chunking: Skip chunking
        skip_embeddings: Skip embeddings
        force_reembed: Force re-embedding
        use_cache: Use embedding cache
        skip_indexing: Skip vector index building
        skip_graph: Skip knowledge graph building
        yolo_batch_size: YOLO batch size (default: 2)
        use_yolo_batching: Enable YOLO batch processing (default: True)
        entity_max_concurrent: Max concurrent entity extractions (default: 4)
        use_async_extraction: Enable async entity extraction (default: True)
        graph_config_path: Path to custom graph configuration
        source_files: Optional list of specific file paths to process (limits processing to these files only)

    Returns:
        Pipeline statistics
    """
    # Initialize pipeline timer
    pipeline_timer = PipelineTimer("ingestion_pipeline").start()

    stats = {
        "document_conversion": {},
        "pdf_conversion": {},
        "yolo_detection": {},
        "text_extraction": {},
        "table_extraction": {},
        "image_description": {},
        "text_chunking": {},
        "table_chunking": {},
        "embeddings": {},
        "indexing": {},
        "graph_extraction": {},
        "graph_building": {},
    }

    # 1. Clear GPU memory
    pipeline_timer.stage("clear_cache")
    clear_pytorch_cache()

    # Track PDF stems for filtering in subsequent stages
    # Compute pdf_filter directly from source_files to ensure filtering works
    # even when conversion is skipped or files are already processed
    pdf_filter = None
    if source_files:
        # Extract stems from source files - for PDFs use stem directly,
        # for other formats the converted PDF will have the same stem
        pdf_filter = [Path(f).stem for f in source_files if Path(f).is_file()]
        if pdf_filter:
            logger.info(f"Will filter subsequent stages to {len(pdf_filter)} file(s): {pdf_filter}")

    # 2. Process source documents (copy PDFs, copy images, convert documents)
    if not skip_conversion:
        pipeline_timer.stage("document_conversion")
        try:
            logger.info("Processing source documents...")
            conversion_stats = process_source_documents(
                sources_dir=SOURCES_DIR,
                pdf_output_dir=PDF_DIR,
                image_output_dir=IMAGE_DIR,  # Images go directly to images folder
                source_files=source_files,  # Limit to specific files if provided
            )
            stats["document_conversion"] = conversion_stats
            logger.info(
                f"Document processing completed: {conversion_stats['pdfs_copied']} PDFs copied, "
                f"{conversion_stats['images_copied']} images copied, "
                f"{conversion_stats['converted']} converted, "
                f"{conversion_stats['skipped_existing']} skipped, "
                f"{conversion_stats['failed']} failed"
            )
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            # Continue with pipeline - conversion failure shouldn't stop PDF processing
            logger.warning("Continuing pipeline without source document processing")
    else:
        logger.info("Skipping source document processing")

    # 3. Convert PDFs to images
    if not skip_pdf:
        pipeline_timer.stage("pdf_conversion")
        try:
            logger.info("Converting PDFs to images...")
            pdf_stats = convert_pdf_to_image(PDF_DIR, IMAGE_DIR, pdf_filter=pdf_filter)
            stats["pdf_conversion"] = pdf_stats
            logger.info(f"PDF conversion completed: {pdf_stats.get('success', 0)} PDFs, {pdf_stats.get('pages', 0)} pages")
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    else:
        logger.info("Skipping PDF conversion")

    # 4. YOLO detection
    if not skip_yolo:
        pipeline_timer.stage("yolo_detection")
        try:
            batch_mode = "batch" if use_yolo_batching else "sequential"
            logger.info(f"Running YOLO detection ({batch_mode} mode, batch_size={yolo_batch_size})...")
            yolo_stats = extract_objects_from_image(
                IMAGE_DIR,
                DETECTION_DIR,
                YOLO_MODEL_PATH,
                YOLO_CONFIDENCE_THRESHOLD,
                YOLO_IOU_THRESHOLD,
                high_quality=True,
                enable_fallback=True,
                batch_size=yolo_batch_size,
                use_batching=use_yolo_batching,
                image_filter=pdf_filter,  # Filter to specific files if provided
            )
            stats["yolo_detection"] = yolo_stats
            logger.info("YOLO detection completed")
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            raise
    else:
        logger.info("Skipping YOLO detection")

    # 5. Extract text and tables
    if not skip_extraction:
        pipeline_timer.stage("content_extraction")
        try:
            logger.info(f"Extracting text from images using {extraction_provider}...")
            text_stats = parse_image_with_text(
                image_dir=DETECTION_DIR,
                output_dir=PROCESSED_DIR,
                provider=extraction_provider,
                image_filter=pdf_filter,  # Filter to specific files if provided
            )
            stats["text_extraction"] = text_stats

            logger.info(f"Extracting tables from images using {extraction_provider}...")
            table_stats = parse_image_with_table(
                image_dir=DETECTION_DIR,
                output_dir=PROCESSED_DIR,
                provider=extraction_provider,
                image_filter=pdf_filter,  # Filter to specific files if provided
            )
            stats["table_extraction"] = table_stats
            logger.info("Content extraction completed")
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            raise
    else:
        logger.info("Skipping content extraction")

    # 6. Generate image descriptions
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
                image_filter=pdf_filter,  # Filter to specific files if provided
            )
            stats["image_description"] = desc_stats
            logger.info("Image description completed")
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            # Continue with pipeline even if descriptions fail
            logger.warning("Continuing pipeline without image descriptions")
    else:
        logger.info("Skipping image descriptions")

    # 7. Chunk text data
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
                SEMANTIC_CHUNK_BUFFER,
                SEMANTIC_BREAKPOINT_METHOD,
                SEMANTIC_BREAKPOINT_VALUE,
                SENTENCE_SPLIT_REGEX,
                verbose=True
            )
            logger.info("Text chunking completed")
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            raise

        # 8. Chunk tables
        pipeline_timer.stage("table_chunking")
        try:
            logger.info("Chunking table data...")
            client = ollama.Client()
            with open(TABLE_SUMMARY_PROMPT_PATH, encoding="utf-8") as f:
                table_prompt = f.read()

            chunk_table_data(
                table_prompt,
                PROCESSED_DIR,
                None,
                MAX_CHUNK_SIZE,
                int(0.25 * MAX_CHUNK_SIZE),
                client,
                OLLAMA_LLM_MODEL,
                {"temperature": TEMPERATURE_GENERATION, "top_p": TOP_P_GENERATION},
                CHUNKS_DIR
            )
            logger.info("Table chunking completed")
        except Exception as e:
            logger.error(f"Table chunking failed: {e}")
            raise
    else:
        logger.info("Skipping chunking")

    # 9. Generate embeddings
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

    # 10. Build vector indexes
    if not skip_indexing:
        pipeline_timer.stage("index_building")
        try:
            logger.info("Building vector indexes...")
            build_indexes(recreate=True)
            stats["indexing"] = {"status": "completed"}
            logger.info("Index building completed")
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            raise
    else:
        logger.info("Skipping index building")

    # 11. Build knowledge graph (GraphRAG)
    if not skip_graph:
        pipeline_timer.stage("graph_extraction")
        try:
            extraction_mode = "async" if use_async_extraction else "sequential"
            logger.info(f"Extracting entities and relationships ({extraction_mode} mode, max_concurrent={entity_max_concurrent})...")

            # Load graph config
            if graph_config_path:
                from .graph_config import reload_graph_config
                graph_config = reload_graph_config(graph_config_path)
            else:
                graph_config = get_graph_config()

            # Extract entities and relationships from chunks
            if use_async_extraction:
                # Use async extraction for improved throughput with cloud LLM providers
                extraction_results = run_extraction_async(
                    chunks_dir=CHUNKS_DIR,
                    config=graph_config,
                    include_parent_chunks=False,
                    max_concurrent=entity_max_concurrent,
                    show_progress=True,
                )
            else:
                # Use sequential extraction (original behavior)
                extraction_results = extract_from_chunks_dir(
                    chunks_dir=CHUNKS_DIR,
                    config=graph_config,
                    include_parent_chunks=False,
                )

            total_entities = sum(len(r.entities) for r in extraction_results)
            total_relationships = sum(len(r.relationships) for r in extraction_results)
            stats["graph_extraction"] = {
                "chunks_processed": len(extraction_results),
                "entities_extracted": total_entities,
                "relationships_extracted": total_relationships,
                "mode": extraction_mode,
                "max_concurrent": entity_max_concurrent if use_async_extraction else 1,
            }
            logger.info(
                f"Extraction completed: {total_entities} entities, "
                f"{total_relationships} relationships from {len(extraction_results)} chunks"
            )

            # Build knowledge graph
            pipeline_timer.stage("graph_building")
            logger.info("Building knowledge graph...")
            kg = build_knowledge_graph(
                extraction_results=extraction_results,
                config=graph_config,
                detect_communities=True,
                generate_summaries=True,
                save_graph=True,
            )

            graph_stats = kg.get_statistics()
            stats["graph_building"] = graph_stats
            logger.info(
                f"Knowledge graph built: {graph_stats['total_nodes']} nodes, "
                f"{graph_stats['total_edges']} edges, "
                f"{graph_stats['total_communities']} communities"
            )

        except Exception as e:
            logger.error(f"Knowledge graph building failed: {e}")
            logger.warning("Continuing without knowledge graph")
            stats["graph_building"] = {"status": "failed", "error": str(e)}
    else:
        logger.info("Skipping knowledge graph building")

    # End pipeline
    pipeline_summary = pipeline_timer.end()

    # Generate and display ingestion report
    report = format_ingestion_report(stats, pipeline_summary)
    print(report)

    return stats


def main():
    """Main entry point."""
    args = parse_args()

    asyncio.run(run_ingestion_pipeline_async(
        vision_provider=args.vision_provider,
        extraction_provider=args.extraction_provider,
        skip_conversion=args.skip_conversion,
        skip_pdf=args.skip_pdf,
        skip_yolo=args.skip_yolo,
        skip_extraction=args.skip_extraction,
        skip_descriptions=args.skip_descriptions,
        skip_chunking=args.skip_chunking,
        skip_embeddings=args.skip_embeddings,
        force_reembed=args.force_reembed,
        use_cache=not args.no_cache,
        skip_indexing=args.skip_indexing,
        skip_graph=args.skip_graph,
        graph_config_path=args.graph_config,
        # Performance optimization parameters
        yolo_batch_size=args.yolo_batch_size,
        use_yolo_batching=not args.no_yolo_batching,
        entity_max_concurrent=args.entity_max_concurrent,
        use_async_extraction=not args.no_async_extraction,
    ))


if __name__ == "__main__":
    main()
