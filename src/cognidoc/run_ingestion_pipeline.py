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
    TEMPERATURE_GENERATION,
    SYSTEM_PROMPT_IMAGE_DESC,
    USER_PROMPT_IMAGE_DESC,
    DEFAULT_VISION_PROVIDER,
    CHECKPOINT_FILE,
    MAX_CONSECUTIVE_QUOTA_ERRORS,
    CHECKPOINT_SAVE_INTERVAL,
    INGESTION_MANIFEST_PATH,
)
from .ingestion_manifest import IngestionManifest
from .checkpoint import PipelineCheckpoint, FailedItem

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
from .knowledge_graph import (
    build_knowledge_graph,
    KnowledgeGraph,
    backup_knowledge_graph,
    has_valid_knowledge_graph,
    get_knowledge_graph_stats,
)
from .graph_config import get_graph_config
from .entity_resolution import resolve_entities

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
        "--skip-resolution",
        action="store_true",
        help="Skip entity resolution (semantic deduplication)"
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
    parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="Force full re-ingestion of all documents (ignore incremental manifest)"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental detection"
    )
    return parser.parse_args()


# =============================================================================
# Pipeline Stage Helpers
# =============================================================================


def _run_document_conversion(source_files: list = None) -> dict:
    """Stage 2: Process source documents (copy PDFs, copy images, convert documents)."""
    try:
        logger.info("Processing source documents...")
        conversion_stats = process_source_documents(
            sources_dir=SOURCES_DIR,
            pdf_output_dir=PDF_DIR,
            image_output_dir=IMAGE_DIR,
            source_files=source_files,
        )
        logger.info(
            f"Document processing completed: {conversion_stats['pdfs_copied']} PDFs copied, "
            f"{conversion_stats['images_copied']} images copied, "
            f"{conversion_stats['converted']} converted, "
            f"{conversion_stats['skipped_existing']} skipped, "
            f"{conversion_stats['failed']} failed"
        )
        return conversion_stats
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        logger.warning("Continuing pipeline without source document processing")
        return {}


def _run_pdf_conversion(pdf_filter: list = None) -> dict:
    """Stage 3: Convert PDFs to images."""
    logger.info("Converting PDFs to images...")
    pdf_stats = convert_pdf_to_image(PDF_DIR, IMAGE_DIR, pdf_filter=pdf_filter)
    logger.info(
        f"PDF conversion completed: {pdf_stats.get('success', 0)} PDFs, "
        f"{pdf_stats.get('pages', 0)} pages"
    )
    return pdf_stats


def _run_yolo_detection(
    yolo_batch_size: int, use_yolo_batching: bool, pdf_filter: list = None
) -> dict:
    """Stage 4: YOLO object detection."""
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
        image_filter=pdf_filter,
    )
    logger.info("YOLO detection completed")
    return yolo_stats


async def _run_content_extraction(
    extraction_provider: str, pdf_filter: list = None
) -> tuple:
    """Stage 5: Extract text and tables in parallel."""
    logger.info(f"Extracting text and tables in parallel using {extraction_provider}...")
    text_task = asyncio.to_thread(
        parse_image_with_text,
        image_dir=DETECTION_DIR,
        output_dir=PROCESSED_DIR,
        provider=extraction_provider,
        image_filter=pdf_filter,
    )
    table_task = asyncio.to_thread(
        parse_image_with_table,
        image_dir=DETECTION_DIR,
        output_dir=PROCESSED_DIR,
        provider=extraction_provider,
        image_filter=pdf_filter,
    )
    text_stats, table_stats = await asyncio.gather(text_task, table_task)
    logger.info("Content extraction completed (parallel)")
    return text_stats, table_stats


async def _run_image_descriptions(
    vision_provider: str, pdf_filter: list = None
) -> dict:
    """Stage 6: Generate image descriptions."""
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
            image_filter=pdf_filter,
        )
        logger.info("Image description completed")
        return desc_stats
    except Exception as e:
        logger.error(f"Image description failed: {e}")
        logger.warning("Continuing pipeline without image descriptions")
        return {}


async def _run_chunking(incremental_stems: list = None):
    """Stage 7: Chunk text and table data in parallel."""
    logger.info("Chunking text and table data in parallel...")
    with open(TABLE_SUMMARY_PROMPT_PATH, encoding="utf-8") as f:
        table_prompt = f.read()

    def run_text_chunking():
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
            verbose=True,
            file_filter=incremental_stems,
        )

    def run_table_chunking():
        chunk_table_data(
            table_prompt,
            PROCESSED_DIR,
            None,
            MAX_CHUNK_SIZE,
            int(0.25 * MAX_CHUNK_SIZE),
            CHUNKS_DIR,
            use_unified_llm=True,
            temperature=TEMPERATURE_GENERATION,
            file_filter=incremental_stems,
        )

    await asyncio.gather(
        asyncio.to_thread(run_text_chunking),
        asyncio.to_thread(run_table_chunking),
    )
    logger.info("Chunking completed (parallel)")


def _run_embeddings(
    use_cache: bool, force_reembed: bool, incremental_stems: list = None
) -> dict:
    """Stage 8: Generate embeddings."""
    logger.info("Creating embeddings...")
    embed_stats = create_embeddings(
        CHUNKS_DIR,
        EMBEDDINGS_DIR,
        EMBED_MODEL,
        use_cache=use_cache,
        force_reembed=force_reembed,
        file_filter=incremental_stems,
    )
    logger.info("Embedding generation completed")
    return embed_stats


def _run_index_building():
    """Stage 9: Build vector indexes."""
    logger.info("Building vector indexes...")
    build_indexes(recreate=True)
    logger.info("Index building completed")


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
    skip_resolution: bool = False,
    graph_config_path: str = None,
    yolo_batch_size: int = 2,
    use_yolo_batching: bool = True,
    entity_max_concurrent: int = 4,
    use_async_extraction: bool = True,
    source_files: list = None,
    # Incremental ingestion parameters
    incremental: bool = True,
    full_reindex: bool = False,
    # Checkpoint/resume parameters
    resume_from_checkpoint: bool = True,
    max_consecutive_quota_errors: int = None,
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
        skip_resolution: Skip entity resolution (semantic deduplication)
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

    # === INCREMENTAL INGESTION LOGIC ===
    manifest_path = Path(INGESTION_MANIFEST_PATH)
    manifest = None
    incremental_stems = None  # None = process everything (full mode)

    if not full_reindex and incremental and not source_files:
        manifest = IngestionManifest.load(manifest_path)
        if manifest is not None:
            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(
                Path(SOURCES_DIR), source_files
            )
            if not new_files and not modified_files:
                logger.info("No new or modified files detected. Nothing to ingest.")
                pipeline_summary = pipeline_timer.end()
                return stats

            incremental_stems = list(new_stems)
            logger.info(
                f"Incremental mode: {len(new_files)} new, {len(modified_files)} modified files "
                f"({len(incremental_stems)} stems to process)"
            )

            # Clean up old intermediate files for modified documents
            for mod_file in modified_files:
                _cleanup_intermediate_files(mod_file.stem)

            # Override source_files to only process new/modified
            source_files = [str(f) for f in new_files + modified_files]
        else:
            logger.info("First ingestion detected (no manifest). Running full pipeline.")
    elif full_reindex:
        logger.info("Full reindex mode. Processing all files.")

    # Create manifest for tracking (will be saved at end)
    if manifest is None:
        manifest = IngestionManifest()

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

    # --- Stage 2: Document conversion ---
    if not skip_conversion:
        pipeline_timer.stage("document_conversion")
        stats["document_conversion"] = _run_document_conversion(source_files)
    else:
        logger.info("Skipping source document processing")

    # --- Stage 3: PDF to images ---
    if not skip_pdf:
        pipeline_timer.stage("pdf_conversion")
        stats["pdf_conversion"] = _run_pdf_conversion(pdf_filter)
    else:
        logger.info("Skipping PDF conversion")

    # --- Stage 4: YOLO detection ---
    if not skip_yolo:
        pipeline_timer.stage("yolo_detection")
        stats["yolo_detection"] = _run_yolo_detection(
            yolo_batch_size, use_yolo_batching, pdf_filter
        )
    else:
        logger.info("Skipping YOLO detection")

    # --- Stage 5: Content extraction ---
    if not skip_extraction:
        pipeline_timer.stage("content_extraction")
        text_stats, table_stats = await _run_content_extraction(
            extraction_provider, pdf_filter
        )
        stats["text_extraction"] = text_stats
        stats["table_extraction"] = table_stats
    else:
        logger.info("Skipping content extraction")

    # --- Stage 6: Image descriptions ---
    if not skip_descriptions:
        pipeline_timer.stage("image_description")
        stats["image_description"] = await _run_image_descriptions(
            vision_provider, pdf_filter
        )
    else:
        logger.info("Skipping image descriptions")

    # --- Stage 7: Chunking ---
    if not skip_chunking:
        pipeline_timer.stage("text_chunking")
        await _run_chunking(incremental_stems)
    else:
        logger.info("Skipping chunking")

    # --- Stage 8: Embeddings ---
    if not skip_embeddings:
        pipeline_timer.stage("embedding_generation")
        stats["embeddings"] = _run_embeddings(
            use_cache, force_reembed, incremental_stems
        )
    else:
        logger.info("Skipping embeddings")

    # --- Stage 9: Index building ---
    if not skip_indexing:
        pipeline_timer.stage("index_building")
        _run_index_building()
        stats["indexing"] = {"status": "completed"}
    else:
        logger.info("Skipping index building")

    # 11. Build knowledge graph (GraphRAG) with checkpoint support
    if not skip_graph:
        pipeline_timer.stage("graph_extraction")

        # Set up checkpoint parameters
        if max_consecutive_quota_errors is None:
            max_consecutive_quota_errors = MAX_CONSECUTIVE_QUOTA_ERRORS

        checkpoint_path = Path(CHECKPOINT_FILE)
        checkpoint = None
        extraction_interrupted = False

        # Load existing checkpoint if resuming
        if resume_from_checkpoint:
            checkpoint = PipelineCheckpoint.load(checkpoint_path)
            if checkpoint and checkpoint.status == "interrupted":
                logger.info(f"Resuming from checkpoint: {checkpoint.get_summary()}")

        # Initialize new checkpoint if none exists
        if checkpoint is None:
            checkpoint = PipelineCheckpoint()

        # Load graph config first (needed for both paths)
        if graph_config_path:
            from .graph_config import reload_graph_config
            graph_config = reload_graph_config(graph_config_path)
        else:
            graph_config = get_graph_config()

        # =====================================================================
        # DATA PROTECTION: Check if we should load existing graph instead of rebuild
        # =====================================================================
        # If checkpoint shows we're past entity_extraction (e.g., at community_summaries)
        # AND we have valid graph data, load it instead of rebuilding from scratch
        # This prevents data loss when resuming after quota errors
        resume_from_graph = False
        if (checkpoint and
            checkpoint.status == "interrupted" and
            checkpoint.pipeline_stage in ("community_summaries", "graph_building") and
            has_valid_knowledge_graph()):
            kg_stats = get_knowledge_graph_stats()
            logger.info(
                f"Found existing knowledge graph with {kg_stats['nodes']} nodes, "
                f"{kg_stats['edges']} edges. Loading instead of rebuilding."
            )
            resume_from_graph = True

        # =====================================================================
        # DATA PROTECTION: Backup existing graph before any modifications
        # =====================================================================
        if has_valid_knowledge_graph() and not resume_from_graph:
            kg_stats = get_knowledge_graph_stats()
            logger.info(
                f"Backing up existing knowledge graph ({kg_stats['nodes']} nodes, "
                f"{kg_stats['edges']} edges) before modifications..."
            )
            backup_path = backup_knowledge_graph()
            if backup_path:
                logger.info(f"Backup created: {backup_path}")
            else:
                logger.warning("Failed to create backup, proceeding anyway...")

        try:
            # If resuming from existing graph (e.g., interrupted at community_summaries)
            if resume_from_graph:
                logger.info("Loading existing knowledge graph for resume...")
                kg = KnowledgeGraph.load(config=graph_config)
                stats["graph_extraction"] = {
                    "status": "resumed_from_existing",
                    "nodes": len(kg.nodes),
                    "edges": kg.graph.number_of_edges(),
                }
                # Skip entity extraction, go directly to community summaries
                extraction_interrupted = False
            else:
                # Normal path: run entity extraction
                extraction_mode = "async" if use_async_extraction else "sequential"
                logger.info(f"Extracting entities and relationships ({extraction_mode} mode, max_concurrent={entity_max_concurrent})...")

                # Update checkpoint stage
                checkpoint.set_stage("entity_extraction")

                # Get processed chunk IDs from checkpoint for resume
                # Use set for O(1) lookup instead of O(n) list search
                processed_chunk_ids = set(checkpoint.entity_extraction.processed_item_ids)

                # Track progress for periodic checkpoint saving
                chunks_since_last_save = [0]  # Use list for mutable closure

                def on_extraction_progress(chunk_id: str, success: bool, error_type: str = None):
                    """Callback to save checkpoint periodically during extraction."""
                    if success:
                        # Add to checkpoint (use set for O(1) lookup, then sync to list)
                        if chunk_id not in processed_chunk_ids:
                            processed_chunk_ids.add(chunk_id)
                            checkpoint.entity_extraction.processed_item_ids.append(chunk_id)
                        chunks_since_last_save[0] += 1

                        # Save checkpoint every N chunks
                        if chunks_since_last_save[0] >= CHECKPOINT_SAVE_INTERVAL:
                            checkpoint.save(checkpoint_path)
                            logger.debug(f"Checkpoint saved ({len(checkpoint.entity_extraction.processed_item_ids)} chunks processed)")
                            chunks_since_last_save[0] = 0

                # Extract entities and relationships from chunks
                # Default: extract from parent chunks (512 tokens) + descriptions, skip children (64 tokens)
                if use_async_extraction:
                    # Use async extraction for improved throughput with cloud LLM providers
                    extraction_results, extraction_state = run_extraction_async(
                        chunks_dir=CHUNKS_DIR,
                        config=graph_config,
                        max_concurrent=entity_max_concurrent,
                        show_progress=True,
                        processed_chunk_ids=processed_chunk_ids,
                        max_consecutive_quota_errors=max_consecutive_quota_errors,
                        on_progress_callback=on_extraction_progress,
                        file_filter=incremental_stems,
                    )

                    # Update checkpoint with extraction state
                    checkpoint.entity_extraction.processed_item_ids = list(extraction_state["processed_chunk_ids"])
                    checkpoint.entity_extraction.consecutive_quota_errors = extraction_state["consecutive_quota_errors"]
                    for fc in extraction_state.get("failed_chunks", []):
                        checkpoint.entity_extraction.failed_items.append(
                            FailedItem(
                                item_id=fc["chunk_id"],
                                error_type=fc["error_type"],
                                error_message=fc["error_message"],
                            )
                        )

                    extraction_interrupted = extraction_state.get("interrupted", False)
                else:
                    # Use sequential extraction (original behavior - no checkpoint support)
                    extraction_results = extract_from_chunks_dir(
                        chunks_dir=CHUNKS_DIR,
                        config=graph_config,
                        file_filter=incremental_stems,
                    )
                    extraction_state = {"interrupted": False}

                total_entities = sum(len(r.entities) for r in extraction_results)
                total_relationships = sum(len(r.relationships) for r in extraction_results)
                stats["graph_extraction"] = {
                    "chunks_processed": len(extraction_results),
                    "entities_extracted": total_entities,
                    "relationships_extracted": total_relationships,
                    "mode": extraction_mode,
                    "max_concurrent": entity_max_concurrent if use_async_extraction else 1,
                    "resumed_from_checkpoint": len(processed_chunk_ids) > 0,
                    "interrupted": extraction_interrupted,
                }

                if extraction_interrupted:
                    # Save checkpoint and stop
                    checkpoint.set_interrupted("quota_exhausted")
                    checkpoint.save(checkpoint_path)
                    logger.warning(
                        f"Entity extraction interrupted due to quota errors. "
                        f"Checkpoint saved to {checkpoint_path}. "
                        f"Resume later by running the pipeline again."
                    )
                    stats["graph_building"] = {"status": "interrupted", "reason": "quota_exhausted"}
                else:
                    logger.info(
                        f"Extraction completed: {total_entities} entities, "
                        f"{total_relationships} relationships from {len(extraction_results)} chunks"
                    )

                    # Build knowledge graph
                    pipeline_timer.stage("graph_building")
                    checkpoint.set_stage("graph_building")

                    if incremental_stems and has_valid_knowledge_graph():
                        # INCREMENTAL: Load existing graph and merge new entities
                        logger.info("Incremental mode: loading existing graph and merging new entities...")
                        kg = KnowledgeGraph.load(config=graph_config)
                        merge_stats = kg.build_from_extraction_results(extraction_results)
                        logger.info(
                            f"Merged into existing graph: {merge_stats.get('entities_added', 0)} added, "
                            f"{merge_stats.get('entities_merged', 0)} merged"
                        )
                        # Re-detect communities on the merged graph
                        if kg.config.graph.enable_communities:
                            kg.detect_communities()
                    else:
                        # FULL: Build new graph from scratch
                        logger.info("Building knowledge graph...")
                        kg = build_knowledge_graph(
                            extraction_results=extraction_results,
                            config=graph_config,
                            detect_communities=True,
                            generate_summaries=False,  # We'll do this with checkpoint support
                            save_graph=False,  # We'll save after all steps
                        )

            # =====================================================================
            # COMMON PATH: Generate community summaries (for both resume and new graph)
            # =====================================================================
            # At this point, kg is defined either from:
            # - Loading existing graph (resume_from_graph=True)
            # - Building new graph from extraction results (extraction_interrupted=False)
            if not extraction_interrupted:
                # Generate community summaries with checkpoint support
                checkpoint.set_stage("community_summaries")
                processed_community_ids = set(checkpoint.community_summaries.processed_item_ids)
                checkpoint.community_summaries.total_items = len(kg.communities)

                generated, skipped, quota_errors, summaries_interrupted = kg.generate_community_summaries(
                    compute_embeddings=True,
                    skip_existing=True,
                    processed_community_ids=processed_community_ids,
                    max_consecutive_quota_errors=max_consecutive_quota_errors,
                    # Periodic save to prevent data loss (saves every 100 communities)
                    save_interval=100,
                )

                # Update checkpoint
                for comm_id, comm in kg.communities.items():
                    if comm.summary and not comm.summary.startswith("Community of "):
                        if comm_id not in processed_community_ids:
                            checkpoint.community_summaries.processed_item_ids.append(comm_id)

                if summaries_interrupted:
                    checkpoint.set_interrupted("quota_exhausted")
                    checkpoint.save(checkpoint_path)
                    kg.save()  # Save partial graph
                    logger.warning(
                        f"Community summary generation interrupted. "
                        f"Checkpoint saved to {checkpoint_path}."
                    )
                    stats["graph_building"] = {"status": "interrupted", "reason": "quota_exhausted"}
                else:
                    # Entity resolution (semantic deduplication)
                    if not skip_resolution and graph_config.entity_resolution.enabled:
                        pipeline_timer.stage("entity_resolution")
                        logger.info("Running entity resolution (semantic deduplication)...")

                        try:
                            resolution_result = await resolve_entities(
                                kg,
                                config=graph_config.entity_resolution,
                                show_progress=True,
                            )

                            stats["entity_resolution"] = {
                                "original_entities": resolution_result.original_entity_count,
                                "final_entities": resolution_result.final_entity_count,
                                "candidates_found": resolution_result.candidates_found,
                                "clusters_merged": resolution_result.clusters_found,
                                "entities_merged": resolution_result.entities_merged,
                                "llm_calls": resolution_result.llm_calls_made,
                                "cache_hits": resolution_result.cache_hits,
                                "duration_seconds": resolution_result.duration_seconds,
                            }

                            logger.info(
                                f"Entity resolution: {resolution_result.original_entity_count} → "
                                f"{resolution_result.final_entity_count} entities "
                                f"({resolution_result.entities_merged} merged)"
                            )
                        except Exception as e:
                            logger.error(f"Entity resolution failed: {e}")
                            stats["entity_resolution"] = {"status": "failed", "error": str(e)}
                            # Continue - resolution failure shouldn't stop pipeline
                    else:
                        logger.info("Skipping entity resolution")

                    # Save completed graph
                    kg.save()

                    # Clear checkpoint after successful completion
                    PipelineCheckpoint.clear(checkpoint_path)

                    graph_stats = kg.get_statistics()
                    stats["graph_building"] = graph_stats
                    logger.info(
                        f"Knowledge graph built: {graph_stats['total_nodes']} nodes, "
                        f"{graph_stats['total_edges']} edges, "
                        f"{graph_stats['total_communities']} communities"
                    )

        except Exception as e:
            logger.error(f"Knowledge graph building failed: {e}")
            # Save checkpoint on error
            if checkpoint:
                checkpoint.set_interrupted(f"error: {str(e)[:100]}")
                checkpoint.save(checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
            logger.warning("Continuing without knowledge graph")
            stats["graph_building"] = {"status": "failed", "error": str(e)}
    else:
        logger.info("Skipping knowledge graph building")

    # Update ingestion manifest after successful processing
    try:
        if source_files:
            for src_file in source_files:
                src_path = Path(src_file)
                if src_path.exists():
                    manifest.record_file(src_path, Path(SOURCES_DIR), src_path.stem)
        else:
            manifest.record_all_sources(Path(SOURCES_DIR))
        manifest.save(manifest_path)
    except Exception as e:
        logger.warning(f"Failed to save ingestion manifest: {e}")

    # End pipeline
    pipeline_summary = pipeline_timer.end()

    # Generate and display ingestion report
    report = format_ingestion_report(stats, pipeline_summary)
    print(report)

    return stats


def _cleanup_intermediate_files(stem: str):
    """Remove intermediate files for a given PDF stem (for re-ingestion of modified files)."""
    dirs_and_patterns = [
        (PROCESSED_DIR, f"{stem}_*"),
        (CHUNKS_DIR, f"{stem}_*"),
        (EMBEDDINGS_DIR, f"{stem}_*"),
    ]
    for dir_path, pattern in dirs_and_patterns:
        dir_p = Path(dir_path)
        if dir_p.exists():
            for f in dir_p.glob(pattern):
                f.unlink()
                logger.debug(f"Cleaned up intermediate file: {f.name}")


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
        skip_resolution=args.skip_resolution,
        graph_config_path=args.graph_config,
        # Performance optimization parameters
        yolo_batch_size=args.yolo_batch_size,
        use_yolo_batching=not args.no_yolo_batching,
        entity_max_concurrent=args.entity_max_concurrent,
        use_async_extraction=not args.no_async_extraction,
        # Incremental ingestion parameters
        incremental=not args.no_incremental,
        full_reindex=args.full_reindex,
    ))


if __name__ == "__main__":
    main()
