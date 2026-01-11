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

    # 2. Process source documents (copy PDFs, copy images, convert documents)
    if not skip_conversion:
        pipeline_timer.stage("document_conversion")
        try:
            logger.info("Processing source documents...")
            conversion_stats = process_source_documents(
                sources_dir=SOURCES_DIR,
                pdf_output_dir=PDF_DIR,
                image_output_dir=IMAGE_DIR,  # Images go directly to images folder
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
            convert_pdf_to_image(PDF_DIR, IMAGE_DIR)
            logger.info("PDF conversion completed")
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
            )
            stats["text_extraction"] = text_stats

            logger.info(f"Extracting tables from images using {extraction_provider}...")
            table_stats = parse_image_with_table(
                image_dir=DETECTION_DIR,
                output_dir=PROCESSED_DIR,
                provider=extraction_provider,
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
