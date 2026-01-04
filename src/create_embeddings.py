"""
Embedding generation module with caching support.

This module creates embeddings for text chunks and stores them with metadata.
Uses SHA256 content hashing to avoid re-embedding unchanged content.
"""

from pathlib import Path
from typing import Dict, List
import json

import ollama

from .utils.logger import logger, timer, PipelineTimer
from .utils.embedding_cache import get_embedding_cache


def get_embeddings(text: str, embed_model: str, use_cache: bool = True) -> List[float]:
    """
    Generate an embedding for a given text, with optional caching.

    Args:
        text: The text to embed.
        embed_model: The name of the embedding model.
        use_cache: Whether to use the embedding cache.

    Returns:
        The embedding vector as a list of floats.
    """
    # Check cache first
    if use_cache:
        cache = get_embedding_cache()
        cached = cache.get(text, embed_model)
        if cached is not None:
            return cached

    # Generate embedding
    response = ollama.embeddings(
        model=embed_model,
        prompt=text
    )
    embed_vector = response['embedding']

    # Cache the result
    if use_cache:
        cache.set(text, embed_vector, embed_model)

    return embed_vector


def make_metadata(chunk_filename: str) -> dict:
    """
    Build the metadata dict based on the chunk filename.

    Args:
        chunk_filename: Name of the chunk file.

    Returns:
        Metadata dictionary with child, parent, and source information.
    """
    # Determine parent filename
    if "_child_chunk_" in chunk_filename:
        parent = chunk_filename.split("_child_chunk_")[0] + ".txt"
    elif "_description_" in chunk_filename:
        parent = chunk_filename.split("_chunk_")[0] + ".txt"
    elif "_Table_" in chunk_filename:
        parent = chunk_filename.split("_chunk_")[0] + ".md"
    else:
        parent = None

    # Extract document and page
    document = None
    page = None
    if "_page_" in chunk_filename:
        try:
            before, after = chunk_filename.split("_page_", 1)
            document = before
            page = after.split("_")[0]
        except ValueError:
            pass

    return {
        "child": chunk_filename,
        "parent": parent,
        "source": {
            "document": document,
            "page": page
        }
    }


def create_embeddings(
    chunks_dir: str,
    embeddings_dir: str,
    embed_model: str,
    use_cache: bool = True,
    force_reembed: bool = False
) -> Dict[str, int]:
    """
    Generate embeddings for all chunk files in chunks_dir.

    Uses content-based caching to skip unchanged files.

    Args:
        chunks_dir: Directory containing chunk files.
        embeddings_dir: Directory to save embedding JSON files.
        embed_model: Name of the embedding model to use.
        use_cache: Whether to use the embedding cache.
        force_reembed: If True, ignore cache and re-embed everything.

    Returns:
        Statistics dictionary with counts.
    """
    chunks_path = Path(chunks_dir)
    embeddings_path = Path(embeddings_dir)

    chunks_path.mkdir(parents=True, exist_ok=True)
    embeddings_path.mkdir(parents=True, exist_ok=True)

    # Initialize timer
    pipeline_timer = PipelineTimer("embedding_generation").start()

    stats = {
        "total_files": 0,
        "embedded": 0,
        "skipped_parent": 0,
        "skipped_short": 0,
        "cached": 0,
        "errors": 0,
    }

    logger.info(f"Processing files in {chunks_path}...")
    logger.info(f"Using embedding model: {embed_model}")
    logger.info(f"Cache enabled: {use_cache}, Force re-embed: {force_reembed}")

    # Get cache stats before
    if use_cache:
        cache = get_embedding_cache()
        cache_stats_before = cache.get_stats()
        logger.info(f"Cache stats before: {cache_stats_before['total_embeddings']} embeddings")

    pipeline_timer.stage("processing_files")

    for file_path in chunks_path.rglob("*"):
        stats["total_files"] += 1

        # Skip directories and parent chunks
        if not file_path.is_file():
            continue

        if "_parent_chunk_" in file_path.name:
            stats["skipped_parent"] += 1
            continue

        # Read file content
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            stats["errors"] += 1
            continue

        # Skip trivially short files
        if not text.strip() or len(text.split()) < 3:
            stats["skipped_short"] += 1
            continue

        # Check if already embedded (by cache)
        if use_cache and not force_reembed:
            cached = cache.get(text, embed_model)
            if cached is not None:
                # Still write the embedding file if it doesn't exist
                embedding_file = embeddings_path / f"{file_path.stem}_embedding.json"
                if not embedding_file.exists():
                    meta = make_metadata(file_path.name)
                    data = {"embedding": cached, "metadata": meta}
                    with open(embedding_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                stats["cached"] += 1
                logger.debug(f"Using cached embedding for {file_path.name}")
                continue

        # Generate embedding
        logger.info(f"Calculating embedding for: {file_path.name}...")
        try:
            with timer(f"embed {file_path.name}"):
                embed_vector = get_embeddings(text, embed_model, use_cache=use_cache)
        except Exception as e:
            logger.error(f"Error generating embedding for {file_path.name}: {e}")
            stats["errors"] += 1
            continue

        # Build metadata and write JSON
        meta = make_metadata(file_path.name)
        data = {
            "embedding": embed_vector,
            "metadata": meta
        }

        embedding_file = embeddings_path / f"{file_path.stem}_embedding.json"
        try:
            with open(embedding_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            stats["embedded"] += 1
            logger.debug(f"Saved embedding: {embedding_file.name}")
        except Exception as e:
            logger.error(f"Error writing {embedding_file.name}: {e}")
            stats["errors"] += 1

    # End timing
    pipeline_timer.end()

    # Get cache stats after
    if use_cache:
        cache_stats_after = cache.get_stats()
        new_cached = cache_stats_after['total_embeddings'] - cache_stats_before['total_embeddings']
        logger.info(f"New embeddings cached: {new_cached}")

    # Log summary
    logger.info(f"""
    Embedding Generation Complete:
    - Total files processed: {stats['total_files']}
    - Newly embedded: {stats['embedded']}
    - From cache: {stats['cached']}
    - Skipped (parent chunks): {stats['skipped_parent']}
    - Skipped (too short): {stats['skipped_short']}
    - Errors: {stats['errors']}
    - Output directory: {embeddings_dir}
    """)

    return stats
