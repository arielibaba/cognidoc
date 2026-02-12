"""
Embedding generation module with batching and caching support.

This module creates embeddings for text chunks and stores them with metadata.
Uses SHA256 content hashing to avoid re-embedding unchanged content.
Supports batched async processing for improved performance.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

from .utils.logger import logger, timer, PipelineTimer
from .utils.embedding_cache import get_embedding_cache
from .utils.embedding_providers import (
    OllamaEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingProvider,
    get_embedding_provider,
)


# Path separator used to encode relative paths in filenames
PATH_SEPARATOR = "__"

# Default batch size for embedding generation
DEFAULT_BATCH_SIZE = 32

# Max concurrent requests for async embedding
MAX_CONCURRENT_REQUESTS = 4

# Max characters to send to embedding model (~8K tokens for qwen3-embedding)
# Chunks exceeding this are truncated with a warning
MAX_EMBED_CHARS = 30000


@dataclass
class ChunkToEmbed:
    """Represents a chunk file to be embedded."""

    file_path: Path
    text: str
    metadata: dict


def decode_document_path(encoded_name: str) -> str:
    """
    Decode a path-encoded document name back to its relative path.

    For "projet_A__doc", returns "projet_A/doc"
    For "doc", returns "doc"

    Args:
        encoded_name: The encoded document name (without extension)

    Returns:
        The decoded relative path (without extension)
    """
    return encoded_name.replace(PATH_SEPARATOR, "/")


def make_metadata(
    chunk_filename: str,
    stem_dates: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Build the metadata dict based on the chunk filename.

    Handles path-encoded filenames where subdirectory paths are encoded
    using "__" separator (e.g., "projet_A__doc_page_1" -> document "projet_A/doc").

    Args:
        chunk_filename: Name of the chunk file.
        stem_dates: Optional mapping of document stem -> file modification time (UNIX timestamp).

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
            document = decode_document_path(before)
            page = after.split("_")[0]
        except ValueError as e:
            logger.debug(f"Could not parse page info from chunk filename '{chunk_filename}': {e}")

    # Lookup file date from stem_dates mapping
    file_date = None
    if stem_dates and document:
        # Try exact match first, then try encoded stem
        encoded_stem = document.replace("/", PATH_SEPARATOR)
        file_date = stem_dates.get(document) or stem_dates.get(encoded_stem)

    source: Dict[str, Any] = {"document": document, "page": page}
    if file_date is not None:
        source["file_date"] = file_date

    return {
        "child": chunk_filename,
        "parent": parent,
        "source": source,
    }


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
    import ollama

    # Check cache first
    if use_cache:
        cache = get_embedding_cache()
        cached = cache.get(text, embed_model)
        if cached is not None:
            return cached

    # Generate embedding
    response = ollama.embeddings(model=embed_model, prompt=text)
    embed_vector = response["embedding"]

    # Cache the result
    if use_cache:
        cache.set(text, embed_vector, embed_model)

    result: list[float] = embed_vector
    return result


def collect_chunks_to_embed(
    chunks_path: Path,
    embeddings_path: Path,
    embed_model: str,
    use_cache: bool,
    force_reembed: bool,
    cache,
    file_filter: Optional[list] = None,
    stem_dates: Optional[Dict[str, float]] = None,
) -> Tuple[List[ChunkToEmbed], Dict[str, int]]:
    """
    Collect all chunks that need to be embedded.

    Args:
        chunks_path: Directory containing chunk files
        embeddings_path: Directory for embedding outputs
        embed_model: Embedding model name
        use_cache: Whether to use cache
        force_reembed: Whether to force re-embedding
        cache: Embedding cache instance

    Returns:
        Tuple of (chunks_to_embed, stats)
    """
    chunks_to_embed = []
    stats = {
        "total_files": 0,
        "to_embed": 0,
        "skipped_parent": 0,
        "skipped_short": 0,
        "cached": 0,
        "errors": 0,
    }

    for file_path in chunks_path.rglob("*"):
        stats["total_files"] += 1

        # Skip directories
        if not file_path.is_file():
            continue

        # Skip files not matching incremental filter
        if file_filter and not any(file_path.name.startswith(stem) for stem in file_filter):
            continue

        # Skip parent chunks (but not child chunks)
        if "_parent_chunk_" in file_path.name and "_child_chunk_" not in file_path.name:
            stats["skipped_parent"] += 1
            continue

        # Read file content
        try:
            text = file_path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            stats["errors"] += 1
            continue

        # Skip trivially short files
        if not text.strip() or len(text.split()) < 3:
            stats["skipped_short"] += 1
            continue

        # Check cache
        if use_cache and not force_reembed:
            cached_embedding = cache.get(text, embed_model)
            if cached_embedding is not None:
                # Write embedding file if it doesn't exist
                embedding_file = embeddings_path / f"{file_path.stem}_embedding.json"
                if not embedding_file.exists():
                    meta = make_metadata(file_path.name, stem_dates=stem_dates)
                    data = {"embedding": cached_embedding, "metadata": meta}
                    with open(embedding_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                stats["cached"] += 1
                continue

        # Add to list of chunks to embed
        metadata = make_metadata(file_path.name, stem_dates=stem_dates)
        chunks_to_embed.append(
            ChunkToEmbed(
                file_path=file_path,
                text=text,
                metadata=metadata,
            )
        )
        stats["to_embed"] += 1

    return chunks_to_embed, stats


async def embed_batch_async(
    chunks: List[ChunkToEmbed],
    embeddings_path: Path,
    embed_model: str,
    use_cache: bool,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
) -> Tuple[int, int]:
    """
    Embed a batch of chunks asynchronously.

    Uses a shared HTTP client with connection pooling for improved performance.

    Args:
        chunks: List of chunks to embed
        embeddings_path: Output directory
        embed_model: Model name
        use_cache: Whether to cache results
        max_concurrent: Max concurrent requests

    Returns:
        Tuple of (success_count, error_count)
    """
    import httpx

    cache = get_embedding_cache() if use_cache else None
    from .constants import OLLAMA_URL

    host = OLLAMA_URL
    success = 0
    errors = 0

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create shared HTTP client with connection pooling
    # This avoids the overhead of creating a new connection per chunk
    async with httpx.AsyncClient(
        timeout=60.0,
        limits=httpx.Limits(
            max_connections=max_concurrent, max_keepalive_connections=max_concurrent
        ),
    ) as client:

        async def process_chunk(chunk: ChunkToEmbed) -> bool:
            nonlocal success, errors
            async with semaphore:
                try:
                    text = chunk.text
                    if len(text) > MAX_EMBED_CHARS:
                        logger.warning(
                            f"Truncating oversized chunk {chunk.file_path.name} "
                            f"({len(text)} chars → {MAX_EMBED_CHARS})"
                        )
                        text = text[:MAX_EMBED_CHARS]

                    response = await client.post(
                        f"{host}/api/embeddings", json={"model": embed_model, "prompt": text}
                    )
                    response.raise_for_status()
                    embedding = response.json()["embedding"]

                    # Cache the result
                    if cache:
                        cache.set(chunk.text, embedding, embed_model)

                    # Write to file
                    data = {"embedding": embedding, "metadata": chunk.metadata}
                    embedding_file = embeddings_path / f"{chunk.file_path.stem}_embedding.json"
                    with open(embedding_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)

                    success += 1
                    return True

                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"HTTP error embedding {chunk.file_path.name}: "
                        f"{e.response.status_code} - {e.response.text[:200]}"
                    )
                    errors += 1
                    return False
                except (ValueError, ConnectionError, TimeoutError, OSError) as e:
                    logger.error(f"Error embedding {chunk.file_path.name}: {e}")
                    errors += 1
                    return False

        await asyncio.gather(*[process_chunk(c) for c in chunks])

    return success, errors


def create_embeddings(
    chunks_dir: str,
    embeddings_dir: str,
    embed_model: str,
    use_cache: bool = True,
    force_reembed: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    file_filter: Optional[list] = None,
    stem_dates: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """
    Generate embeddings for all chunk files in chunks_dir.

    Uses batched async processing for improved performance.
    Uses content-based caching to skip unchanged files.

    Args:
        chunks_dir: Directory containing chunk files.
        embeddings_dir: Directory to save embedding JSON files.
        embed_model: Name of the embedding model to use.
        use_cache: Whether to use the embedding cache.
        force_reembed: If True, ignore cache and re-embed everything.
        batch_size: Number of chunks to process per batch.
        max_concurrent: Max concurrent embedding requests.

    Returns:
        Statistics dictionary with counts.
    """
    chunks_path = Path(chunks_dir)
    embeddings_path = Path(embeddings_dir)

    chunks_path.mkdir(parents=True, exist_ok=True)
    embeddings_path.mkdir(parents=True, exist_ok=True)

    # Initialize timer
    pipeline_timer = PipelineTimer("embedding_generation").start()

    logger.info(f"Processing files in {chunks_path}...")
    logger.info(f"Using embedding model: {embed_model}")
    logger.info(f"Cache enabled: {use_cache}, Force re-embed: {force_reembed}")
    logger.info(f"Batch size: {batch_size}, Max concurrent: {max_concurrent}")

    # Get cache
    cache = get_embedding_cache() if use_cache else None
    if cache:
        cache_stats_before = cache.get_stats()
        logger.info(f"Cache stats before: {cache_stats_before['total_embeddings']} embeddings")

    # Phase 1: Collect chunks to embed
    pipeline_timer.stage("collecting_chunks")
    logger.info("Scanning files and checking cache...")

    chunks_to_embed, stats = collect_chunks_to_embed(
        chunks_path,
        embeddings_path,
        embed_model,
        use_cache,
        force_reembed,
        cache,
        file_filter=file_filter,
        stem_dates=stem_dates,
    )

    logger.info(f"Found {stats['to_embed']} chunks to embed, {stats['cached']} from cache")

    # Phase 2: Process in batches with progress bar
    if chunks_to_embed:
        pipeline_timer.stage("embedding_batches")
        total_success = 0
        total_errors = 0

        # Process in batches
        num_batches = (len(chunks_to_embed) + batch_size - 1) // batch_size

        from .utils.async_utils import run_coroutine

        with tqdm(total=len(chunks_to_embed), desc="Embedding chunks", unit="chunk") as pbar:
            for i in range(0, len(chunks_to_embed), batch_size):
                batch = chunks_to_embed[i : i + batch_size]
                batch_num = i // batch_size + 1

                logger.debug(f"Processing batch {batch_num}/{num_batches} ({len(batch)} chunks)")

                # Run async embedding for this batch
                success, errors = run_coroutine(
                    embed_batch_async(
                        batch, embeddings_path, embed_model, use_cache, max_concurrent
                    )
                )

                total_success += success
                total_errors += errors
                pbar.update(len(batch))

        stats["embedded"] = total_success
        stats["errors"] += total_errors
    else:
        stats["embedded"] = 0

    # End timing
    pipeline_timer.end()

    # Get cache stats after
    if cache:
        cache_stats_after = cache.get_stats()
        new_cached = cache_stats_after["total_embeddings"] - cache_stats_before["total_embeddings"]
        logger.info(f"New embeddings cached: {new_cached}")

    # Log summary
    logger.info(
        f"""
    Embedding Generation Complete:
    - Total files scanned: {stats['total_files']}
    - Newly embedded: {stats['embedded']}
    - From cache: {stats['cached']}
    - Skipped (parent chunks): {stats['skipped_parent']}
    - Skipped (too short): {stats['skipped_short']}
    - Errors: {stats['errors']}
    - Output directory: {embeddings_dir}
    """
    )

    return stats


# Backward compatibility alias
def create_embeddings_sequential(
    chunks_dir: str,
    embeddings_dir: str,
    embed_model: str,
    use_cache: bool = True,
    force_reembed: bool = False,
) -> Dict[str, int]:
    """
    Sequential embedding generation (legacy mode).

    Use create_embeddings() for batched async processing.
    """
    return create_embeddings(
        chunks_dir=chunks_dir,
        embeddings_dir=embeddings_dir,
        embed_model=embed_model,
        use_cache=use_cache,
        force_reembed=force_reembed,
        batch_size=1,
        max_concurrent=1,
    )
