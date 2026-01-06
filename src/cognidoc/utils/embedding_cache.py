"""
Embedding cache with content hash (SHA256).

This module provides caching for embeddings to avoid re-computing
embeddings for content that hasn't changed.

Uses SQLite for cache storage (lightweight, no external dependencies).
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime

from .logger import logger


class EmbeddingCache:
    """
    Cache for embeddings using content hash.

    Uses SQLite to store:
    - Content hash (SHA256)
    - Embedding vector
    - Model name
    - Timestamp
    """

    def __init__(self, cache_dir: str, db_name: str = "embedding_cache.db"):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store the cache database
            db_name: Name of the SQLite database file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content_length INTEGER,
                    source_file TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)
            """)
            conn.commit()

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, content: str, model: str) -> Optional[list[float]]:
        """
        Get cached embedding for content.

        Args:
            content: The text content
            model: The embedding model name

        Returns:
            Embedding vector if cached, None otherwise
        """
        content_hash = self.compute_hash(content)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT embedding FROM embeddings
                WHERE content_hash = ? AND model = ?
                """,
                (content_hash, model)
            )
            row = cursor.fetchone()

        if row:
            logger.debug(f"Cache hit for hash {content_hash[:16]}...")
            return json.loads(row[0])

        logger.debug(f"Cache miss for hash {content_hash[:16]}...")
        return None

    def set(
        self,
        content: str,
        embedding: list[float],
        model: str,
        source_file: Optional[str] = None
    ):
        """
        Store embedding in cache.

        Args:
            content: The text content
            embedding: The embedding vector
            model: The embedding model name
            source_file: Optional source file path for reference
        """
        content_hash = self.compute_hash(content)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (content_hash, embedding, model, created_at, content_length, source_file)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    content_hash,
                    json.dumps(embedding),
                    model,
                    datetime.now().isoformat(),
                    len(content),
                    source_file
                )
            )
            conn.commit()

        logger.debug(f"Cached embedding for hash {content_hash[:16]}...")

    def exists(self, content: str, model: str) -> bool:
        """Check if embedding exists in cache."""
        content_hash = self.compute_hash(content)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM embeddings
                WHERE content_hash = ? AND model = ?
                """,
                (content_hash, model)
            )
            return cursor.fetchone() is not None

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total_count = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT model, COUNT(*) as count
                FROM embeddings
                GROUP BY model
                """
            )
            by_model = dict(cursor.fetchall())

            cursor = conn.execute(
                """
                SELECT SUM(content_length) FROM embeddings
                """
            )
            total_chars = cursor.fetchone()[0] or 0

        return {
            "total_embeddings": total_count,
            "by_model": by_model,
            "total_characters_embedded": total_chars,
            "db_path": str(self.db_path),
        }

    def clear(self, model: Optional[str] = None):
        """
        Clear cache.

        Args:
            model: If provided, only clear embeddings for this model
        """
        with sqlite3.connect(self.db_path) as conn:
            if model:
                conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
                logger.info(f"Cleared cache for model: {model}")
            else:
                conn.execute("DELETE FROM embeddings")
                logger.info("Cleared entire embedding cache")
            conn.commit()

    def cleanup_old(self, days: int = 30):
        """
        Remove embeddings older than specified days.

        Args:
            days: Remove embeddings older than this many days
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM embeddings WHERE created_at < ?",
                (cutoff,)
            )
            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"Cleaned up {deleted} embeddings older than {days} days")
        return deleted


# Global cache instance (lazy initialization)
_cache_instance: Optional[EmbeddingCache] = None


def get_embedding_cache(cache_dir: Optional[str] = None) -> EmbeddingCache:
    """
    Get the global embedding cache instance.

    Args:
        cache_dir: Cache directory (uses default from constants if not provided)

    Returns:
        EmbeddingCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        if cache_dir is None:
            from ..constants import CACHE_DIR
            cache_dir = CACHE_DIR

        _cache_instance = EmbeddingCache(cache_dir)
        logger.info(f"Initialized embedding cache at {cache_dir}")

    return _cache_instance


__all__ = ["EmbeddingCache", "get_embedding_cache"]
