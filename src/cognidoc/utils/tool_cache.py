"""
Persistent tool cache with SQLite backend.

This module provides TTL-based caching for agent tool results.
Replaces the in-memory ToolCache to survive application restarts.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import logger


class PersistentToolCache:
    """
    TTL-based persistent cache for tool results.

    Uses SQLite to store:
    - Cache key (MD5 hash of tool_name + args)
    - Tool name (for filtering/stats)
    - Result (JSON serialized)
    - Timestamp
    - TTL value
    """

    # TTL values in seconds per tool type (tuned for better cache hit rates)
    TTL_CONFIG = {
        "database_stats": 1800,     # 30 minutes - rarely changes during session
        "retrieve_vector": 300,     # 5 minutes - search results stable for a while
        "retrieve_graph": 300,      # 5 minutes - graph results stable for a while
        "lookup_entity": 600,       # 10 minutes - entity data rarely changes
        "compare_entities": 1800,   # 30 minutes - comparison stable for session
    }
    DEFAULT_TTL = 120  # 2 minutes default

    _instance: Optional["PersistentToolCache"] = None

    def __new__(cls, db_path: Optional[str] = None):
        """Singleton pattern - ensure only one cache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the tool cache.

        Args:
            db_path: Path to SQLite database file
        """
        if self._initialized:
            return

        if db_path is None:
            from ..constants import TOOL_CACHE_DB
            db_path = TOOL_CACHE_DB

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._initialized = True
        logger.info(f"Initialized persistent tool cache at {self.db_path}")

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_cache(tool_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON tool_cache(created_at)
            """)
            conn.commit()

    @staticmethod
    def _hash_args(tool_name: str, **kwargs) -> str:
        """Create a hash key from tool name and arguments."""
        args_str = json.dumps(kwargs, sort_keys=True, default=str)
        hash_input = f"{tool_name}:{args_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    @classmethod
    def get(cls, tool_name: str, **kwargs) -> Optional[Any]:
        """
        Get cached result if valid, None otherwise.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
            Cached result if valid and not expired, None otherwise
        """
        instance = cls()
        cache_key = cls._hash_args(tool_name, **kwargs)

        with sqlite3.connect(instance.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT result, created_at, ttl FROM tool_cache
                WHERE cache_key = ?
                """,
                (cache_key,)
            )
            row = cursor.fetchone()

        if row:
            result_json, created_at, ttl = row
            elapsed = time.time() - created_at

            if elapsed < ttl:
                logger.debug(f"Cache HIT for {tool_name} (key={cache_key[:8]}...)")
                instance._record_hit()
                return json.loads(result_json)
            else:
                # Expired - remove from cache
                instance._delete_key(cache_key)
                logger.debug(f"Cache EXPIRED for {tool_name}")

        instance._record_miss()
        return None

    def _record_hit(self) -> None:
        """Record a cache hit in metrics."""
        try:
            from .metrics import get_performance_metrics
            get_performance_metrics().record_cache_hit()
        except Exception:
            pass  # Metrics not available

    def _record_miss(self) -> None:
        """Record a cache miss in metrics."""
        try:
            from .metrics import get_performance_metrics
            get_performance_metrics().record_cache_miss()
        except Exception:
            pass  # Metrics not available

    @classmethod
    def set(cls, tool_name: str, result: Any, **kwargs) -> None:
        """
        Store result in cache.

        Args:
            tool_name: Name of the tool
            result: Result to cache (must be JSON serializable)
            **kwargs: Tool arguments
        """
        instance = cls()
        cache_key = cls._hash_args(tool_name, **kwargs)
        ttl = cls.TTL_CONFIG.get(tool_name, cls.DEFAULT_TTL)

        with sqlite3.connect(instance.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tool_cache
                (cache_key, tool_name, result, created_at, ttl)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    tool_name,
                    json.dumps(result, default=str),
                    time.time(),
                    ttl
                )
            )
            conn.commit()

        logger.debug(f"Cache SET for {tool_name} (key={cache_key[:8]}...)")

    def _delete_key(self, cache_key: str) -> None:
        """Delete a specific cache key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tool_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()

    def clear(self, tool_name: Optional[str] = None) -> None:
        """
        Clear cache for specific tool or all tools.

        Args:
            tool_name: If provided, only clear cache for this tool
        """
        with sqlite3.connect(self.db_path) as conn:
            if tool_name:
                conn.execute("DELETE FROM tool_cache WHERE tool_name = ?", (tool_name,))
                logger.info(f"Cache cleared for {tool_name}")
            else:
                conn.execute("DELETE FROM tool_cache")
                logger.info("Cache cleared for all tools")
            conn.commit()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM tool_cache
                WHERE (? - created_at) >= ttl
                """,
                (current_time,)
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired cache entries")
        return deleted

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        current_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            cursor = conn.execute("SELECT COUNT(*) FROM tool_cache")
            total_count = cursor.fetchone()[0]

            # Valid (non-expired) entries
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM tool_cache
                WHERE (? - created_at) < ttl
                """,
                (current_time,)
            )
            valid_count = cursor.fetchone()[0]

            # By tool
            cursor = conn.execute(
                """
                SELECT tool_name, COUNT(*) as count
                FROM tool_cache
                WHERE (? - created_at) < ttl
                GROUP BY tool_name
                """,
                (current_time,)
            )
            by_tool = dict(cursor.fetchall())

        return {
            "total_entries": total_count,
            "valid_entries": valid_count,
            "expired_entries": total_count - valid_count,
            "by_tool": by_tool,
            "db_path": str(self.db_path),
        }


# Backwards compatibility alias
ToolCache = PersistentToolCache


# Global cache instance (lazy initialization)
_cache_instance: Optional[PersistentToolCache] = None


def get_tool_cache(db_path: Optional[str] = None) -> PersistentToolCache:
    """
    Get the global tool cache instance.

    Args:
        db_path: Database path (uses default from constants if not provided)

    Returns:
        PersistentToolCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = PersistentToolCache(db_path)

    return _cache_instance


__all__ = ["PersistentToolCache", "ToolCache", "get_tool_cache"]
