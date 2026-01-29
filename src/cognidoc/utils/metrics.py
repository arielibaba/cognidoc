"""
Performance metrics collection and storage.

This module provides centralized metrics collection for query performance,
cache statistics, and agent execution tracking.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import logger


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""

    # Routing info
    path: str  # fast, enhanced, agent
    query_type: Optional[str] = None  # factual, relational, exploratory, procedural
    complexity_score: Optional[float] = None

    # Timing (milliseconds)
    total_time_ms: float = 0.0
    rewrite_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    llm_time_ms: float = 0.0

    # Cache stats
    cache_hits: int = 0
    cache_misses: int = 0

    # Agent stats
    agent_steps: Optional[int] = None
    tools_used: List[str] = field(default_factory=list)


class PerformanceMetrics:
    """
    Centralized performance metrics collection.

    Stores metrics in SQLite for historical analysis and dashboard display.
    """

    _instance: Optional["PerformanceMetrics"] = None

    def __new__(cls, db_path: Optional[str] = None):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize metrics storage.

        Args:
            db_path: Path to SQLite database
        """
        if self._initialized:
            return

        if db_path is None:
            from ..constants import METRICS_DB

            db_path = METRICS_DB

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._initialized = True

        # In-memory counters for current session
        self._session_cache_hits = 0
        self._session_cache_misses = 0

        logger.info(f"Initialized performance metrics at {self.db_path}")

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    query_hash TEXT NOT NULL,

                    -- Routing
                    path TEXT NOT NULL,
                    query_type TEXT,
                    complexity_score REAL,

                    -- Timing (ms)
                    total_time_ms REAL NOT NULL,
                    rewrite_time_ms REAL,
                    retrieval_time_ms REAL,
                    rerank_time_ms REAL,
                    llm_time_ms REAL,

                    -- Cache
                    cache_hits INTEGER DEFAULT 0,
                    cache_misses INTEGER DEFAULT 0,

                    -- Agent
                    agent_steps INTEGER,
                    tools_used TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON query_metrics(timestamp)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_path ON query_metrics(path)
            """
            )
            conn.commit()

    def log_query(self, query: str, metrics: QueryMetrics) -> None:
        """
        Log metrics for a query execution.

        Args:
            query: The user query (will be hashed)
            metrics: QueryMetrics dataclass with timing and stats
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO query_metrics (
                    timestamp, query_hash, path, query_type, complexity_score,
                    total_time_ms, rewrite_time_ms, retrieval_time_ms,
                    rerank_time_ms, llm_time_ms,
                    cache_hits, cache_misses,
                    agent_steps, tools_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    query_hash,
                    metrics.path,
                    metrics.query_type,
                    metrics.complexity_score,
                    metrics.total_time_ms,
                    metrics.rewrite_time_ms,
                    metrics.retrieval_time_ms,
                    metrics.rerank_time_ms,
                    metrics.llm_time_ms,
                    metrics.cache_hits,
                    metrics.cache_misses,
                    metrics.agent_steps,
                    json.dumps(metrics.tools_used) if metrics.tools_used else None,
                ),
            )
            conn.commit()

        logger.debug(
            f"Logged metrics for query {query_hash}: {metrics.path}, {metrics.total_time_ms:.0f}ms"
        )

    def record_cache_hit(self) -> None:
        """Record a cache hit in the current session."""
        self._session_cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss in the current session."""
        self._session_cache_misses += 1

    def get_session_cache_stats(self) -> Dict[str, Any]:
        """Get cache stats for the current session."""
        total = self._session_cache_hits + self._session_cache_misses
        hit_rate = (self._session_cache_hits / total * 100) if total > 0 else 0

        return {
            "hits": self._session_cache_hits,
            "misses": self._session_cache_misses,
            "total": total,
            "hit_rate": round(hit_rate, 1),
        }

    def reset_session_cache_stats(self) -> None:
        """Reset session cache counters."""
        self._session_cache_hits = 0
        self._session_cache_misses = 0

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all recorded queries.

        Returns:
            Dictionary with global stats
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total queries
            cursor = conn.execute("SELECT COUNT(*) as count FROM query_metrics")
            total_queries = cursor.fetchone()["count"]

            if total_queries == 0:
                return {
                    "total_queries": 0,
                    "avg_latency_ms": 0,
                    "path_distribution": {},
                    "query_type_distribution": {},
                    "cache_hit_rate": 0,
                }

            # Average latency
            cursor = conn.execute("SELECT AVG(total_time_ms) as avg FROM query_metrics")
            avg_latency = cursor.fetchone()["avg"] or 0

            # Latency by path
            cursor = conn.execute(
                """
                SELECT path, AVG(total_time_ms) as avg_latency, COUNT(*) as count
                FROM query_metrics
                GROUP BY path
            """
            )
            path_stats = {
                row["path"]: {"avg_latency_ms": round(row["avg_latency"], 1), "count": row["count"]}
                for row in cursor.fetchall()
            }

            # Query type distribution
            cursor = conn.execute(
                """
                SELECT query_type, COUNT(*) as count
                FROM query_metrics
                WHERE query_type IS NOT NULL
                GROUP BY query_type
            """
            )
            query_types = {row["query_type"]: row["count"] for row in cursor.fetchall()}

            # Cache stats
            cursor = conn.execute(
                """
                SELECT SUM(cache_hits) as hits, SUM(cache_misses) as misses
                FROM query_metrics
            """
            )
            row = cursor.fetchone()
            total_hits = row["hits"] or 0
            total_misses = row["misses"] or 0
            total_cache_ops = total_hits + total_misses
            cache_hit_rate = (total_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0

            # Agent stats
            cursor = conn.execute(
                """
                SELECT AVG(agent_steps) as avg_steps, COUNT(*) as agent_queries
                FROM query_metrics
                WHERE agent_steps IS NOT NULL
            """
            )
            row = cursor.fetchone()
            avg_agent_steps = row["avg_steps"] or 0
            agent_queries = row["agent_queries"] or 0

        return {
            "total_queries": total_queries,
            "avg_latency_ms": round(avg_latency, 1),
            "path_distribution": path_stats,
            "query_type_distribution": query_types,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "total_cache_hits": total_hits,
            "total_cache_misses": total_misses,
            "avg_agent_steps": round(avg_agent_steps, 1),
            "agent_queries": agent_queries,
        }

    def get_latency_over_time(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get latency data over time for charting.

        Args:
            limit: Maximum number of data points

        Returns:
            List of {timestamp, total_time_ms, path} dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT timestamp, total_time_ms, path
                FROM query_metrics
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {
                "timestamp": datetime.fromtimestamp(row["timestamp"]).isoformat(),
                "total_time_ms": row["total_time_ms"],
                "path": row["path"],
            }
            for row in reversed(rows)  # Oldest first for charts
        ]

    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent queries for the dashboard table.

        Args:
            limit: Maximum number of queries

        Returns:
            List of query details
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    timestamp, query_hash, path, query_type,
                    total_time_ms, cache_hits, cache_misses,
                    agent_steps, tools_used
                FROM query_metrics
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {
                "timestamp": datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "query_hash": row["query_hash"],
                "path": row["path"],
                "query_type": row["query_type"] or "-",
                "latency_ms": round(row["total_time_ms"], 0),
                "cache_hits": row["cache_hits"],
                "cache_misses": row["cache_misses"],
                "agent_steps": row["agent_steps"] or "-",
                "tools": json.loads(row["tools_used"]) if row["tools_used"] else [],
            }
            for row in rows
        ]

    def get_path_distribution(self) -> Dict[str, int]:
        """Get query count by path for pie chart."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT path, COUNT(*) as count
                FROM query_metrics
                GROUP BY path
            """
            )
            return dict(cursor.fetchall())

    def clear(self) -> None:
        """Clear all metrics data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM query_metrics")
            conn.commit()
        self._session_cache_hits = 0
        self._session_cache_misses = 0
        logger.info("Cleared all performance metrics")

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export all metrics to CSV format.

        Args:
            filepath: Optional path to save CSV file

        Returns:
            CSV content as string (also saves to file if filepath provided)
        """
        import csv
        from io import StringIO

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    datetime(timestamp, 'unixepoch', 'localtime') as datetime,
                    query_hash, path, query_type, complexity_score,
                    total_time_ms, rewrite_time_ms, retrieval_time_ms,
                    rerank_time_ms, llm_time_ms,
                    cache_hits, cache_misses,
                    agent_steps, tools_used
                FROM query_metrics
                ORDER BY timestamp DESC
            """
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        # Get column names
        columns = rows[0].keys()

        # Write to StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[col] for col in columns])

        csv_content = output.getvalue()

        # Save to file if path provided
        if filepath:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)
            logger.info(f"Exported {len(rows)} metrics to {filepath}")

        return csv_content

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export all metrics to JSON format.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON content as string (also saves to file if filepath provided)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    datetime(timestamp, 'unixepoch', 'localtime') as datetime,
                    timestamp, query_hash, path, query_type, complexity_score,
                    total_time_ms, rewrite_time_ms, retrieval_time_ms,
                    rerank_time_ms, llm_time_ms,
                    cache_hits, cache_misses,
                    agent_steps, tools_used
                FROM query_metrics
                ORDER BY timestamp DESC
            """
            )
            rows = cursor.fetchall()

        # Convert to list of dicts
        data = []
        for row in rows:
            record = dict(row)
            # Parse tools_used JSON if present
            if record.get("tools_used"):
                try:
                    record["tools_used"] = json.loads(record["tools_used"])
                except json.JSONDecodeError:
                    pass
            data.append(record)

        # Include global stats
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_records": len(data),
            "global_stats": self.get_global_stats(),
            "queries": data,
        }

        json_content = json.dumps(export_data, indent=2, ensure_ascii=False)

        # Save to file if path provided
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_content)
            logger.info(f"Exported {len(data)} metrics to {filepath}")

        return json_content


# Global metrics instance (lazy initialization)
_metrics_instance: Optional[PerformanceMetrics] = None


def get_performance_metrics(db_path: Optional[str] = None) -> PerformanceMetrics:
    """
    Get the global performance metrics instance.

    Args:
        db_path: Database path (uses default from constants if not provided)

    Returns:
        PerformanceMetrics instance
    """
    global _metrics_instance

    if _metrics_instance is None:
        _metrics_instance = PerformanceMetrics(db_path)

    return _metrics_instance


__all__ = ["QueryMetrics", "PerformanceMetrics", "get_performance_metrics"]
