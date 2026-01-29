"""
Structured logging utility with timing metrics using loguru.

This module provides:
- Structured JSON logging for production
- Colorful console output for development
- Automatic file rotation
- Timing decorators and context managers for performance profiling
- Retrieval metrics tracking
"""

import sys
import time
import functools
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Optional
from datetime import datetime

from loguru import logger

# Remove default handler
logger.remove()

# Configuration
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Console handler (colorful, human-readable)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File handler (JSON format for production, with rotation)
logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

# Metrics file (structured performance data)
logger.add(
    LOG_DIR / "metrics_{time:YYYY-MM-DD}.log",
    format="{message}",
    level="INFO",
    filter=lambda record: record["extra"].get("metrics", False),
    rotation="10 MB",
    retention="30 days",
)


class PipelineTimer:
    """Track timing for pipeline stages."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stages: dict[str, float] = {}
        self.start_time: Optional[float] = None
        self.current_stage: Optional[str] = None
        self.stage_start: Optional[float] = None

    def start(self) -> "PipelineTimer":
        """Start the pipeline timer."""
        self.start_time = time.perf_counter()
        logger.info(f"Pipeline '{self.pipeline_name}' started")
        return self

    def stage(self, name: str) -> "PipelineTimer":
        """Start timing a new stage."""
        now = time.perf_counter()

        # End previous stage if exists
        if self.current_stage and self.stage_start:
            elapsed = now - self.stage_start
            self.stages[self.current_stage] = elapsed
            logger.debug(f"Stage '{self.current_stage}' completed in {elapsed:.2f}s")

        # Start new stage
        self.current_stage = name
        self.stage_start = now
        logger.info(f"Starting stage: {name}")
        return self

    def end(self) -> dict[str, float]:
        """End the pipeline and return timing summary."""
        now = time.perf_counter()

        # End final stage if exists
        if self.current_stage and self.stage_start:
            elapsed = now - self.stage_start
            self.stages[self.current_stage] = elapsed

        total_time = now - (self.start_time or now)

        # Log summary
        summary = {
            "pipeline": self.pipeline_name,
            "total_seconds": round(total_time, 2),
            "stages": {k: round(v, 2) for k, v in self.stages.items()},
            "timestamp": datetime.now().isoformat(),
        }

        logger.bind(metrics=True).info(f"PIPELINE_METRICS: {summary}")

        # Pretty print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline '{self.pipeline_name}' completed")
        logger.info(f"{'='*60}")
        for stage_name, duration in self.stages.items():
            logger.info(f"  {stage_name}: {duration:.2f}s")
        logger.info(f"{'='*60}")
        logger.info(f"  TOTAL: {total_time:.2f}s")
        logger.info(f"{'='*60}\n")

        return summary


@contextmanager
def timer(operation: str):
    """Context manager for timing operations."""
    start = time.perf_counter()
    logger.debug(f"Starting: {operation}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"Completed: {operation} in {elapsed:.2f}s")


def timed(func: Callable) -> Callable:
    """Decorator for timing function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        logger.debug(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise

    return wrapper


async def timed_async(func: Callable) -> Callable:
    """Decorator for timing async function execution."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        logger.debug(f"Calling async {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise

    return wrapper


class RetrievalMetrics:
    """Track and log retrieval metrics."""

    def __init__(self):
        self.queries: list[dict] = []

    def log_retrieval(
        self,
        query: str,
        num_retrieved: int,
        num_after_rerank: int,
        retrieval_time: float,
        rerank_time: float,
        top_scores: Optional[list[float]] = None,
    ):
        """Log a retrieval operation."""
        metrics = {
            "query_length": len(query),
            "num_retrieved": num_retrieved,
            "num_after_rerank": num_after_rerank,
            "retrieval_time_s": round(retrieval_time, 3),
            "rerank_time_s": round(rerank_time, 3),
            "top_scores": top_scores[:5] if top_scores else [],
            "timestamp": datetime.now().isoformat(),
        }
        self.queries.append(metrics)
        logger.bind(metrics=True).info(f"RETRIEVAL_METRICS: {metrics}")

        # Console summary
        logger.info(
            f"Retrieval: {num_retrieved} docs -> {num_after_rerank} after rerank "
            f"({retrieval_time:.2f}s + {rerank_time:.2f}s rerank)"
        )

    def summary(self) -> dict:
        """Get summary statistics."""
        if not self.queries:
            return {}

        total_retrievals = len(self.queries)
        avg_retrieval_time = sum(q["retrieval_time_s"] for q in self.queries) / total_retrievals
        avg_rerank_time = sum(q["rerank_time_s"] for q in self.queries) / total_retrievals

        return {
            "total_queries": total_retrievals,
            "avg_retrieval_time_s": round(avg_retrieval_time, 3),
            "avg_rerank_time_s": round(avg_rerank_time, 3),
        }


# Global instances
retrieval_metrics = RetrievalMetrics()


def log_error_with_context(error: Exception, context: dict):
    """Log an error with additional context."""
    logger.error(f"Error: {error}")
    logger.error(f"Context: {context}")
    logger.exception("Stack trace:")


# Export logger for direct use
__all__ = [
    "logger",
    "PipelineTimer",
    "timer",
    "timed",
    "timed_async",
    "RetrievalMetrics",
    "retrieval_metrics",
    "log_error_with_context",
]
