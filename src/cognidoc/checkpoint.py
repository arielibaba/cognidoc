"""
Checkpoint management for resumable pipeline execution.

Provides persistent state tracking for long-running pipeline stages:
- Entity extraction
- Community summary generation
- Entity embedding computation

Checkpoints are saved atomically to prevent corruption on interruption.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .utils.logger import logger


CHECKPOINT_VERSION = "1.0"


@dataclass
class FailedItem:
    """Record of a failed processing item."""

    item_id: str
    error_type: str  # From ErrorType enum value
    error_message: str
    attempts: int = 1
    last_attempt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailedItem":
        return cls(**data)


@dataclass
class StageCheckpoint:
    """Checkpoint state for a single pipeline stage."""

    total_items: int = 0
    processed_item_ids: List[str] = field(default_factory=list)
    failed_items: List[FailedItem] = field(default_factory=list)
    consecutive_quota_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "processed_item_ids": self.processed_item_ids,
            "failed_items": [f.to_dict() for f in self.failed_items],
            "consecutive_quota_errors": self.consecutive_quota_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageCheckpoint":
        return cls(
            total_items=data.get("total_items", 0),
            processed_item_ids=data.get("processed_item_ids", []),
            failed_items=[
                FailedItem.from_dict(f) for f in data.get("failed_items", [])
            ],
            consecutive_quota_errors=data.get("consecutive_quota_errors", 0),
        )

    def get_processed_ids_set(self) -> Set[str]:
        """Return processed IDs as a set for O(1) lookup."""
        return set(self.processed_item_ids)

    def mark_processed(self, item_id: str) -> None:
        """Mark an item as successfully processed."""
        if item_id not in self.processed_item_ids:
            self.processed_item_ids.append(item_id)
        self.consecutive_quota_errors = 0  # Reset on success

    def mark_failed(
        self, item_id: str, error_type: str, error_message: str
    ) -> None:
        """Mark an item as failed."""
        # Check if already in failed items
        for f in self.failed_items:
            if f.item_id == item_id:
                f.attempts += 1
                f.error_type = error_type
                f.error_message = error_message
                f.last_attempt = datetime.now(timezone.utc).isoformat()
                return

        # Add new failed item
        self.failed_items.append(
            FailedItem(
                item_id=item_id,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def increment_quota_errors(self) -> int:
        """Increment consecutive quota error count and return new value."""
        self.consecutive_quota_errors += 1
        return self.consecutive_quota_errors


@dataclass
class PipelineCheckpoint:
    """
    Complete checkpoint state for the ingestion pipeline.

    Tracks progress across all checkpoint-enabled stages:
    - entity_extraction
    - community_summaries
    - entity_embeddings
    """

    version: str = CHECKPOINT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_stage: str = ""  # Current stage name
    status: str = "running"  # running, interrupted, completed
    interrupt_reason: str = ""  # quota_exhausted, rate_limited, error

    entity_extraction: StageCheckpoint = field(default_factory=StageCheckpoint)
    community_summaries: StageCheckpoint = field(default_factory=StageCheckpoint)
    entity_embeddings: StageCheckpoint = field(default_factory=StageCheckpoint)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pipeline_stage": self.pipeline_stage,
            "status": self.status,
            "interrupt_reason": self.interrupt_reason,
            "entity_extraction": self.entity_extraction.to_dict(),
            "community_summaries": self.community_summaries.to_dict(),
            "entity_embeddings": self.entity_embeddings.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineCheckpoint":
        return cls(
            version=data.get("version", CHECKPOINT_VERSION),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            pipeline_stage=data.get("pipeline_stage", ""),
            status=data.get("status", "running"),
            interrupt_reason=data.get("interrupt_reason", ""),
            entity_extraction=StageCheckpoint.from_dict(
                data.get("entity_extraction", {})
            ),
            community_summaries=StageCheckpoint.from_dict(
                data.get("community_summaries", {})
            ),
            entity_embeddings=StageCheckpoint.from_dict(
                data.get("entity_embeddings", {})
            ),
        )

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_stage(self, stage: str) -> None:
        """Set the current pipeline stage."""
        self.pipeline_stage = stage
        self.update_timestamp()

    def set_interrupted(self, reason: str) -> None:
        """Mark the checkpoint as interrupted."""
        self.status = "interrupted"
        self.interrupt_reason = reason
        self.update_timestamp()

    def set_completed(self) -> None:
        """Mark the checkpoint as completed."""
        self.status = "completed"
        self.interrupt_reason = ""
        self.update_timestamp()

    @classmethod
    def load(cls, path: Path) -> Optional["PipelineCheckpoint"]:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to the checkpoint JSON file

        Returns:
            PipelineCheckpoint if file exists and is valid, None otherwise
        """
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            checkpoint = cls.from_dict(data)

            # Check version compatibility
            if checkpoint.version != CHECKPOINT_VERSION:
                logger.warning(
                    f"Checkpoint version mismatch: {checkpoint.version} != {CHECKPOINT_VERSION}. "
                    "Checkpoint may not be fully compatible."
                )

            logger.info(
                f"Loaded checkpoint: stage={checkpoint.pipeline_stage}, "
                f"status={checkpoint.status}, "
                f"entity_extraction={len(checkpoint.entity_extraction.processed_item_ids)} processed"
            )

            return checkpoint

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse checkpoint file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def save(self, path: Path) -> bool:
        """
        Save checkpoint to disk atomically.

        Uses write-to-temp-then-rename pattern to prevent corruption.

        Args:
            path: Path to save the checkpoint JSON file

        Returns:
            True if saved successfully, False otherwise
        """
        self.update_timestamp()

        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix="checkpoint_",
                dir=path.parent,
            )

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

                # Atomic rename
                os.replace(temp_path, path)

                logger.debug(f"Checkpoint saved: {path}")
                return True

            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    @staticmethod
    def clear(path: Path) -> bool:
        """
        Delete checkpoint file.

        Called after successful pipeline completion.

        Args:
            path: Path to the checkpoint file to delete

        Returns:
            True if deleted or didn't exist, False on error
        """
        try:
            if path.exists():
                path.unlink()
                logger.info(f"Checkpoint cleared: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}")
            return False

    def get_summary(self) -> str:
        """Return a human-readable summary of the checkpoint state."""
        lines = [
            f"Checkpoint Status: {self.status}",
            f"Current Stage: {self.pipeline_stage or 'None'}",
        ]

        if self.interrupt_reason:
            lines.append(f"Interrupt Reason: {self.interrupt_reason}")

        ee = self.entity_extraction
        if ee.total_items > 0:
            lines.append(
                f"Entity Extraction: {len(ee.processed_item_ids)}/{ee.total_items} "
                f"({len(ee.failed_items)} failed, {ee.consecutive_quota_errors} consecutive errors)"
            )

        cs = self.community_summaries
        if cs.total_items > 0:
            lines.append(
                f"Community Summaries: {len(cs.processed_item_ids)}/{cs.total_items} "
                f"({len(cs.failed_items)} failed)"
            )

        emb = self.entity_embeddings
        if emb.total_items > 0:
            lines.append(
                f"Entity Embeddings: {len(emb.processed_item_ids)}/{emb.total_items} "
                f"({len(emb.failed_items)} failed)"
            )

        return "\n".join(lines)
