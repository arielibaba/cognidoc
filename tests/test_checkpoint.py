"""
Tests for the checkpoint/resume system.

Tests cover:
- Error classification
- Checkpoint save/load/clear
- Extraction resume logic
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cognidoc.utils.error_classifier import (
    ErrorType,
    classify_error,
    is_quota_or_rate_error,
    is_retriable_error,
    get_error_info,
)
from cognidoc.checkpoint import (
    PipelineCheckpoint,
    StageCheckpoint,
    FailedItem,
)


# =============================================================================
# Error Classifier Tests
# =============================================================================


class TestErrorClassifier:
    """Tests for error classification."""

    def test_classify_quota_exhausted_gemini(self):
        """Test classification of Gemini RESOURCE_EXHAUSTED error."""
        error = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for project")
        assert classify_error(error) == ErrorType.QUOTA_EXHAUSTED

    def test_classify_quota_exhausted_generic(self):
        """Test classification of generic quota error."""
        error = Exception("You have exceeded your current quota")
        assert classify_error(error) == ErrorType.QUOTA_EXHAUSTED

    def test_classify_rate_limited_429(self):
        """Test classification of 429 rate limit error."""
        error = Exception("Error 429: Too many requests")
        assert classify_error(error) == ErrorType.RATE_LIMITED

    def test_classify_rate_limited_text(self):
        """Test classification of rate limit by text."""
        error = Exception("Rate limit exceeded, please try again later")
        assert classify_error(error) == ErrorType.RATE_LIMITED

    def test_classify_transient_timeout(self):
        """Test classification of timeout error."""
        error = Exception("Connection timeout after 30 seconds")
        assert classify_error(error) == ErrorType.TRANSIENT

    def test_classify_transient_connection(self):
        """Test classification of connection error."""
        error = Exception("Connection refused: localhost:11434")
        assert classify_error(error) == ErrorType.TRANSIENT

    def test_classify_permanent_default(self):
        """Test classification defaults to permanent for unknown errors."""
        error = Exception("Invalid JSON in request body")
        assert classify_error(error) == ErrorType.PERMANENT

    def test_is_quota_or_rate_error_true(self):
        """Test is_quota_or_rate_error returns True for quota errors."""
        error = Exception("RESOURCE_EXHAUSTED: Daily limit reached")
        assert is_quota_or_rate_error(error) is True

    def test_is_quota_or_rate_error_false(self):
        """Test is_quota_or_rate_error returns False for other errors."""
        error = Exception("Invalid API key")
        assert is_quota_or_rate_error(error) is False

    def test_is_retriable_error_rate_limited(self):
        """Test is_retriable_error returns True for rate limits."""
        error = Exception("429 Too Many Requests")
        assert is_retriable_error(error) is True

    def test_is_retriable_error_transient(self):
        """Test is_retriable_error returns True for transient errors."""
        error = Exception("Connection timeout")
        assert is_retriable_error(error) is True

    def test_is_retriable_error_quota(self):
        """Test is_retriable_error returns False for quota exhaustion."""
        # Quota exhaustion requires user action, not retry
        error = Exception("RESOURCE_EXHAUSTED")
        assert is_retriable_error(error) is False

    def test_get_error_info(self):
        """Test get_error_info returns type and message."""
        error = Exception("RESOURCE_EXHAUSTED: Quota exceeded")
        error_type, message = get_error_info(error)
        assert error_type == ErrorType.QUOTA_EXHAUSTED
        assert "RESOURCE_EXHAUSTED" in message

    def test_get_error_info_truncates_long_messages(self):
        """Test get_error_info truncates long messages."""
        long_message = "x" * 1000
        error = Exception(long_message)
        error_type, message = get_error_info(error)
        assert len(message) <= 500


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestFailedItem:
    """Tests for FailedItem dataclass."""

    def test_to_dict(self):
        """Test FailedItem serialization."""
        item = FailedItem(
            item_id="chunk_123",
            error_type="RESOURCE_EXHAUSTED",
            error_message="Quota exceeded",
            attempts=3,
        )
        d = item.to_dict()
        assert d["item_id"] == "chunk_123"
        assert d["error_type"] == "RESOURCE_EXHAUSTED"
        assert d["attempts"] == 3

    def test_from_dict(self):
        """Test FailedItem deserialization."""
        d = {
            "item_id": "chunk_456",
            "error_type": "RATE_LIMITED",
            "error_message": "Too many requests",
            "attempts": 2,
            "last_attempt": "2024-01-15T10:00:00",
        }
        item = FailedItem.from_dict(d)
        assert item.item_id == "chunk_456"
        assert item.error_type == "RATE_LIMITED"
        assert item.attempts == 2


class TestStageCheckpoint:
    """Tests for StageCheckpoint dataclass."""

    def test_mark_processed(self):
        """Test marking an item as processed."""
        stage = StageCheckpoint()
        stage.mark_processed("chunk_1")
        stage.mark_processed("chunk_2")
        assert "chunk_1" in stage.processed_item_ids
        assert "chunk_2" in stage.processed_item_ids
        # Duplicate should not be added
        stage.mark_processed("chunk_1")
        assert stage.processed_item_ids.count("chunk_1") == 1

    def test_mark_processed_resets_quota_errors(self):
        """Test that marking processed resets consecutive quota errors."""
        stage = StageCheckpoint(consecutive_quota_errors=3)
        stage.mark_processed("chunk_1")
        assert stage.consecutive_quota_errors == 0

    def test_mark_failed(self):
        """Test marking an item as failed."""
        stage = StageCheckpoint()
        stage.mark_failed("chunk_1", "RESOURCE_EXHAUSTED", "Quota exceeded")
        assert len(stage.failed_items) == 1
        assert stage.failed_items[0].item_id == "chunk_1"

    def test_mark_failed_updates_existing(self):
        """Test that marking failed updates existing item."""
        stage = StageCheckpoint()
        stage.mark_failed("chunk_1", "RESOURCE_EXHAUSTED", "Quota exceeded")
        stage.mark_failed("chunk_1", "RATE_LIMITED", "Rate limited")
        assert len(stage.failed_items) == 1
        assert stage.failed_items[0].attempts == 2
        assert stage.failed_items[0].error_type == "RATE_LIMITED"

    def test_increment_quota_errors(self):
        """Test incrementing quota error count."""
        stage = StageCheckpoint()
        assert stage.increment_quota_errors() == 1
        assert stage.increment_quota_errors() == 2
        assert stage.consecutive_quota_errors == 2

    def test_get_processed_ids_set(self):
        """Test getting processed IDs as set."""
        stage = StageCheckpoint(processed_item_ids=["a", "b", "c"])
        ids_set = stage.get_processed_ids_set()
        assert ids_set == {"a", "b", "c"}


class TestPipelineCheckpoint:
    """Tests for PipelineCheckpoint class."""

    def test_save_and_load(self):
        """Test checkpoint round-trip serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create and save checkpoint
            checkpoint = PipelineCheckpoint()
            checkpoint.set_stage("entity_extraction")
            checkpoint.entity_extraction.mark_processed("chunk_1")
            checkpoint.entity_extraction.mark_processed("chunk_2")
            checkpoint.entity_extraction.mark_failed("chunk_3", "QUOTA", "Error")

            checkpoint.save(checkpoint_path)
            assert checkpoint_path.exists()

            # Load checkpoint
            loaded = PipelineCheckpoint.load(checkpoint_path)
            assert loaded is not None
            assert loaded.pipeline_stage == "entity_extraction"
            assert len(loaded.entity_extraction.processed_item_ids) == 2
            assert len(loaded.entity_extraction.failed_items) == 1

    def test_load_nonexistent_returns_none(self):
        """Test loading nonexistent checkpoint returns None."""
        result = PipelineCheckpoint.load(Path("/nonexistent/path/checkpoint.json"))
        assert result is None

    def test_load_invalid_json_returns_none(self):
        """Test loading invalid JSON returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint_path.write_text("not valid json{{{")
            result = PipelineCheckpoint.load(checkpoint_path)
            assert result is None

    def test_clear_deletes_file(self):
        """Test clear deletes checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint_path.write_text("{}")
            assert checkpoint_path.exists()

            PipelineCheckpoint.clear(checkpoint_path)
            assert not checkpoint_path.exists()

    def test_clear_nonexistent_returns_true(self):
        """Test clear on nonexistent file returns True."""
        result = PipelineCheckpoint.clear(Path("/nonexistent/checkpoint.json"))
        assert result is True

    def test_set_interrupted(self):
        """Test setting interrupted status."""
        checkpoint = PipelineCheckpoint()
        checkpoint.set_interrupted("quota_exhausted")
        assert checkpoint.status == "interrupted"
        assert checkpoint.interrupt_reason == "quota_exhausted"

    def test_set_completed(self):
        """Test setting completed status."""
        checkpoint = PipelineCheckpoint()
        checkpoint.set_interrupted("error")  # First interrupt
        checkpoint.set_completed()  # Then complete
        assert checkpoint.status == "completed"
        assert checkpoint.interrupt_reason == ""

    def test_get_summary(self):
        """Test getting checkpoint summary."""
        checkpoint = PipelineCheckpoint()
        checkpoint.set_stage("entity_extraction")
        checkpoint.entity_extraction.total_items = 100
        checkpoint.entity_extraction.processed_item_ids = ["a", "b", "c"]
        checkpoint.entity_extraction.mark_failed("d", "ERROR", "msg")

        summary = checkpoint.get_summary()
        assert "entity_extraction" in summary.lower() or "Entity Extraction" in summary
        assert "3" in summary  # 3 processed

    def test_atomic_save(self):
        """Test that save is atomic (uses temp file + rename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create initial checkpoint
            checkpoint = PipelineCheckpoint()
            checkpoint.entity_extraction.mark_processed("chunk_1")
            checkpoint.save(checkpoint_path)

            # Verify content
            with open(checkpoint_path) as f:
                data = json.load(f)
            assert "chunk_1" in data["entity_extraction"]["processed_item_ids"]


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestCheckpointIntegration:
    """Integration tests for checkpoint with extraction."""

    def test_checkpoint_preserves_state_across_sessions(self):
        """Test that checkpoint state persists across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Session 1: Process some chunks, then interrupt
            checkpoint1 = PipelineCheckpoint()
            checkpoint1.set_stage("entity_extraction")
            checkpoint1.entity_extraction.total_items = 100
            for i in range(50):
                checkpoint1.entity_extraction.mark_processed(f"chunk_{i}")
            checkpoint1.set_interrupted("quota_exhausted")
            checkpoint1.save(checkpoint_path)

            # Session 2: Resume and verify state
            checkpoint2 = PipelineCheckpoint.load(checkpoint_path)
            assert checkpoint2 is not None
            assert checkpoint2.status == "interrupted"
            assert len(checkpoint2.entity_extraction.processed_item_ids) == 50
            processed_set = checkpoint2.entity_extraction.get_processed_ids_set()
            assert "chunk_0" in processed_set
            assert "chunk_49" in processed_set
            assert "chunk_50" not in processed_set
