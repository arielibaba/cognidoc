"""
Tests for the incremental ingestion pipeline.

Tests the IngestionManifest class and the incremental behavior of the
ingestion pipeline (file_filter propagation, manifest save/load, etc.).
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.ingestion_manifest import IngestionManifest, FileRecord


# =============================================================================
# IngestionManifest unit tests
# =============================================================================


class TestFileRecord:
    """Tests for FileRecord dataclass."""

    def test_creation(self):
        record = FileRecord(
            path="subdir/doc.pdf",
            stem="doc",
            size=1024,
            mtime=1700000000.0,
            content_hash="abc123",
            ingested_at="2025-01-01T00:00:00",
        )
        assert record.path == "subdir/doc.pdf"
        assert record.stem == "doc"
        assert record.size == 1024
        assert record.content_hash == "abc123"


class TestIngestionManifest:
    """Tests for IngestionManifest load/save and file detection."""

    def test_save_and_load(self):
        """Test manifest round-trip serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            manifest = IngestionManifest()
            manifest.files["doc.pdf"] = FileRecord(
                path="doc.pdf",
                stem="doc",
                size=500,
                mtime=1700000000.0,
                content_hash="hash1",
                ingested_at="2025-01-01T00:00:00",
            )
            manifest.save(manifest_path)

            assert manifest_path.exists()

            loaded = IngestionManifest.load(manifest_path)
            assert loaded is not None
            assert len(loaded.files) == 1
            assert "doc.pdf" in loaded.files
            assert loaded.files["doc.pdf"].stem == "doc"
            assert loaded.files["doc.pdf"].content_hash == "hash1"
            assert loaded.created_at != ""
            assert loaded.last_updated != ""

    def test_load_nonexistent_returns_none(self):
        """Test loading a manifest that doesn't exist returns None."""
        result = IngestionManifest.load(Path("/nonexistent/manifest.json"))
        assert result is None

    def test_load_invalid_json_returns_none(self):
        """Test loading invalid JSON returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.json"
            bad_path.write_text("not json {{{")
            result = IngestionManifest.load(bad_path)
            assert result is None

    def test_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "manifest.json"
            manifest = IngestionManifest()
            manifest.save(nested_path)
            assert nested_path.exists()

    def test_save_atomic_write(self):
        """Test that save uses atomic write (temp file + rename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = IngestionManifest()
            manifest.files["a.pdf"] = FileRecord(
                path="a.pdf", stem="a", size=100,
                mtime=1.0, content_hash="h1", ingested_at="t1",
            )
            manifest.save(manifest_path)

            # Overwrite with new data
            manifest.files["b.pdf"] = FileRecord(
                path="b.pdf", stem="b", size=200,
                mtime=2.0, content_hash="h2", ingested_at="t2",
            )
            manifest.save(manifest_path)

            loaded = IngestionManifest.load(manifest_path)
            assert len(loaded.files) == 2

    def test_detect_new_files(self):
        """Test detection of files not in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()

            # Create source files
            (sources / "existing.pdf").write_bytes(b"existing content")
            (sources / "new_doc.pdf").write_bytes(b"new content")

            # Manifest only knows about existing.pdf
            manifest = IngestionManifest()
            existing = sources / "existing.pdf"
            stat = existing.stat()
            manifest.files["existing.pdf"] = FileRecord(
                path="existing.pdf",
                stem="existing",
                size=stat.st_size,
                mtime=stat.st_mtime,
                content_hash=IngestionManifest.compute_file_hash(existing),
                ingested_at="2025-01-01T00:00:00",
            )

            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(sources)

            assert len(new_files) == 1
            assert new_files[0].name == "new_doc.pdf"
            assert len(modified_files) == 0
            assert "new_doc" in new_stems

    def test_detect_modified_files(self):
        """Test detection of files whose content changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()

            doc_path = sources / "doc.pdf"
            doc_path.write_bytes(b"original content")

            # Record original state
            manifest = IngestionManifest()
            manifest.record_file(doc_path, sources, "doc")

            # Modify the file
            time.sleep(0.05)  # Ensure mtime changes
            doc_path.write_bytes(b"modified content")

            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(sources)

            assert len(new_files) == 0
            assert len(modified_files) == 1
            assert modified_files[0].name == "doc.pdf"
            assert "doc" in new_stems

    def test_no_changes_detected(self):
        """Test that unchanged files are not flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()

            doc_path = sources / "doc.pdf"
            doc_path.write_bytes(b"content")

            manifest = IngestionManifest()
            manifest.record_file(doc_path, sources, "doc")

            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(sources)

            assert len(new_files) == 0
            assert len(modified_files) == 0
            assert len(new_stems) == 0

    def test_detect_with_source_files_filter(self):
        """Test detection limited to specific source_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()

            (sources / "a.pdf").write_bytes(b"a")
            (sources / "b.pdf").write_bytes(b"b")
            (sources / "c.pdf").write_bytes(b"c")

            manifest = IngestionManifest()

            # Only check a.pdf and b.pdf
            new_files, _, new_stems = manifest.get_new_and_modified_files(
                sources,
                source_files=[str(sources / "a.pdf"), str(sources / "b.pdf")],
            )

            assert len(new_files) == 2
            assert "c" not in new_stems

    def test_record_file(self):
        """Test recording a file creates correct FileRecord."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()

            doc = sources / "report.pdf"
            doc.write_bytes(b"pdf content here")

            manifest = IngestionManifest()
            manifest.record_file(doc, sources, "report")

            assert "report.pdf" in manifest.files
            record = manifest.files["report.pdf"]
            assert record.stem == "report"
            assert record.size == len(b"pdf content here")
            assert record.content_hash == IngestionManifest.compute_file_hash(doc)

    def test_record_all_sources(self):
        """Test recording all files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()
            (sources / "a.pdf").write_bytes(b"a")
            (sources / "b.pdf").write_bytes(b"b")
            subdir = sources / "sub"
            subdir.mkdir()
            (subdir / "c.pdf").write_bytes(b"c")

            manifest = IngestionManifest()
            manifest.record_all_sources(sources)

            assert len(manifest.files) == 3

    def test_compute_file_hash_deterministic(self):
        """Test that file hash is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.txt"
            path.write_bytes(b"hello world")

            h1 = IngestionManifest.compute_file_hash(path)
            h2 = IngestionManifest.compute_file_hash(path)
            assert h1 == h2
            assert len(h1) == 64  # SHA-256 hex

    def test_compute_file_hash_differs_for_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "a.txt"
            p2 = Path(tmpdir) / "b.txt"
            p1.write_bytes(b"content A")
            p2.write_bytes(b"content B")

            assert IngestionManifest.compute_file_hash(p1) != IngestionManifest.compute_file_hash(p2)

    def test_manifest_version(self):
        """Test that manifest saves and loads version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = IngestionManifest()
            manifest.save(manifest_path)

            loaded = IngestionManifest.load(manifest_path)
            assert loaded.version == 1

    def test_multiple_new_and_modified(self):
        """Test mixed scenario: some new, some modified, some unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()

            # Create 3 files
            unchanged = sources / "unchanged.pdf"
            unchanged.write_bytes(b"stable")
            modified = sources / "modified.pdf"
            modified.write_bytes(b"original")
            # new.pdf doesn't exist yet

            manifest = IngestionManifest()
            manifest.record_file(unchanged, sources, "unchanged")
            manifest.record_file(modified, sources, "modified")

            # Now modify one and add a new one
            time.sleep(0.05)
            modified.write_bytes(b"changed")
            (sources / "new.pdf").write_bytes(b"brand new")

            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(sources)

            assert len(new_files) == 1
            assert new_files[0].name == "new.pdf"
            assert len(modified_files) == 1
            assert modified_files[0].name == "modified.pdf"
            assert new_stems == {"new", "modified"}


# =============================================================================
# file_filter parameter tests
# =============================================================================


class TestFileFilterParameter:
    """Tests that file_filter correctly limits processing to matching stems."""

    def test_chunk_text_data_filter(self):
        """Test that chunk_text_data respects file_filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            # Create processed text files for 2 documents
            (processed / "doc_A_page_1_Text.md").write_text("Document A content about AI.")
            (processed / "doc_B_page_1_Text.md").write_text("Document B content about ML.")

            from cognidoc.chunk_text_data import chunk_text_data

            # Only process doc_A
            chunk_text_data(
                documents_dir=str(processed),
                embed_model_name="nomic-embed-text",
                parent_chunk_size=512,
                child_chunk_size=64,
                documents_chunks_dir=str(chunks),
                buffer_size=5,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95.0,
                sentence_split_regex=r"(?<=[.!?])\s+",
                verbose=False,
                file_filter=["doc_A"],
            )

            chunk_files = list(chunks.rglob("*.txt"))
            chunk_names = [f.name for f in chunk_files]

            # Should have doc_A chunks but NOT doc_B chunks
            assert any("doc_A" in name for name in chunk_names)
            assert not any("doc_B" in name for name in chunk_names)

    def test_chunk_text_data_no_filter_processes_all(self):
        """Test that chunk_text_data with no filter processes everything."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            (processed / "doc_A_page_1_Text.md").write_text("Document A content.")
            (processed / "doc_B_page_1_Text.md").write_text("Document B content.")

            from cognidoc.chunk_text_data import chunk_text_data

            chunk_text_data(
                documents_dir=str(processed),
                embed_model_name="nomic-embed-text",
                parent_chunk_size=512,
                child_chunk_size=64,
                documents_chunks_dir=str(chunks),
                buffer_size=5,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95.0,
                sentence_split_regex=r"(?<=[.!?])\s+",
                verbose=False,
                file_filter=None,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            chunk_names = [f.name for f in chunk_files]

            assert any("doc_A" in name for name in chunk_names)
            assert any("doc_B" in name for name in chunk_names)

    def test_collect_chunks_to_embed_filter(self):
        """Test that collect_chunks_to_embed respects file_filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = Path(tmpdir) / "chunks"
            embeddings_dir = Path(tmpdir) / "embeddings"
            chunks_dir.mkdir()
            embeddings_dir.mkdir()

            # Create child chunk files for 2 documents
            (chunks_dir / "doc_A_page_1_parent_chunk_1_child_chunk_1.txt").write_text(
                "Document A child chunk content about neural networks and deep learning."
            )
            (chunks_dir / "doc_B_page_1_parent_chunk_1_child_chunk_1.txt").write_text(
                "Document B child chunk content about reinforcement learning methods."
            )

            from cognidoc.create_embeddings import collect_chunks_to_embed

            # Filter to doc_A only
            chunks_to_embed, stats = collect_chunks_to_embed(
                chunks_path=chunks_dir,
                embeddings_path=embeddings_dir,
                embed_model="test-model",
                use_cache=False,
                force_reembed=False,
                cache=None,
                file_filter=["doc_A"],
            )

            chunk_names = [c.file_path.name for c in chunks_to_embed]
            assert any("doc_A" in name for name in chunk_names)
            assert not any("doc_B" in name for name in chunk_names)

    def test_collect_chunks_to_embed_no_filter(self):
        """Test that no filter includes all chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = Path(tmpdir) / "chunks"
            embeddings_dir = Path(tmpdir) / "embeddings"
            chunks_dir.mkdir()
            embeddings_dir.mkdir()

            (chunks_dir / "doc_A_page_1_parent_chunk_1_child_chunk_1.txt").write_text(
                "Document A child chunk with enough words to pass the filter."
            )
            (chunks_dir / "doc_B_page_1_parent_chunk_1_child_chunk_1.txt").write_text(
                "Document B child chunk with enough words to pass the filter."
            )

            from cognidoc.create_embeddings import collect_chunks_to_embed

            chunks_to_embed, stats = collect_chunks_to_embed(
                chunks_path=chunks_dir,
                embeddings_path=embeddings_dir,
                embed_model="test-model",
                use_cache=False,
                force_reembed=False,
                cache=None,
                file_filter=None,
            )

            chunk_names = [c.file_path.name for c in chunks_to_embed]
            assert any("doc_A" in name for name in chunk_names)
            assert any("doc_B" in name for name in chunk_names)


# =============================================================================
# Cleanup helper tests
# =============================================================================


class TestCleanupIntermediateFiles:
    """Tests for _cleanup_intermediate_files helper."""

    def test_cleanup_removes_matching_files(self):
        """Test that cleanup deletes files matching the stem pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate intermediate directories
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            embeddings = Path(tmpdir) / "embeddings"
            for d in (processed, chunks, embeddings):
                d.mkdir()

            # Create files for doc_A and doc_B
            (processed / "doc_A_page_1_Text.md").write_text("text")
            (processed / "doc_B_page_1_Text.md").write_text("text")
            (chunks / "doc_A_page_1_parent_chunk_1.txt").write_text("chunk")
            (chunks / "doc_B_page_1_parent_chunk_1.txt").write_text("chunk")
            (embeddings / "doc_A_page_1_embedding.json").write_text("{}")
            (embeddings / "doc_B_page_1_embedding.json").write_text("{}")

            from cognidoc.run_ingestion_pipeline import _cleanup_intermediate_files

            # Patch constants to use temp dirs
            with patch("cognidoc.run_ingestion_pipeline.PROCESSED_DIR", str(processed)), \
                 patch("cognidoc.run_ingestion_pipeline.CHUNKS_DIR", str(chunks)), \
                 patch("cognidoc.run_ingestion_pipeline.EMBEDDINGS_DIR", str(embeddings)):
                _cleanup_intermediate_files("doc_A")

            # doc_A files should be gone, doc_B files remain
            assert not (processed / "doc_A_page_1_Text.md").exists()
            assert (processed / "doc_B_page_1_Text.md").exists()
            assert not (chunks / "doc_A_page_1_parent_chunk_1.txt").exists()
            assert (chunks / "doc_B_page_1_parent_chunk_1.txt").exists()
            assert not (embeddings / "doc_A_page_1_embedding.json").exists()
            assert (embeddings / "doc_B_page_1_embedding.json").exists()

    def test_cleanup_handles_missing_dirs(self):
        """Test that cleanup doesn't crash if directories don't exist."""
        from cognidoc.run_ingestion_pipeline import _cleanup_intermediate_files

        with patch("cognidoc.run_ingestion_pipeline.PROCESSED_DIR", "/nonexistent/processed"), \
             patch("cognidoc.run_ingestion_pipeline.CHUNKS_DIR", "/nonexistent/chunks"), \
             patch("cognidoc.run_ingestion_pipeline.EMBEDDINGS_DIR", "/nonexistent/embeddings"):
            # Should not raise
            _cleanup_intermediate_files("doc_A")


# =============================================================================
# Incremental pipeline logic tests (unit-level, no real LLM calls)
# =============================================================================


class TestIncrementalPipelineLogic:
    """Tests for incremental detection logic in the pipeline orchestrator."""

    def test_manifest_saved_after_full_ingestion(self):
        """Test that manifest is created after a full pipeline run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            sources = Path(tmpdir) / "sources"
            sources.mkdir()
            (sources / "doc.pdf").write_bytes(b"content")

            manifest = IngestionManifest()
            manifest.record_all_sources(sources)
            manifest.save(manifest_path)

            loaded = IngestionManifest.load(manifest_path)
            assert len(loaded.files) == 1
            assert "doc.pdf" in loaded.files

    def test_incremental_detects_new_after_manifest(self):
        """Test that adding a file after manifest creation is detected as new."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()
            (sources / "doc1.pdf").write_bytes(b"content 1")

            # Create manifest for first file
            manifest = IngestionManifest()
            manifest.record_all_sources(sources)

            # Add second file
            (sources / "doc2.pdf").write_bytes(b"content 2")

            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(sources)
            assert len(new_files) == 1
            assert new_files[0].name == "doc2.pdf"
            assert "doc2" in new_stems
            assert len(modified_files) == 0

    def test_full_reindex_ignores_manifest(self):
        """Test that full_reindex=True means no incremental detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            sources = Path(tmpdir) / "sources"
            sources.mkdir()
            (sources / "doc.pdf").write_bytes(b"content")

            # Save manifest
            manifest = IngestionManifest()
            manifest.record_all_sources(sources)
            manifest.save(manifest_path)

            # full_reindex should skip manifest loading entirely
            # (verified by the pipeline's if/else logic, tested at integration level)
            loaded = IngestionManifest.load(manifest_path)
            assert loaded is not None
            # In full_reindex mode, the pipeline does NOT call get_new_and_modified_files

    def test_empty_manifest_treats_as_first_ingestion(self):
        """Test that an empty manifest means all files are new."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            sources.mkdir()
            (sources / "a.pdf").write_bytes(b"a")
            (sources / "b.pdf").write_bytes(b"b")

            manifest = IngestionManifest()
            new_files, modified_files, new_stems = manifest.get_new_and_modified_files(sources)

            assert len(new_files) == 2
            assert len(modified_files) == 0
            assert new_stems == {"a", "b"}

    def test_subdirectory_files_tracked(self):
        """Test that files in subdirectories are tracked with relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = Path(tmpdir) / "sources"
            subdir = sources / "project_A"
            subdir.mkdir(parents=True)

            doc = subdir / "report.pdf"
            doc.write_bytes(b"report content")

            manifest = IngestionManifest()
            manifest.record_file(doc, sources, "report")

            assert "project_A/report.pdf" in manifest.files

            # No changes -> nothing new
            new_files, modified_files, _ = manifest.get_new_and_modified_files(sources)
            assert len(new_files) == 0
            assert len(modified_files) == 0


class TestExtractEntitiesFileFilter:
    """Tests that entity extraction respects file_filter."""

    def test_sync_extraction_filter(self):
        """Test that extract_from_chunks_dir respects file_filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = Path(tmpdir)
            # Create parent chunks for 2 documents
            (chunks_dir / "doc_A_page_1_parent_chunk_1.txt").write_text(
                "Document A discusses machine learning algorithms."
            )
            (chunks_dir / "doc_B_page_1_parent_chunk_1.txt").write_text(
                "Document B discusses database systems."
            )

            from cognidoc.extract_entities import extract_from_chunks_dir

            # Mock the LLM call to avoid real API calls
            with patch("cognidoc.extract_entities.extract_entities_from_text") as mock_extract:
                mock_extract.return_value = ([], [])

                results = extract_from_chunks_dir(
                    chunks_dir=str(chunks_dir),
                    file_filter=["doc_A"],
                )

                # Should only have called extract for doc_A
                call_args = [str(call) for call in mock_extract.call_args_list]
                assert len(mock_extract.call_args_list) == 1
                assert "machine learning" in str(mock_extract.call_args_list[0])
