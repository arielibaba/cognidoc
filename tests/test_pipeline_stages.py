"""
Unit tests for individual pipeline stages.

Tests the extracted helper functions from run_ingestion_pipeline.py:
- _run_document_conversion
- _run_pdf_conversion
- _run_yolo_detection
- _run_content_extraction
- _run_image_descriptions
- _run_chunking
- _run_embeddings
- _run_index_building
- format_ingestion_report
"""

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# =============================================================================
# 1. Document Conversion Stage Tests
# =============================================================================


class TestDocumentConversion:
    """Tests for _run_document_conversion stage."""

    @patch("cognidoc.run_ingestion_pipeline.process_source_documents")
    def test_returns_stats_on_success(self, mock_process):
        from cognidoc.run_ingestion_pipeline import _run_document_conversion

        mock_process.return_value = {
            "pdfs_copied": 3,
            "images_copied": 1,
            "converted": 2,
            "skipped_existing": 0,
            "failed": 0,
        }
        result = _run_document_conversion()
        assert result["pdfs_copied"] == 3
        assert result["converted"] == 2
        mock_process.assert_called_once()

    @patch("cognidoc.run_ingestion_pipeline.process_source_documents")
    def test_passes_source_files_filter(self, mock_process):
        from cognidoc.run_ingestion_pipeline import _run_document_conversion

        mock_process.return_value = {"pdfs_copied": 0, "images_copied": 0, "converted": 0, "skipped_existing": 0, "failed": 0}
        files = ["/tmp/doc1.pdf", "/tmp/doc2.docx"]
        _run_document_conversion(source_files=files)
        _, kwargs = mock_process.call_args
        assert kwargs["source_files"] == files

    @patch("cognidoc.run_ingestion_pipeline.process_source_documents")
    def test_returns_empty_dict_on_failure(self, mock_process):
        from cognidoc.run_ingestion_pipeline import _run_document_conversion

        mock_process.side_effect = Exception("LibreOffice not found")
        result = _run_document_conversion()
        assert result == {}


# =============================================================================
# 2. PDF Conversion Stage Tests
# =============================================================================


class TestPdfConversion:
    """Tests for _run_pdf_conversion stage."""

    @patch("cognidoc.run_ingestion_pipeline.convert_pdf_to_image")
    def test_returns_stats(self, mock_convert):
        from cognidoc.run_ingestion_pipeline import _run_pdf_conversion

        mock_convert.return_value = {"success": 5, "pages": 42, "failed": 0}
        result = _run_pdf_conversion()
        assert result["success"] == 5
        assert result["pages"] == 42

    @patch("cognidoc.run_ingestion_pipeline.convert_pdf_to_image")
    def test_passes_pdf_filter(self, mock_convert):
        from cognidoc.run_ingestion_pipeline import _run_pdf_conversion

        mock_convert.return_value = {"success": 1, "pages": 3}
        _run_pdf_conversion(pdf_filter=["doc1", "doc2"])
        _, kwargs = mock_convert.call_args
        assert kwargs["pdf_filter"] == ["doc1", "doc2"]

    @patch("cognidoc.run_ingestion_pipeline.convert_pdf_to_image")
    def test_propagates_exception(self, mock_convert):
        from cognidoc.run_ingestion_pipeline import _run_pdf_conversion

        mock_convert.side_effect = RuntimeError("poppler not found")
        with pytest.raises(RuntimeError):
            _run_pdf_conversion()


# =============================================================================
# 3. YOLO Detection Stage Tests
# =============================================================================


class TestYoloDetection:
    """Tests for _run_yolo_detection stage."""

    @patch("cognidoc.run_ingestion_pipeline.extract_objects_from_image")
    def test_batch_mode_params(self, mock_yolo):
        from cognidoc.run_ingestion_pipeline import _run_yolo_detection

        mock_yolo.return_value = {"images": 10, "text_regions": 15, "table_regions": 3}
        result = _run_yolo_detection(yolo_batch_size=4, use_yolo_batching=True)
        assert result["text_regions"] == 15
        _, kwargs = mock_yolo.call_args
        assert kwargs["batch_size"] == 4
        assert kwargs["use_batching"] is True

    @patch("cognidoc.run_ingestion_pipeline.extract_objects_from_image")
    def test_sequential_mode(self, mock_yolo):
        from cognidoc.run_ingestion_pipeline import _run_yolo_detection

        mock_yolo.return_value = {}
        _run_yolo_detection(yolo_batch_size=1, use_yolo_batching=False)
        _, kwargs = mock_yolo.call_args
        assert kwargs["use_batching"] is False

    @patch("cognidoc.run_ingestion_pipeline.extract_objects_from_image")
    def test_passes_image_filter(self, mock_yolo):
        from cognidoc.run_ingestion_pipeline import _run_yolo_detection

        mock_yolo.return_value = {}
        _run_yolo_detection(yolo_batch_size=2, use_yolo_batching=True, pdf_filter=["stem1"])
        _, kwargs = mock_yolo.call_args
        assert kwargs["image_filter"] == ["stem1"]


# =============================================================================
# 4. Content Extraction Stage Tests
# =============================================================================


class TestContentExtraction:
    """Tests for _run_content_extraction stage."""

    @patch("cognidoc.run_ingestion_pipeline.parse_image_with_table")
    @patch("cognidoc.run_ingestion_pipeline.parse_image_with_text")
    def test_runs_text_and_table_in_parallel(self, mock_text, mock_table):
        from cognidoc.run_ingestion_pipeline import _run_content_extraction

        mock_text.return_value = {"text_regions": 5}
        mock_table.return_value = {"table_regions": 2}
        text_stats, table_stats = asyncio.run(
            _run_content_extraction("gemini")
        )
        assert text_stats["text_regions"] == 5
        assert table_stats["table_regions"] == 2
        mock_text.assert_called_once()
        mock_table.assert_called_once()

    @patch("cognidoc.run_ingestion_pipeline.parse_image_with_table")
    @patch("cognidoc.run_ingestion_pipeline.parse_image_with_text")
    def test_passes_filter_to_both(self, mock_text, mock_table):
        from cognidoc.run_ingestion_pipeline import _run_content_extraction

        mock_text.return_value = {}
        mock_table.return_value = {}
        asyncio.run(_run_content_extraction("gemini", pdf_filter=["doc1"]))
        assert mock_text.call_args[1]["image_filter"] == ["doc1"]
        assert mock_table.call_args[1]["image_filter"] == ["doc1"]


# =============================================================================
# 5. Image Descriptions Stage Tests
# =============================================================================


class TestImageDescriptions:
    """Tests for _run_image_descriptions stage."""

    @patch("cognidoc.run_ingestion_pipeline.create_image_descriptions_async", new_callable=AsyncMock)
    @patch("cognidoc.run_ingestion_pipeline.load_prompt")
    def test_returns_stats(self, mock_prompt, mock_desc):
        from cognidoc.run_ingestion_pipeline import _run_image_descriptions

        mock_prompt.return_value = "prompt text"
        mock_desc.return_value = {"described": 8, "skipped": 2}
        result = asyncio.run(_run_image_descriptions("gemini"))
        assert result["described"] == 8

    @patch("cognidoc.run_ingestion_pipeline.create_image_descriptions_async", new_callable=AsyncMock)
    @patch("cognidoc.run_ingestion_pipeline.load_prompt")
    def test_returns_empty_on_failure(self, mock_prompt, mock_desc):
        from cognidoc.run_ingestion_pipeline import _run_image_descriptions

        mock_prompt.return_value = "prompt"
        mock_desc.side_effect = Exception("Vision API error")
        result = asyncio.run(_run_image_descriptions("gemini"))
        assert result == {}


# =============================================================================
# 6. Embeddings Stage Tests
# =============================================================================


class TestEmbeddings:
    """Tests for _run_embeddings stage."""

    @patch("cognidoc.run_ingestion_pipeline.create_embeddings")
    def test_returns_stats(self, mock_embed):
        from cognidoc.run_ingestion_pipeline import _run_embeddings

        mock_embed.return_value = {"cached": 10, "to_embed": 5, "embedded": 5}
        result = _run_embeddings(use_cache=True, force_reembed=False)
        assert result["cached"] == 10
        assert result["embedded"] == 5

    @patch("cognidoc.run_ingestion_pipeline.create_embeddings")
    def test_passes_force_reembed(self, mock_embed):
        from cognidoc.run_ingestion_pipeline import _run_embeddings

        mock_embed.return_value = {}
        _run_embeddings(use_cache=False, force_reembed=True)
        _, kwargs = mock_embed.call_args
        assert kwargs["use_cache"] is False
        assert kwargs["force_reembed"] is True

    @patch("cognidoc.run_ingestion_pipeline.create_embeddings")
    def test_passes_incremental_stems(self, mock_embed):
        from cognidoc.run_ingestion_pipeline import _run_embeddings

        mock_embed.return_value = {}
        _run_embeddings(use_cache=True, force_reembed=False, incremental_stems=["doc1", "doc2"])
        _, kwargs = mock_embed.call_args
        assert kwargs["file_filter"] == ["doc1", "doc2"]


# =============================================================================
# 7. Index Building Stage Tests
# =============================================================================


class TestIndexBuilding:
    """Tests for _run_index_building stage."""

    @patch("cognidoc.run_ingestion_pipeline.build_indexes")
    def test_calls_with_recreate(self, mock_build):
        from cognidoc.run_ingestion_pipeline import _run_index_building

        _run_index_building()
        mock_build.assert_called_once_with(recreate=True)

    @patch("cognidoc.run_ingestion_pipeline.build_indexes")
    def test_propagates_exception(self, mock_build):
        from cognidoc.run_ingestion_pipeline import _run_index_building

        mock_build.side_effect = RuntimeError("Qdrant locked")
        with pytest.raises(RuntimeError):
            _run_index_building()


# =============================================================================
# 8. Ingestion Report Tests
# =============================================================================


class TestIngestionReport:
    """Tests for format_ingestion_report."""

    def test_basic_report(self):
        from cognidoc.run_ingestion_pipeline import format_ingestion_report

        stats = {
            "document_conversion": {"pdfs_copied": 2, "converted": 1, "images_copied": 0},
            "pdf_conversion": {"success": 3, "pages": 15},
            "yolo_detection": {"images": 15, "text_regions": 20, "table_regions": 5, "picture_regions": 2},
            "text_extraction": {"extracted": 20},
            "table_extraction": {"extracted": 5},
            "image_description": {"described": 2},
            "embeddings": {"cached": 5, "to_embed": 10, "embedded": 10},
            "graph_extraction": {
                "chunks_processed": 15,
                "entities_extracted": 40,
                "relationships_extracted": 30,
            },
            "graph_building": {
                "total_nodes": 35,
                "total_edges": 28,
                "total_communities": 12,
                "entity_types": {"Person": 5, "Concept": 10},
            },
        }
        timing = {
            "stages": {
                "document_conversion": 1.5,
                "pdf_conversion": 6.0,
                "yolo_detection": 17.0,
            },
            "total_seconds": 90.0,
        }
        report = format_ingestion_report(stats, timing)
        assert "INGESTION REPORT" in report
        assert "15" in report  # pages
        assert "90" in report or "1m" in report  # total time

    def test_empty_stats(self):
        from cognidoc.run_ingestion_pipeline import format_ingestion_report

        stats = {}
        timing = {"stages": {}, "total_seconds": 0.5}
        report = format_ingestion_report(stats, timing)
        assert "INGESTION REPORT" in report

    def test_report_with_missing_sections(self):
        from cognidoc.run_ingestion_pipeline import format_ingestion_report

        stats = {
            "document_conversion": {},
            "pdf_conversion": {"success": 1, "pages": 5},
        }
        timing = {"stages": {"pdf_conversion": 2.0}, "total_seconds": 2.0}
        report = format_ingestion_report(stats, timing)
        assert "INGESTION REPORT" in report


# =============================================================================
# 9. Incremental Ingestion Logic Tests
# =============================================================================


class TestIncrementalIngestion:
    """Tests for incremental ingestion manifest logic."""

    def test_cleanup_intermediate_files(self, tmp_path):
        """Test that _cleanup_intermediate_files removes old files."""
        from cognidoc.run_ingestion_pipeline import _cleanup_intermediate_files

        # Create mock intermediate files in expected directories
        with patch("cognidoc.run_ingestion_pipeline.PROCESSED_DIR", str(tmp_path / "processed")), \
             patch("cognidoc.run_ingestion_pipeline.CHUNKS_DIR", str(tmp_path / "chunks")), \
             patch("cognidoc.run_ingestion_pipeline.EMBEDDINGS_DIR", str(tmp_path / "embeddings")):

            # Create dirs and files
            for d in ["processed", "chunks", "embeddings"]:
                (tmp_path / d).mkdir()
                (tmp_path / d / "mydoc_page1.json").write_text("{}")
                (tmp_path / d / "mydoc_page2.json").write_text("{}")
                (tmp_path / d / "otherdoc_page1.json").write_text("{}")

            _cleanup_intermediate_files("mydoc")

            # mydoc files should be gone, otherdoc should remain
            for d in ["processed", "chunks", "embeddings"]:
                remaining = list((tmp_path / d).glob("*"))
                names = [f.name for f in remaining]
                assert "otherdoc_page1.json" in names
                assert "mydoc_page1.json" not in names
                assert "mydoc_page2.json" not in names
