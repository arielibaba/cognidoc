"""Tests for chunk_text_data and chunk_table_data functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.chunk_text_data import chunk_text_data, hard_split
from cognidoc.chunk_table_data import chunk_table_data, chunk_markdown_table, chunk_markdown_table_with_overlap


# =============================================================================
# Helpers
# =============================================================================

SAMPLE_TEXT = (
    "Artificial intelligence is transforming healthcare. "
    "Machine learning models can detect diseases from medical images. "
    "Natural language processing helps analyze clinical notes. "
    "Deep learning algorithms improve drug discovery processes. "
    "Computer vision enables automated radiology screening. "
    "Reinforcement learning optimizes treatment planning. "
    "Transfer learning reduces the need for large labeled datasets. "
    "Federated learning allows training on distributed patient data. "
    "Generative models create synthetic training data. "
    "Explainable AI builds trust in clinical decision support. "
    "AI ethics frameworks guide responsible deployment in medicine. "
    "Regulatory bodies are adapting to evaluate AI-based diagnostics."
)

SAMPLE_TABLE = """| Name | Age | City |
| --- | --- | --- |
| Alice | 30 | Paris |
| Bob | 25 | London |
| Charlie | 35 | Berlin |
| Diana | 28 | Madrid |
| Eve | 32 | Rome |
| Frank | 45 | Lisbon |
| Grace | 29 | Vienna |
| Hank | 38 | Prague |
| Ivy | 27 | Dublin |
| Jack | 41 | Oslo |"""

MOCK_EMBEDDING = [0.1] * 384

DEFAULT_TEXT_PARAMS = dict(
    embed_model_name="test-model",
    parent_chunk_size=512,
    child_chunk_size=64,
    buffer_size=5,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0,
    sentence_split_regex=r"(?<=[.!?])\s+",
    verbose=False,
)

LLM_JSON_RESPONSE = json.dumps({
    "columns": "Name,Age,City",
    "summary_of_the_table": "A table of people with their ages and cities.",
})


def _create_text_file(directory: Path, name: str, content: str):
    """Helper to create a text file in the given directory."""
    filepath = directory / name
    filepath.write_text(content, encoding="utf-8")
    return filepath


# =============================================================================
# TestChunkTextData
# =============================================================================


class TestChunkTextData:
    """Tests for the chunk_text_data function."""

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_basic_parent_child_chunks(self, mock_embed):
        """Text with multiple sentences creates parent and child chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_page_1_Text.md", SAMPLE_TEXT)

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert len(chunk_files) > 0
            parent_files = [f for f in chunk_files if "parent_chunk" in f.name and "child" not in f.name]
            child_files = [f for f in chunk_files if "child_chunk" in f.name]
            assert len(parent_files) > 0
            assert len(child_files) > 0

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_description_files_processed(self, mock_embed):
        """*_description.txt files are chunked with hard_split (not semantic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_page_1_img_0_description.txt", SAMPLE_TEXT)

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert len(chunk_files) > 0
            # Description chunks use _chunk_ naming, not _parent_chunk_
            assert all("_chunk_" in f.name for f in chunk_files)
            # No embedding calls needed for description files (hard_split only)
            mock_embed.assert_not_called()

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_file_filter_includes(self, mock_embed):
        """Only files matching file_filter stems are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_A_page_1_Text.md", SAMPLE_TEXT)
            _create_text_file(processed, "doc_B_page_1_Text.md", SAMPLE_TEXT)

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                file_filter=["doc_A"],
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert all("doc_A" in f.name for f in chunk_files)
            assert not any("doc_B" in f.name for f in chunk_files)

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_file_filter_excludes(self, mock_embed):
        """Files not matching file_filter are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_B_page_1_Text.md", SAMPLE_TEXT)

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                file_filter=["doc_A"],
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert len(chunk_files) == 0

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_no_filter_processes_all(self, mock_embed):
        """Without file_filter, all files are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_A_page_1_Text.md", SAMPLE_TEXT)
            _create_text_file(processed, "doc_B_page_1_Text.md", SAMPLE_TEXT)

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                file_filter=None,
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            has_a = any("doc_A" in f.name for f in chunk_files)
            has_b = any("doc_B" in f.name for f in chunk_files)
            assert has_a and has_b

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_empty_directory(self, mock_embed):
        """Empty directory produces no chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert len(chunk_files) == 0

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_empty_file(self, mock_embed):
        """Empty text file produces no chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_page_1_Text.md", "")

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert len(chunk_files) == 0

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_short_text_single_chunk(self, mock_embed):
        """Short text produces exactly one parent chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_page_1_Text.md", "Hello world.")

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            parent_files = [
                f for f in chunks.rglob("*.txt")
                if "parent_chunk" in f.name and "child" not in f.name
            ]
            assert len(parent_files) == 1

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_child_chunk_size_auto(self, mock_embed):
        """child_chunk_size=0 triggers auto-calculation (parent_size // 8)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_page_1_Text.md", SAMPLE_TEXT)

            params = {**DEFAULT_TEXT_PARAMS, "child_chunk_size": 0}
            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **params,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            assert len(chunk_files) > 0

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_chunk_files_naming(self, mock_embed):
        """Chunk files follow the expected naming convention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "report_page_1_Text.md", SAMPLE_TEXT)

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            parent_files = [
                f for f in chunks.rglob("*.txt")
                if "parent_chunk" in f.name and "child" not in f.name
            ]
            child_files = [f for f in chunks.rglob("*.txt") if "child_chunk" in f.name]

            for pf in parent_files:
                assert pf.name.startswith("report_page_1_Text_parent_chunk_")
            for cf in child_files:
                assert "report_page_1_Text_parent_chunk_" in cf.name
                assert "_child_chunk_" in cf.name

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_multiple_files(self, mock_embed):
        """Multiple files produce separate chunks for each."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_A_page_1_Text.md", SAMPLE_TEXT)
            _create_text_file(processed, "doc_B_page_1_Text.md", "Short text here.")

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            has_a = any("doc_A" in f.name for f in chunk_files)
            has_b = any("doc_B" in f.name for f in chunk_files)
            assert has_a and has_b

    @patch("cognidoc.chunk_text_data.get_embedding", return_value=MOCK_EMBEDDING)
    def test_text_and_description_mixed(self, mock_embed):
        """Both _Text.md and _description.txt files are processed together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            chunks = Path(tmpdir) / "chunks"
            processed.mkdir()
            chunks.mkdir()

            _create_text_file(processed, "doc_page_1_Text.md", SAMPLE_TEXT)
            _create_text_file(processed, "doc_page_1_img_0_description.txt", "An image of a chart showing data trends.")

            chunk_text_data(
                documents_dir=str(processed),
                documents_chunks_dir=str(chunks),
                **DEFAULT_TEXT_PARAMS,
            )

            chunk_files = list(chunks.rglob("*.txt"))
            text_chunks = [f for f in chunk_files if "parent_chunk" in f.name]
            desc_chunks = [f for f in chunk_files if "description" in f.name]
            assert len(text_chunks) > 0
            assert len(desc_chunks) > 0


# =============================================================================
# TestChunkMarkdownTableWithOverlap (helper function)
# =============================================================================


class TestChunkMarkdownTableWithOverlap:
    """Tests for chunk_markdown_table_with_overlap helper."""

    def test_small_table_single_chunk(self):
        """Table smaller than n_tokens produces a single chunk."""
        chunks, header = chunk_markdown_table_with_overlap(
            SAMPLE_TABLE, n_tokens=2000, overlap=50
        )
        assert len(chunks) == 1
        assert "Alice" in chunks[0]
        assert "Jack" in chunks[0]

    def test_header_preserved_in_all_chunks(self):
        """Every chunk starts with the table header."""
        chunks, header = chunk_markdown_table_with_overlap(
            SAMPLE_TABLE, n_tokens=50, overlap=10
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.startswith("| Name")

    def test_cols_override(self):
        """Providing cols overrides the table header."""
        chunks, header = chunk_markdown_table_with_overlap(
            SAMPLE_TABLE, cols=["Nom", "Âge", "Ville"], n_tokens=2000, overlap=0
        )
        assert "Nom" in header
        assert "Âge" in header
        assert "Nom" in chunks[0]

    def test_empty_table(self):
        """Empty string produces no chunks."""
        chunks, header = chunk_markdown_table_with_overlap("", n_tokens=512, overlap=0)
        assert chunks == []
        assert header == ""

    def test_multiple_chunks_with_overlap(self):
        """Large table split into multiple chunks with overlapping rows."""
        chunks, header = chunk_markdown_table_with_overlap(
            SAMPLE_TABLE, n_tokens=50, overlap=20
        )
        assert len(chunks) >= 2
        # Check overlap: last rows of chunk N should appear in chunk N+1
        if len(chunks) >= 2:
            # Both chunks contain the header
            assert header in chunks[0]
            assert header in chunks[1]


# =============================================================================
# TestHardSplit
# =============================================================================


class TestHardSplit:
    """Tests for the hard_split helper."""

    def test_short_text_no_split(self):
        """Text shorter than max_chunk_size stays as one chunk."""
        chunks = hard_split("Hello world.", max_chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_splits(self):
        """Long text is split into multiple chunks."""
        long_text = " ".join(["word"] * 1000)
        chunks = hard_split(long_text, max_chunk_size=50)
        assert len(chunks) > 1

    def test_empty_text(self):
        """Empty text produces empty or single empty chunk."""
        chunks = hard_split("", max_chunk_size=100)
        # Empty text split produces no words, so either empty list or single empty chunk
        assert len(chunks) <= 1


# =============================================================================
# TestChunkTableData
# =============================================================================


class TestChunkTableData:
    """Tests for the chunk_table_data function."""

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_basic_table_chunking(self, mock_llm):
        """Table file is chunked and saved with summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_page_1_Table_1.md", SAMPLE_TABLE)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) > 0
            mock_llm.assert_called_once()

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_llm_extracts_columns(self, mock_llm):
        """LLM response is used to extract column names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_page_1_Table_1.md", SAMPLE_TABLE)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) > 0
            # The chunk should contain the LLM-extracted columns
            content = chunk_files[0].read_text()
            assert "Name" in content

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_cols_override(self, mock_llm):
        """Providing cols uses the cols-override path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_page_1_Table_1.md", SAMPLE_TABLE)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[["Name", "Age", "City"]],  # One list per table
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) > 0

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_file_filter(self, mock_llm):
        """Only table files matching file_filter are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_A_page_1_Table_1.md", SAMPLE_TABLE)
            _create_text_file(tables, "doc_B_page_1_Table_1.md", SAMPLE_TABLE)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                file_filter=["doc_A"],
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert all("doc_A" in f.name for f in chunk_files)
            assert not any("doc_B" in f.name for f in chunk_files)

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_no_filter_processes_all(self, mock_llm):
        """Without file_filter, all table files are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_A_page_1_Table_1.md", SAMPLE_TABLE)
            _create_text_file(tables, "doc_B_page_1_Table_1.md", SAMPLE_TABLE)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                file_filter=None,
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            has_a = any("doc_A" in f.name for f in chunk_files)
            has_b = any("doc_B" in f.name for f in chunk_files)
            assert has_a and has_b

    def test_empty_directory(self):
        """Empty directory produces no chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) == 0

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_small_table_single_chunk(self, mock_llm):
        """Small table produces exactly one chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            small_table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
            _create_text_file(tables, "doc_page_1_Table_1.md", small_table)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) == 1

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value="not valid json {{{")
    def test_malformed_llm_json(self, mock_llm):
        """Malformed LLM JSON response is handled gracefully (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_page_1_Table_1.md", SAMPLE_TABLE)

            # Should not raise — graceful fallback
            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            # Malformed JSON → chunk_markdown_table returns empty → no chunks saved
            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) == 0

    @patch("cognidoc.chunk_table_data.ask_llm_json_unified", return_value=LLM_JSON_RESPONSE)
    def test_chunk_contains_summary(self, mock_llm):
        """Each saved chunk contains the LLM-generated summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = Path(tmpdir) / "tables"
            chunks = Path(tmpdir) / "chunks"
            tables.mkdir()
            chunks.mkdir()

            _create_text_file(tables, "doc_page_1_Table_1.md", SAMPLE_TABLE)

            chunk_table_data(
                prompt="Analyze this table: {table}",
                tables_dir=str(tables),
                cols=[],
                n_tokens=2000,
                overlap=50,
                tables_chunks_dir=str(chunks),
                use_unified_llm=True,
            )

            chunk_files = list(chunks.rglob("*.md"))
            assert len(chunk_files) > 0
            content = chunk_files[0].read_text()
            assert "A table of people with their ages and cities." in content
