"""Tests for the embedding generation module."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.create_embeddings import (
    make_metadata,
    decode_document_path,
    collect_chunks_to_embed,
    create_embeddings,
    ChunkToEmbed,
    PATH_SEPARATOR,
)


# ── make_metadata ─────────────────────────────────────────────────────────


class TestMakeMetadata:
    """Tests for make_metadata()."""

    def test_child_chunk_metadata(self):
        meta = make_metadata("doc_page_1_Text_parent_chunk_1_child_chunk_2.txt")
        assert meta["child"] == "doc_page_1_Text_parent_chunk_1_child_chunk_2.txt"
        assert meta["parent"] == "doc_page_1_Text_parent_chunk_1.txt"
        assert meta["source"]["document"] == "doc"
        assert meta["source"]["page"] == "1"

    def test_description_chunk_metadata(self):
        meta = make_metadata("doc_page_3_description_chunk_1.txt")
        assert meta["parent"] == "doc_page_3_description.txt"
        assert meta["source"]["document"] == "doc"
        assert meta["source"]["page"] == "3"

    def test_table_chunk_metadata(self):
        meta = make_metadata("report_page_2_Table_1_chunk_1.md")
        assert meta["parent"] == "report_page_2_Table_1.md"
        assert meta["source"]["document"] == "report"
        assert meta["source"]["page"] == "2"

    def test_no_parent_for_unknown_format(self):
        meta = make_metadata("random_file.txt")
        assert meta["parent"] is None
        assert meta["source"]["document"] is None
        assert meta["source"]["page"] is None

    def test_path_encoded_document(self):
        meta = make_metadata(
            f"projet_A{PATH_SEPARATOR}doc_page_1_Text_parent_chunk_1_child_chunk_1.txt"
        )
        assert meta["source"]["document"] == "projet_A/doc"
        assert meta["source"]["page"] == "1"

    def test_no_file_date_without_stem_dates(self):
        meta = make_metadata("doc_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        assert "file_date" not in meta["source"]

    def test_file_date_from_stem_dates(self):
        stem_dates = {"doc": 1700000000.0}
        meta = make_metadata(
            "doc_page_1_Text_parent_chunk_1_child_chunk_1.txt",
            stem_dates=stem_dates,
        )
        assert meta["source"]["file_date"] == 1700000000.0

    def test_file_date_encoded_stem(self):
        """Encoded stems like projet_A__doc should match stem_dates['projet_A__doc']."""
        stem_dates = {"projet_A__doc": 1700000000.0}
        meta = make_metadata(
            f"projet_A{PATH_SEPARATOR}doc_page_1_Text_parent_chunk_1_child_chunk_1.txt",
            stem_dates=stem_dates,
        )
        assert meta["source"]["file_date"] == 1700000000.0

    def test_file_date_missing_stem(self):
        stem_dates = {"other_doc": 1700000000.0}
        meta = make_metadata(
            "doc_page_1_Text_parent_chunk_1_child_chunk_1.txt",
            stem_dates=stem_dates,
        )
        assert "file_date" not in meta["source"]


# ── decode_document_path ──────────────────────────────────────────────────


class TestDecodeDocumentPath:
    def test_simple_name(self):
        assert decode_document_path("doc") == "doc"

    def test_encoded_path(self):
        assert decode_document_path("projet_A__doc") == "projet_A/doc"

    def test_deeply_nested(self):
        assert decode_document_path("a__b__c") == "a/b/c"


# ── collect_chunks_to_embed ───────────────────────────────────────────────


class TestCollectChunksToEmbed:
    """Tests for collect_chunks_to_embed()."""

    @pytest.fixture
    def chunks_dir(self, tmp_path):
        d = tmp_path / "chunks"
        d.mkdir()
        return d

    @pytest.fixture
    def embeddings_dir(self, tmp_path):
        d = tmp_path / "embeddings"
        d.mkdir()
        return d

    def _write_chunk(self, chunks_dir, name, content="This is a test chunk with enough words"):
        (chunks_dir / name).write_text(content, encoding="utf-8")

    def test_collects_child_chunks(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", False, False, None
        )
        assert len(chunks) == 1
        assert stats["to_embed"] == 1

    def test_skips_parent_chunks(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc_page_1_Text_parent_chunk_1.txt")
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", False, False, None
        )
        assert len(chunks) == 0
        assert stats["skipped_parent"] == 1

    def test_skips_short_files(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc_page_1_child_chunk_1.txt", "hi")
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", False, False, None
        )
        assert len(chunks) == 0
        assert stats["skipped_short"] == 1

    def test_file_filter(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc1_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        self._write_chunk(chunks_dir, "doc2_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        chunks, stats = collect_chunks_to_embed(
            chunks_dir,
            embeddings_dir,
            "model",
            False,
            False,
            None,
            file_filter=["doc1"],
        )
        assert len(chunks) == 1
        assert chunks[0].file_path.name.startswith("doc1")

    def test_cached_embedding_creates_file(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        mock_cache = MagicMock()
        mock_cache.get.return_value = [0.1, 0.2, 0.3]
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", True, False, mock_cache
        )
        assert len(chunks) == 0
        assert stats["cached"] == 1
        # Embedding file should have been created
        emb_files = list(embeddings_dir.glob("*.json"))
        assert len(emb_files) == 1
        data = json.loads(emb_files[0].read_text())
        assert data["embedding"] == [0.1, 0.2, 0.3]
        assert "child" in data["metadata"]

    def test_force_reembed_ignores_cache(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        mock_cache = MagicMock()
        mock_cache.get.return_value = [0.1, 0.2, 0.3]
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", True, True, mock_cache
        )
        assert len(chunks) == 1
        assert stats["cached"] == 0

    def test_stem_dates_threaded(self, chunks_dir, embeddings_dir):
        self._write_chunk(chunks_dir, "doc_page_1_Text_parent_chunk_1_child_chunk_1.txt")
        stem_dates = {"doc": 1700000000.0}
        chunks, stats = collect_chunks_to_embed(
            chunks_dir,
            embeddings_dir,
            "model",
            False,
            False,
            None,
            stem_dates=stem_dates,
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["source"]["file_date"] == 1700000000.0

    def test_handles_unreadable_files(self, chunks_dir, embeddings_dir):
        bad = chunks_dir / "bad_child_chunk_1.txt"
        bad.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", False, False, None
        )
        assert stats["errors"] == 1

    def test_empty_dir(self, chunks_dir, embeddings_dir):
        chunks, stats = collect_chunks_to_embed(
            chunks_dir, embeddings_dir, "model", False, False, None
        )
        assert len(chunks) == 0
        assert stats["to_embed"] == 0


# ── create_embeddings integration ─────────────────────────────────────────


class TestCreateEmbeddings:
    """Integration tests for create_embeddings() with mocked providers."""

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        chunks = tmp_path / "chunks"
        chunks.mkdir()
        embs = tmp_path / "embeddings"
        embs.mkdir()
        # Write some child chunks
        for i in range(3):
            (chunks / f"doc_page_1_Text_parent_chunk_1_child_chunk_{i + 1}.txt").write_text(
                f"This is child chunk number {i + 1} with enough content to pass filters"
            )
        return chunks, embs

    @patch("cognidoc.create_embeddings.embed_batch_async")
    @patch("cognidoc.create_embeddings.get_embedding_cache")
    def test_returns_stats(self, mock_cache_fn, mock_embed, setup_dirs):
        chunks, embs = setup_dirs
        mock_cache_fn.return_value = MagicMock(
            get=MagicMock(return_value=None),
            get_stats=MagicMock(return_value={"total_embeddings": 0}),
        )

        async def fake_embed(batch, *args, **kwargs):
            for chunk in batch:
                data = {"embedding": [0.1] * 10, "metadata": chunk.metadata}
                emb_file = embs / f"{chunk.file_path.stem}_embedding.json"
                emb_file.write_text(json.dumps(data))
            return len(batch), 0

        mock_embed.side_effect = fake_embed

        stats = create_embeddings(
            str(chunks),
            str(embs),
            "model",
            use_cache=True,
            force_reembed=False,
        )
        assert stats["to_embed"] == 3
        assert stats["embedded"] == 3
        assert stats["errors"] == 0

    def test_empty_chunks_dir(self, tmp_path):
        chunks = tmp_path / "chunks"
        chunks.mkdir()
        embs = tmp_path / "embeddings"
        embs.mkdir()
        stats = create_embeddings(
            str(chunks),
            str(embs),
            "model",
            use_cache=False,
        )
        assert stats["to_embed"] == 0
        assert stats["embedded"] == 0

    @patch("cognidoc.create_embeddings.get_embedding_cache")
    def test_all_cached(self, mock_cache_fn, setup_dirs):
        chunks, embs = setup_dirs
        mock_cache = MagicMock()
        mock_cache.get.return_value = [0.1] * 10
        mock_cache.get_stats.return_value = {"total_embeddings": 3}
        mock_cache_fn.return_value = mock_cache

        stats = create_embeddings(
            str(chunks),
            str(embs),
            "model",
            use_cache=True,
            force_reembed=False,
        )
        assert stats["cached"] == 3
        assert stats["to_embed"] == 0

    @patch("cognidoc.create_embeddings.embed_batch_async")
    @patch("cognidoc.create_embeddings.get_embedding_cache")
    def test_stem_dates_passed_through(self, mock_cache_fn, mock_embed, setup_dirs):
        chunks, embs = setup_dirs
        mock_cache_fn.return_value = MagicMock(
            get=MagicMock(return_value=None),
            get_stats=MagicMock(return_value={"total_embeddings": 0}),
        )

        captured_metadata = []

        async def fake_embed(batch, *args, **kwargs):
            for chunk in batch:
                captured_metadata.append(chunk.metadata)
                data = {"embedding": [0.1] * 10, "metadata": chunk.metadata}
                emb_file = embs / f"{chunk.file_path.stem}_embedding.json"
                emb_file.write_text(json.dumps(data))
            return len(batch), 0

        mock_embed.side_effect = fake_embed

        stem_dates = {"doc": 1700000000.0}
        create_embeddings(
            str(chunks),
            str(embs),
            "model",
            use_cache=True,
            force_reembed=False,
            stem_dates=stem_dates,
        )
        assert all(m["source"]["file_date"] == 1700000000.0 for m in captured_metadata)
