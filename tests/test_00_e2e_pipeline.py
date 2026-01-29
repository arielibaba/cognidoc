"""
End-to-End Pipeline Tests for CogniDoc.

Tests the complete flow: document ingestion → indexing → querying.

Usage:
    # Run fast E2E test (~10-30s)
    pytest tests/test_e2e_pipeline.py -v

    # Run full E2E test with GraphRAG (~2-5 min)
    pytest tests/test_e2e_pipeline.py -v --run-slow
"""

import os
from pathlib import Path

import pytest

# Markers defined in conftest.py
slow = pytest.mark.slow


@pytest.fixture
def test_document_path():
    """Path to the test document fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_article.md"
    if not fixture_path.exists():
        pytest.skip(f"Test fixture not found: {fixture_path}")
    return fixture_path


@pytest.fixture(scope="module")
def project_root():
    """Path to the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def existing_indexes_available(project_root):
    """Check if existing indexes are available for testing."""
    indexes_dir = project_root / "data" / "indexes"
    vector_index = indexes_dir / "child_documents"
    keyword_index = indexes_dir / "parent_documents"

    if not vector_index.exists() or not keyword_index.exists():
        pytest.skip("Existing indexes not found. Run ingestion first.")

    return True


@pytest.fixture
def cognidoc_with_graph(cognidoc_session):
    """
    CogniDoc instance with graph enabled.

    Uses session-scoped instance from conftest.py to avoid Qdrant lock conflicts.
    """
    return cognidoc_session


class TestE2EQueryOnly:
    """Fast E2E tests using existing indexes (~10-30 seconds)."""

    def test_query_on_existing_data(self, cognidoc_with_graph):
        """
        Test query on existing indexed data.

        This test verifies:
        1. CogniDoc can load existing indexes
        2. Query routing works correctly
        3. Response generation works
        """
        # Test a factual query
        result = cognidoc_with_graph.query("Qu'est-ce que CogniDoc ?")

        assert result.answer is not None, "Should return an answer"
        assert len(result.answer) > 20, "Answer should be substantial"

    def test_query_returns_content(self, cognidoc_with_graph):
        """Test that queries return meaningful content."""
        result = cognidoc_with_graph.query("Quels formats de documents sont supportés ?")

        assert result.answer is not None
        # Response should have content (either natural language or knowledge graph)
        assert len(result.answer) > 50, "Response should have substantial content"

    def test_query_handles_english(self, cognidoc_with_graph):
        """Test that English queries work."""
        result = cognidoc_with_graph.query("What is CogniDoc?")

        assert result.answer is not None
        assert len(result.answer) > 20, "Should return a meaningful answer"


class TestE2ETestDocumentContent:
    """Tests using the test document fixture content."""

    def test_fixture_document_readable(self, test_document_path):
        """Verify test document fixture is readable and has content."""
        content = test_document_path.read_text(encoding='utf-8')

        assert len(content) > 100, "Test document should have substantial content"
        assert "Intelligence Artificielle" in content, "Should contain AI topic"
        assert "médecine" in content.lower(), "Should contain medical topic"

    def test_fixture_document_structure(self, test_document_path):
        """Verify test document has expected structure."""
        content = test_document_path.read_text(encoding='utf-8')

        # Should have markdown headers
        assert "# " in content, "Should have H1 header"
        assert "## " in content, "Should have H2 headers"

        # Should have sections
        assert "Introduction" in content
        assert "Conclusion" in content


@slow
class TestE2EFullPipeline:
    """
    Full E2E tests that run the complete ingestion pipeline.

    WARNING: These tests modify the main data directory.
    Run with: pytest tests/test_e2e_pipeline.py -v --run-slow
    """

    def test_full_pipeline_ingestion(self, project_root, test_document_path, release_qdrant_lock):
        """
        Test full pipeline: ingest test document → query.

        This test:
        1. Copies test document to sources
        2. Runs full ingestion pipeline
        3. Queries the ingested document
        """
        import shutil
        from cognidoc import CogniDoc

        # Copy test document to sources
        sources_dir = project_root / "data" / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        dest_path = sources_dir / "test_article_e2e.md"
        shutil.copy(test_document_path, dest_path)

        try:
            doc = CogniDoc(
                llm_provider="gemini",
                embedding_provider="ollama",
                use_yolo=False,
                use_graph=False,  # Skip graph for speed
            )

            # Run ingestion
            result = doc.ingest(
                str(sources_dir),
                skip_schema_wizard=True,
            )

            # Pipeline should complete without errors
            assert len(result.errors) == 0, f"Pipeline errors: {result.errors}"

            # Query the ingested content
            query_result = doc.query("Quels sont les défis éthiques de l'IA médicale ?")

            assert query_result.answer is not None
            assert len(query_result.answer) > 50

        finally:
            # Release Qdrant lock so other tests can access the vector store
            if doc._retriever is not None:
                doc._retriever.close()
                doc._retriever = None
            # Cleanup: remove test document
            if dest_path.exists():
                dest_path.unlink()

    def test_full_pipeline_with_graphrag(self, project_root, test_document_path, release_qdrant_lock):
        """
        Test full pipeline with GraphRAG enabled.

        This is the slowest test (~2-5 minutes).
        """
        import shutil
        from cognidoc import CogniDoc

        sources_dir = project_root / "data" / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        dest_path = sources_dir / "test_article_graphrag.md"
        shutil.copy(test_document_path, dest_path)

        try:
            doc = CogniDoc(
                llm_provider="gemini",
                embedding_provider="ollama",
                use_yolo=False,
                use_graph=True,  # Enable GraphRAG
            )

            result = doc.ingest(
                str(sources_dir),
                skip_schema_wizard=True,
            )

            assert len(result.errors) == 0, f"Pipeline errors: {result.errors}"

            # Test exploratory query (uses graph)
            query_result = doc.query("Quelles sont les applications de l'IA en médecine ?")

            assert query_result.answer is not None

        finally:
            # Release Qdrant lock so other tests can access the vector store
            if doc._retriever is not None:
                doc._retriever.close()
                doc._retriever = None
            if dest_path.exists():
                dest_path.unlink()


class TestE2EEdgeCases:
    """Test edge cases and error handling."""

    def test_short_query(self, cognidoc_with_graph):
        """Test handling of very short queries."""
        result = cognidoc_with_graph.query("CogniDoc?")
        assert result.answer is not None

    def test_empty_results_handling(self, cognidoc_with_graph):
        """Test handling of queries with unlikely matches."""
        # Query about something not in the documents
        result = cognidoc_with_graph.query("What is the recipe for chocolate cake?")

        # Should still return a response (even if it says no info found)
        assert result.answer is not None
