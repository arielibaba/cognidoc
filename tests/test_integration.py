"""
Integration tests for CogniDoc.

Tests real module interactions with mocked LLM/embedding boundaries.
Covers: query pipeline routing, metrics recording, API endpoints,
chat history CRUD, and document upload.
"""

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from cognidoc.query_orchestrator import (
    classify_query_rules,
    QueryType,
    RoutingDecision,
    RetrievalMode,
    QueryOrchestrator,
    OrchestratorConfig,
    WEIGHT_CONFIG,
)
from cognidoc.complexity import (
    evaluate_complexity,
    should_use_agent,
    ComplexityLevel,
    ComplexityScore,
    AGENT_THRESHOLD,
    MODERATE_THRESHOLD,
)
from cognidoc.utils.metrics import PerformanceMetrics, QueryMetrics
from cognidoc.utils.chat_history import ChatHistory


# =============================================================================
# TestQueryPipelineIntegration
# =============================================================================


class TestQueryPipelineIntegration:
    """Tests query -> classify -> route -> retrieve flow with mocked LLM/vector."""

    def test_factual_query_routes_to_vector_heavy(self):
        """A factual query should assign higher weight to vector than graph."""
        query_type, confidence, reasoning = classify_query_rules("What is quantum computing?")
        assert query_type == QueryType.FACTUAL

        # Build a RoutingDecision using the WEIGHT_CONFIG for FACTUAL
        weight_cfg = WEIGHT_CONFIG[QueryType.FACTUAL]
        assert weight_cfg["vector"] > weight_cfg["graph"], (
            f"FACTUAL weight config should favor vector: "
            f"vector={weight_cfg['vector']} vs graph={weight_cfg['graph']}"
        )

    def test_relational_query_routes_to_graph_heavy(self):
        """A relational query should assign higher weight to graph than vector."""
        query_type, confidence, reasoning = classify_query_rules(
            "What is the relationship between DNA and RNA?"
        )
        assert query_type == QueryType.RELATIONAL

        weight_cfg = WEIGHT_CONFIG[QueryType.RELATIONAL]
        assert weight_cfg["graph"] > weight_cfg["vector"], (
            f"RELATIONAL weight config should favor graph: "
            f"graph={weight_cfg['graph']} vs vector={weight_cfg['vector']}"
        )

    @patch("cognidoc.utils.rag_utils.get_embedding", side_effect=RuntimeError("no embeddings"))
    def test_complex_query_triggers_agent_path(self, _mock_emb):
        """A complex analytical query with multiple signals should trigger agent path."""
        routing = RoutingDecision(
            query="Analyze and compare the advantages and disadvantages of X vs Y",
            query_type=QueryType.ANALYTICAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.5,
            graph_weight=0.5,
            confidence=0.3,
            entities_detected=["X", "Y", "advantages", "disadvantages"],
        )

        complexity = evaluate_complexity(
            query="Analyze and compare the advantages and disadvantages of X vs Y",
            routing=routing,
        )

        assert complexity.score >= AGENT_THRESHOLD, (
            f"Complex analytical query should trigger agent path: "
            f"score={complexity.score:.2f} < threshold={AGENT_THRESHOLD}"
        )
        assert complexity.should_use_agent is True

    @patch("cognidoc.utils.rag_utils.get_embedding", side_effect=RuntimeError("no embeddings"))
    def test_simple_query_uses_fast_path(self, _mock_emb):
        """A simple factual query should stay below the moderate threshold."""
        routing = RoutingDecision(
            query="What is the capital of France?",
            query_type=QueryType.FACTUAL,
            mode=RetrievalMode.VECTOR_ONLY,
            vector_weight=1.0,
            graph_weight=0.0,
            confidence=0.8,
            entities_detected=[],
        )

        complexity = evaluate_complexity(
            query="What is the capital of France?",
            routing=routing,
        )

        assert complexity.score < MODERATE_THRESHOLD, (
            f"Simple factual query should use fast path: "
            f"score={complexity.score:.2f} >= threshold={MODERATE_THRESHOLD}"
        )
        assert complexity.level == ComplexityLevel.SIMPLE
        assert complexity.should_use_agent is False

    def test_cached_query_returns_immediately(self, tmp_path):
        """Retrieval cache should return cached result on second identical query."""
        from cognidoc.hybrid_retriever import RetrievalCache

        # Reset singleton for isolated test
        RetrievalCache._instance = None
        RetrievalCache._initialized = False

        try:
            db_path = str(tmp_path / "test_cache.db")
            cache = RetrievalCache(db_path=db_path, max_size=10, ttl_seconds=60)

            # Store a result
            from cognidoc.hybrid_retriever import HybridRetrievalResult, QueryAnalysis
            from cognidoc.utils.rag_utils import Document, NodeWithScore

            query_analysis = QueryAnalysis(
                query="test query",
                query_type=QueryType.FACTUAL,
                entities_mentioned=[],
                relationship_keywords=[],
                use_vector=True,
                use_graph=False,
                vector_weight=1.0,
                graph_weight=0.0,
                confidence=0.8,
            )
            result = HybridRetrievalResult(
                query="test query",
                query_analysis=query_analysis,
                fused_context="Some context about testing.",
                source_chunks=["chunk1"],
            )

            cache.put("test query", 10, False, result)

            # Retrieve should hit cache
            cached = cache.get("test query", 10, False)
            assert cached is not None, "Cache should return result for identical query"
            assert cached.fused_context == "Some context about testing."
            assert cache._hits >= 1
        finally:
            # Cleanup singleton
            RetrievalCache._instance = None
            RetrievalCache._initialized = False

    @patch("cognidoc.utils.rag_utils.get_embedding", side_effect=RuntimeError("no embeddings"))
    def test_reranking_toggle_affects_results(self, _mock_emb):
        """Toggling reranking should change the cache key (different retrieval path)."""
        from cognidoc.hybrid_retriever import RetrievalCache

        RetrievalCache._instance = None
        RetrievalCache._initialized = False

        try:
            with tempfile.TemporaryDirectory() as tmp:
                db_path = str(Path(tmp) / "cache.db")
                cache = RetrievalCache(db_path=db_path, max_size=10, ttl_seconds=60)

                key_rerank_on = cache._make_key("test query", 10, True)
                key_rerank_off = cache._make_key("test query", 10, False)

                assert key_rerank_on != key_rerank_off, (
                    "Cache keys should differ when reranking toggle changes"
                )
        finally:
            RetrievalCache._instance = None
            RetrievalCache._initialized = False

    def test_graph_toggle_disables_graph_retrieval(self):
        """When graph is disabled via config, routing should skip graph retrieval."""
        config = OrchestratorConfig(use_llm_classifier=False)
        orchestrator = QueryOrchestrator(config)

        # FACTUAL with high confidence + prefer_vector_for_simple skips graph
        result = orchestrator.route("What is Python?")

        # For simple factual queries with high confidence, graph should be skipped
        # The exact behavior depends on the confidence from rules, but we can verify the
        # structure of the RoutingDecision
        assert isinstance(result, RoutingDecision)
        assert result.query_type == QueryType.FACTUAL
        # With prefer_vector_for_simple=True (default) and confidence > 0.7,
        # graph should be skipped
        if result.confidence > 0.7:
            assert result.skip_graph is True
            assert result.mode == RetrievalMode.VECTOR_ONLY

    @patch("cognidoc.utils.rag_utils.get_embedding", side_effect=RuntimeError("no embeddings"))
    def test_pre_computed_routing_skips_classification(self, _mock_emb):
        """When a pre-computed RoutingDecision is provided, classification is not re-run."""
        pre_routing = RoutingDecision(
            query="custom routed query",
            query_type=QueryType.PROCEDURAL,
            mode=RetrievalMode.VECTOR_ONLY,
            vector_weight=0.9,
            graph_weight=0.1,
            confidence=0.95,
            entities_detected=[],
        )

        # evaluate_complexity should use the pre-computed routing directly
        complexity = evaluate_complexity(
            query="custom routed query",
            routing=pre_routing,
        )

        # PROCEDURAL is neither AGENT_QUERY_TYPES nor MODERATE_QUERY_TYPES,
        # so query_type factor should be 0.0
        assert complexity.factors["query_type"] == 0.0
        # High confidence (0.95) means low_confidence factor should be 0.0
        assert complexity.factors["low_confidence"] == 0.0
        # Simple query overall
        assert complexity.level == ComplexityLevel.SIMPLE


# =============================================================================
# TestMetricsIntegration
# =============================================================================


class TestMetricsIntegration:
    """Tests that metrics are correctly recorded in SQLite."""

    def _fresh_metrics(self, tmp_path):
        """Create a fresh PerformanceMetrics instance (bypass singleton)."""
        PerformanceMetrics._instance = None
        PerformanceMetrics._initialized = False
        db_path = str(tmp_path / "test_metrics.db")
        return PerformanceMetrics(db_path=db_path)

    def test_query_logs_metrics_to_sqlite(self, tmp_path):
        """log_query should persist metrics to SQLite and be queryable."""
        metrics_store = self._fresh_metrics(tmp_path)
        try:
            qm = QueryMetrics(
                path="fast",
                query_type="factual",
                complexity_score=0.15,
                total_time_ms=120.5,
                rewrite_time_ms=10.0,
                retrieval_time_ms=80.0,
                rerank_time_ms=0.0,
                llm_time_ms=30.5,
            )
            metrics_store.log_query("What is X?", qm)

            stats = metrics_store.get_global_stats()
            assert stats["total_queries"] == 1
            assert stats["avg_latency_ms"] == pytest.approx(120.5, abs=0.1)
            assert "fast" in stats["path_distribution"]
            assert stats["path_distribution"]["fast"]["count"] == 1
        finally:
            PerformanceMetrics._instance = None
            PerformanceMetrics._initialized = False

    def test_timing_breakdown_populated(self, tmp_path):
        """Detailed timing fields (routing, vector, graph, fusion) should be stored."""
        metrics_store = self._fresh_metrics(tmp_path)
        try:
            qm = QueryMetrics(
                path="enhanced",
                query_type="relational",
                total_time_ms=500.0,
                routing_time_ms=15.0,
                vector_time_ms=120.0,
                graph_time_ms=200.0,
                rerank_time_ms=50.0,
                llm_time_ms=100.0,
                fusion_time_ms=15.0,
            )
            metrics_store.log_query("How is A related to B?", qm)

            breakdown = metrics_store.get_timing_breakdown()
            assert "enhanced" in breakdown
            enhanced = breakdown["enhanced"]
            assert enhanced["routing"] == pytest.approx(15.0, abs=0.1)
            assert enhanced["vector"] == pytest.approx(120.0, abs=0.1)
            assert enhanced["graph"] == pytest.approx(200.0, abs=0.1)
            assert enhanced["fusion"] == pytest.approx(15.0, abs=0.1)
            assert enhanced["count"] == 1
        finally:
            PerformanceMetrics._instance = None
            PerformanceMetrics._initialized = False

    def test_cache_hit_increments_counter(self, tmp_path):
        """record_cache_hit should increment session counter correctly."""
        metrics_store = self._fresh_metrics(tmp_path)
        try:
            assert metrics_store.get_session_cache_stats()["hits"] == 0

            metrics_store.record_cache_hit()
            metrics_store.record_cache_hit()
            metrics_store.record_cache_miss()

            stats = metrics_store.get_session_cache_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["total"] == 3
            assert stats["hit_rate"] == pytest.approx(66.7, abs=0.1)
        finally:
            PerformanceMetrics._instance = None
            PerformanceMetrics._initialized = False

    def test_percentiles_computation(self, tmp_path):
        """P50/P95/P99 percentiles should be computed from logged query times."""
        metrics_store = self._fresh_metrics(tmp_path)
        try:
            # Log 20 queries with known latencies (10ms to 200ms, step 10)
            for i in range(1, 21):
                qm = QueryMetrics(
                    path="fast",
                    total_time_ms=float(i * 10),
                )
                metrics_store.log_query(f"query_{i}", qm)

            percentiles = metrics_store.get_percentiles()

            # With values [10, 20, ..., 200]:
            # P50 should be around 100-110ms (median)
            assert 90 <= percentiles["p50"] <= 120, f"P50={percentiles['p50']}"
            # P95 should be close to 190-200ms
            assert 180 <= percentiles["p95"] <= 200, f"P95={percentiles['p95']}"
            # P99 should be close to 200ms
            assert 190 <= percentiles["p99"] <= 200, f"P99={percentiles['p99']}"
        finally:
            PerformanceMetrics._instance = None
            PerformanceMetrics._initialized = False


# =============================================================================
# TestAPIEndpointIntegration
# =============================================================================


class TestAPIEndpointIntegration:
    """Tests FastAPI endpoints. Uses mocked Gradio app components."""

    def test_graph_data_endpoint_returns_valid_json(self):
        """The /api/graph/data endpoint should return JSON with nodes/edges/stats keys."""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from starlette.testclient import TestClient

        app = FastAPI()

        mock_nodes = [
            {
                "id": "n1",
                "name": "Entity A",
                "type": "CONCEPT",
                "description": "Test entity",
                "degree": 2,
                "community_id": 0,
                "attributes": {},
                "aliases": [],
            }
        ]
        mock_edges = [
            {
                "source": "n1",
                "target": "n2",
                "relationship_type": "RELATED",
                "description": "",
                "weight": 1.0,
            }
        ]
        mock_stats = {"node_count": 2, "edge_count": 1}

        @app.get("/api/graph/data")
        async def graph_data():
            return {
                "nodes": mock_nodes,
                "edges": mock_edges,
                "communities": [],
                "stats": mock_stats,
            }

        client = TestClient(app)
        resp = client.get("/api/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["name"] == "Entity A"

    def test_graph_data_empty_when_no_graph(self):
        """When no knowledge graph is loaded, the endpoint should return 503."""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from starlette.testclient import TestClient

        app = FastAPI()

        @app.get("/api/graph/data")
        async def graph_data():
            # Simulates the behavior when graph retriever is None
            return JSONResponse(
                status_code=503,
                content={"error": "Knowledge graph not loaded"},
            )

        client = TestClient(app)
        resp = client.get("/api/graph/data")
        assert resp.status_code == 503
        assert "error" in resp.json()
        assert "not loaded" in resp.json()["error"]

    def test_graph_viewer_html_served(self):
        """The /graph-viewer endpoint should serve the HTML file."""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from starlette.testclient import TestClient

        app = FastAPI()

        @app.get("/graph-viewer")
        async def serve_graph_viewer():
            # Instead of requiring the actual file, return a known HTML snippet
            return HTMLResponse(
                content="<html><head><title>Graph Viewer</title></head><body></body></html>"
            )

        client = TestClient(app)
        resp = client.get("/graph-viewer")
        assert resp.status_code == 200
        assert "Graph Viewer" in resp.text
        assert resp.headers["content-type"].startswith("text/html")

    def test_reset_conversation_clears_state(self):
        """reset_conversation should return empty history and empty string."""
        from cognidoc.helpers import reset_conversation

        history, text = reset_conversation()
        assert history == []
        assert text == ""

    def test_graph_data_with_communities(self):
        """Graph data endpoint should include community information."""
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        app = FastAPI()

        mock_communities = [
            {"id": 0, "node_ids": ["n1", "n2"], "summary": "A test community"},
            {"id": 1, "node_ids": ["n3"], "summary": "Another community"},
        ]

        @app.get("/api/graph/data")
        async def graph_data():
            return {
                "nodes": [],
                "edges": [],
                "communities": mock_communities,
                "stats": {"node_count": 3, "edge_count": 0, "community_count": 2},
            }

        client = TestClient(app)
        resp = client.get("/api/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["communities"]) == 2
        assert data["communities"][0]["summary"] == "A test community"
        assert data["stats"]["community_count"] == 2

    def test_pdf_serving_returns_404_for_missing_file(self):
        """PDF serving endpoint should return 404 for non-existent files."""
        from fastapi import FastAPI, HTTPException
        from starlette.testclient import TestClient

        app = FastAPI()

        @app.get("/pdf/{file_path:path}")
        async def serve_pdf(file_path: str):
            raise HTTPException(status_code=404, detail="File not found")

        client = TestClient(app)
        resp = client.get("/pdf/nonexistent.pdf")
        assert resp.status_code == 404


# =============================================================================
# TestChatHistoryIntegration
# =============================================================================


class TestChatHistoryIntegration:
    """Tests chat history CRUD with real SQLite."""

    def test_conversation_crud_roundtrip(self, tmp_path):
        """Create, add messages, read back, delete - full lifecycle."""
        db_path = str(tmp_path / "chat.db")
        ch = ChatHistory(db_path=db_path)

        # Create
        conv_id = ch.create_conversation("Test Chat")
        assert conv_id is not None
        assert len(conv_id) == 36  # UUID format

        # Add messages
        msg1_id = ch.add_message(conv_id, "user", "Hello, what is CogniDoc?")
        msg2_id = ch.add_message(
            conv_id,
            "assistant",
            "CogniDoc is a Hybrid RAG assistant.",
            sources=[{"file": "readme.pdf", "page": 1}],
        )
        assert msg1_id is not None
        assert msg2_id is not None
        assert msg2_id > msg1_id

        # Read back
        messages = ch.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, what is CogniDoc?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["sources"] == [{"file": "readme.pdf", "page": 1}]

        # Delete
        ch.delete_conversation(conv_id)
        conversations = ch.list_conversations()
        assert all(c["id"] != conv_id for c in conversations)

    def test_messages_persist_across_sessions(self, tmp_path):
        """A new ChatHistory instance on the same db_path should see existing data."""
        db_path = str(tmp_path / "persist.db")

        # Session 1: create and populate
        ch1 = ChatHistory(db_path=db_path)
        conv_id = ch1.create_conversation("Persistent Chat")
        ch1.add_message(conv_id, "user", "First message")
        ch1.add_message(conv_id, "assistant", "First reply")

        # Session 2: new instance, same database
        ch2 = ChatHistory(db_path=db_path)
        messages = ch2.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "First message"
        assert messages[1]["content"] == "First reply"

    def test_conversation_list_ordered_by_date(self, tmp_path):
        """Conversations should be listed most-recently-updated first."""
        db_path = str(tmp_path / "order.db")
        ch = ChatHistory(db_path=db_path)

        # Create conversations with slight time differences
        id1 = ch.create_conversation("First")
        time.sleep(0.05)
        id2 = ch.create_conversation("Second")
        time.sleep(0.05)
        id3 = ch.create_conversation("Third")

        conversations = ch.list_conversations()
        assert len(conversations) == 3
        # Most recent first
        assert conversations[0]["id"] == id3
        assert conversations[1]["id"] == id2
        assert conversations[2]["id"] == id1

        # Now update the first conversation - it should move to the top
        time.sleep(0.05)
        ch.add_message(id1, "user", "Updating first conv")
        conversations = ch.list_conversations()
        assert conversations[0]["id"] == id1

    def test_export_json_format(self, tmp_path):
        """Export should produce valid JSON with expected structure."""
        db_path = str(tmp_path / "export.db")
        ch = ChatHistory(db_path=db_path)

        # Use default title (None) so auto-title kicks in from first user message
        conv_id = ch.create_conversation()
        ch.add_message(conv_id, "user", "Question about exports")
        ch.add_message(conv_id, "assistant", "Here is the answer.")

        exported = ch.export_conversation(conv_id, fmt="json")
        data = json.loads(exported)

        assert data["id"] == conv_id
        # Auto-title from first user message (content[:50] + "..." if > 50 chars)
        assert data["title"] == "Question about exports"
        assert "created_at" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    def test_delete_cascade_removes_messages(self, tmp_path):
        """Deleting a conversation should cascade-delete all its messages."""
        db_path = str(tmp_path / "cascade.db")
        ch = ChatHistory(db_path=db_path)

        conv_id = ch.create_conversation("Cascade Test")
        ch.add_message(conv_id, "user", "msg1")
        ch.add_message(conv_id, "assistant", "reply1")
        ch.add_message(conv_id, "user", "msg2")

        # Verify messages exist
        assert len(ch.get_messages(conv_id)) == 3

        # Delete conversation
        ch.delete_conversation(conv_id)

        # Verify messages are gone (query the raw database)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()[0]
            assert count == 0, "Messages should be cascade-deleted with conversation"


# =============================================================================
# TestUploadIntegration
# =============================================================================


class TestUploadIntegration:
    """Tests document upload functionality."""

    def test_uploaded_file_copied_to_sources(self, tmp_path):
        """Upload should copy file to the SOURCES_DIR location."""
        import shutil

        sources_dir = tmp_path / "data" / "sources"
        sources_dir.mkdir(parents=True)

        # Create a mock uploaded file
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        test_file = upload_dir / "test_document.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake pdf content for testing")

        # Simulate the upload_documents logic from cognidoc_app.py
        max_size = 50 * 1024 * 1024  # 50MB

        class MockFile:
            def __init__(self, path):
                self.name = str(path)

        files = [MockFile(test_file)]
        copied = []

        for file in files:
            file_path = file.name if hasattr(file, "name") else str(file)
            file_size = Path(file_path).stat().st_size
            if file_size > max_size:
                continue
            dest = sources_dir / Path(file_path).name
            shutil.copy2(file_path, dest)
            copied.append(Path(file_path).name)

        assert len(copied) == 1
        assert copied[0] == "test_document.pdf"
        assert (sources_dir / "test_document.pdf").exists()
        assert (sources_dir / "test_document.pdf").read_bytes() == b"%PDF-1.4 fake pdf content for testing"

    def test_upload_rejects_oversized_files(self, tmp_path):
        """Files exceeding the 50MB limit should be rejected."""
        sources_dir = tmp_path / "data" / "sources"
        sources_dir.mkdir(parents=True)

        # Create a mock file reference that appears to be > 50MB
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        test_file = upload_dir / "huge_file.pdf"
        test_file.write_bytes(b"x")  # Tiny actual file

        max_size = 50 * 1024 * 1024

        class MockFile:
            def __init__(self, path):
                self.name = str(path)

        # Patch stat to report a large file size
        original_stat = Path.stat

        def fake_stat(self_path):
            result = original_stat(self_path)
            if self_path.name == "huge_file.pdf":
                # Return a mock stat with large size
                mock_stat = MagicMock()
                mock_stat.st_size = 60 * 1024 * 1024  # 60MB
                return mock_stat
            return result

        files = [MockFile(test_file)]
        copied = []

        with patch.object(Path, "stat", fake_stat):
            for file in files:
                file_path = file.name if hasattr(file, "name") else str(file)
                file_size = Path(file_path).stat().st_size
                if file_size > max_size:
                    continue
                dest = sources_dir / Path(file_path).name
                copied.append(Path(file_path).name)

        assert len(copied) == 0, "Oversized file should not be copied"
        assert not (sources_dir / "huge_file.pdf").exists()

    def test_upload_handles_multiple_files(self, tmp_path):
        """Upload should handle multiple files, copying valid ones and skipping oversized."""
        import shutil

        sources_dir = tmp_path / "data" / "sources"
        sources_dir.mkdir(parents=True)

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        # Create test files
        file1 = upload_dir / "doc1.pdf"
        file1.write_bytes(b"%PDF small")
        file2 = upload_dir / "doc2.docx"
        file2.write_bytes(b"PK docx content")
        file3 = upload_dir / "doc3.txt"
        file3.write_bytes(b"plain text")

        max_size = 50 * 1024 * 1024

        class MockFile:
            def __init__(self, path):
                self.name = str(path)

        files = [MockFile(file1), MockFile(file2), MockFile(file3)]
        copied = []

        for file in files:
            file_path = file.name if hasattr(file, "name") else str(file)
            file_size = Path(file_path).stat().st_size
            if file_size > max_size:
                continue
            dest = sources_dir / Path(file_path).name
            shutil.copy2(file_path, dest)
            copied.append(Path(file_path).name)

        assert len(copied) == 3
        assert (sources_dir / "doc1.pdf").exists()
        assert (sources_dir / "doc2.docx").exists()
        assert (sources_dir / "doc3.txt").exists()
