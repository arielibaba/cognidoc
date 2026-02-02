"""Tests for graph_retrieval.py — cache, retriever, result dataclass."""

import time
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.graph_retrieval import (
    GraphRetrievalResult,
    GraphRetrievalCache,
    GraphRetriever,
)
from cognidoc.knowledge_graph import GraphNode, Community


# ===========================================================================
# GraphRetrievalResult
# ===========================================================================


class TestGraphRetrievalResult:
    """Tests for GraphRetrievalResult dataclass."""

    def test_default_fields(self):
        r = GraphRetrievalResult(query="test", retrieval_type="entity")
        assert r.entities == []
        assert r.relationships == []
        assert r.communities == []
        assert r.paths == []
        assert r.context == ""
        assert r.confidence == 0.0

    def test_get_context_text_entities(self):
        node = GraphNode(id="n1", name="Python", type="LANGUAGE")
        r = GraphRetrievalResult(
            query="test",
            retrieval_type="entity",
            entities=[node],
            context="Python is a language",
        )
        ctx = r.get_context_text()
        assert "Python" in ctx

    def test_get_context_text_relationships(self):
        r = GraphRetrievalResult(
            query="test",
            retrieval_type="relationship",
            relationships=[("A", "USES", "B")],
        )
        ctx = r.get_context_text()
        assert "A" in ctx
        assert "B" in ctx

    def test_get_context_text_empty(self):
        r = GraphRetrievalResult(query="test", retrieval_type="none")
        ctx = r.get_context_text()
        assert isinstance(ctx, str)


# ===========================================================================
# GraphRetrievalCache
# ===========================================================================


class TestGraphRetrievalCache:
    """Tests for in-memory LRU cache."""

    def test_put_and_get(self):
        cache = GraphRetrievalCache(max_size=10, ttl_seconds=60)
        result = GraphRetrievalResult(query="test", retrieval_type="entity", context="ctx")
        cache.put("What is X?", result)
        cached = cache.get("What is X?")
        assert cached is not None
        assert cached.context == "ctx"

    def test_miss_unknown(self):
        cache = GraphRetrievalCache()
        assert cache.get("unknown query") is None

    def test_normalization(self):
        cache = GraphRetrievalCache()
        result = GraphRetrievalResult(query="q", retrieval_type="entity")
        cache.put("What  is  X?", result)
        assert cache.get("what is x?") is not None

    def test_ttl_expiry(self):
        cache = GraphRetrievalCache(ttl_seconds=0.1)
        result = GraphRetrievalResult(query="q", retrieval_type="entity")
        cache.put("query", result)
        time.sleep(0.15)
        assert cache.get("query") is None

    def test_lru_eviction(self):
        cache = GraphRetrievalCache(max_size=2)
        for i in range(3):
            cache.put(f"q{i}", GraphRetrievalResult(query=f"q{i}", retrieval_type="entity"))
        assert cache.get("q0") is None  # evicted
        assert cache.get("q2") is not None  # still present

    def test_clear(self):
        cache = GraphRetrievalCache()
        cache.put("q", GraphRetrievalResult(query="q", retrieval_type="entity"))
        cache.clear()
        assert cache.get("q") is None
        assert cache.stats()["size"] == 0
        assert cache.stats()["hits"] == 0

    def test_stats(self):
        cache = GraphRetrievalCache()
        cache.put("q", GraphRetrievalResult(query="q", retrieval_type="entity"))
        cache.get("q")  # hit
        cache.get("other")  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1

    def test_move_to_end_on_access(self):
        cache = GraphRetrievalCache(max_size=2)
        cache.put("q0", GraphRetrievalResult(query="q0", retrieval_type="entity"))
        cache.put("q1", GraphRetrievalResult(query="q1", retrieval_type="entity"))
        cache.get("q0")  # access q0, making q1 the LRU
        cache.put("q2", GraphRetrievalResult(query="q2", retrieval_type="entity"))
        assert cache.get("q1") is None  # q1 evicted
        assert cache.get("q0") is not None  # q0 kept


# ===========================================================================
# GraphRetriever
# ===========================================================================


class TestGraphRetriever:
    """Tests for GraphRetriever class."""

    def test_not_loaded_initially(self):
        with patch("cognidoc.graph_retrieval.get_graph_config"):
            retriever = GraphRetriever.__new__(GraphRetriever)
            retriever.kg = None
            retriever._cache = GraphRetrievalCache()
            retriever.graph_path = "/nonexistent"
            retriever.config = MagicMock()
            assert not retriever.is_loaded()

    def test_retrieve_when_not_loaded_returns_error(self):
        with patch("cognidoc.graph_retrieval.get_graph_config"):
            retriever = GraphRetriever.__new__(GraphRetriever)
            retriever.kg = None
            retriever._cache = GraphRetrievalCache()
            retriever.graph_path = "/nonexistent"
            retriever.config = MagicMock()
            with patch.object(retriever, "load", return_value=False):
                result = retriever.retrieve("test query")
                assert result.retrieval_type == "error"
                assert result.confidence == 0.0

    def test_retrieve_uses_cache(self):
        with patch("cognidoc.graph_retrieval.get_graph_config"):
            retriever = GraphRetriever.__new__(GraphRetriever)
            retriever.kg = MagicMock()
            retriever.kg.nodes = {"n1": GraphNode(id="n1", name="X", type="T")}
            retriever._cache = GraphRetrievalCache()
            retriever.graph_path = "/test"
            retriever.config = MagicMock()

            cached_result = GraphRetrievalResult(
                query="test", retrieval_type="entity", context="cached"
            )
            retriever._cache.put("test query", cached_result)
            result = retriever.retrieve("test query")
            assert result.context == "cached"

    def test_get_statistics_not_loaded(self):
        with patch("cognidoc.graph_retrieval.get_graph_config"):
            retriever = GraphRetriever.__new__(GraphRetriever)
            retriever.kg = None
            retriever._cache = GraphRetrievalCache()
            stats = retriever.get_statistics()
            assert stats["status"] == "not_loaded"
