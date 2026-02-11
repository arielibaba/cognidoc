"""Tests for api.py — CogniDocConfig, QueryResult, IngestionResult, CogniDoc public API."""

import warnings
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.api import CogniDocConfig, QueryResult, IngestionResult, CogniDoc


# ===========================================================================
# CogniDocConfig
# ===========================================================================


class TestCogniDocConfig:
    """Tests for CogniDocConfig dataclass."""

    def test_default_values(self):
        config = CogniDocConfig()
        assert config.llm_provider == "gemini"
        assert config.embedding_provider == "ollama"
        assert config.use_graph is True
        assert config.use_reranking is True
        assert config.top_k == 10
        assert config.dense_weight == 0.6

    def test_custom_values(self):
        config = CogniDocConfig(llm_provider="openai", top_k=20, use_graph=False)
        assert config.llm_provider == "openai"
        assert config.top_k == 20
        assert config.use_graph is False

    def test_validation_dense_weight_rejects_above_1(self):
        with pytest.raises(ValueError, match="dense_weight"):
            CogniDocConfig(dense_weight=1.5)

    def test_validation_dense_weight_rejects_negative(self):
        with pytest.raises(ValueError, match="dense_weight"):
            CogniDocConfig(dense_weight=-0.5)

    def test_vision_provider_defaults_to_llm(self):
        config = CogniDocConfig(llm_provider="openai", vision_provider=None)
        assert config.vision_provider == "openai"


# ===========================================================================
# QueryResult and IngestionResult
# ===========================================================================


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_default_fields(self):
        r = QueryResult(answer="test answer")
        assert r.answer == "test answer"
        assert r.sources == []
        assert r.query_type is None
        assert r.retrieval_stats is None

    def test_with_sources(self):
        r = QueryResult(
            answer="answer",
            sources=[{"text": "chunk", "source": "doc.pdf"}],
            query_type="FACTUAL",
        )
        assert len(r.sources) == 1
        assert r.query_type == "FACTUAL"


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_default_fields(self):
        r = IngestionResult()
        assert r.documents_processed == 0
        assert r.chunks_created == 0
        assert r.entities_extracted == 0
        assert r.errors == []

    def test_custom_fields(self):
        r = IngestionResult(
            documents_processed=5,
            chunks_created=50,
            entities_extracted=100,
            relationships_extracted=30,
            errors=["error1"],
        )
        assert r.documents_processed == 5
        assert len(r.errors) == 1


# ===========================================================================
# CogniDoc deprecation warnings
# ===========================================================================


# ===========================================================================
# CogniDoc.get_info
# ===========================================================================


class TestCogniDocGetInfo:
    """Tests for get_info() method."""

    def test_get_info_returns_dict(self):
        with patch("cognidoc.api.CogniDoc._setup_providers"):
            cd = CogniDoc.__new__(CogniDoc)
            cd.config = CogniDocConfig(data_dir="/tmp/test")
            with patch.object(cd, "_get_use_yolo", return_value=False):
                info = cd.get_info()
                assert isinstance(info, dict)
                assert "llm_provider" in info
                assert "data_dir" in info
