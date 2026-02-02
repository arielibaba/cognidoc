"""Tests for query_orchestrator.py — routing, classification, weight config, fallback."""

from unittest.mock import patch, MagicMock

import pytest

from cognidoc.query_orchestrator import (
    QueryType,
    RetrievalMode,
    RoutingDecision,
    OrchestratorConfig,
    QueryOrchestrator,
    classify_query_rules,
    classify_query_llm,
    route_query,
    WEIGHT_CONFIG,
    QUERY_PATTERNS,
)


# ===========================================================================
# Rule-based classification
# ===========================================================================


class TestClassifyQueryRules:
    """Tests for classify_query_rules()."""

    def test_relational_pattern(self):
        qt, conf, _ = classify_query_rules("What is the relationship between A and B?")
        assert qt == QueryType.RELATIONAL
        assert conf >= 0.7

    def test_comparative_pattern(self):
        qt, conf, _ = classify_query_rules("Compare Python vs Java")
        assert qt == QueryType.COMPARATIVE

    def test_exploratory_pattern(self):
        qt, conf, _ = classify_query_rules("Summarize the main topics")
        assert qt == QueryType.EXPLORATORY

    def test_procedural_pattern(self):
        qt, conf, _ = classify_query_rules("How to install Python?")
        assert qt == QueryType.PROCEDURAL

    def test_analytical_pattern(self):
        qt, conf, _ = classify_query_rules("Analyze the performance of this system")
        assert qt == QueryType.ANALYTICAL

    def test_factual_question_word(self):
        qt, conf, _ = classify_query_rules("What is machine learning?")
        assert qt == QueryType.FACTUAL

    def test_unknown_no_match(self):
        qt, conf, _ = classify_query_rules("xyz123")
        assert qt == QueryType.UNKNOWN
        assert conf <= 0.5

    # French
    def test_french_relational(self):
        qt, _, _ = classify_query_rules("Quelle est la relation entre A et B?")
        assert qt == QueryType.RELATIONAL

    def test_french_exploratory(self):
        qt, _, _ = classify_query_rules("Résumé des thèmes principaux")
        assert qt == QueryType.EXPLORATORY

    def test_french_question_word(self):
        qt, _, _ = classify_query_rules("Qu'est-ce que l'apprentissage automatique?")
        assert qt == QueryType.FACTUAL

    # Spanish
    def test_spanish_comparative(self):
        qt, _, _ = classify_query_rules("Diferencia entre A y B")
        assert qt == QueryType.COMPARATIVE

    # German
    def test_german_procedural(self):
        qt, _, _ = classify_query_rules("Wie kann ich Python installieren?")
        assert qt == QueryType.PROCEDURAL


# ===========================================================================
# LLM-based classification
# ===========================================================================


class TestClassifyQueryLLM:
    """Tests for classify_query_llm()."""

    @patch("cognidoc.query_orchestrator.llm_chat")
    def test_parses_response(self, mock_llm):
        mock_llm.return_value = (
            "TYPE: RELATIONAL\n"
            "CONFIDENCE: 0.9\n"
            "ENTITIES: Python, Java\n"
            "REASONING: Asks about relationship"
        )
        qt, conf, reasoning, entities = classify_query_llm("How is A related to B?")
        assert qt == QueryType.RELATIONAL
        assert conf == 0.9
        assert entities == ["Python", "Java"]
        assert "relationship" in reasoning.lower()

    @patch("cognidoc.query_orchestrator.llm_chat")
    def test_handles_no_entities(self, mock_llm):
        mock_llm.return_value = (
            "TYPE: FACTUAL\nCONFIDENCE: 0.8\nENTITIES: none\nREASONING: Simple fact"
        )
        _, _, _, entities = classify_query_llm("What is X?")
        assert entities == []

    @patch("cognidoc.query_orchestrator.llm_chat")
    def test_fallback_on_error(self, mock_llm):
        mock_llm.side_effect = ConnectionError("API error")
        qt, conf, _, _ = classify_query_llm("What is X?")
        # Should fallback to rule-based
        assert isinstance(qt, QueryType)

    @patch("cognidoc.query_orchestrator.llm_chat")
    def test_handles_unknown_type(self, mock_llm):
        mock_llm.return_value = "TYPE: INVALID\nCONFIDENCE: 0.5\nENTITIES: none\nREASONING: x"
        qt, _, _, _ = classify_query_llm("q")
        assert qt == QueryType.UNKNOWN


# ===========================================================================
# Weight configuration
# ===========================================================================


class TestWeightConfig:
    """Tests for WEIGHT_CONFIG defaults."""

    def test_all_query_types_have_weights(self):
        for qt in QueryType:
            assert qt in WEIGHT_CONFIG

    def test_weights_sum_to_one_or_less(self):
        for qt, cfg in WEIGHT_CONFIG.items():
            total = cfg["vector"] + cfg["graph"]
            assert total <= 1.01, f"{qt}: vector+graph={total}"

    def test_factual_favors_vector(self):
        cfg = WEIGHT_CONFIG[QueryType.FACTUAL]
        assert cfg["vector"] > cfg["graph"]

    def test_relational_favors_graph(self):
        cfg = WEIGHT_CONFIG[QueryType.RELATIONAL]
        assert cfg["graph"] > cfg["vector"]

    def test_exploratory_is_graph_only(self):
        cfg = WEIGHT_CONFIG[QueryType.EXPLORATORY]
        assert cfg["mode"] == RetrievalMode.GRAPH_ONLY


# ===========================================================================
# QueryOrchestrator routing
# ===========================================================================


class TestQueryOrchestrator:
    """Tests for QueryOrchestrator.route() and should_fallback()."""

    def _make_orchestrator(self, use_llm=False):
        config = OrchestratorConfig(use_llm_classifier=use_llm)
        return QueryOrchestrator(config)

    def test_route_factual_hybrid(self):
        """Rule-based FACTUAL with confidence=0.6 uses hybrid (not vector-only)."""
        orch = self._make_orchestrator()
        decision = orch.route("What is machine learning?")
        assert decision.query_type == QueryType.FACTUAL
        # Rule-based confidence is 0.6 (< 0.7 threshold), so graph is NOT skipped
        assert decision.vector_weight >= 0.7
        assert decision.mode == RetrievalMode.HYBRID

    def test_route_relational(self):
        orch = self._make_orchestrator()
        decision = orch.route("What is the relationship between A and B?")
        assert decision.query_type == QueryType.RELATIONAL
        assert decision.graph_weight >= 0.5

    def test_route_exploratory_skips_vector(self):
        orch = self._make_orchestrator()
        decision = orch.route("Summarize all main topics")
        assert decision.skip_vector is True
        assert decision.mode == RetrievalMode.GRAPH_ONLY

    def test_skip_threshold(self):
        config = OrchestratorConfig(skip_threshold=0.15, use_llm_classifier=False)
        orch = QueryOrchestrator(config)
        decision = orch.route("List all key concepts")
        # Exploratory: vector=0.1 < 0.15 → skip_vector
        assert decision.skip_vector is True

    def test_should_fallback_low_vector(self):
        orch = self._make_orchestrator()
        decision = RoutingDecision(
            query="q",
            query_type=QueryType.FACTUAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.7,
            graph_weight=0.3,
        )
        adjusted = orch.should_fallback(decision, vector_confidence=0.1, graph_confidence=0.8)
        # Graph should be boosted
        assert adjusted.graph_weight > decision.graph_weight

    def test_should_fallback_both_low(self):
        orch = self._make_orchestrator()
        decision = RoutingDecision(
            query="q",
            query_type=QueryType.FACTUAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.5,
            graph_weight=0.5,
        )
        adjusted = orch.should_fallback(decision, vector_confidence=0.1, graph_confidence=0.1)
        assert adjusted.mode == RetrievalMode.ADAPTIVE

    def test_should_fallback_no_change_when_good(self):
        orch = self._make_orchestrator()
        decision = RoutingDecision(
            query="q",
            query_type=QueryType.FACTUAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.7,
            graph_weight=0.3,
        )
        adjusted = orch.should_fallback(decision, vector_confidence=0.9, graph_confidence=0.8)
        assert adjusted.vector_weight == decision.vector_weight
        assert adjusted.graph_weight == decision.graph_weight


# ===========================================================================
# Context fusion (deduplication)
# ===========================================================================


class TestContextFusion:
    """Tests for fuse_contexts deduplication."""

    def test_deduplicates_similar(self):
        orch = QueryOrchestrator(OrchestratorConfig(dedup_similarity_threshold=0.85))
        decision = RoutingDecision(
            query="q",
            query_type=QueryType.ANALYTICAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.5,
            graph_weight=0.5,
        )
        ctx = orch.fuse_contexts(
            "This is a paragraph about machine learning.",
            "This is a paragraph about machine learning!",  # near-duplicate
            decision,
        )
        # Should not contain the duplicate
        assert ctx.count("machine learning") == 1

    def test_keeps_unique_content(self):
        orch = QueryOrchestrator()
        decision = RoutingDecision(
            query="q",
            query_type=QueryType.ANALYTICAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.5,
            graph_weight=0.5,
        )
        ctx = orch.fuse_contexts("Vector result about Python", "Graph result about Java", decision)
        assert "Python" in ctx
        assert "Java" in ctx


# ===========================================================================
# Convenience functions
# ===========================================================================


class TestConvenienceFunctions:
    """Tests for route_query()."""

    def test_route_query_returns_decision(self):
        with patch("cognidoc.query_orchestrator.llm_chat") as mock_llm:
            mock_llm.return_value = (
                "TYPE: FACTUAL\nCONFIDENCE: 0.8\nENTITIES: none\nREASONING: fact"
            )
            decision = route_query("What is X?")
            assert isinstance(decision, RoutingDecision)
            assert isinstance(decision.query_type, QueryType)
