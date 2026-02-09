"""
Unit tests for entity resolution module.

Tests the 4-phase entity resolution pipeline:
1. Blocking (embedding similarity)
2. Matching (LLM verification)
3. Clustering (Union-Find)
4. Merging (enrichment)
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from cognidoc.entity_resolution import (
    # Data structures
    CandidatePair,
    ResolutionDecision,
    MergedEntity,
    EntityResolutionResult,
    # Phase 1: Blocking
    find_candidate_pairs,
    # Phase 2: Matching
    get_entity_relations_summary,
    _parse_resolution_response,
    # Phase 3: Clustering
    UnionFind,
    build_entity_clusters,
    # Phase 4: Merging
    merge_descriptions,
    merge_attributes,
    _concatenate_descriptions_smart,
    _word_overlap,
    # Orchestration
    resolve_entities,
)
from cognidoc.knowledge_graph import GraphNode, KnowledgeGraph
from cognidoc.graph_config import EntityResolutionConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        GraphNode(
            id="1",
            name="Machine Learning",
            type="CONCEPT",
            description="A subset of AI that enables systems to learn from data",
        ),
        GraphNode(
            id="2",
            name="ML",
            type="CONCEPT",
            description="Technique d'apprentissage automatique",
        ),
        GraphNode(
            id="3",
            name="Python",
            type="LANGUAGE",
            description="A popular programming language",
        ),
        GraphNode(
            id="4",
            name="Deep Learning",
            type="CONCEPT",
            description="A subset of machine learning using neural networks",
        ),
    ]


@pytest.fixture
def mock_graph(sample_entities):
    """Create a mock knowledge graph."""
    graph = KnowledgeGraph()
    for entity in sample_entities:
        graph.nodes[entity.id] = entity
        graph._backend.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
        )
    # Add some relationships
    graph.add_edge_raw("1", "3", relationship_type="USES")
    graph.add_edge_raw("2", "3", relationship_type="IMPLEMENTED_IN")
    graph.add_edge_raw("4", "1", relationship_type="PART_OF")
    return graph


@pytest.fixture
def resolution_config():
    """Create a test configuration."""
    return EntityResolutionConfig(
        enabled=True,
        similarity_threshold=0.75,
        llm_confidence_threshold=0.7,
        max_concurrent_llm=2,
        use_llm_for_descriptions=False,  # Use simple concat for tests
        cache_decisions=False,
    )


# =============================================================================
# Phase 1: Blocking Tests
# =============================================================================


class TestBlocking:
    """Tests for Phase 1: Blocking (embedding similarity)."""

    def test_find_candidate_pairs_basic(self, sample_entities):
        """Should find similar entity pairs based on embeddings."""
        # Mock embeddings where entities 1 and 2 are similar
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Machine Learning
                [0.95, 0.1, 0.0],  # ML (similar to Machine Learning)
                [0.0, 0.0, 1.0],  # Python (different)
                [0.8, 0.2, 0.0],  # Deep Learning (somewhat similar)
            ]
        )

        candidates = find_candidate_pairs(
            sample_entities,
            embeddings,
            similarity_threshold=0.9,
        )

        # Should find ML similar to Machine Learning
        assert len(candidates) >= 1
        pair_ids = {(c.entity_a_id, c.entity_b_id) for c in candidates}
        assert ("1", "2") in pair_ids or ("2", "1") in pair_ids

    def test_find_candidate_pairs_no_self_match(self, sample_entities):
        """Should not match entity with itself."""
        # Identity embeddings
        embeddings = np.eye(len(sample_entities))

        candidates = find_candidate_pairs(
            sample_entities,
            embeddings,
            similarity_threshold=0.99,
        )

        # No candidates because each entity only matches itself
        assert len(candidates) == 0

    def test_find_candidate_pairs_threshold(self, sample_entities):
        """Should respect similarity threshold."""
        # Use orthogonal embeddings that have known similarities
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Entity 1
                [0.6, 0.8, 0.0, 0.0],  # Entity 2 - cos sim with 1 is 0.6
                [0.0, 0.0, 1.0, 0.0],  # Entity 3 - orthogonal
                [0.0, 0.0, 0.0, 1.0],  # Entity 4 - orthogonal
            ]
        )

        # High threshold (0.95): no matches
        high_threshold = find_candidate_pairs(
            sample_entities,
            embeddings,
            similarity_threshold=0.95,
        )
        assert len(high_threshold) == 0

        # Lower threshold (0.5): should find entity 1-2 match
        low_threshold = find_candidate_pairs(
            sample_entities,
            embeddings,
            similarity_threshold=0.5,
        )
        assert len(low_threshold) > 0

    def test_find_candidate_pairs_sorted_by_similarity(self, sample_entities):
        """Candidates should be sorted by similarity (highest first)."""
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],  # High similarity with 1
                [0.7, 0.3, 0.0],  # Medium similarity with 1
                [0.5, 0.5, 0.0],  # Lower similarity with 1
            ]
        )

        candidates = find_candidate_pairs(
            sample_entities,
            embeddings,
            similarity_threshold=0.5,
        )

        # Should be sorted by similarity descending
        for i in range(len(candidates) - 1):
            assert candidates[i].similarity_score >= candidates[i + 1].similarity_score


# =============================================================================
# Phase 2: Matching Tests
# =============================================================================


class TestMatching:
    """Tests for Phase 2: Matching (LLM verification)."""

    def test_get_entity_relations_summary(self, mock_graph):
        """Should get relationship summary for entity."""
        entity = mock_graph.nodes["1"]  # Machine Learning
        summary = get_entity_relations_summary(entity, mock_graph, max_relations=5)

        assert "Python" in summary
        assert "USES" in summary

    def test_get_entity_relations_summary_no_relations(self):
        """Should handle entity with no relationships."""
        graph = KnowledgeGraph()
        entity = GraphNode(id="1", name="Test", type="TEST", description="")
        graph.nodes["1"] = entity
        graph._backend.add_node("1", name="Test")

        summary = get_entity_relations_summary(entity, graph)
        assert summary == "(no relationships)"

    def test_parse_resolution_response_valid_json(self):
        """Should parse valid JSON response."""
        response = '{"same_entity": true, "confidence": 0.9, "canonical_name": "Machine Learning", "reasoning": "Same concept"}'
        result = _parse_resolution_response(response, "ML")

        assert result["same_entity"] is True
        assert result["confidence"] == 0.9
        assert result["canonical_name"] == "Machine Learning"

    def test_parse_resolution_response_markdown_json(self):
        """Should parse JSON in markdown code block."""
        response = """Here is my analysis:
```json
{"same_entity": true, "confidence": 0.85, "canonical_name": "ML", "reasoning": "Abbreviation"}
```"""
        result = _parse_resolution_response(response, "Machine Learning")

        assert result["same_entity"] is True
        assert result["confidence"] == 0.85

    def test_parse_resolution_response_invalid(self):
        """Should handle invalid response gracefully."""
        response = "I think they are the same entity but I cannot output JSON"
        result = _parse_resolution_response(response, "Fallback Name")

        assert result["same_entity"] is False
        assert result["canonical_name"] == "Fallback Name"


# =============================================================================
# Phase 3: Clustering Tests
# =============================================================================


class TestClustering:
    """Tests for Phase 3: Clustering (Union-Find)."""

    def test_union_find_basic(self):
        """Test basic union-find operations."""
        uf = UnionFind()

        uf.union("a", "b", "A")
        uf.union("b", "c", "A")  # Transitive: a, b, c same cluster

        assert uf.find("a") == uf.find("b") == uf.find("c")

    def test_union_find_separate_clusters(self):
        """Test separate clusters remain separate."""
        uf = UnionFind()

        uf.union("a", "b")
        uf.union("c", "d")

        assert uf.find("a") == uf.find("b")
        assert uf.find("c") == uf.find("d")
        assert uf.find("a") != uf.find("c")

    def test_union_find_get_clusters(self):
        """Test cluster extraction."""
        uf = UnionFind()

        uf.union("1", "2")
        uf.union("2", "3")
        uf.union("4", "5")

        clusters = uf.get_clusters()

        # Should have 2 clusters
        assert len(clusters) == 2

        # Find cluster with 1, 2, 3
        found_123 = False
        found_45 = False
        for members in clusters.values():
            if set(members) == {"1", "2", "3"}:
                found_123 = True
            if set(members) == {"4", "5"}:
                found_45 = True

        assert found_123 and found_45

    def test_union_find_canonical_name(self):
        """Test canonical name tracking."""
        uf = UnionFind()

        uf.union("1", "2", "Entity A")
        uf.union("2", "3", "Entity A Updated")  # Should update canonical

        # Canonical name should be tracked
        canonical = uf.get_canonical_name("1")
        assert canonical is not None

    def test_build_entity_clusters(self):
        """Test cluster building from verified pairs."""
        verified = [
            (CandidatePair("1", "2", 0.9), ResolutionDecision(True, 0.95, "Entity A", "")),
            (CandidatePair("2", "3", 0.85), ResolutionDecision(True, 0.9, "Entity A", "")),
        ]

        clusters, names = build_entity_clusters(verified)

        # Should have one cluster with 3 entities
        assert len(clusters) == 1
        cluster = list(clusters.values())[0]
        assert set(cluster) == {"1", "2", "3"}


# =============================================================================
# Phase 4: Merging Tests
# =============================================================================


class TestMerging:
    """Tests for Phase 4: Merging (enrichment)."""

    @pytest.mark.asyncio
    async def test_merge_descriptions_single(self):
        """Single description returns as-is."""
        result = await merge_descriptions(["Only one"], "Entity", use_llm=False)
        assert result == "Only one"

    @pytest.mark.asyncio
    async def test_merge_descriptions_duplicates(self):
        """Duplicate descriptions are deduplicated."""
        result = await merge_descriptions(
            ["Same text.", "Same text.", "Same text."],
            "Entity",
            use_llm=False,
        )
        assert result == "Same text."

    @pytest.mark.asyncio
    async def test_merge_descriptions_concatenate(self):
        """Different descriptions are concatenated."""
        result = await merge_descriptions(
            ["First fact about the entity.", "Second fact about it."],
            "Entity",
            use_llm=False,
        )
        assert "First fact" in result
        assert "Second fact" in result

    @pytest.mark.asyncio
    async def test_merge_descriptions_empty(self):
        """Empty descriptions handled gracefully."""
        result = await merge_descriptions([], "Entity", use_llm=False)
        assert result == ""

        result2 = await merge_descriptions(["", "  ", None], "Entity", use_llm=False)
        assert result2 == ""

    def test_merge_attributes_no_conflict(self):
        """Non-conflicting attributes are merged."""
        attrs = [
            {"a": 1, "b": 2},
            {"c": 3},
        ]

        merged = merge_attributes(attrs)

        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_merge_attributes_conflict_string(self):
        """String conflicts keep longest."""
        attrs = [
            {"name": "short"},
            {"name": "much longer name"},
        ]

        merged = merge_attributes(attrs)

        assert merged["name"] == "much longer name"

    def test_merge_attributes_conflict_list(self):
        """List conflicts are merged."""
        attrs = [
            {"tags": ["a", "b"]},
            {"tags": ["b", "c"]},
        ]

        merged = merge_attributes(attrs)

        assert set(merged["tags"]) == {"a", "b", "c"}

    def test_merge_attributes_conflict_number(self):
        """Number conflicts keep max."""
        attrs = [
            {"count": 5},
            {"count": 10},
        ]

        merged = merge_attributes(attrs)

        assert merged["count"] == 10

    def test_merge_attributes_empty(self):
        """Empty attributes handled gracefully."""
        assert merge_attributes([]) == {}
        assert merge_attributes([{}, {}]) == {}

    def test_word_overlap(self):
        """Test word overlap calculation."""
        assert _word_overlap("hello world", "hello world") == 1.0
        assert _word_overlap("hello world", "hello there") == 0.5
        assert _word_overlap("a b c", "d e f") == 0.0

    def test_concatenate_descriptions_smart_dedup(self):
        """Smart concatenation deduplicates similar sentences."""
        descriptions = [
            "Machine learning is a subset of AI.",
            "Machine learning is a subset of artificial intelligence.",  # Similar
            "It enables systems to learn from data.",
        ]

        result = _concatenate_descriptions_smart(descriptions)

        # Should not have both similar sentences
        assert "learn from data" in result
        assert result.count("subset of") == 1  # Deduplicated


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full resolution pipeline."""

    @pytest.mark.asyncio
    async def test_resolve_entities_disabled(self, mock_graph):
        """Resolution disabled returns unchanged graph."""
        config = EntityResolutionConfig(enabled=False)

        result = await resolve_entities(mock_graph, config)

        assert result.clusters_found == 0
        assert result.entities_merged == 0
        assert result.original_entity_count == result.final_entity_count

    @pytest.mark.asyncio
    async def test_resolve_entities_no_candidates(self, mock_graph, resolution_config):
        """Graph with no similar entities returns unchanged."""
        with patch("cognidoc.entity_resolution.compute_resolution_embeddings") as mock_embed:
            # Orthogonal embeddings = no similarity
            mock_embed.return_value = np.eye(len(mock_graph.nodes))

            result = await resolve_entities(mock_graph, resolution_config)

            assert result.candidates_found == 0
            assert result.entities_merged == 0

    @pytest.mark.asyncio
    async def test_resolve_entities_with_merges(self, resolution_config):
        """Test full resolution with actual merges."""
        # Create a graph with obviously similar entities
        graph = KnowledgeGraph()

        # Add similar entities
        e1 = GraphNode(
            id="1",
            name="Machine Learning",
            type="CONCEPT",
            description="AI technique for learning from data",
            source_chunks=["chunk1"],
        )
        e2 = GraphNode(
            id="2",
            name="ML",
            type="CONCEPT",
            description="Apprentissage automatique",
            source_chunks=["chunk2"],
        )
        e3 = GraphNode(
            id="3",
            name="Python",
            type="LANGUAGE",
            description="Programming language",
            source_chunks=["chunk3"],
        )

        for e in [e1, e2, e3]:
            graph.nodes[e.id] = e
            graph._backend.add_node(e.id, name=e.name, type=e.type)
            graph._name_to_id[graph._normalize_name(e.name)] = e.id

        # Add relationships
        graph.add_edge_raw("1", "3", relationship_type="USES")
        graph.add_edge_raw("2", "3", relationship_type="IMPLEMENTED_IN")

        with (
            patch("cognidoc.entity_resolution.compute_resolution_embeddings") as mock_embed,
            patch("cognidoc.entity_resolution.verify_candidates_batch") as mock_verify,
        ):
            # Make entities 1 and 2 similar
            mock_embed.return_value = np.array(
                [
                    [1.0, 0.0],  # Machine Learning
                    [0.95, 0.1],  # ML (similar)
                    [0.0, 1.0],  # Python (different)
                ]
            )

            # Mock LLM verification to confirm merge
            mock_verify.return_value = (
                [
                    (
                        CandidatePair("1", "2", 0.95),
                        ResolutionDecision(True, 0.9, "Machine Learning", "Same concept"),
                    )
                ],
                1,  # llm_calls
                0,  # cache_hits
            )

            result = await resolve_entities(graph, resolution_config)

            assert result.candidates_found >= 1
            assert result.clusters_found == 1
            # Note: entities_merged counts removed nodes, not the total in cluster


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_candidate_pair_creation(self):
        """Test CandidatePair dataclass."""
        pair = CandidatePair("a", "b", 0.9)
        assert pair.entity_a_id == "a"
        assert pair.entity_b_id == "b"
        assert pair.similarity_score == 0.9

    def test_resolution_decision_creation(self):
        """Test ResolutionDecision dataclass."""
        decision = ResolutionDecision(True, 0.95, "Name", "Reason")
        assert decision.same_entity is True
        assert decision.confidence == 0.95
        assert decision.canonical_name == "Name"

    def test_merged_entity_creation(self):
        """Test MergedEntity dataclass."""
        merged = MergedEntity(
            canonical_id="1",
            canonical_name="Entity",
            type="TYPE",
            description="Desc",
            attributes={"key": "value"},
            aliases=["Alias1"],
            source_chunks=["c1", "c2"],
            merged_from=["1", "2"],
            confidence=0.9,
        )
        assert merged.canonical_id == "1"
        assert len(merged.aliases) == 1
        assert len(merged.merged_from) == 2

    def test_empty_embeddings(self, sample_entities):
        """Handle empty or zero embeddings gracefully."""
        # All zeros
        embeddings = np.zeros((len(sample_entities), 3))
        candidates = find_candidate_pairs(sample_entities, embeddings, 0.5)
        # Should not crash, may return empty or NaN-filtered results
        assert isinstance(candidates, list)

    def test_single_entity(self):
        """Single entity should have no candidates."""
        entities = [GraphNode(id="1", name="Only One", type="TEST", description="")]
        embeddings = np.array([[1.0, 0.0]])

        candidates = find_candidate_pairs(entities, embeddings, 0.5)
        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_merge_descriptions_with_none(self):
        """Handle None values in descriptions."""
        result = await merge_descriptions(
            ["This is valid.", None, "This is also valid."],
            "Entity",
            use_llm=False,
        )
        # At minimum, valid descriptions should be preserved
        assert "valid" in result.lower()
        # Should not crash on None values
        assert result != ""
