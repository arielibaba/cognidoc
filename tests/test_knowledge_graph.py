"""Tests for knowledge_graph.py — graph CRUD, communities, persistence, JSON serialization."""

import json
from pathlib import Path

import pytest

from cognidoc.knowledge_graph import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    Community,
    build_knowledge_graph,
    has_valid_knowledge_graph,
    get_knowledge_graph_stats,
)
from cognidoc.extract_entities import Entity, Relationship, ExtractionResult


# ===========================================================================
# Helpers
# ===========================================================================


def _make_entity(name, etype="CONCEPT", desc="", chunk="chunk_1"):
    return Entity(
        id=f"ent_{name.lower().replace(' ', '_')}",
        name=name,
        type=etype,
        description=desc,
        source_chunk=chunk,
    )


def _make_relationship(src, tgt, rel_type="RELATED_TO", chunk="chunk_1"):
    return Relationship(
        source_entity=src,
        target_entity=tgt,
        relationship_type=rel_type,
        source_chunk=chunk,
    )


# ===========================================================================
# GraphNode
# ===========================================================================


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_matches_name_exact(self):
        node = GraphNode(id="1", name="Machine Learning", type="CONCEPT")
        assert node.matches_name("Machine Learning")
        assert node.matches_name("machine learning")

    def test_matches_name_alias(self):
        node = GraphNode(id="1", name="ML", type="CONCEPT", aliases=["Machine Learning"])
        assert node.matches_name("Machine Learning")
        assert node.matches_name("ml")

    def test_no_match(self):
        node = GraphNode(id="1", name="ML", type="CONCEPT")
        assert not node.matches_name("Deep Learning")

    def test_hash_and_eq(self):
        n1 = GraphNode(id="a", name="X", type="T")
        n2 = GraphNode(id="a", name="Y", type="T")
        n3 = GraphNode(id="b", name="X", type="T")
        assert n1 == n2  # same id
        assert n1 != n3
        assert hash(n1) == hash(n2)
        assert len({n1, n2, n3}) == 2


# ===========================================================================
# KnowledgeGraph CRUD
# ===========================================================================


class TestKnowledgeGraphCRUD:
    """Tests for add_entity, add_relationship, build_from_extraction_results."""

    def test_add_entity(self):
        kg = KnowledgeGraph()
        eid = kg.add_entity(_make_entity("Python"))
        assert eid in kg.nodes
        assert kg.nodes[eid].name == "Python"

    def test_add_duplicate_merges(self):
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity("Python", chunk="c1"))
        kg.add_entity(_make_entity("python", chunk="c2"))  # case-insensitive merge
        assert len(kg.nodes) == 1
        node = list(kg.nodes.values())[0]
        assert len(node.source_chunks) == 2

    def test_add_relationship(self):
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity("A"))
        kg.add_entity(_make_entity("B"))
        assert kg.add_relationship(_make_relationship("A", "B", "USES"))
        assert kg.has_edge(kg._find_existing_node("A"), kg._find_existing_node("B"))

    def test_add_relationship_missing_entity(self):
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity("A"))
        assert not kg.add_relationship(_make_relationship("A", "MISSING", "USES"))

    def test_duplicate_edge_increments_weight(self):
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity("A"))
        kg.add_entity(_make_entity("B"))
        kg.add_relationship(_make_relationship("A", "B"))
        kg.add_relationship(_make_relationship("A", "B", chunk="c2"))
        src = kg._find_existing_node("A")
        tgt = kg._find_existing_node("B")
        assert kg.get_edge_data(src, tgt)["weight"] == 2.0

    def test_build_from_extraction_results(self):
        result = ExtractionResult(
            chunk_id="chunk_1",
            chunk_text="test text",
            entities=[_make_entity("X"), _make_entity("Y")],
            relationships=[_make_relationship("X", "Y", "DEPENDS_ON")],
        )
        kg = KnowledgeGraph()
        stats = kg.build_from_extraction_results([result])
        assert stats["entities_added"] == 2
        assert stats["relationships_added"] == 1


# ===========================================================================
# Graph traversal
# ===========================================================================


class TestGraphTraversal:
    """Tests for get_neighbors, find_paths, get_node_by_name."""

    def _build_graph(self):
        kg = KnowledgeGraph()
        for name in ["A", "B", "C"]:
            kg.add_entity(_make_entity(name))
        kg.add_relationship(_make_relationship("A", "B", "R1"))
        kg.add_relationship(_make_relationship("B", "C", "R2"))
        return kg

    def test_get_node_by_name(self):
        kg = self._build_graph()
        node = kg.get_node_by_name("a")
        assert node is not None
        assert node.name == "A"

    def test_get_node_by_name_missing(self):
        kg = self._build_graph()
        assert kg.get_node_by_name("Z") is None

    def test_get_neighbors(self):
        kg = self._build_graph()
        a_id = kg._find_existing_node("A")
        neighbors = kg.get_neighbors(a_id, depth=1, direction="out")
        assert len(neighbors) == 1
        assert neighbors[0][0].name == "B"

    def test_find_paths(self):
        kg = self._build_graph()
        paths = kg.find_paths("A", "C", max_depth=3)
        assert len(paths) >= 1
        # Path should be A -> B -> C
        assert paths[0][0][0] == "A"
        assert paths[0][-1][2] == "C"

    def test_find_paths_no_connection(self):
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity("X"))
        kg.add_entity(_make_entity("Y"))
        paths = kg.find_paths("X", "Y")
        assert paths == []


# ===========================================================================
# Community detection
# ===========================================================================


class TestCommunityDetection:
    """Tests for detect_communities."""

    def test_detect_communities(self):
        kg = KnowledgeGraph()
        for name in ["A", "B", "C", "D"]:
            kg.add_entity(_make_entity(name))
        kg.add_relationship(_make_relationship("A", "B"))
        kg.add_relationship(_make_relationship("C", "D"))
        num = kg.detect_communities()
        assert num >= 1
        assert len(kg.communities) == num

    def test_detect_communities_empty_graph(self):
        kg = KnowledgeGraph()
        assert kg.detect_communities() == 0

    def test_get_community_nodes(self):
        kg = KnowledgeGraph()
        for name in ["A", "B"]:
            kg.add_entity(_make_entity(name))
        kg.add_relationship(_make_relationship("A", "B"))
        kg.detect_communities()
        if kg.communities:
            cid = list(kg.communities.keys())[0]
            nodes = kg.get_community_nodes(cid)
            assert len(nodes) >= 1


# ===========================================================================
# Persistence (JSON, no pickle)
# ===========================================================================


class TestKnowledgeGraphPersistence:
    """Tests for save/load with JSON-based graph serialization."""

    def _build_and_save(self, tmp_path):
        kg = KnowledgeGraph()
        for name in ["Alpha", "Beta"]:
            kg.add_entity(_make_entity(name, desc=f"Description of {name}"))
        kg.add_relationship(_make_relationship("Alpha", "Beta", "LINKED"))
        kg.detect_communities()
        save_path = str(tmp_path / "kg")
        kg.save(save_path)
        return save_path, kg

    def test_save_creates_json_files(self, tmp_path):
        save_path, _ = self._build_and_save(tmp_path)
        p = Path(save_path)
        assert (p / "graph.json").exists()
        assert (p / "nodes.json").exists()
        assert (p / "communities.json").exists()
        assert (p / "name_mapping.json").exists()
        # No pickle file should be created
        assert not (p / "graph.gpickle").exists()

    def test_save_graph_json_is_valid(self, tmp_path):
        save_path, _ = self._build_and_save(tmp_path)
        with open(Path(save_path) / "graph.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data or "directed" in data  # node_link format

    def test_load_roundtrip(self, tmp_path):
        save_path, original = self._build_and_save(tmp_path)
        loaded = KnowledgeGraph.load(save_path)
        assert len(loaded.nodes) == len(original.nodes)
        assert loaded.number_of_edges() == original.number_of_edges()
        assert loaded.get_node_by_name("Alpha") is not None

    def test_load_missing_path(self, tmp_path):
        kg = KnowledgeGraph.load(str(tmp_path / "nonexistent"))
        assert len(kg.nodes) == 0

    def test_has_valid_knowledge_graph(self, tmp_path):
        save_path, _ = self._build_and_save(tmp_path)
        assert has_valid_knowledge_graph(save_path)

    def test_has_valid_knowledge_graph_missing(self, tmp_path):
        assert not has_valid_knowledge_graph(str(tmp_path / "nope"))

    def test_get_knowledge_graph_stats(self, tmp_path):
        save_path, _ = self._build_and_save(tmp_path)
        stats = get_knowledge_graph_stats(save_path)
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        assert stats["communities"] >= 0


# ===========================================================================
# Statistics
# ===========================================================================


class TestGraphStatistics:
    """Tests for get_statistics."""

    def test_statistics(self):
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity("A", etype="PERSON"))
        kg.add_entity(_make_entity("B", etype="CONCEPT"))
        kg.add_relationship(_make_relationship("A", "B"))
        stats = kg.get_statistics()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert "PERSON" in stats["node_types"]
        assert "CONCEPT" in stats["node_types"]

    def test_empty_statistics(self):
        kg = KnowledgeGraph()
        stats = kg.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["avg_degree"] == 0


# ===========================================================================
# Pruning by source stems
# ===========================================================================


class TestPruneBySourceStems:
    """Tests for prune_by_source_stems."""

    def _build_graph(self):
        """Build a graph with entities from two source files (report, memo)."""
        kg = KnowledgeGraph()
        # Entities from report only
        kg.add_entity(_make_entity("Alpha", chunk="report_chunk_1"))
        kg.add_entity(_make_entity("Beta", chunk="report_chunk_2"))
        # Entity from memo only
        kg.add_entity(_make_entity("Gamma", chunk="memo_chunk_1"))
        # Entity from both sources
        kg.add_entity(_make_entity("Delta", chunk="report_chunk_3"))
        kg.add_entity(_make_entity("Delta", chunk="memo_chunk_2"))  # merges
        # Relationships
        kg.add_relationship(_make_relationship("Alpha", "Beta", "R1", chunk="report_chunk_1"))
        kg.add_relationship(_make_relationship("Alpha", "Delta", "R2", chunk="report_chunk_1"))
        kg.add_relationship(_make_relationship("Gamma", "Delta", "R3", chunk="memo_chunk_1"))
        return kg

    def test_remove_single_source_nodes(self):
        kg = self._build_graph()
        stats = kg.prune_by_source_stems({"report"})
        # Alpha and Beta should be removed (report-only)
        assert kg.get_node_by_name("Alpha") is None
        assert kg.get_node_by_name("Beta") is None
        assert stats["nodes_removed"] == 2

    def test_keep_multi_source_nodes(self):
        kg = self._build_graph()
        kg.prune_by_source_stems({"report"})
        # Delta should remain (has memo_chunk_2)
        delta = kg.get_node_by_name("Delta")
        assert delta is not None
        assert "memo_chunk_2" in delta.source_chunks
        assert not any(c.startswith("report_") for c in delta.source_chunks)

    def test_update_multi_source_nodes(self):
        kg = self._build_graph()
        stats = kg.prune_by_source_stems({"report"})
        assert stats["nodes_updated"] >= 1  # Delta was updated

    def test_remove_edges(self):
        kg = self._build_graph()
        stats = kg.prune_by_source_stems({"report"})
        # Alpha->Beta edge should be removed (report-only)
        assert stats["edges_removed"] >= 1

    def test_keep_unrelated_nodes(self):
        kg = self._build_graph()
        kg.prune_by_source_stems({"report"})
        # Gamma should be untouched
        gamma = kg.get_node_by_name("Gamma")
        assert gamma is not None
        assert gamma.source_chunks == ["memo_chunk_1"]

    def test_name_to_id_cleaned(self):
        kg = self._build_graph()
        kg.prune_by_source_stems({"report"})
        assert "alpha" not in kg._name_to_id
        assert "beta" not in kg._name_to_id
        assert "gamma" in kg._name_to_id

    def test_edges_list_cleaned(self):
        kg = self._build_graph()
        initial_edges = len(kg.edges)
        kg.prune_by_source_stems({"report"})
        # At least the report-only edge should be removed
        assert len(kg.edges) < initial_edges

    def test_empty_stems_no_change(self):
        kg = self._build_graph()
        initial_nodes = len(kg.nodes)
        stats = kg.prune_by_source_stems(set())
        assert len(kg.nodes) == initial_nodes
        assert stats["nodes_removed"] == 0

    def test_unknown_stems_no_change(self):
        kg = self._build_graph()
        initial_nodes = len(kg.nodes)
        stats = kg.prune_by_source_stems({"nonexistent_file"})
        assert len(kg.nodes) == initial_nodes
        assert stats["nodes_removed"] == 0

    def test_persistence_roundtrip(self, tmp_path):
        kg = self._build_graph()
        kg.prune_by_source_stems({"report"})
        save_path = str(tmp_path / "pruned_kg")
        kg.save(save_path)
        loaded = KnowledgeGraph.load(save_path)
        assert loaded.get_node_by_name("Alpha") is None
        assert loaded.get_node_by_name("Gamma") is not None
        assert loaded.get_node_by_name("Delta") is not None
