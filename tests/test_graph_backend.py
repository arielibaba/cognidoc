"""Tests for graph backends (NetworkX and Kùzu).

Both backends are tested with identical test cases via pytest parametrize.
Kùzu tests are skipped if kuzu is not installed.
"""

import pytest

from cognidoc.graph_backend_networkx import NetworkXBackend

try:
    from cognidoc.graph_backend_kuzu import KuzuBackend, KUZU_AVAILABLE
except ImportError:
    KUZU_AVAILABLE = False


@pytest.fixture(params=["networkx", "kuzu"])
def backend(request, tmp_path):
    """Parametrized fixture: returns a fresh backend for each type."""
    if request.param == "networkx":
        return NetworkXBackend()
    elif request.param == "kuzu":
        if not KUZU_AVAILABLE:
            pytest.skip("kuzu not installed")
        return KuzuBackend(db_path=str(tmp_path / "test_kuzu_db"))
    else:
        raise ValueError(f"Unknown backend: {request.param}")


@pytest.fixture
def nx_backend():
    """NetworkX-only backend for tests that don't need parametrization."""
    return NetworkXBackend()


# ===========================================================================
# Node CRUD
# ===========================================================================


class TestNodeCRUD:
    def test_add_and_has_node(self, backend):
        assert not backend.has_node("n1")
        backend.add_node("n1", name="Alpha", type="CONCEPT")
        assert backend.has_node("n1")

    def test_number_of_nodes(self, backend):
        assert backend.number_of_nodes() == 0
        backend.add_node("n1", name="A")
        backend.add_node("n2", name="B")
        assert backend.number_of_nodes() == 2

    def test_remove_node(self, backend):
        backend.add_node("n1", name="A")
        backend.add_node("n2", name="B")
        backend.add_edge("n1", "n2", relationship_type="RELATED")
        backend.remove_node("n1")
        assert not backend.has_node("n1")
        assert backend.has_node("n2")
        assert backend.number_of_edges() == 0  # edge also removed

    def test_update_node_attrs(self, backend):
        backend.add_node("n1", name="Old", type="T")
        backend.update_node_attrs("n1", name="New")
        attrs = backend.get_node_attrs("n1")
        assert attrs["name"] == "New"

    def test_get_node_attrs(self, backend):
        backend.add_node("n1", name="Alpha", type="CONCEPT", description="A concept")
        attrs = backend.get_node_attrs("n1")
        assert attrs["name"] == "Alpha"
        assert attrs["type"] == "CONCEPT"
        assert attrs["description"] == "A concept"


# ===========================================================================
# Edge CRUD
# ===========================================================================


class TestEdgeCRUD:
    def test_add_and_has_edge(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        assert not backend.has_edge("a", "b")
        backend.add_edge("a", "b", relationship_type="USES", weight=1.0)
        assert backend.has_edge("a", "b")
        assert not backend.has_edge("b", "a")  # directed

    def test_get_edge_data(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_edge("a", "b", relationship_type="USES", weight=2.5, description="test")
        data = backend.get_edge_data("a", "b")
        assert data is not None
        assert data["relationship_type"] == "USES"
        assert data["weight"] == 2.5

    def test_get_edge_data_nonexistent(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        assert backend.get_edge_data("a", "b") is None

    def test_update_edge_attrs(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_edge("a", "b", relationship_type="R", weight=1.0)
        backend.update_edge_attrs("a", "b", weight=5.0)
        data = backend.get_edge_data("a", "b")
        assert data["weight"] == 5.0

    def test_number_of_edges(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_node("c", name="C")
        assert backend.number_of_edges() == 0
        backend.add_edge("a", "b", relationship_type="R1")
        backend.add_edge("b", "c", relationship_type="R2")
        assert backend.number_of_edges() == 2

    def test_iter_edges_without_data(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_edge("a", "b", relationship_type="R")
        edges = list(backend.iter_edges(data=False))
        assert len(edges) == 1
        assert edges[0] == ("a", "b")

    def test_iter_edges_with_data(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_edge("a", "b", relationship_type="USES", weight=1.0)
        edges = list(backend.iter_edges(data=True))
        assert len(edges) == 1
        src, tgt, data = edges[0]
        assert src == "a"
        assert tgt == "b"
        assert data["relationship_type"] == "USES"

    def test_source_chunks_serialization(self, backend):
        """Source chunks (list) should round-trip correctly."""
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        chunks = ["chunk_1", "chunk_2"]
        backend.add_edge("a", "b", relationship_type="R", source_chunks=chunks)
        data = backend.get_edge_data("a", "b")
        assert data["source_chunks"] == chunks

    def test_remove_edge(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_edge("a", "b", relationship_type="R")
        assert backend.has_edge("a", "b")
        backend.remove_edge("a", "b")
        assert not backend.has_edge("a", "b")
        # Nodes should still exist
        assert backend.has_node("a")
        assert backend.has_node("b")

    def test_remove_edge_nonexistent(self, backend):
        """Removing a non-existent edge should not raise."""
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.remove_edge("a", "b")  # no-op


# ===========================================================================
# Traversal
# ===========================================================================


class TestTraversal:
    def test_successors(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_node("c", name="C")
        backend.add_edge("a", "b", relationship_type="R")
        backend.add_edge("a", "c", relationship_type="R")
        succ = sorted(backend.successors("a"))
        assert succ == ["b", "c"]

    def test_predecessors(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_node("c", name="C")
        backend.add_edge("b", "a", relationship_type="R")
        backend.add_edge("c", "a", relationship_type="R")
        pred = sorted(backend.predecessors("a"))
        assert pred == ["b", "c"]

    def test_find_all_simple_paths(self, backend):
        # a -> b -> c
        for nid in ("a", "b", "c"):
            backend.add_node(nid, name=nid.upper())
        backend.add_edge("a", "b", relationship_type="R")
        backend.add_edge("b", "c", relationship_type="R")
        paths = backend.find_all_simple_paths("a", "c", cutoff=3)
        assert len(paths) >= 1
        assert paths[0] == ["a", "b", "c"]

    def test_find_paths_no_path(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        paths = backend.find_all_simple_paths("a", "b", cutoff=3)
        assert paths == []

    def test_degree(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_node("c", name="C")
        backend.add_edge("a", "b", relationship_type="R")
        backend.add_edge("c", "a", relationship_type="R")
        degrees = backend.degree()
        assert degrees["a"] == 2  # one out, one in
        assert degrees["b"] == 1
        assert degrees["c"] == 1


# ===========================================================================
# Export / Import
# ===========================================================================


class TestExportImport:
    def test_to_undirected_networkx(self, backend):
        backend.add_node("a", name="A")
        backend.add_node("b", name="B")
        backend.add_edge("a", "b", relationship_type="R")
        G = backend.to_undirected_networkx()
        assert len(G.nodes) == 2
        assert G.has_edge("a", "b")
        assert G.has_edge("b", "a")  # undirected

    def test_node_link_roundtrip(self, backend):
        backend.add_node("x", name="X", type="T", description="desc")
        backend.add_node("y", name="Y", type="T")
        backend.add_edge("x", "y", relationship_type="R", weight=2.0)
        data = backend.to_node_link_data()

        # Import into a fresh backend
        fresh = NetworkXBackend()
        fresh.from_node_link_data(data)
        assert fresh.has_node("x")
        assert fresh.has_node("y")
        assert fresh.has_edge("x", "y")
        assert fresh.number_of_edges() == 1

    def test_from_node_link_data_overwrites(self, backend):
        backend.add_node("old", name="Old")
        data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": "new"}],
            "edges": [],
        }
        backend.from_node_link_data(data)
        assert backend.has_node("new")
        # Old data should be replaced
        if isinstance(backend, NetworkXBackend):
            assert not backend.has_node("old")


# ===========================================================================
# NetworkX-specific tests
# ===========================================================================


class TestNetworkXBackend:
    def test_internal_graph_type(self, nx_backend):
        import networkx as nx

        assert isinstance(nx_backend._graph, nx.DiGraph)

    def test_empty_successors(self, nx_backend):
        nx_backend.add_node("a", name="A")
        assert nx_backend.successors("a") == []

    def test_empty_predecessors(self, nx_backend):
        nx_backend.add_node("a", name="A")
        assert nx_backend.predecessors("a") == []


# ===========================================================================
# Kùzu-specific tests
# ===========================================================================


class TestKuzuBackend:
    @pytest.fixture
    def kuzu(self, tmp_path):
        if not KUZU_AVAILABLE:
            pytest.skip("kuzu not installed")
        return KuzuBackend(db_path=str(tmp_path / "kuzu_specific"))

    def test_schema_creation(self, kuzu):
        """Schema should be auto-created on init."""
        assert kuzu.number_of_nodes() == 0
        assert kuzu.number_of_edges() == 0

    def test_cypher_count(self, kuzu):
        """Test direct Cypher counting."""
        kuzu.add_node("a", name="A", type="PERSON")
        kuzu.add_node("b", name="B", type="PERSON")
        kuzu.add_node("c", name="C", type="CONCEPT")

        result = kuzu._conn.execute("MATCH (n:Entity) WHERE n.type = 'PERSON' RETURN count(n)")
        while result.has_next():
            assert result.get_next()[0] == 2
