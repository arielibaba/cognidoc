"""
NetworkX implementation of GraphBackend.

Wraps an nx.DiGraph with the abstract interface.
This is the default backend and preserves the original CogniDoc behavior.
"""

from typing import Any, Dict, Iterator, List, Optional

import networkx as nx

from .graph_backend import GraphBackend


class NetworkXBackend(GraphBackend):
    """Graph backend backed by NetworkX DiGraph."""

    def __init__(self):
        self._graph = nx.DiGraph()

    # ── Nodes ────────────────────────────────────────────────────────────

    def add_node(self, node_id: str, **attrs) -> None:
        self._graph.add_node(node_id, **attrs)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def remove_node(self, node_id: str) -> None:
        self._graph.remove_node(node_id)

    def update_node_attrs(self, node_id: str, **attrs) -> None:
        for key, value in attrs.items():
            self._graph.nodes[node_id][key] = value

    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        return dict(self._graph.nodes[node_id])

    def number_of_nodes(self) -> int:
        return len(self._graph)

    # ── Edges ────────────────────────────────────────────────────────────

    def add_edge(self, src: str, tgt: str, **attrs) -> None:
        self._graph.add_edge(src, tgt, **attrs)

    def has_edge(self, src: str, tgt: str) -> bool:
        return self._graph.has_edge(src, tgt)

    def get_edge_data(self, src: str, tgt: str) -> Optional[Dict[str, Any]]:
        if self._graph.has_edge(src, tgt):
            return dict(self._graph.edges[src, tgt])
        return None

    def update_edge_attrs(self, src: str, tgt: str, **attrs) -> None:
        for key, value in attrs.items():
            self._graph.edges[src, tgt][key] = value

    def remove_edge(self, src: str, tgt: str) -> None:
        if self._graph.has_edge(src, tgt):
            self._graph.remove_edge(src, tgt)

    def iter_edges(self, data: bool = False) -> Iterator:
        return iter(self._graph.edges(data=data))

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    # ── Traversal ────────────────────────────────────────────────────────

    def successors(self, node_id: str) -> List[str]:
        return list(self._graph.successors(node_id))

    def predecessors(self, node_id: str) -> List[str]:
        return list(self._graph.predecessors(node_id))

    def find_all_simple_paths(self, src: str, tgt: str, cutoff: int = 5) -> List[List[str]]:
        try:
            return list(nx.all_simple_paths(self._graph, src, tgt, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []

    def degree(self) -> Dict[str, int]:
        return dict(self._graph.degree())

    # ── Export / import ──────────────────────────────────────────────────

    def to_undirected_networkx(self) -> nx.Graph:
        return self._graph.to_undirected()

    def to_node_link_data(self) -> dict:
        return nx.node_link_data(self._graph)

    def from_node_link_data(self, data: dict) -> None:
        self._graph = nx.node_link_graph(data)
