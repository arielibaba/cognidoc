"""
Abstract base class for graph backends.

Defines the interface that both NetworkX and Kùzu backends must implement.
KnowledgeGraph delegates all low-level graph operations to a backend.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import networkx as nx


class GraphBackend(ABC):
    """Abstract graph backend interface."""

    # ── Nodes ────────────────────────────────────────────────────────────

    @abstractmethod
    def add_node(self, node_id: str, **attrs) -> None:
        """Add a node with optional attributes."""

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges."""

    @abstractmethod
    def update_node_attrs(self, node_id: str, **attrs) -> None:
        """Update attributes on an existing node."""

    @abstractmethod
    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes of a node."""

    @abstractmethod
    def number_of_nodes(self) -> int:
        """Return total number of nodes."""

    # ── Edges ────────────────────────────────────────────────────────────

    @abstractmethod
    def add_edge(self, src: str, tgt: str, **attrs) -> None:
        """Add a directed edge with optional attributes."""

    @abstractmethod
    def has_edge(self, src: str, tgt: str) -> bool:
        """Check if an edge exists."""

    @abstractmethod
    def get_edge_data(self, src: str, tgt: str) -> Optional[Dict[str, Any]]:
        """Get edge attributes. Returns None if edge doesn't exist."""

    @abstractmethod
    def update_edge_attrs(self, src: str, tgt: str, **attrs) -> None:
        """Update attributes on an existing edge."""

    @abstractmethod
    def iter_edges(self, data: bool = False) -> Iterator:
        """Iterate over edges. If data=True, yields (src, tgt, attrs) tuples."""

    @abstractmethod
    def number_of_edges(self) -> int:
        """Return total number of edges."""

    # ── Traversal ────────────────────────────────────────────────────────

    @abstractmethod
    def successors(self, node_id: str) -> List[str]:
        """Get outgoing neighbor IDs."""

    @abstractmethod
    def predecessors(self, node_id: str) -> List[str]:
        """Get incoming neighbor IDs."""

    @abstractmethod
    def find_all_simple_paths(self, src: str, tgt: str, cutoff: int = 5) -> List[List[str]]:
        """Find all simple paths between two nodes up to cutoff depth."""

    @abstractmethod
    def degree(self) -> Dict[str, int]:
        """Return {node_id: degree} mapping."""

    # ── Export / import ──────────────────────────────────────────────────

    @abstractmethod
    def to_undirected_networkx(self) -> nx.Graph:
        """Export as an undirected NetworkX graph (used for Louvain)."""

    @abstractmethod
    def to_node_link_data(self) -> dict:
        """Serialize to NetworkX node-link JSON format."""

    @abstractmethod
    def from_node_link_data(self, data: dict) -> None:
        """Load from NetworkX node-link JSON format."""
