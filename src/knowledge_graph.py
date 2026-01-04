"""
Knowledge Graph module for GraphRAG.

Builds and manages a knowledge graph using NetworkX.
Supports entity merging, community detection, and graph persistence.
"""

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import ollama

from .extract_entities import Entity, Relationship, ExtractionResult
from .graph_config import get_graph_config, GraphConfig
from .constants import INDEX_DIR, LLM
from .utils.logger import logger
from .utils.rag_utils import get_embedding


# Directory for graph storage
GRAPH_DIR = Path(INDEX_DIR) / "knowledge_graph"


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    name: str
    type: str
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    community_id: Optional[int] = None
    embedding: Optional[List[float]] = None


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str
    description: str = ""
    weight: float = 1.0
    source_chunks: List[str] = field(default_factory=list)


@dataclass
class Community:
    """Community of related nodes."""
    id: int
    node_ids: List[str] = field(default_factory=list)
    summary: str = ""
    level: int = 0


class KnowledgeGraph:
    """
    Knowledge Graph implementation using NetworkX.

    Features:
    - Entity deduplication and merging
    - Community detection (Louvain algorithm)
    - Graph persistence
    - Multi-hop traversal
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        """Initialize the knowledge graph."""
        self.config = config or get_graph_config()
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.communities: Dict[int, Community] = {}
        self._name_to_id: Dict[str, str] = {}  # normalized name -> node id

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        return name.lower().strip()

    def _find_existing_node(self, name: str) -> Optional[str]:
        """Find existing node by name (case-insensitive)."""
        normalized = self._normalize_name(name)
        return self._name_to_id.get(normalized)

    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the graph, merging with existing if similar.

        Returns the node ID (new or existing).
        """
        normalized_name = self._normalize_name(entity.name)

        # Check for existing node
        existing_id = self._name_to_id.get(normalized_name)
        if existing_id:
            # Merge with existing node
            existing_node = self.nodes[existing_id]
            if entity.source_chunk not in existing_node.source_chunks:
                existing_node.source_chunks.append(entity.source_chunk)
            # Update description if new one is longer
            if len(entity.description) > len(existing_node.description):
                existing_node.description = entity.description
            return existing_id

        # Create new node
        node = GraphNode(
            id=entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
            attributes=entity.attributes,
            source_chunks=[entity.source_chunk] if entity.source_chunk else [],
        )

        self.nodes[node.id] = node
        self._name_to_id[normalized_name] = node.id

        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            name=node.name,
            type=node.type,
            description=node.description,
        )

        return node.id

    def add_relationship(self, relationship: Relationship) -> bool:
        """
        Add a relationship to the graph.

        Returns True if successful, False if entities not found.
        """
        # Find source and target nodes
        source_id = self._find_existing_node(relationship.source_entity)
        target_id = self._find_existing_node(relationship.target_entity)

        if not source_id or not target_id:
            logger.warning(
                f"Cannot add relationship: entities not found - "
                f"'{relationship.source_entity}' -> '{relationship.target_entity}'"
            )
            return False

        # Check for existing edge
        if self.graph.has_edge(source_id, target_id):
            # Update existing edge
            edge_data = self.graph.edges[source_id, target_id]
            edge_data["weight"] = edge_data.get("weight", 1.0) + 1
            if relationship.source_chunk:
                sources = edge_data.get("source_chunks", [])
                if relationship.source_chunk not in sources:
                    sources.append(relationship.source_chunk)
                    edge_data["source_chunks"] = sources
            return True

        # Create new edge
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship.relationship_type,
            description=relationship.description,
            source_chunks=[relationship.source_chunk] if relationship.source_chunk else [],
        )
        self.edges.append(edge)

        # Add to NetworkX graph
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=edge.relationship_type,
            description=edge.description,
            weight=edge.weight,
            source_chunks=edge.source_chunks,
        )

        return True

    def build_from_extraction_results(
        self,
        results: List[ExtractionResult],
    ) -> Dict[str, int]:
        """
        Build the graph from extraction results.

        Returns statistics about the build process.
        """
        stats = {
            "entities_added": 0,
            "entities_merged": 0,
            "relationships_added": 0,
            "relationships_failed": 0,
        }

        # First pass: add all entities
        for result in results:
            for entity in result.entities:
                existing = self._find_existing_node(entity.name)
                self.add_entity(entity)
                if existing:
                    stats["entities_merged"] += 1
                else:
                    stats["entities_added"] += 1

        # Second pass: add all relationships
        for result in results:
            for rel in result.relationships:
                if self.add_relationship(rel):
                    stats["relationships_added"] += 1
                else:
                    stats["relationships_failed"] += 1

        logger.info(
            f"Graph built: {stats['entities_added']} entities, "
            f"{stats['entities_merged']} merged, "
            f"{stats['relationships_added']} relationships"
        )

        return stats

    def detect_communities(self) -> int:
        """
        Detect communities using Louvain algorithm.

        Returns number of communities found.
        """
        if len(self.graph) == 0:
            return 0

        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()

        try:
            # Use Louvain algorithm
            from networkx.algorithms.community import louvain_communities

            resolution = self.config.graph.community_resolution
            communities = louvain_communities(
                undirected,
                resolution=resolution,
                seed=42,
            )

            # Assign community IDs to nodes
            self.communities.clear()
            for i, community_nodes in enumerate(communities):
                community = Community(
                    id=i,
                    node_ids=list(community_nodes),
                    level=0,
                )
                self.communities[i] = community

                # Update nodes with community ID
                for node_id in community_nodes:
                    if node_id in self.nodes:
                        self.nodes[node_id].community_id = i

            logger.info(f"Detected {len(self.communities)} communities")
            return len(self.communities)

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return 0

    def generate_community_summaries(self, model: str = None) -> None:
        """Generate summaries for each community using LLM."""
        if not self.communities:
            return

        if model is None:
            model = LLM

        for community_id, community in self.communities.items():
            if not community.node_ids:
                continue

            # Get nodes in community
            nodes_info = []
            for node_id in community.node_ids[:20]:  # Limit to 20 nodes
                node = self.nodes.get(node_id)
                if node:
                    nodes_info.append(f"- {node.name} ({node.type}): {node.description}")

            nodes_str = "\n".join(nodes_info)

            # Get relationships within community
            relationships_info = []
            community_set = set(community.node_ids)
            for src, tgt, data in self.graph.edges(data=True):
                if src in community_set and tgt in community_set:
                    src_node = self.nodes.get(src)
                    tgt_node = self.nodes.get(tgt)
                    if src_node and tgt_node:
                        rel_type = data.get("relationship_type", "RELATED_TO")
                        relationships_info.append(
                            f"- {src_node.name} {rel_type} {tgt_node.name}"
                        )

            rels_str = "\n".join(relationships_info[:15])  # Limit relationships

            prompt = f"""Summarize the following group of related entities and their relationships in 2-3 sentences.

ENTITIES:
{nodes_str}

RELATIONSHIPS:
{rels_str}

SUMMARY:"""

            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3},
                )
                community.summary = response["message"]["content"].strip()
                logger.debug(f"Generated summary for community {community_id}")

            except Exception as e:
                logger.error(f"Failed to generate summary for community {community_id}: {e}")
                community.summary = f"Community of {len(community.node_ids)} related entities"

    def get_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Get a node by its name (case-insensitive)."""
        node_id = self._find_existing_node(name)
        return self.nodes.get(node_id) if node_id else None

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        direction: str = "both",
    ) -> List[Tuple[GraphNode, str, int]]:
        """
        Get neighboring nodes up to a certain depth.

        Args:
            node_id: Starting node ID
            depth: Maximum traversal depth
            direction: "in", "out", or "both"

        Returns:
            List of (node, relationship_type, distance) tuples
        """
        if node_id not in self.graph:
            return []

        visited = set()
        results = []
        queue = [(node_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            # Get neighbors based on direction
            neighbors = []
            if direction in ("out", "both"):
                for successor in self.graph.successors(current_id):
                    edge_data = self.graph.edges[current_id, successor]
                    neighbors.append((successor, edge_data.get("relationship_type", ""), "out"))

            if direction in ("in", "both"):
                for predecessor in self.graph.predecessors(current_id):
                    edge_data = self.graph.edges[predecessor, current_id]
                    neighbors.append((predecessor, edge_data.get("relationship_type", ""), "in"))

            for neighbor_id, rel_type, _ in neighbors:
                if neighbor_id not in visited:
                    node = self.nodes.get(neighbor_id)
                    if node:
                        results.append((node, rel_type, current_depth + 1))
                    queue.append((neighbor_id, current_depth + 1))

        return results

    def find_paths(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = None,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find paths between two entities.

        Returns list of paths, where each path is a list of
        (source_name, relationship, target_name) tuples.
        """
        if max_depth is None:
            max_depth = self.config.graph.max_traversal_depth

        source_id = self._find_existing_node(source_name)
        target_id = self._find_existing_node(target_name)

        if not source_id or not target_id:
            return []

        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_depth,
            ))

            result = []
            for path in paths[:10]:  # Limit to 10 paths
                path_tuples = []
                for i in range(len(path) - 1):
                    src_node = self.nodes.get(path[i])
                    tgt_node = self.nodes.get(path[i + 1])
                    edge_data = self.graph.edges.get((path[i], path[i + 1]), {})
                    rel_type = edge_data.get("relationship_type", "RELATED_TO")

                    if src_node and tgt_node:
                        path_tuples.append((src_node.name, rel_type, tgt_node.name))

                if path_tuples:
                    result.append(path_tuples)

            return result

        except nx.NetworkXNoPath:
            return []

    def get_community_nodes(self, community_id: int) -> List[GraphNode]:
        """Get all nodes in a community."""
        community = self.communities.get(community_id)
        if not community:
            return []

        return [
            self.nodes[nid]
            for nid in community.node_ids
            if nid in self.nodes
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": self.graph.number_of_edges(),
            "total_communities": len(self.communities),
            "node_types": dict(self._count_node_types()),
            "relationship_types": dict(self._count_relationship_types()),
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / len(self.graph)
                if len(self.graph) > 0 else 0
            ),
        }

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.type] += 1
        return dict(counts)

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            counts[data.get("relationship_type", "UNKNOWN")] += 1
        return dict(counts)

    def save(self, path: str = None) -> None:
        """Save the knowledge graph to disk."""
        if path is None:
            path = str(GRAPH_DIR)

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save NetworkX graph using pickle (write_gpickle is deprecated)
        with open(save_path / "graph.gpickle", "wb") as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

        # Save nodes
        nodes_data = {nid: {
            "id": n.id,
            "name": n.name,
            "type": n.type,
            "description": n.description,
            "attributes": n.attributes,
            "source_chunks": n.source_chunks,
            "community_id": n.community_id,
        } for nid, n in self.nodes.items()}

        with open(save_path / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)

        # Save communities
        communities_data = {str(cid): {
            "id": c.id,
            "node_ids": c.node_ids,
            "summary": c.summary,
            "level": c.level,
        } for cid, c in self.communities.items()}

        with open(save_path / "communities.json", "w", encoding="utf-8") as f:
            json.dump(communities_data, f, ensure_ascii=False, indent=2)

        # Save name mapping
        with open(save_path / "name_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self._name_to_id, f, ensure_ascii=False, indent=2)

        logger.info(f"Knowledge graph saved to {save_path}")

    @classmethod
    def load(cls, path: str = None, config: Optional[GraphConfig] = None) -> "KnowledgeGraph":
        """Load knowledge graph from disk."""
        if path is None:
            path = str(GRAPH_DIR)

        load_path = Path(path)
        if not load_path.exists():
            logger.warning(f"Graph path not found: {load_path}")
            return cls(config)

        kg = cls(config)

        # Load NetworkX graph using pickle (read_gpickle is deprecated)
        graph_file = load_path / "graph.gpickle"
        if graph_file.exists():
            with open(graph_file, "rb") as f:
                kg.graph = pickle.load(f)

        # Load nodes
        nodes_file = load_path / "nodes.json"
        if nodes_file.exists():
            with open(nodes_file, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)

            for nid, data in nodes_data.items():
                kg.nodes[nid] = GraphNode(
                    id=data["id"],
                    name=data["name"],
                    type=data["type"],
                    description=data.get("description", ""),
                    attributes=data.get("attributes", {}),
                    source_chunks=data.get("source_chunks", []),
                    community_id=data.get("community_id"),
                )

        # Load communities
        communities_file = load_path / "communities.json"
        if communities_file.exists():
            with open(communities_file, "r", encoding="utf-8") as f:
                communities_data = json.load(f)

            for cid, data in communities_data.items():
                kg.communities[int(cid)] = Community(
                    id=data["id"],
                    node_ids=data["node_ids"],
                    summary=data.get("summary", ""),
                    level=data.get("level", 0),
                )

        # Load name mapping
        mapping_file = load_path / "name_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                kg._name_to_id = json.load(f)

        logger.info(
            f"Loaded knowledge graph: {len(kg.nodes)} nodes, "
            f"{kg.graph.number_of_edges()} edges, "
            f"{len(kg.communities)} communities"
        )

        return kg


def build_knowledge_graph(
    extraction_results: List[ExtractionResult],
    config: Optional[GraphConfig] = None,
    detect_communities: bool = True,
    generate_summaries: bool = True,
    save_graph: bool = True,
    output_path: str = None,
) -> KnowledgeGraph:
    """
    Build a complete knowledge graph from extraction results.

    Args:
        extraction_results: List of entity/relationship extractions
        config: Graph configuration
        detect_communities: Whether to run community detection
        generate_summaries: Whether to generate community summaries
        save_graph: Whether to save the graph to disk
        output_path: Custom output path (uses default if not specified)

    Returns:
        Built KnowledgeGraph instance
    """
    kg = KnowledgeGraph(config)

    # Build graph from extractions
    stats = kg.build_from_extraction_results(extraction_results)

    # Detect communities if enabled
    if detect_communities and kg.config.graph.enable_communities:
        num_communities = kg.detect_communities()
        stats["communities"] = num_communities

        # Generate summaries if enabled
        if generate_summaries and kg.config.graph.generate_community_summaries:
            kg.generate_community_summaries()

    # Save if requested
    if save_graph:
        kg.save(output_path)

    logger.info(f"Knowledge graph complete: {kg.get_statistics()}")
    return kg
