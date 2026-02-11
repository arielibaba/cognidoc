"""
Knowledge Graph module for GraphRAG.

Builds and manages a knowledge graph with pluggable backends (NetworkX or Kùzu).
Supports entity merging, community detection, and graph persistence.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .extract_entities import Entity, Relationship, ExtractionResult
from .graph_backend import GraphBackend
from .graph_config import get_graph_config, GraphConfig
from .constants import INDEX_DIR, EMBED_MODEL, MAX_CONSECUTIVE_QUOTA_ERRORS, GRAPH_BACKEND
from .utils.llm_client import llm_chat
from .utils.logger import logger
from .utils.rag_utils import get_embedding
from .utils.error_classifier import is_quota_or_rate_error, get_error_info, ErrorType


# Directory for graph storage
GRAPH_DIR = Path(INDEX_DIR) / "knowledge_graph"


def _create_backend(backend_name: str = None) -> GraphBackend:
    """Factory: create a GraphBackend from its name."""
    if backend_name is None:
        backend_name = GRAPH_BACKEND

    if backend_name == "kuzu":
        from .graph_backend_kuzu import KuzuBackend

        return KuzuBackend()
    else:
        from .graph_backend_networkx import NetworkXBackend

        return NetworkXBackend()


@dataclass
class GraphNode:
    """
    Node in the knowledge graph.

    Attributes:
        id: Unique identifier for the node
        name: Human-readable name of the entity (canonical name after resolution)
        type: Entity type (e.g., "PERSON", "CONCEPT")
        description: Brief description of the entity
        attributes: Additional key-value attributes
        source_chunks: List of chunk IDs where this entity was found
        community_id: ID of the community this node belongs to
        embedding: Pre-computed embedding vector for similarity search
        aliases: Alternative names for this entity (populated by entity resolution)
        merged_from: List of original entity IDs that were merged into this one
    """

    id: str
    name: str
    type: str
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    community_id: Optional[int] = None
    embedding: Optional[List[float]] = None
    aliases: List[str] = field(default_factory=list)
    merged_from: List[str] = field(default_factory=list)

    def __hash__(self):
        """Make GraphNode hashable by using its id."""
        return hash(self.id)

    def __eq__(self, other):
        """Compare nodes by id."""
        if isinstance(other, GraphNode):
            return self.id == other.id
        return False

    def matches_name(self, query: str) -> bool:
        """Check if query matches name or any alias (case-insensitive)."""
        query_lower = query.lower().strip()
        if self.name.lower().strip() == query_lower:
            return True
        return any(alias.lower().strip() == query_lower for alias in self.aliases)


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
    """
    Community of related nodes detected via community detection algorithm.

    Attributes:
        id: Unique community identifier
        node_ids: List of node IDs belonging to this community
        summary: LLM-generated summary of the community's topic
        level: Hierarchical level (0 = base level)
        embedding: Pre-computed embedding of the summary for fast similarity search
    """

    id: int
    node_ids: List[str] = field(default_factory=list)
    summary: str = ""
    level: int = 0
    embedding: Optional[List[float]] = None


class KnowledgeGraph:
    """
    Knowledge Graph with pluggable backend (NetworkX or Kùzu).

    Features:
    - Entity deduplication and merging
    - Community detection (Louvain algorithm)
    - Graph persistence
    - Multi-hop traversal
    """

    def __init__(self, config: Optional[GraphConfig] = None, backend: str = None):
        """Initialize the knowledge graph."""
        self.config = config or get_graph_config()
        self._backend: GraphBackend = _create_backend(backend)
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.communities: Dict[int, Community] = {}
        self._name_to_id: Dict[str, str] = {}  # normalized name -> node id

    @property
    def backend_name(self) -> str:
        """Return the backend type name."""
        return type(self._backend).__name__

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        return name.lower().strip()

    def _find_existing_node(self, name: str) -> Optional[str]:
        """Find existing node by name (case-insensitive)."""
        normalized = self._normalize_name(name)
        return self._name_to_id.get(normalized)

    # ── Public wrappers for entity_resolution / external access ──────────

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the backend graph."""
        return self._backend.has_node(node_id)

    def get_successors(self, node_id: str) -> List[str]:
        """Get outgoing neighbor IDs."""
        return self._backend.successors(node_id)

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get incoming neighbor IDs."""
        return self._backend.predecessors(node_id)

    def get_edge_data(self, src: str, tgt: str) -> Optional[Dict[str, Any]]:
        """Get edge attributes dict (or None)."""
        return self._backend.get_edge_data(src, tgt)

    def has_edge(self, src: str, tgt: str) -> bool:
        """Check if an edge exists."""
        return self._backend.has_edge(src, tgt)

    def add_edge_raw(self, src: str, tgt: str, **attrs) -> None:
        """Add an edge with raw attributes (low-level)."""
        self._backend.add_edge(src, tgt, **attrs)

    def remove_graph_node(self, node_id: str) -> None:
        """Remove a node from the backend graph."""
        self._backend.remove_node(node_id)

    def remove_graph_edge(self, src: str, tgt: str) -> None:
        """Remove an edge from the backend graph."""
        self._backend.remove_edge(src, tgt)

    def update_graph_node_attrs(self, node_id: str, **attrs) -> None:
        """Update attributes on a backend graph node."""
        self._backend.update_node_attrs(node_id, **attrs)

    def update_edge_attrs(self, src: str, tgt: str, **attrs) -> None:
        """Update attributes on a backend graph edge."""
        self._backend.update_edge_attrs(src, tgt, **attrs)

    def iter_edges(self, data: bool = False):
        """Iterate over backend edges."""
        return self._backend.iter_edges(data=data)

    def number_of_edges(self) -> int:
        """Return total number of edges in the backend."""
        return self._backend.number_of_edges()

    # ── CRUD ─────────────────────────────────────────────────────────────

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

        # Add to backend graph
        self._backend.add_node(
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
        if self._backend.has_edge(source_id, target_id):
            # Update existing edge
            edge_data = self._backend.get_edge_data(source_id, target_id)
            new_weight = edge_data.get("weight", 1.0) + 1
            update_attrs = {"weight": new_weight}
            if relationship.source_chunk:
                sources = edge_data.get("source_chunks", [])
                if relationship.source_chunk not in sources:
                    sources.append(relationship.source_chunk)
                    update_attrs["source_chunks"] = sources
            self._backend.update_edge_attrs(source_id, target_id, **update_attrs)
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

        # Add to backend graph
        self._backend.add_edge(
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
        if self._backend.number_of_nodes() == 0:
            return 0

        # Convert to undirected for community detection
        undirected = self._backend.to_undirected_networkx()

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

    def generate_community_summaries(
        self,
        compute_embeddings: bool = True,
        skip_existing: bool = False,
        # Checkpoint parameters
        processed_community_ids: Optional[Set[int]] = None,
        max_consecutive_quota_errors: int = None,
        # Periodic save parameters
        save_interval: int = 100,
        save_path: str = None,
    ) -> Tuple[int, int, int, bool]:
        """
        Generate summaries and embeddings for each community using LLM.

        Uses the unified LLM client (Gemini by default).
        Embeddings are computed in batch for better performance.

        Supports checkpoint/resume: skips already-processed communities and stops
        gracefully when quota errors are detected.

        IMPORTANT: Periodically saves the graph to disk to prevent data loss
        in case of unexpected interruption.

        Args:
            compute_embeddings: Whether to pre-compute embeddings for fast retrieval
            skip_existing: If True, skip communities that already have a valid summary
                          (not the default "Community of X related entities" placeholder)
            processed_community_ids: Set of community IDs already processed (for resume)
            max_consecutive_quota_errors: Stop after this many consecutive quota errors
            save_interval: Save graph to disk every N communities (default: 100)
            save_path: Path to save graph (defaults to GRAPH_DIR)

        Returns:
            Tuple of (generated, skipped, quota_errors, interrupted)
        """
        import asyncio

        if max_consecutive_quota_errors is None:
            max_consecutive_quota_errors = MAX_CONSECUTIVE_QUOTA_ERRORS

        if processed_community_ids is None:
            processed_community_ids = set()

        if not self.communities:
            return 0, 0, 0, False

        skipped = 0
        generated = 0
        quota_errors = 0
        consecutive_quota_errors = 0
        interrupted = False
        communities_to_embed = []  # Collect for batch embedding
        last_save_count = 0  # Track when we last saved

        for community_id, community in self.communities.items():
            if not community.node_ids:
                continue

            # Skip already processed communities (resume support)
            if community_id in processed_community_ids:
                skipped += 1
                continue

            # Skip communities with existing valid summaries
            if (
                skip_existing
                and community.summary
                and not community.summary.startswith("Community of ")
            ):
                skipped += 1
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
            for src, tgt, data in self._backend.iter_edges(data=True):
                if src in community_set and tgt in community_set:
                    src_node = self.nodes.get(src)
                    tgt_node = self.nodes.get(tgt)
                    if src_node and tgt_node:
                        rel_type = data.get("relationship_type", "RELATED_TO")
                        relationships_info.append(f"- {src_node.name} {rel_type} {tgt_node.name}")

            rels_str = "\n".join(relationships_info[:15])  # Limit relationships

            prompt = f"""Summarize the following group of related entities and their relationships in 2-3 sentences.

ENTITIES:
{nodes_str}

RELATIONSHIPS:
{rels_str}

SUMMARY:"""

            try:
                response = llm_chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                community.summary = response.strip()
                generated += 1
                consecutive_quota_errors = 0  # Reset on success
                logger.debug(f"Generated summary for community {community_id}")

                # Collect for batch embedding
                if compute_embeddings and community.summary:
                    communities_to_embed.append((community_id, community.summary))

                # Periodic save to prevent data loss
                if save_interval > 0 and (generated - last_save_count) >= save_interval:
                    logger.info(f"Periodic save: {generated} community summaries generated...")
                    self.save(save_path)
                    last_save_count = generated

            except Exception as e:
                error_type, error_msg = get_error_info(e)
                logger.error(f"Failed to generate summary for community {community_id}: {e}")

                if error_type in (ErrorType.QUOTA_EXHAUSTED, ErrorType.RATE_LIMITED):
                    quota_errors += 1
                    consecutive_quota_errors += 1
                    logger.warning(
                        f"Quota/rate error ({consecutive_quota_errors}/{max_consecutive_quota_errors})"
                    )

                    if consecutive_quota_errors >= max_consecutive_quota_errors:
                        logger.error(
                            f"Stopping community summary generation: {consecutive_quota_errors} consecutive quota errors."
                        )
                        interrupted = True
                        break
                else:
                    consecutive_quota_errors = 0

                community.summary = f"Community of {len(community.node_ids)} related entities"

        # Batch compute embeddings for all communities at once
        if communities_to_embed and not interrupted:
            logger.info(
                f"Computing embeddings for {len(communities_to_embed)} community summaries..."
            )
            from .utils.embedding_providers import get_embedding_provider, OllamaEmbeddingProvider

            provider = get_embedding_provider()

            if isinstance(provider, OllamaEmbeddingProvider):
                # Use batched async embedding
                try:
                    texts = [summary for _, summary in communities_to_embed]
                    from .utils.async_utils import run_coroutine

                    coro = provider.embed_async(texts, max_concurrent=4)
                    embeddings = run_coroutine(coro)
                    for (comm_id, _), embedding in zip(communities_to_embed, embeddings):
                        if embedding:
                            self.communities[comm_id].embedding = embedding
                except Exception as e:
                    logger.warning(f"Batch embedding failed: {e}, falling back to sequential")
                    for comm_id, summary in communities_to_embed:
                        try:
                            self.communities[comm_id].embedding = get_embedding(
                                summary, EMBED_MODEL
                            )
                        except Exception as e2:
                            logger.warning(f"Failed to embed community {comm_id}: {e2}")
            else:
                # Sequential fallback for other providers
                for comm_id, summary in communities_to_embed:
                    try:
                        self.communities[comm_id].embedding = get_embedding(summary, EMBED_MODEL)
                    except Exception as e:
                        logger.warning(f"Failed to embed community {comm_id}: {e}")

        if interrupted:
            logger.warning(
                f"Community summaries interrupted: {generated} generated, {skipped} skipped, "
                f"{quota_errors} quota errors. Resume to continue."
            )
        elif skip_existing or processed_community_ids:
            logger.info(f"Community summaries: {generated} generated, {skipped} skipped")
        else:
            logger.info(f"Community summaries: {generated} generated")

        return generated, skipped, quota_errors, interrupted

    def compute_entity_embeddings(
        self,
        skip_existing: bool = True,
        batch_size: int = 50,
        # Checkpoint parameters
        processed_entity_ids: Optional[Set[str]] = None,
        max_consecutive_quota_errors: int = None,
    ) -> Tuple[int, int, int, bool]:
        """
        Pre-compute embeddings for all entities using batched async requests.

        Embeddings are computed from "{name}: {description}" text.
        This enables fast semantic matching at query time.

        Uses batch async embedding for 5-10x faster processing compared to
        sequential embedding calls.

        Supports checkpoint/resume: skips already-processed entities and stops
        gracefully when quota errors are detected.

        Args:
            skip_existing: If True, skip entities that already have embeddings
            batch_size: Number of entities to embed per batch (default: 50)
            processed_entity_ids: Set of entity IDs already processed (for resume)
            max_consecutive_quota_errors: Stop after this many consecutive quota errors

        Returns:
            Tuple of (computed, skipped, quota_errors, interrupted)
        """
        import asyncio

        if max_consecutive_quota_errors is None:
            max_consecutive_quota_errors = MAX_CONSECUTIVE_QUOTA_ERRORS

        if processed_entity_ids is None:
            processed_entity_ids = set()

        if not self.nodes:
            return 0, 0, 0, False

        # Collect entities that need embeddings
        to_embed = []
        skipped = 0

        for node_id, node in self.nodes.items():
            # Skip already processed entities (resume support)
            if node_id in processed_entity_ids:
                skipped += 1
                continue
            if skip_existing and node.embedding is not None:
                skipped += 1
                continue
            # Build text for embedding: name + description
            text = node.name
            if node.description:
                text = f"{node.name}: {node.description}"
            to_embed.append((node_id, text))

        if not to_embed:
            logger.info(f"Entity embeddings: 0 computed, {skipped} skipped (all cached)")
            return 0, skipped, 0, False

        logger.info(
            f"Computing embeddings for {len(to_embed)} entities (batch_size={batch_size})..."
        )

        # Get embedding provider
        from .utils.embedding_providers import get_embedding_provider, OllamaEmbeddingProvider

        provider = get_embedding_provider()

        computed = 0
        errors = 0
        quota_errors = 0
        consecutive_quota_errors = 0
        interrupted = False

        # Check if provider supports async embedding
        if isinstance(provider, OllamaEmbeddingProvider):
            # Process in batches
            for i in range(0, len(to_embed), batch_size):
                if interrupted:
                    break

                batch = to_embed[i : i + batch_size]
                batch_texts = [text for _, text in batch]

                try:
                    # Run async batch embedding
                    from .utils.async_utils import run_coroutine

                    coro = provider.embed_async(batch_texts, max_concurrent=4)
                    embeddings = run_coroutine(coro)

                    # Assign embeddings to nodes
                    for (node_id, _), embedding in zip(batch, embeddings):
                        if embedding:
                            self.nodes[node_id].embedding = embedding
                            computed += 1

                    consecutive_quota_errors = 0  # Reset on success

                except Exception as e:
                    error_type, error_msg = get_error_info(e)

                    if error_type in (ErrorType.QUOTA_EXHAUSTED, ErrorType.RATE_LIMITED):
                        quota_errors += 1
                        consecutive_quota_errors += 1
                        logger.warning(
                            f"Quota/rate error ({consecutive_quota_errors}/{max_consecutive_quota_errors}): {e}"
                        )

                        if consecutive_quota_errors >= max_consecutive_quota_errors:
                            logger.error(
                                f"Stopping entity embedding: {consecutive_quota_errors} consecutive quota errors."
                            )
                            interrupted = True
                            break
                    else:
                        consecutive_quota_errors = 0

                    logger.warning(f"Batch embedding failed: {e}, falling back to sequential")
                    # Fallback to sequential for this batch
                    for node_id, text in batch:
                        try:
                            self.nodes[node_id].embedding = get_embedding(text, EMBED_MODEL)
                            computed += 1
                            consecutive_quota_errors = 0
                        except Exception as e2:
                            error_type2, _ = get_error_info(e2)
                            logger.warning(
                                f"Failed to compute embedding for {self.nodes[node_id].name}: {e2}"
                            )
                            errors += 1

                            if error_type2 in (ErrorType.QUOTA_EXHAUSTED, ErrorType.RATE_LIMITED):
                                quota_errors += 1
                                consecutive_quota_errors += 1
                                if consecutive_quota_errors >= max_consecutive_quota_errors:
                                    interrupted = True
                                    break
        else:
            # Fallback to sequential embedding for non-Ollama providers
            for node_id, text in to_embed:
                if interrupted:
                    break

                try:
                    self.nodes[node_id].embedding = get_embedding(text, EMBED_MODEL)
                    computed += 1
                    consecutive_quota_errors = 0
                except Exception as e:
                    error_type, _ = get_error_info(e)
                    logger.warning(
                        f"Failed to compute embedding for {self.nodes[node_id].name}: {e}"
                    )
                    errors += 1

                    if error_type in (ErrorType.QUOTA_EXHAUSTED, ErrorType.RATE_LIMITED):
                        quota_errors += 1
                        consecutive_quota_errors += 1
                        if consecutive_quota_errors >= max_consecutive_quota_errors:
                            logger.error(
                                f"Stopping entity embedding: {consecutive_quota_errors} consecutive quota errors."
                            )
                            interrupted = True

        if interrupted:
            logger.warning(
                f"Entity embeddings interrupted: {computed} computed, {skipped} skipped, "
                f"{quota_errors} quota errors. Resume to continue."
            )
        else:
            logger.info(
                f"Entity embeddings: {computed} computed, {skipped} skipped, {errors} errors"
            )

        return computed, skipped, quota_errors, interrupted

    def find_similar_entities(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[GraphNode, float]]:
        """
        Find entities semantically similar to the query.

        Uses pre-computed entity embeddings for fast similarity search.
        Embeddings are computed lazily on first call if not already available.

        Args:
            query: Query text
            top_k: Maximum number of results
            threshold: Minimum similarity score

        Returns:
            List of (entity, similarity_score) tuples, sorted by score
        """
        import numpy as np
        from .utils.rag_utils import get_query_embedding

        # Lazy compute entity embeddings if needed (on first semantic search)
        entities_with_embeddings = sum(1 for n in self.nodes.values() if n.embedding is not None)
        if entities_with_embeddings == 0 and len(self.nodes) > 0:
            logger.info("Lazy loading entity embeddings (first semantic search)...")
            self.compute_entity_embeddings()

        # Get query embedding
        try:
            query_emb = np.array(get_query_embedding(query))
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return []
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return []

        # Score all entities with embeddings
        scored = []
        for node in self.nodes.values():
            if node.embedding is None:
                continue

            entity_emb = np.array(node.embedding)
            entity_norm = np.linalg.norm(entity_emb)
            if entity_norm == 0:
                continue

            similarity = float(np.dot(query_emb, entity_emb) / (query_norm * entity_norm))
            if similarity >= threshold:
                scored.append((node, similarity))

        # Sort by score and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

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
        if not self._backend.has_node(node_id):
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
                for successor in self._backend.successors(current_id):
                    edge_data = self._backend.get_edge_data(current_id, successor) or {}
                    neighbors.append((successor, edge_data.get("relationship_type", ""), "out"))

            if direction in ("in", "both"):
                for predecessor in self._backend.predecessors(current_id):
                    edge_data = self._backend.get_edge_data(predecessor, current_id) or {}
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
            # Find all simple paths via backend
            paths = self._backend.find_all_simple_paths(source_id, target_id, cutoff=max_depth)

            result = []
            for path in paths[:10]:  # Limit to 10 paths
                path_tuples = []
                for i in range(len(path) - 1):
                    src_node = self.nodes.get(path[i])
                    tgt_node = self.nodes.get(path[i + 1])
                    edge_data = self._backend.get_edge_data(path[i], path[i + 1]) or {}
                    rel_type = edge_data.get("relationship_type", "RELATED_TO")

                    if src_node and tgt_node:
                        path_tuples.append((src_node.name, rel_type, tgt_node.name))

                if path_tuples:
                    result.append(path_tuples)

            return result

        except Exception:
            return []

    def get_community_nodes(self, community_id: int) -> List[GraphNode]:
        """Get all nodes in a community."""
        community = self.communities.get(community_id)
        if not community:
            return []

        return [self.nodes[nid] for nid in community.node_ids if nid in self.nodes]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        degrees = self._backend.degree()
        n_nodes = self._backend.number_of_nodes()
        return {
            "total_nodes": len(self.nodes),
            "total_edges": self._backend.number_of_edges(),
            "total_communities": len(self.communities),
            "node_types": dict(self._count_node_types()),
            "relationship_types": dict(self._count_relationship_types()),
            "avg_degree": (sum(degrees.values()) / n_nodes if n_nodes > 0 else 0),
        }

    def prune_by_source_stems(self, deleted_stems: Set[str]) -> Dict[str, int]:
        """
        Remove entities and edges whose source_chunks all belong to deleted file stems.

        For nodes/edges with mixed sources, only the deleted chunks are removed.

        Args:
            deleted_stems: Set of file stems (e.g. {"report_2023", "memo_draft"})

        Returns:
            Stats dict with nodes_removed, nodes_updated, edges_removed, edges_updated
        """
        if not deleted_stems:
            return {"nodes_removed": 0, "nodes_updated": 0, "edges_removed": 0, "edges_updated": 0}

        stats = {"nodes_removed": 0, "nodes_updated": 0, "edges_removed": 0, "edges_updated": 0}

        def _chunk_belongs_to_stems(chunk_id: str, stems: Set[str]) -> bool:
            """Check if a chunk ID belongs to any of the given stems.

            Chunk IDs follow the pattern: {stem}_chunk_{N} (e.g. "report_2023_chunk_5").
            We match by checking if the chunk starts with "{stem}_chunk_".
            """
            for stem in stems:
                if chunk_id.startswith(f"{stem}_chunk_"):
                    return True
            return False

        # Phase 1: Process edges (snapshot to avoid mutation during iteration)
        edge_snapshot = list(self._backend.iter_edges(data=True))
        for src, tgt, data in edge_snapshot:
            source_chunks = data.get("source_chunks", [])
            if not source_chunks:
                continue

            matching = [c for c in source_chunks if _chunk_belongs_to_stems(c, deleted_stems)]
            if not matching:
                continue

            remaining = [c for c in source_chunks if c not in matching]
            if not remaining:
                # All chunks belong to deleted stems — remove edge
                self._backend.remove_edge(src, tgt)
                stats["edges_removed"] += 1
            else:
                # Partial — update source_chunks
                self._backend.update_edge_attrs(src, tgt, source_chunks=remaining)
                stats["edges_updated"] += 1

        # Clean self.edges list
        self.edges = [
            e
            for e in self.edges
            if not all(_chunk_belongs_to_stems(c, deleted_stems) for c in e.source_chunks)
            or not e.source_chunks
        ]
        # Update source_chunks on remaining edges
        for edge in self.edges:
            if edge.source_chunks:
                edge.source_chunks = [
                    c for c in edge.source_chunks if not _chunk_belongs_to_stems(c, deleted_stems)
                ]

        # Phase 2: Process nodes
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            if not node.source_chunks:
                continue

            matching = [c for c in node.source_chunks if _chunk_belongs_to_stems(c, deleted_stems)]
            if not matching:
                continue

            remaining = [c for c in node.source_chunks if c not in matching]
            if not remaining:
                # All chunks belong to deleted stems — mark for removal
                nodes_to_remove.append(node_id)
            else:
                # Partial — update source_chunks
                node.source_chunks = remaining
                stats["nodes_updated"] += 1

        # Remove nodes
        for node_id in nodes_to_remove:
            node = self.nodes.pop(node_id)
            # Clean _name_to_id
            normalized = self._normalize_name(node.name)
            if self._name_to_id.get(normalized) == node_id:
                del self._name_to_id[normalized]
            # Remove from backend (also removes remaining incident edges)
            if self._backend.has_node(node_id):
                self._backend.remove_node(node_id)
            stats["nodes_removed"] += 1

        logger.info(
            f"Knowledge graph pruned: {stats['nodes_removed']} nodes removed, "
            f"{stats['nodes_updated']} updated, {stats['edges_removed']} edges removed, "
            f"{stats['edges_updated']} updated"
        )

        return stats

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.type] += 1
        return dict(counts)

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = defaultdict(int)
        for _, _, data in self._backend.iter_edges(data=True):
            counts[data.get("relationship_type", "UNKNOWN")] += 1
        return dict(counts)

    def save(self, path: str = None) -> None:
        """Save the knowledge graph to disk."""
        if path is None:
            path = str(GRAPH_DIR)

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save backend graph as JSON (node_link format, safe serialization)
        graph_data = self._backend.to_node_link_data()
        with open(save_path / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # Save nodes (including pre-computed embeddings and resolution data)
        nodes_data = {
            nid: {
                "id": n.id,
                "name": n.name,
                "type": n.type,
                "description": n.description,
                "attributes": n.attributes,
                "source_chunks": n.source_chunks,
                "community_id": n.community_id,
                "embedding": n.embedding,
                "aliases": n.aliases,
                "merged_from": n.merged_from,
            }
            for nid, n in self.nodes.items()
        }

        with open(save_path / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)

        # Save communities (including pre-computed embeddings)
        communities_data = {
            str(cid): {
                "id": c.id,
                "node_ids": c.node_ids,
                "summary": c.summary,
                "level": c.level,
                "embedding": c.embedding,
            }
            for cid, c in self.communities.items()
        }

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

        # Load backend graph from JSON (node_link format)
        graph_json = load_path / "graph.json"
        graph_pickle = load_path / "graph.gpickle"  # Legacy fallback
        if graph_json.exists():
            with open(graph_json, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            kg._backend.from_node_link_data(graph_data)
        elif graph_pickle.exists():
            # Legacy pickle format — load but will be saved as JSON next time
            import pickle

            with open(graph_pickle, "rb") as f:
                legacy_graph = pickle.load(f)
            legacy_data = nx.node_link_data(legacy_graph)
            kg._backend.from_node_link_data(legacy_data)
            logger.info("Loaded legacy pickle graph — will be saved as JSON on next save")

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
                    embedding=data.get("embedding"),  # Load pre-computed embedding
                    aliases=data.get("aliases", []),  # Load aliases from resolution
                    merged_from=data.get("merged_from", []),  # Load merge history
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
                    embedding=data.get("embedding"),  # Load pre-computed embedding
                )

        # Load name mapping
        mapping_file = load_path / "name_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                kg._name_to_id = json.load(f)

        logger.info(
            f"Loaded knowledge graph: {len(kg.nodes)} nodes, "
            f"{kg._backend.number_of_edges()} edges, "
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
    compute_embeddings: bool = False,  # Lazy by default - computed on first semantic search
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
        compute_embeddings: Whether to pre-compute entity embeddings (default: False).
                           When False, embeddings are computed lazily on first semantic search.

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

    # Compute entity embeddings for semantic search (optional - lazy by default)
    if compute_embeddings:
        kg.compute_entity_embeddings()

    # Save if requested
    if save_graph:
        kg.save(output_path)

    logger.info(f"Knowledge graph complete: {kg.get_statistics()}")
    return kg


# =============================================================================
# Backup and Recovery Functions
# =============================================================================


def has_valid_knowledge_graph(path: str = None) -> bool:
    """
    Check if a valid knowledge graph exists at the given path.

    A valid KG has nodes.json with at least one entity.

    Args:
        path: Path to knowledge graph directory (defaults to GRAPH_DIR)

    Returns:
        True if valid KG exists, False otherwise
    """
    if path is None:
        path = str(GRAPH_DIR)

    kg_path = Path(path)
    nodes_file = kg_path / "nodes.json"

    if not nodes_file.exists():
        return False

    try:
        with open(nodes_file, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        # Valid if we have at least one entity
        return len(nodes_data) > 0
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def get_knowledge_graph_stats(path: str = None) -> Dict[str, int]:
    """
    Get basic statistics about an existing knowledge graph.

    Args:
        path: Path to knowledge graph directory (defaults to GRAPH_DIR)

    Returns:
        Dict with counts of nodes, communities, etc.
    """
    if path is None:
        path = str(GRAPH_DIR)

    kg_path = Path(path)
    stats = {"nodes": 0, "communities": 0, "edges": 0}

    # Count nodes
    nodes_file = kg_path / "nodes.json"
    if nodes_file.exists():
        try:
            with open(nodes_file, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)
            stats["nodes"] = len(nodes_data)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not read nodes.json: {e}")

    # Count communities
    communities_file = kg_path / "communities.json"
    if communities_file.exists():
        try:
            with open(communities_file, "r", encoding="utf-8") as f:
                communities_data = json.load(f)
            stats["communities"] = len(communities_data)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not read communities.json: {e}")

    # Count edges from graph JSON (with pickle fallback for legacy data)
    graph_json = kg_path / "graph.json"
    graph_pickle = kg_path / "graph.gpickle"
    if graph_json.exists():
        try:
            with open(graph_json, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            graph = nx.node_link_graph(graph_data)
            stats["edges"] = graph.number_of_edges()
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.debug(f"Could not read graph.json: {e}")
    elif graph_pickle.exists():
        try:
            import pickle

            with open(graph_pickle, "rb") as f:
                graph = pickle.load(f)
            stats["edges"] = graph.number_of_edges()
        except (FileNotFoundError, EOFError, OSError) as e:
            logger.debug(f"Could not read graph.gpickle: {e}")

    return stats


def backup_knowledge_graph(path: str = None, backup_dir: str = None) -> Optional[str]:
    """
    Create a timestamped backup of the knowledge graph.

    Backups are stored in INDEX_DIR/knowledge_graph_backups/

    Args:
        path: Path to knowledge graph directory (defaults to GRAPH_DIR)
        backup_dir: Custom backup directory (defaults to INDEX_DIR/knowledge_graph_backups)

    Returns:
        Path to backup directory if successful, None otherwise
    """
    import shutil
    from datetime import datetime

    if path is None:
        path = str(GRAPH_DIR)

    kg_path = Path(path)

    # Check if there's anything to backup
    if not has_valid_knowledge_graph(path):
        logger.info("No valid knowledge graph to backup")
        return None

    # Get stats for logging
    stats = get_knowledge_graph_stats(path)

    # Create backup directory
    if backup_dir is None:
        backup_dir = Path(INDEX_DIR) / "knowledge_graph_backups"
    else:
        backup_dir = Path(backup_dir)

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"kg_backup_{timestamp}"

    try:
        # Copy entire knowledge_graph directory
        shutil.copytree(kg_path, backup_path)

        logger.info(
            f"Knowledge graph backed up to {backup_path} "
            f"({stats['nodes']} nodes, {stats['edges']} edges, {stats['communities']} communities)"
        )

        # Clean up old backups (keep last 5)
        _cleanup_old_backups(backup_dir, keep=5)

        return str(backup_path)

    except Exception as e:
        logger.error(f"Failed to backup knowledge graph: {e}")
        return None


def _cleanup_old_backups(backup_dir: Path, keep: int = 5) -> None:
    """Remove old backups, keeping the most recent ones."""
    import shutil

    backups = sorted(
        [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("kg_backup_")],
        key=lambda x: x.name,
        reverse=True,
    )

    for old_backup in backups[keep:]:
        try:
            shutil.rmtree(old_backup)
            logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            logger.warning(f"Failed to remove old backup {old_backup}: {e}")


def list_knowledge_graph_backups(backup_dir: str = None) -> List[Dict[str, Any]]:
    """
    List available knowledge graph backups.

    Args:
        backup_dir: Custom backup directory (defaults to INDEX_DIR/knowledge_graph_backups)

    Returns:
        List of backup info dicts with path, timestamp, and stats
    """
    if backup_dir is None:
        backup_dir = Path(INDEX_DIR) / "knowledge_graph_backups"
    else:
        backup_dir = Path(backup_dir)

    if not backup_dir.exists():
        return []

    backups = []
    for d in sorted(backup_dir.iterdir(), key=lambda x: x.name, reverse=True):
        if d.is_dir() and d.name.startswith("kg_backup_"):
            try:
                # Parse timestamp from name
                ts_str = d.name.replace("kg_backup_", "")
                from datetime import datetime

                timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

                stats = get_knowledge_graph_stats(str(d))

                backups.append(
                    {
                        "path": str(d),
                        "name": d.name,
                        "timestamp": timestamp.isoformat(),
                        "nodes": stats["nodes"],
                        "edges": stats["edges"],
                        "communities": stats["communities"],
                    }
                )
            except Exception:
                continue

    return backups


def restore_knowledge_graph_backup(backup_path: str, target_path: str = None) -> bool:
    """
    Restore a knowledge graph from backup.

    Args:
        backup_path: Path to backup directory
        target_path: Target directory (defaults to GRAPH_DIR)

    Returns:
        True if successful, False otherwise
    """
    import shutil

    if target_path is None:
        target_path = str(GRAPH_DIR)

    backup = Path(backup_path)
    target = Path(target_path)

    if not backup.exists():
        logger.error(f"Backup not found: {backup_path}")
        return False

    if not has_valid_knowledge_graph(str(backup)):
        logger.error(f"Backup is not a valid knowledge graph: {backup_path}")
        return False

    try:
        # Remove current KG if exists
        if target.exists():
            shutil.rmtree(target)

        # Copy backup to target
        shutil.copytree(backup, target)

        stats = get_knowledge_graph_stats(target_path)
        logger.info(
            f"Knowledge graph restored from {backup_path} "
            f"({stats['nodes']} nodes, {stats['edges']} edges, {stats['communities']} communities)"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to restore knowledge graph: {e}")
        return False
