"""
Graph Retrieval module for GraphRAG.

Provides retrieval capabilities from the knowledge graph including:
- Entity-based retrieval
- Relationship traversal
- Community-based global queries
- Path finding between entities
"""

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .knowledge_graph import KnowledgeGraph, GraphNode, Community
from .graph_config import get_graph_config, GraphConfig
from .constants import EMBED_MODEL
from .utils.llm_client import llm_chat
from .utils.logger import logger
from .utils.rag_utils import get_embedding, get_query_embedding


@dataclass
class GraphRetrievalResult:
    """Result from graph retrieval."""

    query: str
    retrieval_type: str  # "entity", "relationship", "community", "path"
    entities: List[GraphNode] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)  # (source, rel, target)
    communities: List[Community] = field(default_factory=list)
    paths: List[List[Tuple[str, str, str]]] = field(default_factory=list)
    context: str = ""
    confidence: float = 0.0

    def get_context_text(self) -> str:
        """Generate context text from retrieval results."""
        parts = []

        # Add entity information
        if self.entities:
            parts.append("RELEVANT ENTITIES:")
            for entity in self.entities[:10]:
                parts.append(f"- {entity.name} ({entity.type}): {entity.description}")

        # Add relationship information
        if self.relationships:
            parts.append("\nRELATIONSHIPS:")
            for src, rel, tgt in self.relationships[:15]:
                parts.append(f"- {src} {rel} {tgt}")

        # Add path information
        if self.paths:
            parts.append("\nCONNECTION PATHS:")
            for i, path in enumerate(self.paths[:5], 1):
                path_str = " → ".join([f"{s} [{r}]" for s, r, _ in path] + [path[-1][2]])
                parts.append(f"{i}. {path_str}")

        # Add community summaries
        if self.communities:
            parts.append("\nRELATED TOPICS:")
            for comm in self.communities[:3]:
                if comm.summary:
                    parts.append(f"- {comm.summary}")

        return "\n".join(parts) if parts else ""


class GraphRetrievalCache:
    """In-memory LRU cache for graph retrieval results with TTL."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 600.0):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize query for cache key (lowercase, collapse whitespace)."""
        return " ".join(query.lower().strip().split())

    def get(self, query: str) -> Optional[GraphRetrievalResult]:
        """Get cached result. Returns None on miss or expiry."""
        key = self._normalize(query)
        with self._lock:
            if key in self._cache:
                ts, result = self._cache[key]
                if time.monotonic() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return result
                del self._cache[key]
            self._misses += 1
            return None

    def put(self, query: str, result: GraphRetrievalResult) -> None:
        """Cache a result, evicting oldest if at capacity."""
        key = self._normalize(query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.monotonic(), result)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


def extract_entities_from_query(
    query: str,
    kg: KnowledgeGraph,
    config: Optional[GraphConfig] = None,
) -> List[GraphNode]:
    """
    Extract entity mentions from query and match to graph nodes.

    Uses a 2-stage matching approach:
    1. Direct string matching (fastest)
    2. LLM extraction (most accurate)

    Args:
        query: User query
        kg: Knowledge graph to search
        config: Graph configuration

    Returns:
        List of matched GraphNode entities
    """
    if config is None:
        config = get_graph_config()

    matched_entities = []

    # Stage 1: Direct string matching (fastest)
    query_lower = query.lower()
    for node in kg.nodes.values():
        if node.name.lower() in query_lower:
            matched_entities.append(node)

    # If we found direct matches, return them
    if matched_entities:
        return matched_entities

    # Stage 2: Use LLM to extract entity mentions
    entity_names = list(kg._name_to_id.keys())
    if not entity_names:
        return []

    # Sample some entities to show in prompt
    sample_entities = entity_names[:50]

    prompt = f"""Given the following query and a list of known entities, identify which entities are mentioned or relevant.

QUERY: {query}

KNOWN ENTITIES:
{', '.join(sample_entities)}

Return ONLY a JSON list of entity names that are mentioned or directly relevant to the query.
Example: ["Entity1", "Entity2"]

If no entities match, return: []

OUTPUT:"""

    try:
        result_text = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        # Parse JSON list
        import json

        try:
            # Try to extract JSON array
            match = re.search(r"\[.*?\]", result_text, re.DOTALL)
            if match:
                entity_names = json.loads(match.group(0))
                for name in entity_names:
                    node = kg.get_node_by_name(name)
                    if node and node not in matched_entities:
                        matched_entities.append(node)
        except json.JSONDecodeError:
            logger.debug("Could not parse entity names as JSON from LLM response")

    except Exception as e:
        logger.error(f"Entity extraction from query failed: {e}")

    return matched_entities


def retrieve_by_entities(
    query: str,
    kg: KnowledgeGraph,
    max_depth: int = 2,
    max_results: int = 20,
) -> GraphRetrievalResult:
    """
    Retrieve graph context based on entity mentions in query.

    Steps:
    1. Extract entity mentions from query
    2. Get neighbors of mentioned entities
    3. Get relationships between entities
    4. Build context from results
    """
    # Extract entities from query
    mentioned_entities = extract_entities_from_query(query, kg)

    if not mentioned_entities:
        return GraphRetrievalResult(
            query=query,
            retrieval_type="entity",
            context="No relevant entities found in query.",
            confidence=0.0,
        )

    # Collect all related entities and relationships
    all_entities = set(mentioned_entities)
    all_relationships = []

    for entity in mentioned_entities:
        # Get neighbors
        neighbors = kg.get_neighbors(entity.id, depth=max_depth)
        for neighbor_node, rel_type, depth in neighbors:
            all_entities.add(neighbor_node)
            all_relationships.append((entity.name, rel_type, neighbor_node.name))

    # Sort entities by relevance (mentioned first, then by connection count)
    mentioned_ids = {e.id for e in mentioned_entities}
    sorted_entities = sorted(
        all_entities,
        key=lambda e: (e.id not in mentioned_ids, -len(kg.get_neighbors(e.id, depth=1))),
    )[:max_results]

    result = GraphRetrievalResult(
        query=query,
        retrieval_type="entity",
        entities=sorted_entities,
        relationships=all_relationships[:30],
        confidence=min(1.0, len(mentioned_entities) * 0.3),
    )
    result.context = result.get_context_text()

    return result


def retrieve_by_relationship(
    query: str,
    kg: KnowledgeGraph,
    source_entity: str = None,
    target_entity: str = None,
    relationship_type: str = None,
    max_results: int = 20,
) -> GraphRetrievalResult:
    """
    Retrieve based on relationship patterns.

    Can find:
    - All relationships of a specific type
    - All relationships from/to a specific entity
    - Paths between two entities
    """
    entities = []
    seen_entity_ids = set()  # O(1) lookup instead of O(n) list check
    relationships = []
    paths = []

    def _add_entity(node):
        """Add entity if not already seen (O(1) check)."""
        if node and node.id not in seen_entity_ids:
            seen_entity_ids.add(node.id)
            entities.append(node)

    # If both source and target specified, find paths
    if source_entity and target_entity:
        paths = kg.find_paths(source_entity, target_entity)

        # Collect entities from paths
        for path in paths:
            for src, rel, tgt in path:
                _add_entity(kg.get_node_by_name(src))
                _add_entity(kg.get_node_by_name(tgt))
                relationships.append((src, rel, tgt))

    # If only source specified, get outgoing relationships
    elif source_entity:
        node = kg.get_node_by_name(source_entity)
        if node:
            _add_entity(node)
            neighbors = kg.get_neighbors(node.id, depth=1, direction="out")
            for neighbor, rel_type, _ in neighbors:
                if relationship_type is None or rel_type == relationship_type:
                    _add_entity(neighbor)
                    relationships.append((node.name, rel_type, neighbor.name))

    # If only target specified, get incoming relationships
    elif target_entity:
        node = kg.get_node_by_name(target_entity)
        if node:
            _add_entity(node)
            neighbors = kg.get_neighbors(node.id, depth=1, direction="in")
            for neighbor, rel_type, _ in neighbors:
                if relationship_type is None or rel_type == relationship_type:
                    _add_entity(neighbor)
                    relationships.append((neighbor.name, rel_type, node.name))

    # If only relationship type specified, find all of that type
    elif relationship_type:
        for src, tgt, data in kg.iter_edges(data=True):
            if data.get("relationship_type") == relationship_type:
                src_node = kg.nodes.get(src)
                tgt_node = kg.nodes.get(tgt)
                if src_node and tgt_node:
                    _add_entity(src_node)
                    _add_entity(tgt_node)
                    relationships.append((src_node.name, relationship_type, tgt_node.name))

    result = GraphRetrievalResult(
        query=query,
        retrieval_type="relationship",
        entities=entities[:max_results],
        relationships=relationships[:30],
        paths=paths[:10],
        confidence=0.8 if relationships or paths else 0.0,
    )
    result.context = result.get_context_text()

    return result


def retrieve_by_community(
    query: str,
    kg: KnowledgeGraph,
    top_k: int = 3,
) -> GraphRetrievalResult:
    """
    Retrieve based on community summaries (global queries).

    Good for broad questions about themes or topics.
    Uses pre-computed embeddings for fast similarity matching.
    """
    if not kg.communities:
        return GraphRetrievalResult(
            query=query,
            retrieval_type="community",
            context="No communities available.",
            confidence=0.0,
        )

    # Get query embedding with task instruction (improves accuracy for Qwen3-Embedding)
    try:
        query_embedding = get_query_embedding(query)
    except (ValueError, ConnectionError, TimeoutError, OSError) as e:
        logger.error(f"Failed to get query embedding: {e}")
        # Fallback to text matching
        query_embedding = None

    # Score communities by relevance using pre-computed embeddings (O(n) fast comparison)
    q = np.array(query_embedding) if query_embedding else None
    q_norm = np.linalg.norm(q) if q is not None else 0

    scored_communities = []
    for comm_id, community in kg.communities.items():
        if not community.summary:
            continue

        if query_embedding is not None and q_norm > 0:
            # Use pre-computed embedding if available (fast path)
            if community.embedding is not None:
                s = np.array(community.embedding)
                s_norm = np.linalg.norm(s)
                if s_norm > 0:
                    similarity = float(np.dot(q, s) / (q_norm * s_norm))
                    scored_communities.append((similarity, community))
                    continue

            # Fallback: compute embedding on-the-fly (slow path)
            try:
                summary_embedding = get_embedding(community.summary, EMBED_MODEL)
                s = np.array(summary_embedding)
                similarity = float(np.dot(q, s) / (q_norm * np.linalg.norm(s)))
                scored_communities.append((similarity, community))
            except Exception as e:
                logger.debug(
                    f"On-the-fly embedding failed for community, using keyword fallback: {e}"
                )
                # Fallback to keyword matching
                query_words = set(query.lower().split())
                summary_words = set(community.summary.lower().split())
                overlap = len(query_words & summary_words)
                scored_communities.append((overlap / max(len(query_words), 1), community))
        else:
            # Keyword matching fallback
            query_words = set(query.lower().split())
            summary_words = set(community.summary.lower().split())
            overlap = len(query_words & summary_words)
            scored_communities.append((overlap / max(len(query_words), 1), community))

    # Sort by score
    scored_communities.sort(key=lambda x: x[0], reverse=True)
    top_communities = [c for _, c in scored_communities[:top_k]]

    # Get entities from top communities
    entities = []
    for comm in top_communities:
        for node_id in comm.node_ids[:10]:
            node = kg.nodes.get(node_id)
            if node and node not in entities:
                entities.append(node)

    result = GraphRetrievalResult(
        query=query,
        retrieval_type="community",
        entities=entities[:20],
        communities=top_communities,
        confidence=scored_communities[0][0] if scored_communities else 0.0,
    )
    result.context = result.get_context_text()

    return result


def retrieve_from_graph(
    query: str,
    kg: KnowledgeGraph,
    config: Optional[GraphConfig] = None,
) -> GraphRetrievalResult:
    """
    Main retrieval function that combines multiple retrieval strategies.

    Tries entity-based retrieval first (LLM entity extraction → graph lookup),
    then falls back to community-based retrieval if no entities are found.

    Args:
        query: Natural language query.
        kg: Loaded KnowledgeGraph instance.
        config: Optional GraphConfig for schema and retrieval parameters.

    Returns:
        GraphRetrievalResult with context string, confidence score,
        matched entities, and retrieval type ("entity", "community", or "none").
    """
    if config is None:
        config = get_graph_config()

    logger.info(f"Graph retrieval for query: {query[:100]}...")

    # Check for relationship patterns in query
    rel_patterns = [
        r"relationship between (.+?) and (.+)",
        r"how (?:does|is) (.+?) (?:related|connected) to (.+)",
        r"what connects (.+?) (?:to|and) (.+)",
        r"path from (.+?) to (.+)",
    ]

    for pattern in rel_patterns:
        match = re.search(pattern, query.lower())
        if match:
            source = match.group(1).strip()
            target = match.group(2).strip()
            result = retrieve_by_relationship(query, kg, source, target)
            if result.entities or result.paths:
                return result

    # Check for global/summary patterns
    global_patterns = [
        r"what are all",
        r"list all",
        r"summarize",
        r"overview of",
        r"main (?:topics|themes|concepts)",
    ]

    for pattern in global_patterns:
        if re.search(pattern, query.lower()):
            result = retrieve_by_community(query, kg)
            if result.communities:
                return result

    # Default: entity-based retrieval
    result = retrieve_by_entities(query, kg)

    # If entity retrieval has low confidence, try community as supplement
    if result.confidence < 0.3 and kg.communities:
        community_result = retrieve_by_community(query, kg)
        if community_result.confidence > result.confidence:
            # Merge results
            result.communities = community_result.communities
            result.context = result.get_context_text()

    return result


class GraphRetriever:
    """
    Stateful graph retriever that maintains a loaded knowledge graph.

    Usage:
        retriever = GraphRetriever()
        retriever.load()
        result = retriever.retrieve("What is the relationship between X and Y?")
    """

    def __init__(
        self,
        graph_path: str = None,
        config: Optional[GraphConfig] = None,
    ):
        self.graph_path = graph_path
        self.config = config or get_graph_config()
        self.kg: Optional[KnowledgeGraph] = None
        self._cache = GraphRetrievalCache()

    def load(self) -> bool:
        """Load the knowledge graph."""
        try:
            self.kg = KnowledgeGraph.load(self.graph_path, self.config)
            self._cache.clear()
            return len(self.kg.nodes) > 0
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if graph is loaded."""
        return self.kg is not None and len(self.kg.nodes) > 0

    def retrieve(self, query: str) -> GraphRetrievalResult:
        """
        Retrieve context from the knowledge graph for a query.

        Uses an in-memory LRU cache. If the graph is not loaded, attempts
        to load it first. Delegates to ``retrieve_from_graph()`` for the
        actual entity/community/path retrieval logic.

        Args:
            query: Natural language query.

        Returns:
            GraphRetrievalResult with context, confidence, and matched entities.
        """
        if not self.is_loaded():
            logger.warning("Knowledge graph not loaded, attempting to load...")
            if not self.load():
                return GraphRetrievalResult(
                    query=query,
                    retrieval_type="error",
                    context="Knowledge graph not available.",
                    confidence=0.0,
                )

        cached = self._cache.get(query)
        if cached is not None:
            logger.debug(f"Graph cache hit: {query[:50]}...")
            return cached

        result = retrieve_from_graph(query, self.kg, self.config)
        self._cache.put(query, result)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.is_loaded():
            return {"status": "not_loaded"}
        return self.kg.get_statistics()
