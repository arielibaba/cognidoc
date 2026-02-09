"""
Entity Resolution module for Knowledge Graph semantic deduplication.

Implements a 4-phase approach:
1. Blocking: Find candidate pairs using embedding similarity (reduces O(n²) to O(n×k))
2. Matching: Verify candidates using LLM with relationship context
3. Clustering: Build equivalence clusters via Union-Find (transitive closure)
4. Merging: Consolidate entities and relationships with full enrichment

Features:
- Multi-language support via multilingual embeddings
- LLM caching to avoid redundant calls
- Incremental resolution support
- Checkpoint/resume capability
"""

import asyncio
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .constants import (
    ENTITY_RESOLUTION_PROMPT,
    DESCRIPTION_MERGE_PROMPT,
)
from .graph_config import EntityResolutionConfig
from .helpers import load_prompt
from .knowledge_graph import KnowledgeGraph, GraphNode, GraphEdge
from .utils.llm_client import llm_chat_async_ingestion
from .utils.logger import logger
from .utils.tool_cache import PersistentToolCache


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CandidatePair:
    """Candidate pair for LLM verification."""

    entity_a_id: str
    entity_b_id: str
    similarity_score: float


@dataclass
class ResolutionDecision:
    """Decision from LLM verification."""

    same_entity: bool
    confidence: float
    canonical_name: str
    reasoning: str


@dataclass
class MergedEntity:
    """Result of merging a cluster of entities."""

    canonical_id: str
    canonical_name: str
    type: str
    description: str
    attributes: Dict[str, Any]
    aliases: List[str]
    source_chunks: List[str]
    merged_from: List[str]
    confidence: float


@dataclass
class EntityResolutionResult:
    """Complete resolution result with statistics."""

    original_entity_count: int
    final_entity_count: int
    candidates_found: int
    candidates_verified: int
    clusters_found: int
    entities_merged: int
    relationships_updated: int
    llm_calls_made: int
    cache_hits: int
    duration_seconds: float


# =============================================================================
# Phase 1: Blocking (Embedding Similarity)
# =============================================================================


async def compute_resolution_embeddings(
    entities: List[GraphNode],
    batch_size: int = 50,
) -> np.ndarray:
    """
    Compute embeddings for entity resolution.

    Text format: "{name} ({type}): {description}"
    Uses the configured embedding provider.

    Args:
        entities: List of graph nodes to embed
        batch_size: Batch size for embedding computation

    Returns:
        NumPy array of embeddings (n_entities, embedding_dim)
    """
    from .utils.embedding_providers import get_embedding_provider, OllamaEmbeddingProvider

    # Build text for each entity
    texts = []
    for entity in entities:
        text = entity.name
        if entity.type:
            text = f"{entity.name} ({entity.type})"
        if entity.description:
            text = f"{text}: {entity.description}"
        texts.append(text)

    provider = get_embedding_provider()

    # Use async embedding if available (Ollama)
    if isinstance(provider, OllamaEmbeddingProvider):
        logger.info(f"Computing embeddings for {len(texts)} entities (async batch)...")
        embeddings = await provider.embed_async(texts, max_concurrent=4)
    else:
        # Fallback to sync batching
        logger.info(f"Computing embeddings for {len(texts)} entities (sync)...")
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = provider.embed(batch)
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def find_candidate_pairs(
    entities: List[GraphNode],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.75,
) -> List[CandidatePair]:
    """
    Find candidate entity pairs using cosine similarity.

    Optimizations:
    - Batch matrix operations for efficiency
    - Only keeps upper triangle (avoids A,B and B,A duplicates)
    - Skips self-comparisons

    Args:
        entities: List of graph nodes
        embeddings: NumPy array of embeddings
        similarity_threshold: Minimum cosine similarity

    Returns:
        List of CandidatePair sorted by similarity (highest first)
    """
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    candidates = []
    n = len(entities)
    batch_size = 500  # Process in batches to manage memory

    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch = normalized[i:batch_end]

        # Similarity with all entities
        similarities = batch @ normalized.T  # (batch_size, n)

        for j, entity_idx in enumerate(range(i, batch_end)):
            sim_scores = similarities[j]
            sim_scores[entity_idx] = -1  # Exclude self

            # Find indices above threshold (only upper triangle)
            for cand_idx in range(entity_idx + 1, n):
                if sim_scores[cand_idx] >= similarity_threshold:
                    candidates.append(
                        CandidatePair(
                            entity_a_id=entities[entity_idx].id,
                            entity_b_id=entities[cand_idx].id,
                            similarity_score=float(sim_scores[cand_idx]),
                        )
                    )

    # Sort by similarity (highest first) for prioritization
    candidates.sort(key=lambda c: c.similarity_score, reverse=True)

    return candidates


# =============================================================================
# Phase 2: Matching (LLM Verification)
# =============================================================================


def get_entity_relations_summary(
    entity: GraphNode,
    graph: KnowledgeGraph,
    max_relations: int = 5,
) -> str:
    """
    Get a concise summary of entity relationships for context.

    Args:
        entity: The entity to summarize
        graph: Knowledge graph
        max_relations: Maximum number of relations to include

    Returns:
        String summary of relationships
    """
    relations = []

    # Outgoing relationships
    if graph.has_node(entity.id):
        for successor in graph.get_successors(entity.id)[:max_relations]:
            edge_data = graph.get_edge_data(entity.id, successor) or {}
            target = graph.nodes.get(successor)
            if target:
                rel_type = edge_data.get("relationship_type", "RELATED_TO")
                relations.append(f"--[{rel_type}]--> {target.name}")

        # Incoming relationships
        remaining = max_relations - len(relations)
        if remaining > 0:
            for predecessor in graph.get_predecessors(entity.id)[:remaining]:
                edge_data = graph.get_edge_data(predecessor, entity.id) or {}
                source = graph.nodes.get(predecessor)
                if source:
                    rel_type = edge_data.get("relationship_type", "RELATED_TO")
                    relations.append(f"<--[{rel_type}]-- {source.name}")

    return "; ".join(relations) if relations else "(no relationships)"


def _get_cache_key(entity_a_id: str, entity_b_id: str) -> str:
    """Generate a consistent cache key for an entity pair."""
    # Sort IDs to ensure A,B and B,A produce the same key
    sorted_ids = tuple(sorted([entity_a_id, entity_b_id]))
    key_str = f"{sorted_ids[0]}:{sorted_ids[1]}"
    return hashlib.md5(key_str.encode()).hexdigest()


async def verify_candidate_pair(
    entity_a: GraphNode,
    entity_b: GraphNode,
    graph: KnowledgeGraph,
    cache: Optional[PersistentToolCache] = None,
    cache_ttl_hours: int = 24,
) -> Tuple[ResolutionDecision, bool]:
    """
    Verify a candidate pair using LLM.

    Args:
        entity_a: First entity
        entity_b: Second entity
        graph: Knowledge graph for context
        cache: Optional cache for decisions
        cache_ttl_hours: Cache TTL in hours

    Returns:
        Tuple of (ResolutionDecision, was_cached)
    """
    # Check cache
    if cache is not None:
        cache_key = _get_cache_key(entity_a.id, entity_b.id)
        cached = PersistentToolCache.get("entity_resolution", cache_key=cache_key)
        if cached is not None:
            return ResolutionDecision(**cached), True

    # Get relationship context
    relations_a = get_entity_relations_summary(entity_a, graph, max_relations=5)
    relations_b = get_entity_relations_summary(entity_b, graph, max_relations=5)

    # Load and format prompt
    prompt_template = load_prompt(ENTITY_RESOLUTION_PROMPT)
    prompt = prompt_template.format(
        name_a=entity_a.name,
        type_a=entity_a.type or "(unknown)",
        desc_a=entity_a.description or "(no description)",
        relations_a=relations_a,
        name_b=entity_b.name,
        type_b=entity_b.type or "(unknown)",
        desc_b=entity_b.description or "(no description)",
        relations_b=relations_b,
    )

    # Call LLM
    try:
        response = await llm_chat_async_ingestion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            json_mode=True,
        )

        # Parse response
        result = _parse_resolution_response(response, entity_a.name)

        decision = ResolutionDecision(
            same_entity=result.get("same_entity", False),
            confidence=result.get("confidence", 0.0),
            canonical_name=result.get("canonical_name", entity_a.name),
            reasoning=result.get("reasoning", ""),
        )

        # Cache result
        if cache is not None:
            cache_key = _get_cache_key(entity_a.id, entity_b.id)
            PersistentToolCache.set(
                "entity_resolution",
                asdict(decision),
                ttl=cache_ttl_hours * 3600,
                cache_key=cache_key,
            )

        return decision, False

    except Exception as e:
        logger.warning(f"LLM verification failed for {entity_a.name} vs {entity_b.name}: {e}")
        return (
            ResolutionDecision(
                same_entity=False,
                confidence=0.0,
                canonical_name=entity_a.name,
                reasoning=f"Error: {str(e)}",
            ),
            False,
        )


def _parse_resolution_response(response: str, fallback_name: str) -> Dict[str, Any]:
    """Parse LLM response for resolution decision."""
    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parse failed, trying code block extraction")

    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            logger.debug("Code block JSON parse failed, trying raw object extraction")

    # Try to find JSON object
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.debug("Raw JSON object parse failed")

    logger.warning(f"Failed to parse resolution response: {response[:200]}...")
    return {
        "same_entity": False,
        "confidence": 0.0,
        "canonical_name": fallback_name,
        "reasoning": "Failed to parse LLM response",
    }


async def verify_candidates_batch(
    candidates: List[CandidatePair],
    graph: KnowledgeGraph,
    config: EntityResolutionConfig,
    show_progress: bool = True,
) -> Tuple[List[Tuple[CandidatePair, ResolutionDecision]], int, int]:
    """
    Verify candidates in parallel with caching.

    Args:
        candidates: List of candidate pairs
        graph: Knowledge graph
        config: Resolution configuration
        show_progress: Whether to show progress

    Returns:
        Tuple of (verified_pairs, llm_calls_made, cache_hits)
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    cache = PersistentToolCache() if config.cache_decisions else None

    results = []
    llm_calls = 0
    cache_hits = 0

    async def verify_one(pair: CandidatePair) -> Tuple[CandidatePair, ResolutionDecision, bool]:
        async with semaphore:
            entity_a = graph.nodes.get(pair.entity_a_id)
            entity_b = graph.nodes.get(pair.entity_b_id)

            if not entity_a or not entity_b:
                return pair, ResolutionDecision(False, 0.0, "", "Entity not found"), False

            decision, was_cached = await verify_candidate_pair(
                entity_a, entity_b, graph, cache, config.cache_ttl_hours
            )
            return pair, decision, was_cached

    # Process all candidates
    tasks = [verify_one(pair) for pair in candidates]

    if show_progress:
        # Simple progress logging
        total = len(tasks)
        logger.info(f"Verifying {total} candidate pairs...")

    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in task_results:
        if isinstance(result, Exception):
            logger.warning(f"Verification task failed: {result}")
            continue

        pair, decision, was_cached = result
        if was_cached:
            cache_hits += 1
        else:
            llm_calls += 1

        # Keep only verified matches above confidence threshold
        if decision.same_entity and decision.confidence >= config.llm_confidence_threshold:
            results.append((pair, decision))

    logger.info(
        f"Verification complete: {len(results)}/{len(candidates)} verified as same entity "
        f"(LLM calls: {llm_calls}, cache hits: {cache_hits})"
    )

    return results, llm_calls, cache_hits


# =============================================================================
# Phase 3: Clustering (Union-Find)
# =============================================================================


class UnionFind:
    """Union-Find data structure with canonical name tracking."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        self.canonical_names: Dict[str, str] = {}

    def find(self, x: str) -> str:
        """Find root with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str, canonical_name: str = None) -> None:
        """Union two elements with optional canonical name."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        # Track canonical name (prefer new name if provided)
        if canonical_name:
            self.canonical_names[root_x] = canonical_name
        elif root_y in self.canonical_names and root_x not in self.canonical_names:
            self.canonical_names[root_x] = self.canonical_names[root_y]

    def get_clusters(self) -> Dict[str, List[str]]:
        """Get all clusters as {root_id: [member_ids]}."""
        clusters = defaultdict(list)
        for x in self.parent:
            clusters[self.find(x)].append(x)
        # Only return clusters with more than one member
        return {k: v for k, v in clusters.items() if len(v) > 1}

    def get_canonical_name(self, x: str) -> Optional[str]:
        """Get canonical name for an element's cluster."""
        return self.canonical_names.get(self.find(x))


def build_entity_clusters(
    verified_pairs: List[Tuple[CandidatePair, ResolutionDecision]],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Build clusters from verified pairs using Union-Find.

    Returns:
        Tuple of (clusters, canonical_names) where:
        - clusters: {canonical_id: [member_ids]}
        - canonical_names: {canonical_id: canonical_name}
    """
    uf = UnionFind()

    for pair, decision in verified_pairs:
        uf.union(pair.entity_a_id, pair.entity_b_id, decision.canonical_name)

    clusters = uf.get_clusters()
    canonical_names = {}
    for root_id in clusters:
        name = uf.get_canonical_name(root_id)
        if name:
            canonical_names[root_id] = name

    return clusters, canonical_names


# =============================================================================
# Phase 4: Merging (Enrichment)
# =============================================================================


async def merge_descriptions(
    descriptions: List[str],
    entity_name: str,
    use_llm: bool = True,
) -> str:
    """
    Merge multiple descriptions preserving all information.

    Args:
        descriptions: List of descriptions to merge
        entity_name: Entity name for context
        use_llm: Whether to use LLM for synthesis

    Returns:
        Merged description
    """
    # Filter and deduplicate
    valid = [d.strip() for d in descriptions if d and d.strip()]
    if not valid:
        return ""
    if len(valid) == 1:
        return valid[0]

    # Remove exact duplicates
    unique = list(dict.fromkeys(valid))
    if len(unique) == 1:
        return unique[0]

    if not use_llm:
        return _concatenate_descriptions_smart(unique)

    # Use LLM for intelligent merging
    try:
        prompt_template = load_prompt(DESCRIPTION_MERGE_PROMPT)
        prompt = prompt_template.format(
            entity_name=entity_name,
            descriptions="\n".join(f"- {d}" for d in unique),
        )

        response = await llm_chat_async_ingestion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.strip()

    except Exception as e:
        logger.warning(f"LLM description merge failed: {e}, using concatenation")
        return _concatenate_descriptions_smart(unique)


def _concatenate_descriptions_smart(descriptions: List[str]) -> str:
    """Fallback: concatenate with sentence-level deduplication."""
    all_sentences = []
    seen_normalized = set()

    for desc in descriptions:
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", desc)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            normalized = sent.lower()
            # Skip if we've seen a very similar sentence
            if normalized not in seen_normalized:
                if not any(_word_overlap(normalized, s) > 0.8 for s in seen_normalized):
                    all_sentences.append(sent)
                    seen_normalized.add(normalized)

    return " ".join(all_sentences)


def _word_overlap(s1: str, s2: str) -> float:
    """Calculate word overlap ratio between two strings."""
    w1, w2 = set(s1.split()), set(s2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / min(len(w1), len(w2))


def merge_attributes(attributes_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge attributes with conflict resolution.

    Strategy:
    - Non-conflicting: add to merged dict
    - Lists: merge and deduplicate
    - Numbers: keep maximum
    - Strings: keep longest
    - Incompatible: keep as list
    """
    if not attributes_list:
        return {}

    merged = {}
    all_values = defaultdict(list)

    for attrs in attributes_list:
        if attrs:
            for key, value in attrs.items():
                all_values[key].append(value)

    for key, values in all_values.items():
        unique = list(dict.fromkeys(str(v) for v in values))

        if len(unique) == 1:
            merged[key] = values[0]
        else:
            merged[key] = _resolve_attribute_conflict(key, values)

    return merged


def _resolve_attribute_conflict(key: str, values: List[Any]) -> Any:
    """Resolve conflicting attribute values."""
    types = set(type(v) for v in values)

    if len(types) == 1:
        val_type = types.pop()

        if val_type == list:
            # Merge lists
            merged = []
            for v in values:
                merged.extend(v)
            return list(dict.fromkeys(merged))

        if val_type in (int, float):
            return max(values)

        if val_type == str:
            return max(values, key=lambda s: len(s) if s else 0)

    # Incompatible: keep all as list
    return list(dict.fromkeys(str(v) for v in values))


async def merge_entity_cluster(
    cluster: List[str],
    graph: KnowledgeGraph,
    canonical_name: Optional[str],
    config: EntityResolutionConfig,
) -> MergedEntity:
    """
    Merge a cluster of entities with full enrichment.

    Args:
        cluster: List of entity IDs to merge
        graph: Knowledge graph
        canonical_name: Canonical name from LLM (optional)
        config: Resolution configuration

    Returns:
        MergedEntity with all information preserved
    """
    entities = [graph.nodes[eid] for eid in cluster if eid in graph.nodes]
    if not entities:
        raise ValueError(f"No valid entities in cluster: {cluster}")

    # 1. Canonical name (from LLM or most frequent/longest)
    if not canonical_name:
        name_counts = Counter(e.name for e in entities)
        canonical_name = max(name_counts.keys(), key=lambda n: (name_counts[n], len(n)))

    # 2. Collect all aliases
    aliases = list(set(e.name for e in entities if e.name != canonical_name))

    # 3. Merge descriptions (ENRICHED - not just longest)
    all_descriptions = [e.description for e in entities]
    merged_description = await merge_descriptions(
        all_descriptions,
        canonical_name,
        use_llm=config.use_llm_for_descriptions,
    )

    # 4. Merge attributes (ENRICHED)
    all_attributes = [e.attributes for e in entities if e.attributes]
    merged_attributes = merge_attributes(all_attributes)

    # 5. Type (majority vote)
    type_counts = Counter(e.type for e in entities)
    canonical_type = type_counts.most_common(1)[0][0]

    # 6. Merge source chunks
    all_chunks = []
    for e in entities:
        all_chunks.extend(e.source_chunks)
    unique_chunks = list(dict.fromkeys(all_chunks))

    # 7. Average confidence (from embeddings if available)
    confidences = [getattr(e, "confidence", 1.0) for e in entities]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    return MergedEntity(
        canonical_id=cluster[0],
        canonical_name=canonical_name,
        type=canonical_type,
        description=merged_description,
        attributes=merged_attributes,
        aliases=aliases,
        source_chunks=unique_chunks,
        merged_from=cluster,
        confidence=avg_confidence,
    )


# =============================================================================
# Orchestration
# =============================================================================


async def apply_merges(
    graph: KnowledgeGraph,
    clusters: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    config: EntityResolutionConfig,
) -> Dict[str, int]:
    """
    Apply merges to the knowledge graph.

    Args:
        graph: Knowledge graph to modify
        clusters: {canonical_id: [member_ids]}
        canonical_names: {canonical_id: canonical_name}
        config: Resolution configuration

    Returns:
        Statistics dict
    """
    stats = {
        "entities_merged": 0,
        "relationships_updated": 0,
    }

    # Build reverse mapping for relationship resolution
    id_to_canonical = {}
    for canonical_id, members in clusters.items():
        for member in members:
            id_to_canonical[member] = canonical_id

    for canonical_id, member_ids in clusters.items():
        # Merge entity
        merged = await merge_entity_cluster(
            member_ids,
            graph,
            canonical_names.get(canonical_id),
            config,
        )

        # Update graph node
        node = graph.nodes[canonical_id]
        node.name = merged.canonical_name
        node.description = merged.description
        node.attributes = merged.attributes
        node.aliases = merged.aliases
        node.source_chunks = merged.source_chunks
        node.merged_from = merged.merged_from

        # Update backend graph node attributes
        graph.update_graph_node_attrs(
            canonical_id, name=merged.canonical_name, description=merged.description
        )

        # Process other members (redirect edges, remove nodes)
        for member_id in member_ids:
            if member_id != canonical_id:
                # Redirect edges
                stats["relationships_updated"] += _redirect_edges(graph, member_id, canonical_id)

                # Remove member node
                if member_id in graph.nodes:
                    # Update name mapping for member's name
                    member_name = graph.nodes[member_id].name
                    normalized = graph._normalize_name(member_name)
                    if normalized in graph._name_to_id:
                        del graph._name_to_id[normalized]

                    del graph.nodes[member_id]

                if graph.has_node(member_id):
                    graph.remove_graph_node(member_id)

                stats["entities_merged"] += 1

        # Register aliases in name mapping
        for alias in merged.aliases:
            normalized = graph._normalize_name(alias)
            graph._name_to_id[normalized] = canonical_id

        # Update canonical name mapping
        normalized_canonical = graph._normalize_name(merged.canonical_name)
        graph._name_to_id[normalized_canonical] = canonical_id

    return stats


def _redirect_edges(graph: KnowledgeGraph, old_id: str, new_id: str) -> int:
    """
    Redirect all edges from old_id to new_id.

    Returns number of edges updated.
    """
    updated = 0

    if not graph.has_node(old_id):
        return 0

    # Outgoing edges
    for successor in list(graph.get_successors(old_id)):
        if successor == new_id:
            continue  # Skip self-loops

        edge_data = dict(graph.get_edge_data(old_id, successor) or {})

        if not graph.has_edge(new_id, successor):
            graph.add_edge_raw(new_id, successor, **edge_data)
        else:
            # Merge edge data
            existing = graph.get_edge_data(new_id, successor) or {}
            merged_weight = existing.get("weight", 1) + edge_data.get("weight", 1)
            chunks = list(existing.get("source_chunks", []))
            chunks.extend(edge_data.get("source_chunks", []))
            merged_chunks = list(dict.fromkeys(chunks))

            # Merge descriptions if different
            old_desc = edge_data.get("description", "")
            new_desc = existing.get("description", "")
            merged_desc = new_desc
            if old_desc and old_desc != new_desc:
                merged_desc = _concatenate_descriptions_smart([new_desc, old_desc])

            graph.update_edge_attrs(
                new_id,
                successor,
                weight=merged_weight,
                source_chunks=merged_chunks,
                description=merged_desc,
            )

        updated += 1

    # Incoming edges
    for predecessor in list(graph.get_predecessors(old_id)):
        if predecessor == new_id:
            continue  # Skip self-loops

        edge_data = dict(graph.get_edge_data(predecessor, old_id) or {})

        if not graph.has_edge(predecessor, new_id):
            graph.add_edge_raw(predecessor, new_id, **edge_data)
        else:
            # Merge edge data
            existing = graph.get_edge_data(predecessor, new_id) or {}
            merged_weight = existing.get("weight", 1) + edge_data.get("weight", 1)
            chunks = list(existing.get("source_chunks", []))
            chunks.extend(edge_data.get("source_chunks", []))
            merged_chunks = list(dict.fromkeys(chunks))

            # Merge descriptions
            old_desc = edge_data.get("description", "")
            new_desc = existing.get("description", "")
            merged_desc = new_desc
            if old_desc and old_desc != new_desc:
                merged_desc = _concatenate_descriptions_smart([new_desc, old_desc])

            graph.update_edge_attrs(
                predecessor,
                new_id,
                weight=merged_weight,
                source_chunks=merged_chunks,
                description=merged_desc,
            )

        updated += 1

    return updated


async def resolve_entities(
    graph: KnowledgeGraph,
    config: Optional[EntityResolutionConfig] = None,
    show_progress: bool = True,
) -> EntityResolutionResult:
    """
    Run complete entity resolution pipeline.

    This is the main entry point for entity resolution.

    Args:
        graph: Knowledge graph to deduplicate (modified in place)
        config: Resolution configuration (uses defaults if not provided)
        show_progress: Whether to show progress

    Returns:
        EntityResolutionResult with statistics
    """
    start_time = time.time()

    if config is None:
        config = EntityResolutionConfig.from_env()

    if not config.enabled:
        logger.info("Entity resolution is disabled")
        return EntityResolutionResult(
            original_entity_count=len(graph.nodes),
            final_entity_count=len(graph.nodes),
            candidates_found=0,
            candidates_verified=0,
            clusters_found=0,
            entities_merged=0,
            relationships_updated=0,
            llm_calls_made=0,
            cache_hits=0,
            duration_seconds=0.0,
        )

    original_count = len(graph.nodes)
    entities = list(graph.nodes.values())

    logger.info(f"Starting entity resolution for {original_count} entities...")

    # Phase 1: Compute embeddings and find candidates
    logger.info("Phase 1: Computing embeddings and finding candidates...")
    embeddings = await compute_resolution_embeddings(
        entities,
        batch_size=config.batch_size,
    )

    candidates = find_candidate_pairs(
        entities,
        embeddings,
        config.similarity_threshold,
    )
    logger.info(f"Found {len(candidates)} candidate pairs")

    if not candidates:
        duration = time.time() - start_time
        logger.info(f"No candidates found. Entity resolution complete in {duration:.1f}s")
        return EntityResolutionResult(
            original_entity_count=original_count,
            final_entity_count=original_count,
            candidates_found=0,
            candidates_verified=0,
            clusters_found=0,
            entities_merged=0,
            relationships_updated=0,
            llm_calls_made=0,
            cache_hits=0,
            duration_seconds=duration,
        )

    # Phase 2: LLM verification
    logger.info("Phase 2: Verifying candidates with LLM...")
    verified, llm_calls, cache_hits = await verify_candidates_batch(
        candidates,
        graph,
        config,
        show_progress=show_progress,
    )

    if not verified:
        duration = time.time() - start_time
        logger.info(f"No verified matches. Entity resolution complete in {duration:.1f}s")
        return EntityResolutionResult(
            original_entity_count=original_count,
            final_entity_count=original_count,
            candidates_found=len(candidates),
            candidates_verified=0,
            clusters_found=0,
            entities_merged=0,
            relationships_updated=0,
            llm_calls_made=llm_calls,
            cache_hits=cache_hits,
            duration_seconds=duration,
        )

    # Phase 3: Clustering
    logger.info("Phase 3: Building entity clusters...")
    clusters, canonical_names = build_entity_clusters(verified)
    logger.info(f"Found {len(clusters)} clusters to merge")

    # Phase 4: Merging
    logger.info("Phase 4: Merging entities and relationships...")
    merge_stats = await apply_merges(graph, clusters, canonical_names, config)

    duration = time.time() - start_time

    result = EntityResolutionResult(
        original_entity_count=original_count,
        final_entity_count=len(graph.nodes),
        candidates_found=len(candidates),
        candidates_verified=len(verified),
        clusters_found=len(clusters),
        entities_merged=merge_stats["entities_merged"],
        relationships_updated=merge_stats["relationships_updated"],
        llm_calls_made=llm_calls,
        cache_hits=cache_hits,
        duration_seconds=duration,
    )

    logger.info(
        f"Entity resolution complete: {original_count} → {result.final_entity_count} entities "
        f"({result.entities_merged} merged from {len(clusters)} clusters) in {duration:.1f}s"
    )

    return result
