"""
Graph configuration loader for GraphRAG.

Loads and validates the graph schema configuration from YAML.
Provides default values and configuration access throughout the system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import PROJECT_DIR
from .utils.logger import logger


# Default configuration file path (in project directory, not data directory)
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config/graph_schema.yaml"


@dataclass
class EntityType:
    """Definition of an entity type."""

    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)


@dataclass
class RelationshipType:
    """Definition of a relationship type."""

    name: str
    description: str
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class DomainConfig:
    """Domain configuration."""

    name: str = "General Knowledge Base"
    description: str = "General-purpose document collection"
    language: str = "en"
    document_types: List[str] = field(default_factory=list)


@dataclass
class ExtractionConfig:
    """Entity/relationship extraction settings."""

    max_entities_per_chunk: int = 15
    max_relationships_per_chunk: int = 20
    min_confidence: float = 0.7
    extract_implicit_relations: bool = True
    resolve_coreferences: bool = True
    extraction_temperature: float = 0.1


@dataclass
class GraphSettings:
    """Knowledge graph settings."""

    merge_similar_entities: bool = True
    entity_similarity_threshold: float = 0.85
    enable_communities: bool = True
    community_algorithm: str = "louvain"
    community_resolution: float = 1.0
    max_traversal_depth: int = 3
    generate_community_summaries: bool = True


@dataclass
class RoutingConfig:
    """Query routing settings."""

    strategy: str = "hybrid"
    graph_query_patterns: List[str] = field(default_factory=list)
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    always_include_vector: bool = True


@dataclass
class CustomPrompts:
    """Custom extraction prompts."""

    entity_extraction: str = ""
    relationship_extraction: str = ""
    community_summary: str = ""


@dataclass
class EntityResolutionConfig:
    """
    Entity resolution settings for semantic deduplication.

    The resolution process has 4 phases:
    1. Blocking: Find candidate pairs using embedding similarity
    2. Matching: Verify candidates using LLM with relationship context
    3. Clustering: Build equivalence clusters via transitive closure
    4. Merging: Merge entities with full information enrichment
    """

    # Enable/disable entity resolution
    enabled: bool = True

    # Phase 1: Blocking - cosine similarity threshold for candidates
    # Lower = more candidates (higher recall, more LLM calls)
    # Higher = fewer candidates (higher precision, fewer LLM calls)
    similarity_threshold: float = 0.75

    # Phase 2: Matching - minimum LLM confidence to merge
    llm_confidence_threshold: float = 0.7

    # Maximum concurrent LLM calls for verification
    max_concurrent_llm: int = 4

    # Phase 4: Merging - use LLM to synthesize merged descriptions
    # If False, uses smart concatenation instead (faster, cheaper)
    use_llm_for_descriptions: bool = True

    # Batch size for embedding computation
    batch_size: int = 500

    # Cache LLM resolution decisions
    cache_decisions: bool = True

    # Cache TTL in hours
    cache_ttl_hours: int = 24

    @classmethod
    def from_env(cls) -> "EntityResolutionConfig":
        """Create config from environment variables."""
        from .constants import (
            ENABLE_ENTITY_RESOLUTION,
            ENTITY_RESOLUTION_SIMILARITY_THRESHOLD,
            ENTITY_RESOLUTION_LLM_CONFIDENCE,
            ENTITY_RESOLUTION_MAX_CONCURRENT,
            ENTITY_RESOLUTION_USE_LLM_DESCRIPTIONS,
            ENTITY_RESOLUTION_CACHE_ENABLED,
            ENTITY_RESOLUTION_CACHE_TTL_HOURS,
            ENTITY_RESOLUTION_BATCH_SIZE,
        )

        return cls(
            enabled=ENABLE_ENTITY_RESOLUTION,
            similarity_threshold=ENTITY_RESOLUTION_SIMILARITY_THRESHOLD,
            llm_confidence_threshold=ENTITY_RESOLUTION_LLM_CONFIDENCE,
            max_concurrent_llm=ENTITY_RESOLUTION_MAX_CONCURRENT,
            use_llm_for_descriptions=ENTITY_RESOLUTION_USE_LLM_DESCRIPTIONS,
            batch_size=ENTITY_RESOLUTION_BATCH_SIZE,
            cache_decisions=ENTITY_RESOLUTION_CACHE_ENABLED,
            cache_ttl_hours=ENTITY_RESOLUTION_CACHE_TTL_HOURS,
        )


@dataclass
class GraphConfig:
    """Complete graph configuration."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    entities: List[EntityType] = field(default_factory=list)
    relationships: List[RelationshipType] = field(default_factory=list)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    graph: GraphSettings = field(default_factory=GraphSettings)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    custom_prompts: CustomPrompts = field(default_factory=CustomPrompts)
    entity_resolution: EntityResolutionConfig = field(default_factory=EntityResolutionConfig)

    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """Get entity type by name."""
        for et in self.entities:
            if et.name.lower() == name.lower():
                return et
        return None

    def get_relationship_type(self, name: str) -> Optional[RelationshipType]:
        """Get relationship type by name."""
        for rt in self.relationships:
            if rt.name.lower() == name.lower():
                return rt
        return None

    def get_entity_names(self) -> List[str]:
        """Get list of entity type names."""
        return [e.name for e in self.entities]

    def get_relationship_names(self) -> List[str]:
        """Get list of relationship type names."""
        return [r.name for r in self.relationships]


def load_graph_config(config_path: Optional[str] = None) -> GraphConfig:
    """
    Load graph configuration from YAML file.

    Args:
        config_path: Path to configuration file. Uses default if not specified.

    Returns:
        GraphConfig object with all settings.
    """
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH.resolve())

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {path}, using defaults")
        return GraphConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return GraphConfig()

    # Parse domain
    domain_data = data.get("domain", {})
    domain = DomainConfig(
        name=domain_data.get("name", "General Knowledge Base"),
        description=domain_data.get("description", ""),
        language=domain_data.get("language", "en"),
        document_types=domain_data.get("document_types", []),
    )

    # Parse entities
    entities = []
    for e in data.get("entities", []):
        entities.append(
            EntityType(
                name=e.get("name", ""),
                description=e.get("description", ""),
                examples=e.get("examples", []),
                attributes=e.get("attributes", []),
            )
        )

    # Parse relationships
    relationships = []
    for r in data.get("relationships", []):
        relationships.append(
            RelationshipType(
                name=r.get("name", ""),
                description=r.get("description", ""),
                source_types=r.get("source_types", []),
                target_types=r.get("target_types", []),
                examples=r.get("examples", []),
            )
        )

    # Parse extraction settings
    extraction_data = data.get("extraction", {})
    extraction = ExtractionConfig(
        max_entities_per_chunk=extraction_data.get("max_entities_per_chunk", 15),
        max_relationships_per_chunk=extraction_data.get("max_relationships_per_chunk", 20),
        min_confidence=extraction_data.get("min_confidence", 0.7),
        extract_implicit_relations=extraction_data.get("extract_implicit_relations", True),
        resolve_coreferences=extraction_data.get("resolve_coreferences", True),
        extraction_temperature=extraction_data.get("extraction_temperature", 0.1),
    )

    # Parse graph settings
    graph_data = data.get("graph", {})
    graph_settings = GraphSettings(
        merge_similar_entities=graph_data.get("merge_similar_entities", True),
        entity_similarity_threshold=graph_data.get("entity_similarity_threshold", 0.85),
        enable_communities=graph_data.get("enable_communities", True),
        community_algorithm=graph_data.get("community_algorithm", "louvain"),
        community_resolution=graph_data.get("community_resolution", 1.0),
        max_traversal_depth=graph_data.get("max_traversal_depth", 3),
        generate_community_summaries=graph_data.get("generate_community_summaries", True),
    )

    # Parse routing settings
    routing_data = data.get("routing", {})
    routing = RoutingConfig(
        strategy=routing_data.get("strategy", "hybrid"),
        graph_query_patterns=routing_data.get("graph_query_patterns", []),
        vector_weight=routing_data.get("vector_weight", 0.5),
        graph_weight=routing_data.get("graph_weight", 0.5),
        always_include_vector=routing_data.get("always_include_vector", True),
    )

    # Parse custom prompts
    prompts_data = data.get("custom_prompts", {})
    custom_prompts = CustomPrompts(
        entity_extraction=prompts_data.get("entity_extraction", ""),
        relationship_extraction=prompts_data.get("relationship_extraction", ""),
        community_summary=prompts_data.get("community_summary", ""),
    )

    # Parse entity resolution settings (with env var defaults)
    resolution_data = data.get("entity_resolution", {})
    # Start with env-based defaults, then override with YAML values
    entity_resolution = EntityResolutionConfig.from_env()
    if resolution_data:
        entity_resolution = EntityResolutionConfig(
            enabled=resolution_data.get("enabled", entity_resolution.enabled),
            similarity_threshold=resolution_data.get(
                "similarity_threshold", entity_resolution.similarity_threshold
            ),
            llm_confidence_threshold=resolution_data.get(
                "llm_confidence_threshold", entity_resolution.llm_confidence_threshold
            ),
            max_concurrent_llm=resolution_data.get(
                "max_concurrent_llm", entity_resolution.max_concurrent_llm
            ),
            use_llm_for_descriptions=resolution_data.get(
                "use_llm_for_descriptions", entity_resolution.use_llm_for_descriptions
            ),
            batch_size=resolution_data.get("batch_size", entity_resolution.batch_size),
            cache_decisions=resolution_data.get(
                "cache_decisions", entity_resolution.cache_decisions
            ),
            cache_ttl_hours=resolution_data.get(
                "cache_ttl_hours", entity_resolution.cache_ttl_hours
            ),
        )

    return GraphConfig(
        domain=domain,
        entities=entities,
        relationships=relationships,
        extraction=extraction,
        graph=graph_settings,
        routing=routing,
        custom_prompts=custom_prompts,
        entity_resolution=entity_resolution,
    )


# Global config instance (lazy-loaded)
_config: Optional[GraphConfig] = None


def get_graph_config() -> GraphConfig:
    """Get the global graph configuration (lazy-loaded singleton)."""
    global _config
    if _config is None:
        _config = load_graph_config()
    return _config


def reload_graph_config(config_path: Optional[str] = None) -> GraphConfig:
    """Reload the graph configuration from file."""
    global _config
    _config = load_graph_config(config_path)
    return _config
