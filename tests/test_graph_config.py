"""Tests for graph_config.py — schema loading, dataclass methods, singleton thread safety."""

import threading
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cognidoc.graph_config import (
    EntityAttribute,
    EntityType,
    RelationshipType,
    DomainConfig,
    ExtractionConfig,
    GraphSettings,
    RoutingConfig,
    CustomPrompts,
    EntityResolutionConfig,
    GraphConfig,
    load_graph_config,
    get_graph_config,
    reload_graph_config,
)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestEntityAttribute:

    def test_defaults(self):
        attr = EntityAttribute(name="date")
        assert attr.type == "string"
        assert attr.description == ""

    def test_full_init(self):
        attr = EntityAttribute(name="population", type="number", description="City population")
        assert attr.name == "population"
        assert attr.type == "number"
        assert attr.description == "City population"


class TestDataclassDefaults:

    def test_entity_type_defaults(self):
        et = EntityType(name="Person", description="A human")
        assert et.examples == []
        assert et.attributes == []

    def test_relationship_type_defaults(self):
        rt = RelationshipType(name="KNOWS", description="Acquaintance")
        assert rt.source_types == []
        assert rt.target_types == []
        assert rt.examples == []

    def test_domain_config_defaults(self):
        dc = DomainConfig()
        assert dc.name == "General Knowledge Base"
        assert dc.language == "en"

    def test_extraction_config_defaults(self):
        ec = ExtractionConfig()
        assert ec.max_entities_per_chunk == 15
        assert ec.min_confidence == 0.7
        assert ec.extraction_temperature == 0.1

    def test_graph_settings_defaults(self):
        gs = GraphSettings()
        assert gs.merge_similar_entities is True
        assert gs.community_algorithm == "louvain"
        assert gs.max_traversal_depth == 3

    def test_routing_config_defaults(self):
        rc = RoutingConfig()
        assert rc.strategy == "hybrid"
        assert rc.vector_weight == 0.5
        assert rc.always_include_vector is True

    def test_custom_prompts_defaults(self):
        cp = CustomPrompts()
        assert cp.entity_extraction == ""

    def test_entity_resolution_defaults(self):
        er = EntityResolutionConfig()
        assert er.enabled is True
        assert er.similarity_threshold == 0.75
        assert er.batch_size == 500


# ---------------------------------------------------------------------------
# GraphConfig methods
# ---------------------------------------------------------------------------


class TestGraphConfigMethods:

    @pytest.fixture
    def config(self):
        return GraphConfig(
            entities=[
                EntityType(name="Person", description="A person"),
                EntityType(name="Organization", description="An org"),
            ],
            relationships=[
                RelationshipType(name="WORKS_FOR", description="Employment"),
                RelationshipType(name="KNOWS", description="Acquaintance"),
            ],
        )

    def test_get_entity_type_found(self, config):
        et = config.get_entity_type("person")
        assert et is not None
        assert et.name == "Person"

    def test_get_entity_type_not_found(self, config):
        assert config.get_entity_type("Alien") is None

    def test_get_entity_type_case_insensitive(self, config):
        assert config.get_entity_type("ORGANIZATION") is not None

    def test_get_relationship_type_found(self, config):
        rt = config.get_relationship_type("works_for")
        assert rt is not None
        assert rt.name == "WORKS_FOR"

    def test_get_relationship_type_not_found(self, config):
        assert config.get_relationship_type("HATES") is None

    def test_get_entity_names(self, config):
        assert config.get_entity_names() == ["Person", "Organization"]

    def test_get_relationship_names(self, config):
        assert config.get_relationship_names() == ["WORKS_FOR", "KNOWS"]


# ---------------------------------------------------------------------------
# load_graph_config — file-based
# ---------------------------------------------------------------------------


class TestLoadGraphConfig:

    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_graph_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, GraphConfig)
        assert cfg.domain.name == "General Knowledge Base"
        assert cfg.entities == []

    def test_empty_file_returns_defaults(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        cfg = load_graph_config(str(f))
        assert isinstance(cfg, GraphConfig)
        assert cfg.entities == []

    def test_full_yaml_loads(self, tmp_path):
        schema = {
            "domain": {
                "name": "Test Domain",
                "description": "Testing",
                "language": "fr",
                "document_types": ["pdf", "docx"],
            },
            "entities": [
                {
                    "name": "Concept",
                    "description": "An idea",
                    "examples": ["AI", "ML"],
                    "attributes": ["field"],
                }
            ],
            "relationships": [
                {
                    "name": "RELATED_TO",
                    "description": "General relation",
                    "source_types": ["Concept"],
                    "target_types": ["Concept"],
                    "examples": ["AI RELATED_TO ML"],
                }
            ],
            "extraction": {
                "max_entities_per_chunk": 20,
                "min_confidence": 0.8,
            },
            "graph": {
                "enable_communities": False,
                "max_traversal_depth": 5,
            },
            "routing": {
                "strategy": "graph_only",
                "vector_weight": 0.2,
                "graph_weight": 0.8,
            },
            "custom_prompts": {
                "entity_extraction": "Extract entities from {text}",
            },
        }
        f = tmp_path / "schema.yaml"
        f.write_text(yaml.dump(schema), encoding="utf-8")

        cfg = load_graph_config(str(f))
        assert cfg.domain.name == "Test Domain"
        assert cfg.domain.language == "fr"
        assert len(cfg.entities) == 1
        assert cfg.entities[0].name == "Concept"
        assert cfg.entities[0].examples == ["AI", "ML"]
        assert len(cfg.entities[0].attributes) == 1
        assert cfg.entities[0].attributes[0].name == "field"
        assert isinstance(cfg.entities[0].attributes[0], EntityAttribute)
        assert len(cfg.relationships) == 1
        assert cfg.relationships[0].source_types == ["Concept"]
        assert cfg.extraction.max_entities_per_chunk == 20
        assert cfg.extraction.min_confidence == 0.8
        assert cfg.graph.enable_communities is False
        assert cfg.graph.max_traversal_depth == 5
        assert cfg.routing.strategy == "graph_only"
        assert cfg.routing.graph_weight == 0.8
        assert cfg.custom_prompts.entity_extraction == "Extract entities from {text}"

    def test_partial_yaml_uses_defaults(self, tmp_path):
        """YAML with only domain section → other sections get defaults."""
        f = tmp_path / "partial.yaml"
        f.write_text(
            yaml.dump({"domain": {"name": "Partial"}}),
            encoding="utf-8",
        )
        cfg = load_graph_config(str(f))
        assert cfg.domain.name == "Partial"
        assert cfg.extraction.max_entities_per_chunk == 15  # default
        assert cfg.graph.merge_similar_entities is True  # default

    def test_entity_resolution_from_yaml(self, tmp_path):
        schema = {
            "entity_resolution": {
                "enabled": False,
                "similarity_threshold": 0.9,
                "batch_size": 100,
            }
        }
        f = tmp_path / "res.yaml"
        f.write_text(yaml.dump(schema), encoding="utf-8")
        cfg = load_graph_config(str(f))
        assert cfg.entity_resolution.enabled is False
        assert cfg.entity_resolution.similarity_threshold == 0.9
        assert cfg.entity_resolution.batch_size == 100

    def test_none_config_path_uses_default(self):
        """load_graph_config(None) uses DEFAULT_CONFIG_PATH."""
        with patch("cognidoc.graph_config.DEFAULT_CONFIG_PATH") as mock_default:
            mock_default.resolve.return_value = Path("/fake/schema.yaml")
            cfg = load_graph_config(None)
            # The file won't exist, so we get defaults
            assert isinstance(cfg, GraphConfig)

    def test_attributes_as_strings_backward_compat(self, tmp_path):
        """Old-style string attributes are parsed into EntityAttribute."""
        schema = {
            "entities": [
                {
                    "name": "Concept",
                    "description": "An idea",
                    "attributes": ["field", "topic"],
                }
            ]
        }
        f = tmp_path / "schema.yaml"
        f.write_text(yaml.dump(schema), encoding="utf-8")
        cfg = load_graph_config(str(f))
        assert len(cfg.entities[0].attributes) == 2
        attr0 = cfg.entities[0].attributes[0]
        assert isinstance(attr0, EntityAttribute)
        assert attr0.name == "field"
        assert attr0.type == "string"
        assert attr0.description == ""

    def test_attributes_as_typed_dicts(self, tmp_path):
        """New-style typed attributes are parsed correctly."""
        schema = {
            "entities": [
                {
                    "name": "Document",
                    "description": "A source document",
                    "attributes": [
                        {
                            "name": "publication_date",
                            "type": "date",
                            "description": "Date of publication",
                        },
                        {
                            "name": "page_count",
                            "type": "number",
                            "description": "Number of pages",
                        },
                    ],
                }
            ]
        }
        f = tmp_path / "schema.yaml"
        f.write_text(yaml.dump(schema), encoding="utf-8")
        cfg = load_graph_config(str(f))
        attrs = cfg.entities[0].attributes
        assert len(attrs) == 2
        assert attrs[0].name == "publication_date"
        assert attrs[0].type == "date"
        assert attrs[0].description == "Date of publication"
        assert attrs[1].name == "page_count"
        assert attrs[1].type == "number"

    def test_mixed_attribute_formats(self, tmp_path):
        """Mix of string and dict attributes is handled."""
        schema = {
            "entities": [
                {
                    "name": "Thing",
                    "description": "A thing",
                    "attributes": [
                        "simple_attr",
                        {"name": "typed_attr", "type": "number"},
                    ],
                }
            ]
        }
        f = tmp_path / "schema.yaml"
        f.write_text(yaml.dump(schema), encoding="utf-8")
        cfg = load_graph_config(str(f))
        attrs = cfg.entities[0].attributes
        assert len(attrs) == 2
        assert attrs[0].name == "simple_attr"
        assert attrs[0].type == "string"
        assert attrs[1].name == "typed_attr"
        assert attrs[1].type == "number"


# ---------------------------------------------------------------------------
# Singleton thread safety
# ---------------------------------------------------------------------------


class TestSingletonThreadSafety:

    def test_get_graph_config_returns_same_instance(self):
        """get_graph_config() returns the same object on repeated calls."""
        import cognidoc.graph_config as gc_mod

        # Reset the global
        gc_mod._config = None
        try:
            cfg1 = get_graph_config()
            cfg2 = get_graph_config()
            assert cfg1 is cfg2
        finally:
            gc_mod._config = None

    def test_reload_replaces_instance(self, tmp_path):
        """reload_graph_config() replaces the cached instance."""
        import cognidoc.graph_config as gc_mod

        gc_mod._config = None
        try:
            cfg1 = get_graph_config()

            f = tmp_path / "new.yaml"
            f.write_text(
                yaml.dump({"domain": {"name": "Reloaded"}}),
                encoding="utf-8",
            )
            cfg2 = reload_graph_config(str(f))
            assert cfg2.domain.name == "Reloaded"
            assert cfg2 is not cfg1
            # Subsequent calls should return the reloaded one
            assert get_graph_config() is cfg2
        finally:
            gc_mod._config = None

    def test_concurrent_access(self):
        """Multiple threads calling get_graph_config() don't race."""
        import cognidoc.graph_config as gc_mod

        gc_mod._config = None
        results = []
        errors = []

        def worker():
            try:
                cfg = get_graph_config()
                results.append(id(cfg))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert not errors, f"Errors in threads: {errors}"
            # All threads should see the same instance
            assert len(set(results)) == 1
        finally:
            gc_mod._config = None


# ---------------------------------------------------------------------------
# EntityResolutionConfig.from_env
# ---------------------------------------------------------------------------


class TestEntityResolutionFromEnv:

    def test_from_env_uses_constants(self):
        """from_env() reads values from constants module."""
        with patch.dict("os.environ", {}, clear=False):
            cfg = EntityResolutionConfig.from_env()
            assert isinstance(cfg, EntityResolutionConfig)
            assert isinstance(cfg.enabled, bool)
            assert 0 <= cfg.similarity_threshold <= 1
            assert cfg.batch_size > 0
