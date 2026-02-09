"""Tests for extract_entities.py — dataclasses, JSON parsing, prompt building."""

import json
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.extract_entities import (
    Entity,
    Relationship,
    ExtractionResult,
    extract_json_from_response,
    build_entity_extraction_prompt,
    build_relationship_extraction_prompt,
    extract_entities_from_text,
    extract_relationships_from_text,
    extract_from_chunk,
    get_optimal_concurrency,
    save_extraction_results,
    load_extraction_results,
)


# ===========================================================================
# Dataclass serialization
# ===========================================================================


class TestEntityDataclass:
    """Tests for Entity dataclass."""

    def test_to_dict(self):
        e = Entity(id="e1", name="Python", type="LANGUAGE", description="A language")
        d = e.to_dict()
        assert d["id"] == "e1"
        assert d["name"] == "Python"
        assert d["type"] == "LANGUAGE"

    def test_default_fields(self):
        e = Entity(id="e1", name="X", type="T")
        assert e.description == ""
        assert e.attributes == {}
        assert e.confidence == 1.0

    def test_to_dict_with_attributes(self):
        e = Entity(id="e1", name="X", type="T", attributes={"key": "val"})
        d = e.to_dict()
        assert d["attributes"] == {"key": "val"}


class TestRelationshipDataclass:
    """Tests for Relationship dataclass."""

    def test_to_dict(self):
        r = Relationship(
            source_entity="A", target_entity="B", relationship_type="USES", source_chunk="c1"
        )
        d = r.to_dict()
        assert d["source_entity"] == "A"
        assert d["target_entity"] == "B"
        assert d["relationship_type"] == "USES"

    def test_default_fields(self):
        r = Relationship(source_entity="A", target_entity="B", relationship_type="R")
        assert r.description == ""
        assert r.confidence == 1.0


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_to_dict_truncates_text(self):
        long_text = "x" * 300
        result = ExtractionResult(
            chunk_id="c1",
            chunk_text=long_text,
            entities=[Entity(id="e1", name="X", type="T")],
            relationships=[],
        )
        d = result.to_dict()
        assert len(d["chunk_text"]) <= 203  # 200 + "..."

    def test_to_dict_counts(self):
        result = ExtractionResult(
            chunk_id="c1",
            chunk_text="text",
            entities=[Entity(id="e1", name="X", type="T"), Entity(id="e2", name="Y", type="T")],
            relationships=[
                Relationship(source_entity="X", target_entity="Y", relationship_type="R")
            ],
        )
        d = result.to_dict()
        assert len(d["entities"]) == 2
        assert len(d["relationships"]) == 1


# ===========================================================================
# JSON parsing
# ===========================================================================


class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response()."""

    def test_parses_json_block(self):
        response = '```json\n{"entities": [{"name": "X", "type": "T"}]}\n```'
        result = extract_json_from_response(response, "entities")
        assert "entities" in result

    def test_parses_raw_json(self):
        response = '{"entities": [{"name": "X", "type": "T"}]}'
        result = extract_json_from_response(response, "entities")
        assert "entities" in result

    def test_handles_invalid_json(self):
        response = "This is not JSON at all"
        result = extract_json_from_response(response, "entities")
        assert result == {}

    def test_parses_french_json(self):
        response = '{"entités": [{"nom": "Python", "type_entité": "LANGUAGE"}]}'
        result = extract_json_from_response(response, "entities")
        # extract_json_from_response returns raw parsed JSON; field normalization
        # happens downstream in extract_entities_from_text
        assert "entités" in result


# ===========================================================================
# Prompt building
# ===========================================================================


class TestBuildPrompts:
    """Tests for prompt building functions."""

    def test_entity_prompt_contains_text(self):
        from cognidoc.graph_config import get_graph_config

        config = get_graph_config()
        prompt = build_entity_extraction_prompt("Sample text about AI", config)
        assert "Sample text about AI" in prompt

    def test_relationship_prompt_contains_entities(self):
        from cognidoc.graph_config import get_graph_config

        config = get_graph_config()
        entities = [Entity(id="e1", name="Python", type="LANGUAGE")]
        prompt = build_relationship_extraction_prompt("text", entities, config)
        assert "Python" in prompt


# ===========================================================================
# Extraction with mocked LLM
# ===========================================================================


class TestExtractEntitiesFromText:
    """Tests for extract_entities_from_text() with mocked LLM."""

    @patch("cognidoc.extract_entities.llm_chat")
    def test_extracts_entities(self, mock_llm):
        mock_llm.return_value = (
            '{"entities": [{"name": "Python", "type": "LANGUAGE", "description": "A language"}]}'
        )
        entities = extract_entities_from_text("Python is a language", "c1")
        assert len(entities) >= 1
        assert any(e.name == "Python" for e in entities)

    @patch("cognidoc.extract_entities.llm_chat")
    def test_handles_llm_error(self, mock_llm):
        mock_llm.side_effect = Exception("API error")
        entities = extract_entities_from_text("text", "c1")
        assert entities == []


class TestExtractRelationshipsFromText:
    """Tests for extract_relationships_from_text() with mocked LLM."""

    @patch("cognidoc.extract_entities.llm_chat")
    def test_extracts_relationships(self, mock_llm):
        mock_llm.return_value = (
            '{"relationships": [{"source": "A", "target": "B", "type": "USES"}]}'
        )
        entities = [
            Entity(id="e1", name="A", type="T"),
            Entity(id="e2", name="B", type="T"),
        ]
        rels = extract_relationships_from_text("A uses B", entities, "c1")
        assert len(rels) >= 1

    @patch("cognidoc.extract_entities.llm_chat")
    def test_handles_llm_error(self, mock_llm):
        mock_llm.side_effect = Exception("API error")
        rels = extract_relationships_from_text("text", [], "c1")
        assert rels == []


class TestExtractFromChunk:
    """Tests for extract_from_chunk()."""

    @patch("cognidoc.extract_entities.llm_chat")
    def test_returns_extraction_result(self, mock_llm):
        mock_llm.return_value = '{"entities": [{"name": "X", "type": "T"}]}'
        result = extract_from_chunk("Some text about X", "c1")
        assert isinstance(result, ExtractionResult)
        assert result.chunk_id == "c1"


# ===========================================================================
# Utilities
# ===========================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_get_optimal_concurrency(self):
        n = get_optimal_concurrency()
        assert 2 <= n <= 8

    def test_save_and_load_results(self, tmp_path):
        results = [
            ExtractionResult(
                chunk_id="c1",
                chunk_text="text",
                entities=[Entity(id="e1", name="X", type="T")],
                relationships=[],
            )
        ]
        path = str(tmp_path / "results.json")
        save_extraction_results(results, path)
        loaded = load_extraction_results(path)
        assert len(loaded) == 1
        assert loaded[0].chunk_id == "c1"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_extraction_results(str(tmp_path / "nonexistent.json"))


# ===========================================================================
# Attribute extraction
# ===========================================================================


class TestAttributeExtraction:
    """Tests for entity attribute extraction and parsing."""

    @patch("cognidoc.extract_entities.llm_chat")
    def test_attributes_parsed_from_llm_response(self, mock_llm):
        """Attributes dict is populated from LLM response."""
        mock_llm.return_value = json.dumps(
            {
                "entities": [
                    {
                        "name": "Treaty of Rome",
                        "type": "EVENT",
                        "description": "Founding treaty",
                        "confidence": 0.95,
                        "attributes": {
                            "date": "1957",
                            "location": "Rome",
                            "status": "ratified",
                        },
                    }
                ]
            }
        )
        entities = extract_entities_from_text("The Treaty of Rome was signed in 1957.", "c1")
        assert len(entities) == 1
        assert entities[0].attributes["date"] == "1957"
        assert entities[0].attributes["location"] == "Rome"
        assert entities[0].attributes["status"] == "ratified"

    @patch("cognidoc.extract_entities.llm_chat")
    def test_missing_attributes_defaults_to_empty_dict(self, mock_llm):
        """When LLM omits attributes, they default to empty dict."""
        mock_llm.return_value = json.dumps(
            {
                "entities": [
                    {
                        "name": "Python",
                        "type": "LANGUAGE",
                        "description": "A programming language",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        entities = extract_entities_from_text("Python is a language.", "c1")
        assert len(entities) == 1
        assert entities[0].attributes == {}

    @patch("cognidoc.extract_entities.llm_chat")
    def test_non_dict_attributes_ignored(self, mock_llm):
        """When LLM returns non-dict attributes, they are treated as empty dict."""
        mock_llm.return_value = json.dumps(
            {
                "entities": [
                    {
                        "name": "X",
                        "type": "T",
                        "description": "desc",
                        "confidence": 0.9,
                        "attributes": "not_a_dict",
                    }
                ]
            }
        )
        entities = extract_entities_from_text("text", "c1")
        assert len(entities) == 1
        assert entities[0].attributes == {}

    def test_prompt_contains_attributes_instructions(self):
        """Prompt includes instructions for attribute extraction."""
        from cognidoc.graph_config import GraphConfig

        config = GraphConfig()
        prompt = build_entity_extraction_prompt("Sample text", config)
        assert "attributes" in prompt.lower()
        assert "date" in prompt.lower() or "Date" in prompt

    def test_prompt_includes_schema_attributes(self):
        """When entity types have typed attributes, they appear in the prompt."""
        from cognidoc.graph_config import GraphConfig, EntityType, EntityAttribute

        config = GraphConfig(
            entities=[
                EntityType(
                    name="Document",
                    description="A document",
                    attributes=[
                        EntityAttribute(
                            name="publication_date", type="date", description="Date of publication"
                        ),
                        EntityAttribute(
                            name="page_count", type="number", description="Number of pages"
                        ),
                    ],
                )
            ]
        )
        prompt = build_entity_extraction_prompt("Sample text", config)
        assert "publication_date (date)" in prompt
        assert "page_count (number)" in prompt

    def test_json_parser_normalizes_french_attributes(self):
        """extract_json_from_response normalizes 'attributs' to 'attributes'."""
        response = json.dumps(
            {
                "entities": [
                    {
                        "nom": "Paris",
                        "type": "LIEU",
                        "description": "Capitale",
                        "confiance": 0.9,
                        "attributs": {"population": 2161000},
                    }
                ]
            }
        )
        result = extract_json_from_response(response, "entities")
        entity = result["entities"][0]
        assert "attributes" in entity
        assert entity["attributes"]["population"] == 2161000

    def test_save_load_preserves_attributes(self, tmp_path):
        """Attributes survive save/load round-trip."""
        results = [
            ExtractionResult(
                chunk_id="c1",
                chunk_text="text",
                entities=[
                    Entity(
                        id="e1",
                        name="X",
                        type="T",
                        attributes={"date": "2024", "count": 42},
                    )
                ],
                relationships=[],
            )
        ]
        path = str(tmp_path / "results.json")
        save_extraction_results(results, path)
        loaded = load_extraction_results(path)
        assert loaded[0].entities[0].attributes == {"date": "2024", "count": 42}
