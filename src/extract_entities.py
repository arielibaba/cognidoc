"""
Entity and relationship extraction module for GraphRAG.

Uses LLM to extract structured entities and relationships from text chunks
based on the graph schema configuration.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import ollama

from .graph_config import get_graph_config, GraphConfig
from .constants import LLM, CHUNKS_DIR, PROCESSED_DIR
from .utils.logger import logger


@dataclass
class Entity:
    """Extracted entity from text."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: str = ""
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_chunk: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relationship:
    """Extracted relationship between entities."""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_entity: str = ""  # Entity name
    target_entity: str = ""  # Entity name
    relationship_type: str = ""
    description: str = ""
    source_chunk: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction from a chunk."""
    chunk_id: str
    chunk_text: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text[:200] + "..." if len(self.chunk_text) > 200 else self.chunk_text,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
        }


def build_entity_extraction_prompt(
    text: str,
    config: GraphConfig,
) -> str:
    """Build the prompt for entity extraction."""
    # Build entity types description
    entity_types_desc = []
    for et in config.entities:
        examples_str = ", ".join(et.examples[:3]) if et.examples else "N/A"
        entity_types_desc.append(
            f"- {et.name}: {et.description} (examples: {examples_str})"
        )
    entity_types_str = "\n".join(entity_types_desc)

    # Use custom prompt if provided
    if config.custom_prompts.entity_extraction:
        return config.custom_prompts.entity_extraction.format(
            text=text,
            entity_types=entity_types_str,
            domain=config.domain.description,
        )

    # Default prompt
    prompt = f"""You are an expert entity extractor. Your task is to identify and extract entities from the given text.

DOMAIN CONTEXT:
{config.domain.description}

ENTITY TYPES TO EXTRACT:
{entity_types_str}

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

INSTRUCTIONS:
1. Identify all entities in the text that match the entity types above
2. For each entity, provide:
   - name: The canonical name of the entity
   - type: One of the entity types listed above
   - description: A brief description based on the text context
   - confidence: Your confidence in this extraction (0.0-1.0)
3. Be precise and only extract entities that are clearly mentioned
4. Normalize entity names (e.g., "ML" and "Machine Learning" should be "Machine Learning")

OUTPUT FORMAT (JSON):
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "EntityType",
      "description": "Brief description from context",
      "confidence": 0.95
    }}
  ]
}}

Extract entities from the text above. Output ONLY valid JSON, no explanations."""

    return prompt


def build_relationship_extraction_prompt(
    text: str,
    entities: List[Entity],
    config: GraphConfig,
) -> str:
    """Build the prompt for relationship extraction."""
    # Build relationship types description
    rel_types_desc = []
    for rt in config.relationships:
        examples_str = "; ".join(rt.examples[:2]) if rt.examples else "N/A"
        rel_types_desc.append(
            f"- {rt.name}: {rt.description} (examples: {examples_str})"
        )
    rel_types_str = "\n".join(rel_types_desc)

    # Build entities list
    entities_str = "\n".join([f"- {e.name} ({e.type})" for e in entities])

    # Use custom prompt if provided
    if config.custom_prompts.relationship_extraction:
        return config.custom_prompts.relationship_extraction.format(
            text=text,
            entities=entities_str,
            relationship_types=rel_types_str,
        )

    # Default prompt
    prompt = f"""You are an expert relationship extractor. Your task is to identify relationships between entities in the given text.

ENTITIES FOUND IN TEXT:
{entities_str}

RELATIONSHIP TYPES TO EXTRACT:
{rel_types_str}

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

INSTRUCTIONS:
1. Identify relationships between the entities listed above
2. Only extract relationships that are explicitly or implicitly stated in the text
3. For each relationship, provide:
   - source: The source entity name (must be from the entities list)
   - target: The target entity name (must be from the entities list)
   - type: One of the relationship types listed above
   - description: Brief description of the relationship
   - confidence: Your confidence in this extraction (0.0-1.0)
4. Do not invent relationships that are not supported by the text

OUTPUT FORMAT (JSON):
{{
  "relationships": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "type": "RELATIONSHIP_TYPE",
      "description": "Brief description of the relationship",
      "confidence": 0.9
    }}
  ]
}}

Extract relationships from the text above. Output ONLY valid JSON, no explanations."""

    return prompt


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling common issues."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Return empty result
    logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
    return {}


def extract_entities_from_text(
    text: str,
    chunk_id: str,
    config: Optional[GraphConfig] = None,
    model: str = None,
) -> List[Entity]:
    """
    Extract entities from text using LLM.

    Args:
        text: Text to extract entities from
        chunk_id: ID of the source chunk
        config: Graph configuration (uses global if not provided)
        model: LLM model to use (uses default if not provided)

    Returns:
        List of extracted entities
    """
    if config is None:
        config = get_graph_config()
    if model is None:
        model = LLM

    prompt = build_entity_extraction_prompt(text, config)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": config.extraction.extraction_temperature},
        )
        result_text = response["message"]["content"]
        result = extract_json_from_response(result_text)

        entities = []
        for e in result.get("entities", []):
            # Filter by confidence
            confidence = float(e.get("confidence", 1.0))
            if confidence < config.extraction.min_confidence:
                continue

            entity = Entity(
                name=e.get("name", "").strip(),
                type=e.get("type", "").strip(),
                description=e.get("description", "").strip(),
                confidence=confidence,
                source_chunk=chunk_id,
            )
            if entity.name and entity.type:
                entities.append(entity)

        # Limit to max entities
        return entities[:config.extraction.max_entities_per_chunk]

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []


def extract_relationships_from_text(
    text: str,
    entities: List[Entity],
    chunk_id: str,
    config: Optional[GraphConfig] = None,
    model: str = None,
) -> List[Relationship]:
    """
    Extract relationships from text using LLM.

    Args:
        text: Text to extract relationships from
        entities: Entities already extracted from the text
        chunk_id: ID of the source chunk
        config: Graph configuration (uses global if not provided)
        model: LLM model to use (uses default if not provided)

    Returns:
        List of extracted relationships
    """
    if not entities:
        return []

    if config is None:
        config = get_graph_config()
    if model is None:
        model = LLM

    prompt = build_relationship_extraction_prompt(text, entities, config)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": config.extraction.extraction_temperature},
        )
        result_text = response["message"]["content"]
        result = extract_json_from_response(result_text)

        # Build entity name set for validation
        entity_names = {e.name.lower() for e in entities}

        relationships = []
        for r in result.get("relationships", []):
            # Filter by confidence
            confidence = float(r.get("confidence", 1.0))
            if confidence < config.extraction.min_confidence:
                continue

            source = r.get("source", "").strip()
            target = r.get("target", "").strip()

            # Validate entities exist
            if source.lower() not in entity_names or target.lower() not in entity_names:
                continue

            rel = Relationship(
                source_entity=source,
                target_entity=target,
                relationship_type=r.get("type", "RELATED_TO").strip(),
                description=r.get("description", "").strip(),
                confidence=confidence,
                source_chunk=chunk_id,
            )
            relationships.append(rel)

        # Limit to max relationships
        return relationships[:config.extraction.max_relationships_per_chunk]

    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        return []


def extract_from_chunk(
    text: str,
    chunk_id: str,
    config: Optional[GraphConfig] = None,
    model: str = None,
) -> ExtractionResult:
    """
    Extract entities and relationships from a single chunk.

    Args:
        text: Chunk text
        chunk_id: Chunk identifier
        config: Graph configuration
        model: LLM model to use

    Returns:
        ExtractionResult with entities and relationships
    """
    logger.info(f"Extracting from chunk: {chunk_id}")

    # Extract entities first
    entities = extract_entities_from_text(text, chunk_id, config, model)
    logger.debug(f"Extracted {len(entities)} entities")

    # Then extract relationships using found entities
    relationships = extract_relationships_from_text(text, entities, chunk_id, config, model)
    logger.debug(f"Extracted {len(relationships)} relationships")

    return ExtractionResult(
        chunk_id=chunk_id,
        chunk_text=text,
        entities=entities,
        relationships=relationships,
    )


def extract_from_chunks_dir(
    chunks_dir: str = None,
    config: Optional[GraphConfig] = None,
    model: str = None,
    include_parent_chunks: bool = False,
) -> List[ExtractionResult]:
    """
    Extract entities and relationships from all chunks in a directory.

    Args:
        chunks_dir: Directory containing chunk files
        config: Graph configuration
        model: LLM model to use
        include_parent_chunks: Whether to process parent chunks (default: False)

    Returns:
        List of extraction results
    """
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR

    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        logger.warning(f"Chunks directory not found: {chunks_dir}")
        return []

    results = []

    for chunk_file in sorted(chunks_path.rglob("*.txt")):
        # Skip parent chunks unless explicitly requested
        if not include_parent_chunks:
            if "_parent_chunk_" in chunk_file.name and "_child_chunk_" not in chunk_file.name:
                continue

        try:
            text = chunk_file.read_text(encoding="utf-8")
            if not text.strip():
                continue

            result = extract_from_chunk(
                text=text,
                chunk_id=chunk_file.stem,
                config=config,
                model=model,
            )
            results.append(result)

            logger.info(
                f"Processed {chunk_file.name}: "
                f"{len(result.entities)} entities, "
                f"{len(result.relationships)} relationships"
            )

        except Exception as e:
            logger.error(f"Error processing {chunk_file.name}: {e}")
            continue

    logger.info(f"Extraction complete: {len(results)} chunks processed")
    return results


def save_extraction_results(
    results: List[ExtractionResult],
    output_path: str,
) -> None:
    """Save extraction results to JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_chunks": len(results),
        "total_entities": sum(len(r.entities) for r in results),
        "total_relationships": sum(len(r.relationships) for r in results),
        "results": [r.to_dict() for r in results],
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved extraction results to {output}")


def load_extraction_results(input_path: str) -> List[ExtractionResult]:
    """Load extraction results from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for r in data.get("results", []):
        entities = [
            Entity(
                id=e.get("id", str(uuid4())),
                name=e["name"],
                type=e["type"],
                description=e.get("description", ""),
                attributes=e.get("attributes", {}),
                source_chunk=e.get("source_chunk", ""),
                confidence=e.get("confidence", 1.0),
            )
            for e in r.get("entities", [])
        ]
        relationships = [
            Relationship(
                id=rel.get("id", str(uuid4())),
                source_entity=rel["source_entity"],
                target_entity=rel["target_entity"],
                relationship_type=rel["relationship_type"],
                description=rel.get("description", ""),
                source_chunk=rel.get("source_chunk", ""),
                confidence=rel.get("confidence", 1.0),
            )
            for rel in r.get("relationships", [])
        ]
        results.append(ExtractionResult(
            chunk_id=r["chunk_id"],
            chunk_text=r.get("chunk_text", ""),
            entities=entities,
            relationships=relationships,
        ))

    return results
