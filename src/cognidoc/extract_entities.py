"""
Entity and relationship extraction module for GraphRAG.

Uses LLM to extract structured entities and relationships from text chunks
based on the graph schema configuration.

Supports async extraction with configurable concurrency for improved throughput
when using cloud LLM providers (Gemini, OpenAI, etc.).
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from .graph_config import get_graph_config, GraphConfig
from .constants import CHUNKS_DIR, PROCESSED_DIR
from .utils.llm_client import llm_chat, llm_chat_async
from .utils.logger import logger


def get_optimal_concurrency() -> int:
    """
    Compute optimal concurrency based on system resources.

    Returns:
        Recommended max_concurrent value:
        - For powerful systems (8+ cores): up to 8
        - For standard systems (4 cores): 4
        - For limited systems (2 cores): 2
        - Minimum: 2
    """
    try:
        cpu_count = os.cpu_count() or 4
        # Use min(8, cpu_count - 1) to leave one core for system
        # But at least 2 for parallelism benefit
        optimal = max(2, min(8, cpu_count - 1))
        return optimal
    except Exception:
        return 4  # Safe default


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


def extract_json_from_response(response: str, key: str = "entities") -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling common issues.

    Args:
        response: Raw LLM response text
        key: Expected key for list wrapping (e.g., "entities" or "relationships")

    Returns:
        Dict with the extracted JSON, normalized to have the expected structure
    """
    parsed = None

    # Try direct parse first
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    if parsed is None:
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Try to find JSON array
    if parsed is None:
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Try to find JSON object
    if parsed is None:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Handle parsing failure
    if parsed is None:
        logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
        return {}

    # Normalize structure: if we got a list, wrap it in {key: list}
    if isinstance(parsed, list):
        parsed = {key: parsed}

    # Normalize field names (handle French field names from Gemini)
    if key in parsed and isinstance(parsed[key], list):
        normalized = []
        for item in parsed[key]:
            if isinstance(item, dict):
                norm_item = {}
                for k, v in item.items():
                    # Map common French field names to English
                    k_lower = k.lower()
                    if k_lower in ("nom", "name"):
                        norm_item["name"] = v
                    elif k_lower in ("type", "type_entite", "type_entité"):
                        norm_item["type"] = v
                    elif k_lower in ("description",):
                        norm_item["description"] = v
                    elif k_lower in ("confiance", "confidence"):
                        norm_item["confidence"] = v
                    elif k_lower in ("source", "source_entity", "entite_source", "entité_source"):
                        norm_item["source"] = v
                    elif k_lower in ("target", "target_entity", "cible", "entite_cible", "entité_cible"):
                        norm_item["target"] = v
                    elif k_lower in ("relation", "type", "relationship_type", "type_relation"):
                        norm_item["type"] = v
                    else:
                        norm_item[k] = v
                normalized.append(norm_item)
        parsed[key] = normalized

    return parsed


def extract_entities_from_text(
    text: str,
    chunk_id: str,
    config: Optional[GraphConfig] = None,
) -> List[Entity]:
    """
    Extract entities from text using LLM.

    Uses the unified LLM client (Gemini by default).

    Args:
        text: Text to extract entities from
        chunk_id: ID of the source chunk
        config: Graph configuration (uses global if not provided)

    Returns:
        List of extracted entities
    """
    if config is None:
        config = get_graph_config()

    prompt = build_entity_extraction_prompt(text, config)

    try:
        result_text = llm_chat(
            messages=[
                {"role": "system", "content": "You are a JSON entity extractor. You MUST respond with ONLY valid JSON, no explanations or text before/after."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.extraction.extraction_temperature,
            json_mode=True,
        )
        result = extract_json_from_response(result_text, key="entities")

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
) -> List[Relationship]:
    """
    Extract relationships from text using LLM.

    Uses the unified LLM client (Gemini by default).

    Args:
        text: Text to extract relationships from
        entities: Entities already extracted from the text
        chunk_id: ID of the source chunk
        config: Graph configuration (uses global if not provided)

    Returns:
        List of extracted relationships
    """
    if not entities:
        return []

    if config is None:
        config = get_graph_config()

    prompt = build_relationship_extraction_prompt(text, entities, config)

    try:
        result_text = llm_chat(
            messages=[
                {"role": "system", "content": "You are a JSON relationship extractor. You MUST respond with ONLY valid JSON, no explanations or text before/after."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.extraction.extraction_temperature,
            json_mode=True,
        )
        result = extract_json_from_response(result_text, key="relationships")

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
) -> ExtractionResult:
    """
    Extract entities and relationships from a single chunk.

    Uses the unified LLM client (Gemini by default).

    Args:
        text: Chunk text
        chunk_id: Chunk identifier
        config: Graph configuration

    Returns:
        ExtractionResult with entities and relationships
    """
    logger.info(f"Extracting from chunk: {chunk_id}")

    # Extract entities first
    entities = extract_entities_from_text(text, chunk_id, config)
    logger.debug(f"Extracted {len(entities)} entities")

    # Then extract relationships using found entities
    relationships = extract_relationships_from_text(text, entities, chunk_id, config)
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
    include_parent_chunks: bool = True,
    include_child_chunks: bool = False,
    include_descriptions: bool = True,
) -> List[ExtractionResult]:
    """
    Extract entities and relationships from all chunks in a directory.

    Uses the unified LLM client (Gemini by default).

    By default, extracts from parent chunks (512 tokens) and descriptions,
    skipping child chunks (64 tokens) to avoid redundant extractions.

    Args:
        chunks_dir: Directory containing chunk files
        config: Graph configuration
        include_parent_chunks: Whether to process parent chunks (default: True)
        include_child_chunks: Whether to process child chunks (default: False)
        include_descriptions: Whether to process description chunks (default: True)

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
        # Classify chunk type
        is_parent_only = "_parent_chunk_" in chunk_file.name and "_child_chunk_" not in chunk_file.name
        is_child = "_child_chunk_" in chunk_file.name
        is_description = "_description" in chunk_file.name

        # Apply filters
        if is_parent_only and not include_parent_chunks:
            continue
        if is_child and not include_child_chunks:
            continue
        if is_description and not include_descriptions:
            continue

        try:
            text = chunk_file.read_text(encoding="utf-8")
            if not text.strip():
                continue

            result = extract_from_chunk(
                text=text,
                chunk_id=chunk_file.stem,
                config=config,
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


# =============================================================================
# ASYNC EXTRACTION FUNCTIONS
# =============================================================================


async def extract_entities_from_text_async(
    text: str,
    chunk_id: str,
    config: Optional[GraphConfig] = None,
) -> List[Entity]:
    """
    Async version of extract_entities_from_text.

    Uses llm_chat_async for non-blocking LLM calls, enabling concurrent
    extraction across multiple chunks.

    Args:
        text: Text to extract entities from
        chunk_id: ID of the source chunk
        config: Graph configuration (uses global if not provided)

    Returns:
        List of extracted entities
    """
    if config is None:
        config = get_graph_config()

    prompt = build_entity_extraction_prompt(text, config)

    try:
        result_text = await llm_chat_async(
            messages=[
                {"role": "system", "content": "You are a JSON entity extractor. You MUST respond with ONLY valid JSON, no explanations or text before/after."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.extraction.extraction_temperature,
            json_mode=True,
        )
        result = extract_json_from_response(result_text, key="entities")

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
        logger.error(f"Async entity extraction failed for {chunk_id}: {e}")
        return []


async def extract_relationships_from_text_async(
    text: str,
    entities: List[Entity],
    chunk_id: str,
    config: Optional[GraphConfig] = None,
) -> List[Relationship]:
    """
    Async version of extract_relationships_from_text.

    Uses llm_chat_async for non-blocking LLM calls.

    Args:
        text: Text to extract relationships from
        entities: Entities already extracted from the text
        chunk_id: ID of the source chunk
        config: Graph configuration (uses global if not provided)

    Returns:
        List of extracted relationships
    """
    if not entities:
        return []

    if config is None:
        config = get_graph_config()

    prompt = build_relationship_extraction_prompt(text, entities, config)

    try:
        result_text = await llm_chat_async(
            messages=[
                {"role": "system", "content": "You are a JSON relationship extractor. You MUST respond with ONLY valid JSON, no explanations or text before/after."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.extraction.extraction_temperature,
            json_mode=True,
        )
        result = extract_json_from_response(result_text, key="relationships")

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
        logger.error(f"Async relationship extraction failed for {chunk_id}: {e}")
        return []


async def extract_from_chunk_async(
    text: str,
    chunk_id: str,
    config: Optional[GraphConfig] = None,
) -> ExtractionResult:
    """
    Async version of extract_from_chunk.

    Extracts entities and relationships from a single chunk using async LLM calls.

    Args:
        text: Chunk text
        chunk_id: Chunk identifier
        config: Graph configuration

    Returns:
        ExtractionResult with entities and relationships
    """
    logger.debug(f"Async extracting from chunk: {chunk_id}")

    # Extract entities first
    entities = await extract_entities_from_text_async(text, chunk_id, config)

    # Then extract relationships using found entities
    relationships = await extract_relationships_from_text_async(text, entities, chunk_id, config)

    return ExtractionResult(
        chunk_id=chunk_id,
        chunk_text=text,
        entities=entities,
        relationships=relationships,
    )


async def extract_from_chunks_dir_async(
    chunks_dir: str = None,
    config: Optional[GraphConfig] = None,
    include_parent_chunks: bool = True,
    include_child_chunks: bool = False,
    include_descriptions: bool = True,
    max_concurrent: int = None,
    show_progress: bool = True,
) -> List[ExtractionResult]:
    """
    Async version of extract_from_chunks_dir with concurrent extraction.

    Processes multiple chunks concurrently using a semaphore to control
    the number of simultaneous LLM calls. This significantly improves
    throughput when using cloud LLM providers like Gemini.

    By default, extracts from parent chunks (512 tokens) and descriptions,
    skipping child chunks (64 tokens) to avoid redundant extractions.

    Args:
        chunks_dir: Directory containing chunk files
        config: Graph configuration
        include_parent_chunks: Whether to process parent chunks (default: True)
        include_child_chunks: Whether to process child chunks (default: False)
        include_descriptions: Whether to process description chunks (default: True)
        max_concurrent: Maximum number of concurrent extractions (default: auto)
                       Auto-detected based on CPU cores. Override for specific needs:
                       4-8 for Gemini API, 2-4 for local Ollama
        show_progress: Show progress bar (default: True)

    Returns:
        List of extraction results
    """
    if chunks_dir is None:
        chunks_dir = CHUNKS_DIR

    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        logger.warning(f"Chunks directory not found: {chunks_dir}")
        return []

    # Collect all chunk files to process
    chunk_files = []
    for chunk_file in sorted(chunks_path.rglob("*.txt")):
        # Classify chunk type
        is_parent_only = "_parent_chunk_" in chunk_file.name and "_child_chunk_" not in chunk_file.name
        is_child = "_child_chunk_" in chunk_file.name
        is_description = "_description" in chunk_file.name

        # Apply filters
        if is_parent_only and not include_parent_chunks:
            continue
        if is_child and not include_child_chunks:
            continue
        if is_description and not include_descriptions:
            continue
        chunk_files.append(chunk_file)

    if not chunk_files:
        logger.warning(f"No chunk files found in {chunks_dir}")
        return []

    # Use adaptive concurrency if not specified
    if max_concurrent is None:
        max_concurrent = get_optimal_concurrency()

    logger.info(f"Starting async extraction for {len(chunk_files)} chunks (max_concurrent={max_concurrent})")

    # Semaphore to limit concurrent extractions
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_chunk(chunk_file: Path) -> Optional[ExtractionResult]:
        """Process a single chunk with semaphore control."""
        async with semaphore:
            try:
                text = chunk_file.read_text(encoding="utf-8")
                if not text.strip():
                    return None

                result = await extract_from_chunk_async(
                    text=text,
                    chunk_id=chunk_file.stem,
                    config=config,
                )

                logger.debug(
                    f"Processed {chunk_file.name}: "
                    f"{len(result.entities)} entities, "
                    f"{len(result.relationships)} relationships"
                )
                return result

            except Exception as e:
                logger.error(f"Error processing {chunk_file.name}: {e}")
                return None

    # Create tasks for all chunks
    tasks = [process_chunk(cf) for cf in chunk_files]

    # Run all tasks concurrently with optional progress bar
    if show_progress:
        results_raw = await async_tqdm.gather(
            *tasks,
            desc="Entity extraction",
            unit="chunk",
        )
    else:
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and exceptions
    results = []
    errors = 0
    for r in results_raw:
        if isinstance(r, Exception):
            logger.error(f"Chunk extraction failed: {r}")
            errors += 1
        elif r is not None:
            results.append(r)

    total_entities = sum(len(r.entities) for r in results)
    total_relationships = sum(len(r.relationships) for r in results)

    logger.info(
        f"Async extraction complete: {len(results)} chunks processed, "
        f"{total_entities} entities, {total_relationships} relationships, "
        f"{errors} errors"
    )
    return results


def run_extraction_async(
    chunks_dir: str = None,
    config: Optional[GraphConfig] = None,
    include_parent_chunks: bool = True,
    include_child_chunks: bool = False,
    include_descriptions: bool = True,
    max_concurrent: int = None,
    show_progress: bool = True,
) -> List[ExtractionResult]:
    """
    Synchronous wrapper for async extraction.

    Convenience function that handles the async event loop setup,
    making it easy to call from synchronous code.

    By default, extracts from parent chunks (512 tokens) and descriptions,
    skipping child chunks (64 tokens) to avoid redundant extractions.

    Args:
        chunks_dir: Directory containing chunk files
        config: Graph configuration
        include_parent_chunks: Whether to process parent chunks (default: True)
        include_child_chunks: Whether to process child chunks (default: False)
        include_descriptions: Whether to process description chunks (default: True)
        max_concurrent: Maximum number of concurrent extractions (default: auto)
                       Auto-detected based on CPU cores.
        show_progress: Show progress bar (default: True)

    Returns:
        List of extraction results

    Example:
        # Instead of:
        # results = extract_from_chunks_dir(chunks_dir)  # Sequential, slow

        # Use:
        results = run_extraction_async(chunks_dir)  # 2-4x faster with auto concurrency
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context, use ThreadPoolExecutor to avoid conflicts
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                extract_from_chunks_dir_async(
                    chunks_dir=chunks_dir,
                    config=config,
                    include_parent_chunks=include_parent_chunks,
                    include_child_chunks=include_child_chunks,
                    include_descriptions=include_descriptions,
                    max_concurrent=max_concurrent,
                    show_progress=show_progress,
                )
            )
            return future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run
        return asyncio.run(
            extract_from_chunks_dir_async(
                chunks_dir=chunks_dir,
                config=config,
                include_parent_chunks=include_parent_chunks,
                include_child_chunks=include_child_chunks,
                include_descriptions=include_descriptions,
                max_concurrent=max_concurrent,
                show_progress=show_progress,
            )
        )


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
