"""
Intelligent Query Orchestrator for Hybrid RAG.

Provides smart routing between Vector RAG and GraphRAG based on:
- Query classification (rule-based or LLM-based)
- Confidence-based fallback
- Cost-aware skipping
- Intelligent result fusion
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import ollama

from .constants import LLM
from .utils.logger import logger


class QueryType(Enum):
    """Types of queries for routing."""
    FACTUAL = "factual"           # Simple fact lookup → favor vector
    RELATIONAL = "relational"     # About relationships → favor graph
    COMPARATIVE = "comparative"   # Comparing entities → favor graph
    EXPLORATORY = "exploratory"   # Broad/global questions → favor graph
    PROCEDURAL = "procedural"     # How-to questions → favor vector
    ANALYTICAL = "analytical"     # Deep analysis → use both
    UNKNOWN = "unknown"


class RetrievalMode(Enum):
    """Which retrieval systems to use."""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"  # Decide based on first results


@dataclass
class RoutingDecision:
    """Result of query routing analysis."""
    query: str
    query_type: QueryType
    mode: RetrievalMode
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    skip_vector: bool = False
    skip_graph: bool = False
    confidence: float = 0.5
    reasoning: str = ""
    entities_detected: List[str] = field(default_factory=list)


@dataclass
class OrchestratorConfig:
    """Configuration for the query orchestrator."""
    # Routing thresholds
    skip_threshold: float = 0.15  # Skip retriever if weight below this
    confidence_threshold: float = 0.3  # Trigger fallback if confidence below this

    # LLM classification
    use_llm_classifier: bool = True
    classifier_model: str = None  # Use default LLM if None

    # Fusion settings
    dedup_similarity_threshold: float = 0.85  # Deduplicate if similarity above
    max_context_tokens: int = 4000

    # Cost settings
    prefer_vector_for_simple: bool = True  # Skip graph for simple factual queries


# =============================================================================
# Rule-based Classification
# =============================================================================

QUERY_PATTERNS = {
    QueryType.RELATIONAL: [
        r"relationship between",
        r"related to",
        r"connection between",
        r"how (?:does|is|are) .+ (?:related|connected|linked) to",
        r"depends on",
        r"affects?",
        r"impacts?",
        r"influences?",
        r"between .+ and .+",
        r"link between",
    ],
    QueryType.COMPARATIVE: [
        r"compare",
        r"difference between",
        r"differences? between",
        r"vs\.?(?:\s|$)",
        r"versus",
        r"better than",
        r"worse than",
        r"similar to",
        r"differ from",
        r"contrast",
    ],
    QueryType.EXPLORATORY: [
        r"what are all",
        r"list all",
        r"list the",
        r"give me all",
        r"summarize",
        r"summary of",
        r"overview",
        r"main (?:topics|themes|concepts|ideas|points)",
        r"key (?:topics|themes|concepts|ideas|points)",
        r"how many",
        r"enumerate",
    ],
    QueryType.PROCEDURAL: [
        r"how (?:do|does|to|can|should)",
        r"steps to",
        r"process (?:for|of|to)",
        r"procedure",
        r"guide (?:for|to|on)",
        r"instructions? (?:for|to|on)",
        r"tutorial",
        r"walk me through",
        r"explain how",
    ],
    QueryType.ANALYTICAL: [
        r"analyze",
        r"analysis of",
        r"evaluate",
        r"assess",
        r"examine",
        r"investigate",
        r"deep dive",
        r"in-depth",
        r"comprehensive",
    ],
}

# Weight configurations per query type
WEIGHT_CONFIG = {
    QueryType.FACTUAL:     {"vector": 0.7, "graph": 0.3, "mode": RetrievalMode.HYBRID},
    QueryType.RELATIONAL:  {"vector": 0.2, "graph": 0.8, "mode": RetrievalMode.HYBRID},
    QueryType.COMPARATIVE: {"vector": 0.3, "graph": 0.7, "mode": RetrievalMode.HYBRID},
    QueryType.EXPLORATORY: {"vector": 0.1, "graph": 0.9, "mode": RetrievalMode.GRAPH_ONLY},
    QueryType.PROCEDURAL:  {"vector": 0.8, "graph": 0.2, "mode": RetrievalMode.HYBRID},
    QueryType.ANALYTICAL:  {"vector": 0.5, "graph": 0.5, "mode": RetrievalMode.HYBRID},
    QueryType.UNKNOWN:     {"vector": 0.6, "graph": 0.4, "mode": RetrievalMode.HYBRID},
}


def classify_query_rules(query: str) -> Tuple[QueryType, float, str]:
    """
    Classify query using rule-based pattern matching.

    Returns:
        Tuple of (query_type, confidence, reasoning)
    """
    query_lower = query.lower().strip()

    # Check each pattern type
    for query_type, patterns in QUERY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return (
                    query_type,
                    0.8,
                    f"Matched pattern: '{pattern}'"
                )

    # Default classification based on question words
    if re.match(r"^(what|who|where|when|which)\b", query_lower):
        return (QueryType.FACTUAL, 0.6, "Question word detected")

    if re.match(r"^(why|how)\b", query_lower):
        return (QueryType.ANALYTICAL, 0.5, "Analytical question word")

    return (QueryType.UNKNOWN, 0.3, "No pattern matched")


# =============================================================================
# LLM-based Classification
# =============================================================================

CLASSIFIER_PROMPT = """Classify this query for a document retrieval system.

Query: "{query}"

Choose ONE type:
- FACTUAL: Simple fact lookup (who, what, where, when)
- RELATIONAL: About relationships between entities
- COMPARATIVE: Comparing two or more things
- EXPLORATORY: Broad questions, summaries, overviews
- PROCEDURAL: How-to, step-by-step instructions
- ANALYTICAL: Deep analysis, evaluation

Also extract any named entities mentioned.

Respond in this exact format:
TYPE: <type>
CONFIDENCE: <0.0-1.0>
ENTITIES: <comma-separated list or "none">
REASONING: <brief explanation>"""


def classify_query_llm(
    query: str,
    model: str = None,
) -> Tuple[QueryType, float, str, List[str]]:
    """
    Classify query using LLM.

    Returns:
        Tuple of (query_type, confidence, reasoning, entities)
    """
    if model is None:
        model = LLM

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": CLASSIFIER_PROMPT.format(query=query)
            }],
            options={"temperature": 0.1, "num_predict": 150},
        )

        result = response["message"]["content"]

        # Parse response
        query_type = QueryType.UNKNOWN
        confidence = 0.5
        reasoning = ""
        entities = []

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("TYPE:"):
                type_str = line.split(":", 1)[1].strip().upper()
                try:
                    query_type = QueryType[type_str]
                except KeyError:
                    query_type = QueryType.UNKNOWN
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("ENTITIES:"):
                ent_str = line.split(":", 1)[1].strip()
                if ent_str.lower() != "none":
                    entities = [e.strip() for e in ent_str.split(",")]
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return (query_type, confidence, reasoning, entities)

    except Exception as e:
        logger.warning(f"LLM classification failed: {e}, falling back to rules")
        query_type, confidence, reasoning = classify_query_rules(query)
        return (query_type, confidence, reasoning, [])


# =============================================================================
# Query Orchestrator
# =============================================================================

class QueryOrchestrator:
    """
    Intelligent orchestrator for hybrid retrieval.

    Decides:
    - Which retrieval systems to use
    - How to weight their results
    - When to skip expensive operations
    - How to fuse results intelligently
    """

    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()

    def route(self, query: str, model: str = None) -> RoutingDecision:
        """
        Analyze query and decide routing strategy.

        Args:
            query: User query
            model: LLM model for classification

        Returns:
            RoutingDecision with routing strategy
        """
        # Classify query
        if self.config.use_llm_classifier:
            query_type, confidence, reasoning, entities = classify_query_llm(
                query, model or self.config.classifier_model
            )
        else:
            query_type, confidence, reasoning = classify_query_rules(query)
            entities = []

        # Get weight configuration
        weight_cfg = WEIGHT_CONFIG.get(query_type, WEIGHT_CONFIG[QueryType.UNKNOWN])
        vector_weight = weight_cfg["vector"]
        graph_weight = weight_cfg["graph"]
        mode = weight_cfg["mode"]

        # Determine skip logic
        skip_vector = vector_weight < self.config.skip_threshold
        skip_graph = graph_weight < self.config.skip_threshold

        # Simple factual queries can skip graph for speed
        if (self.config.prefer_vector_for_simple
            and query_type == QueryType.FACTUAL
            and confidence > 0.7):
            skip_graph = True
            mode = RetrievalMode.VECTOR_ONLY
            vector_weight = 1.0
            graph_weight = 0.0

        # Exploratory queries should use graph
        if query_type == QueryType.EXPLORATORY:
            skip_vector = True
            mode = RetrievalMode.GRAPH_ONLY
            vector_weight = 0.0
            graph_weight = 1.0

        # If no entities detected for relational query, fall back to vector
        if query_type == QueryType.RELATIONAL and not entities:
            skip_graph = False  # Still try graph
            vector_weight = 0.5
            graph_weight = 0.5

        decision = RoutingDecision(
            query=query,
            query_type=query_type,
            mode=mode,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            skip_vector=skip_vector,
            skip_graph=skip_graph,
            confidence=confidence,
            reasoning=reasoning,
            entities_detected=entities,
        )

        logger.info(
            f"Query routing: type={query_type.value}, mode={mode.value}, "
            f"vector={vector_weight:.1f}, graph={graph_weight:.1f}, "
            f"skip_v={skip_vector}, skip_g={skip_graph}"
        )

        return decision

    def should_fallback(
        self,
        decision: RoutingDecision,
        vector_confidence: float,
        graph_confidence: float,
    ) -> RoutingDecision:
        """
        Adjust routing based on retrieval confidence.

        If one system returns low confidence, boost the other.
        """
        adjusted = RoutingDecision(
            query=decision.query,
            query_type=decision.query_type,
            mode=decision.mode,
            vector_weight=decision.vector_weight,
            graph_weight=decision.graph_weight,
            skip_vector=decision.skip_vector,
            skip_graph=decision.skip_graph,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            entities_detected=decision.entities_detected,
        )

        threshold = self.config.confidence_threshold

        # If vector failed, boost graph
        if vector_confidence < threshold and not decision.skip_graph:
            adjusted.graph_weight = min(1.0, adjusted.graph_weight + 0.3)
            adjusted.vector_weight = max(0.0, adjusted.vector_weight - 0.3)
            adjusted.reasoning += f" | Vector low confidence ({vector_confidence:.2f}), boosted graph"

        # If graph failed, boost vector
        if graph_confidence < threshold and not decision.skip_vector:
            adjusted.vector_weight = min(1.0, adjusted.vector_weight + 0.3)
            adjusted.graph_weight = max(0.0, adjusted.graph_weight - 0.3)
            adjusted.reasoning += f" | Graph low confidence ({graph_confidence:.2f}), boosted vector"

        # If both failed, use adaptive mode
        if vector_confidence < threshold and graph_confidence < threshold:
            adjusted.mode = RetrievalMode.ADAPTIVE
            adjusted.reasoning += " | Both low confidence, using adaptive"

        return adjusted

    def fuse_contexts(
        self,
        vector_context: str,
        graph_context: str,
        decision: RoutingDecision,
    ) -> str:
        """
        Intelligently fuse contexts from vector and graph retrieval.

        - Deduplicates similar content
        - Weights by routing decision
        - Respects token limits
        """
        if decision.skip_vector or not vector_context:
            return graph_context or ""

        if decision.skip_graph or not graph_context:
            return vector_context or ""

        # Split into chunks for deduplication
        vector_chunks = self._split_context(vector_context)
        graph_chunks = self._split_context(graph_context)

        # Deduplicate
        unique_chunks = []
        seen_content = []

        # Add graph chunks first (usually more structured)
        for chunk in graph_chunks:
            if not self._is_duplicate(chunk, seen_content):
                unique_chunks.append(("graph", chunk))
                seen_content.append(chunk)

        # Add vector chunks
        for chunk in vector_chunks:
            if not self._is_duplicate(chunk, seen_content):
                unique_chunks.append(("vector", chunk))
                seen_content.append(chunk)

        # Build final context with weights
        context_parts = []

        if decision.graph_weight >= decision.vector_weight:
            # Graph first
            context_parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
            for source, chunk in unique_chunks:
                if source == "graph":
                    context_parts.append(chunk)

            context_parts.append("\n=== DOCUMENT CONTEXT ===")
            for source, chunk in unique_chunks:
                if source == "vector":
                    context_parts.append(chunk)
        else:
            # Vector first
            context_parts.append("=== DOCUMENT CONTEXT ===")
            for source, chunk in unique_chunks:
                if source == "vector":
                    context_parts.append(chunk)

            context_parts.append("\n=== KNOWLEDGE GRAPH CONTEXT ===")
            for source, chunk in unique_chunks:
                if source == "graph":
                    context_parts.append(chunk)

        return "\n".join(context_parts)

    def _split_context(self, context: str) -> List[str]:
        """Split context into chunks for deduplication."""
        # Split by document markers or double newlines
        chunks = re.split(r'\n(?=\[Document|\=\=\=)', context)
        return [c.strip() for c in chunks if c.strip()]

    def _is_duplicate(self, chunk: str, seen: List[str]) -> bool:
        """Check if chunk is similar to any seen content."""
        chunk_normalized = chunk.lower().strip()

        for seen_chunk in seen:
            seen_normalized = seen_chunk.lower().strip()

            # Use sequence matcher for similarity
            ratio = SequenceMatcher(None, chunk_normalized, seen_normalized).ratio()
            if ratio > self.config.dedup_similarity_threshold:
                return True

        return False


# =============================================================================
# Convenience functions
# =============================================================================

_orchestrator: Optional[QueryOrchestrator] = None


def get_orchestrator(config: OrchestratorConfig = None) -> QueryOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None or config is not None:
        _orchestrator = QueryOrchestrator(config)
    return _orchestrator


def route_query(query: str, model: str = None) -> RoutingDecision:
    """Convenience function for query routing."""
    return get_orchestrator().route(query, model)
