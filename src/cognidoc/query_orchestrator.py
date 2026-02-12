"""
Intelligent Query Orchestrator for Hybrid RAG.

Provides smart routing between Vector RAG and GraphRAG based on:
- Query classification (rule-based or LLM-based)
- Confidence-based fallback
- Cost-aware skipping
- Intelligent result fusion
"""

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

from pathlib import Path

from .constants import PROJECT_DIR
from .utils.llm_client import llm_chat
from .utils.logger import logger


class QueryType(Enum):
    """Types of queries for routing."""

    FACTUAL = "factual"  # Simple fact lookup → favor vector
    RELATIONAL = "relational"  # About relationships → favor graph
    COMPARATIVE = "comparative"  # Comparing entities → favor graph
    EXPLORATORY = "exploratory"  # Broad/global questions → favor graph
    PROCEDURAL = "procedural"  # How-to questions → favor vector
    ANALYTICAL = "analytical"  # Deep analysis → use both
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
        # French
        r"relation entre",
        r"li[ée][es]? [àa]",
        r"rapport entre",
        r"lien entre",
        # Spanish
        r"relaci[oó]n entre",
        r"relacionado con",
        r"conexi[oó]n entre",
        # German
        r"beziehung zwischen",
        r"verbunden mit",
        r"zusammenhang zwischen",
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
        # French
        r"compar[eé]",
        r"diff[ée]rence entre",
        r"par rapport [àa]",
        # Spanish
        r"comparar",
        r"diferencia entre",
        r"mejor que",
        r"peor que",
        # German
        r"vergleich",
        r"unterschied zwischen",
        r"besser als",
        r"schlechter als",
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
        # French
        r"r[ée]sum[ée]",
        r"aper[çc]u",
        r"liste[rz]? (?:tous|toutes)",
        r"combien",
        r"vue d'ensemble",
        # Spanish
        r"resumen",
        r"listar todos",
        r"cu[aá]ntos",
        r"enumerar",
        # German
        r"zusammenfassung",
        r"[üu]berblick",
        r"alle auflisten",
        r"wie viele",
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
        # French
        r"comment (?:faire|proc[ée]der|r[ée]aliser)",
        r"[ée]tapes pour",
        r"proc[ée]dure",
        r"guide pour",
        # Spanish
        r"c[oó]mo (?:hacer|realizar|proceder)",
        r"pasos para",
        r"procedimiento",
        r"gu[ií]a para",
        # German
        r"wie (?:kann|soll|mach)",
        r"schritte (?:f[üu]r|zu)",
        r"anleitung",
        r"verfahren",
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
        # French
        r"analyser",
        r"analyse de",
        r"[ée]valuer",
        r"examiner",
        # Spanish
        r"analizar",
        r"an[aá]lisis de",
        r"evaluar",
        r"examinar",
        # German
        r"analysieren",
        r"analyse von",
        r"bewerten",
        r"untersuchen",
    ],
}

# Weight configurations per query type
WEIGHT_CONFIG = {
    QueryType.FACTUAL: {"vector": 0.7, "graph": 0.3, "mode": RetrievalMode.HYBRID},
    QueryType.RELATIONAL: {"vector": 0.2, "graph": 0.8, "mode": RetrievalMode.HYBRID},
    QueryType.COMPARATIVE: {"vector": 0.3, "graph": 0.7, "mode": RetrievalMode.HYBRID},
    QueryType.EXPLORATORY: {"vector": 0.1, "graph": 0.9, "mode": RetrievalMode.GRAPH_ONLY},
    QueryType.PROCEDURAL: {"vector": 0.8, "graph": 0.2, "mode": RetrievalMode.HYBRID},
    QueryType.ANALYTICAL: {"vector": 0.5, "graph": 0.5, "mode": RetrievalMode.HYBRID},
    QueryType.UNKNOWN: {"vector": 0.6, "graph": 0.4, "mode": RetrievalMode.HYBRID},
}

_weights_loaded = False
_weights_lock = threading.Lock()


def _load_custom_weights() -> None:
    """Load custom query weights from graph_schema.yaml if available (thread-safe)."""
    global _weights_loaded, WEIGHT_CONFIG
    if _weights_loaded:
        return
    with _weights_lock:
        if _weights_loaded:
            return
        _weights_loaded = True
    try:
        config_path = PROJECT_DIR / "config" / "graph_schema.yaml"
        if not config_path.exists():
            return
        import yaml

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        custom = data.get("query_weights", {})
        if not custom:
            return
        type_map = {t.value: t for t in QueryType}
        for key, weights in custom.items():
            qt = type_map.get(key)
            if qt and isinstance(weights, dict):
                cfg = WEIGHT_CONFIG.get(qt, {}).copy()
                if "vector" in weights:
                    cfg["vector"] = float(weights["vector"])
                if "graph" in weights:
                    cfg["graph"] = float(weights["graph"])
                WEIGHT_CONFIG[qt] = cfg
        logger.info(f"Loaded custom query weights for {len(custom)} query types")
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        logger.debug(f"Could not load custom query weights: {e}, using defaults")


@lru_cache(maxsize=256)
def classify_query_rules(query: str) -> Tuple[QueryType, float, str]:
    """
    Classify query using rule-based pattern matching.

    Results are cached via LRU (deterministic, no TTL needed).

    Returns:
        Tuple of (query_type, confidence, reasoning)
    """
    query_lower = query.lower().strip()

    # Check each pattern type
    for query_type, patterns in QUERY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return (query_type, 0.8, f"Matched pattern: '{pattern}'")

    # Default classification based on question words
    if re.match(r"^(what|who|where|when|which)\b", query_lower):
        return (QueryType.FACTUAL, 0.6, "Question word detected")

    if re.match(r"^(why|how)\b", query_lower):
        return (QueryType.ANALYTICAL, 0.5, "Analytical question word")

    # French question words
    if re.match(r"^(qu[e']|quel[le]?s?|o[uù]|quand|qui|combien)\b", query_lower):
        return (QueryType.FACTUAL, 0.6, "French question word detected")
    if re.match(r"^(pourquoi|comment)\b", query_lower):
        return (QueryType.ANALYTICAL, 0.5, "French analytical question word")

    # Spanish question words
    if re.match(r"^(qu[eé]|cu[aá]l|d[oó]nde|cu[aá]ndo|qui[eé]n|cu[aá]nto)\b", query_lower):
        return (QueryType.FACTUAL, 0.6, "Spanish question word detected")
    if re.match(r"^(por qu[eé]|c[oó]mo)\b", query_lower):
        return (QueryType.ANALYTICAL, 0.5, "Spanish analytical question word")

    # German question words
    if re.match(r"^(was|wer|wo|wann|welche[rs]?)\b", query_lower):
        return (QueryType.FACTUAL, 0.6, "German question word detected")
    if re.match(r"^(warum|wie)\b", query_lower):
        return (QueryType.ANALYTICAL, 0.5, "German analytical question word")

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


_llm_classify_cache: Dict[str, Tuple[Tuple[QueryType, float, str, List[str]], float]] = {}
_LLM_CLASSIFY_TTL = 300  # 5 minutes


def classify_query_llm(query: str) -> Tuple[QueryType, float, str, List[str]]:
    """
    Classify query using LLM with TTL cache.

    Uses the unified LLM client (Gemini by default).

    Args:
        query: The query to classify

    Returns:
        Tuple of (query_type, confidence, reasoning, entities)
    """
    cache_key = query.lower().strip()
    now = time.time()
    if cache_key in _llm_classify_cache:
        cached_result, cached_at = _llm_classify_cache[cache_key]
        if now - cached_at < _LLM_CLASSIFY_TTL:
            logger.debug(f"LLM classification cache hit for: {cache_key[:50]}")
            return cached_result

    try:
        result = llm_chat(
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(query=query)}],
            temperature=0.1,
        )

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

        result = (query_type, confidence, reasoning, entities)
        _llm_classify_cache[cache_key] = (result, time.time())
        return result

    except (TimeoutError, ConnectionError, ValueError, KeyError, OSError) as e:
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

    def route(self, query: str) -> RoutingDecision:
        """
        Analyze query and decide routing strategy.

        Uses the unified LLM client (Gemini by default) for classification.

        Args:
            query: User query

        Returns:
            RoutingDecision with routing strategy
        """
        _load_custom_weights()

        # Classify query
        if self.config.use_llm_classifier:
            query_type, confidence, reasoning, entities = classify_query_llm(query)
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
        if (
            self.config.prefer_vector_for_simple
            and query_type == QueryType.FACTUAL
            and confidence > 0.7
        ):
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

    def _confidence_shift(self, confidence: float) -> float:
        """Proportional shift: larger when confidence is further below threshold."""
        threshold = self.config.confidence_threshold
        if confidence >= threshold:
            return 0.0
        return 0.3 * (1.0 - confidence / threshold)

    def should_fallback(
        self,
        decision: RoutingDecision,
        vector_confidence: float,
        graph_confidence: float,
    ) -> RoutingDecision:
        """
        Adjust routing based on retrieval confidence.

        If one system returns low confidence, boost the other proportionally.
        The shift scales from 0 (at threshold) to 0.3 (at zero confidence).
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

        # If vector failed, boost graph proportionally
        if vector_confidence < threshold and not decision.skip_graph:
            shift = self._confidence_shift(vector_confidence)
            adjusted.graph_weight = min(1.0, adjusted.graph_weight + shift)
            adjusted.vector_weight = max(0.0, adjusted.vector_weight - shift)
            adjusted.reasoning += (
                f" | Vector low confidence ({vector_confidence:.2f}), boosted graph by {shift:.2f}"
            )

        # If graph failed, boost vector proportionally
        if graph_confidence < threshold and not decision.skip_vector:
            shift = self._confidence_shift(graph_confidence)
            adjusted.vector_weight = min(1.0, adjusted.vector_weight + shift)
            adjusted.graph_weight = max(0.0, adjusted.graph_weight - shift)
            adjusted.reasoning += (
                f" | Graph low confidence ({graph_confidence:.2f}), boosted vector by {shift:.2f}"
            )

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
        chunks = re.split(r"\n(?=\[Document|\=\=\=)", context)
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


def route_query(query: str) -> RoutingDecision:
    """Convenience function for query routing."""
    return get_orchestrator().route(query)
