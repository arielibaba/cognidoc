"""
Query Complexity Evaluator for Agentic Routing.

Determines whether a query should use the fast path (standard RAG pipeline)
or the agent path (multi-step reasoning with tools).

Complexity signals:
- Query type (ANALYTICAL, COMPARATIVE trigger agent)
- Number of entities detected
- Number of sub-questions from rewriting
- Presence of complex reasoning keywords
- Low confidence from orchestrator
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .query_orchestrator import QueryType, RoutingDecision
from .utils.logger import logger


class ComplexityLevel(Enum):
    """Complexity levels for routing decisions."""
    SIMPLE = "simple"          # Fast path: standard RAG
    MODERATE = "moderate"      # Fast path with enhanced retrieval
    COMPLEX = "complex"        # Agent path: multi-step reasoning
    AMBIGUOUS = "ambiguous"    # Agent path: needs clarification


@dataclass
class ComplexityScore:
    """Result of complexity evaluation."""
    score: float  # 0.0 (simple) to 1.0 (complex)
    level: ComplexityLevel
    factors: dict = field(default_factory=dict)
    reasoning: str = ""

    @property
    def should_use_agent(self) -> bool:
        """Whether this query should use the agent path."""
        return self.level in (ComplexityLevel.COMPLEX, ComplexityLevel.AMBIGUOUS)


# =============================================================================
# Complexity Signals Configuration
# =============================================================================

# Query types that automatically trigger agent path
AGENT_QUERY_TYPES = {
    QueryType.ANALYTICAL,
    QueryType.COMPARATIVE,
}

# Query types that may trigger agent with other signals
MODERATE_QUERY_TYPES = {
    QueryType.RELATIONAL,
    QueryType.EXPLORATORY,
}

# Keywords indicating complex reasoning needs
COMPLEX_KEYWORDS = [
    # Causal reasoning
    r"\bpourquoi\b",
    r"\bwhy\b",
    r"\bcause[sd]?\b",
    r"\breason(?:s|ing)?\b",
    r"\bconséquence[s]?\b",
    r"\bconsequence[s]?\b",
    r"\bimpact[s]?\b",
    r"\beffet[s]?\b",
    r"\beffect[s]?\b",

    # Analytical
    r"\banalyse[rz]?\b",
    r"\banalyz[es]?\b",
    r"\bévalue[rz]?\b",
    r"\bevaluat[es]?\b",
    r"\bexplique[rz]?\b",
    r"\bexplain\b",
    r"\bjustifi[ez]?\b",
    r"\bjustify\b",

    # Comparative
    r"\bcompare[rz]?\b",
    r"\bcompare\b",
    r"\bdifférence[s]?\b",
    r"\bdifference[s]?\b",
    r"\bavantage[s]?\b",
    r"\badvantage[s]?\b",
    r"\binconvénient[s]?\b",
    r"\bdisadvantage[s]?\b",
    r"\bmeilleur[es]?\b",
    r"\bbetter\b",
    r"\bbest\b",

    # Multi-step
    r"\bétape[s]?\b",
    r"\bstep[s]?\b",
    r"\bprocessus\b",
    r"\bprocess\b",
    r"\bcomment .+ puis\b",
    r"\bfirst .+ then\b",

    # Synthesis
    r"\brésume[rz]?\b",
    r"\bsummariz[es]?\b",
    r"\bsynthèse\b",
    r"\bsynthesi[sz]e\b",
    r"\bvue d'ensemble\b",
    r"\boverview\b",
]

# Patterns indicating ambiguous/unclear queries
AMBIGUOUS_PATTERNS = [
    r"^.{1,15}$",  # Very short queries (< 15 chars)
    r"\?.*\?",     # Multiple question marks
    r"\bou\b.*\bou\b",  # Multiple "or" (ambiguous choice)
    r"\bor\b.*\bor\b",
]

# Database meta-questions that FORCE agent path (require database_stats tool)
# These cannot be answered by document retrieval alone
DATABASE_META_PATTERNS = [
    # French patterns - flexible matching
    r"\bcombien de doc",  # Matches "combien de documents", "combien de docs", typos
    r"\bcombien.{0,20}base\b",  # "combien...base" with up to 20 chars between
    r"\bnombre de doc",  # "nombre de documents/docs"
    r"\bbase.{0,15}comprend",  # "cette base comprend", "la base comprend-elle"
    r"\bbase.{0,15}contient",  # "la base contient"
    r"\btaille de (?:la |cette )?base\b",  # "taille de la base", "taille de cette base"
    r"\bstatistiques?\b.*\bbase\b",  # "statistiques de la base"
    r"\bcette base\b.*\bcombien\b",  # "cette base...combien"
    r"\bla base\b.*\bcombien\b",  # "la base...combien"

    # English patterns
    r"\bhow many doc",  # "how many documents/docs"
    r"\bdocument count\b",
    r"\bsize of (?:the )?(?:database|knowledge base)\b",
    r"\bstats?\b.*\b(?:database|knowledge base)\b",
    r"\btotal (?:de )?(?:documents?|fichiers?)\b",
    r"\btotal (?:documents?|files?)\b",
    r"\bnumber of doc",  # "number of documents"

    # Generic database meta patterns
    r"\b(?:database|base)\b.*\b(?:size|taille|count|nombre)\b",
    r"\b(?:count|nombre)\b.*\b(?:database|base)\b",
]

# Weights for complexity factors
COMPLEXITY_WEIGHTS = {
    "query_type": 0.25,
    "entity_count": 0.20,
    "subquestion_count": 0.20,
    "keyword_matches": 0.20,
    "low_confidence": 0.15,
}

# Thresholds
AGENT_THRESHOLD = 0.55  # Score above this → agent path
MODERATE_THRESHOLD = 0.35  # Score above this → enhanced retrieval
MIN_ENTITIES_FOR_COMPLEX = 3
MIN_SUBQUESTIONS_FOR_COMPLEX = 3
LOW_CONFIDENCE_THRESHOLD = 0.4


# =============================================================================
# Complexity Evaluation Functions
# =============================================================================

def count_complex_keywords(query: str) -> Tuple[int, List[str]]:
    """
    Count complex reasoning keywords in query.

    Returns:
        Tuple of (count, list of matched keywords)
    """
    query_lower = query.lower()
    matches = []

    for pattern in COMPLEX_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            matches.append(pattern)

    return len(matches), matches


def count_subquestions(rewritten_query: str) -> int:
    """
    Count sub-questions from rewritten query.

    The query rewriter outputs bullet points like:
    - Sub-question 1
    - Sub-question 2
    """
    if not rewritten_query:
        return 1

    # Count bullet points or numbered items
    bullets = re.findall(r"^[\s]*[-•*]\s+.+", rewritten_query, re.MULTILINE)
    numbers = re.findall(r"^[\s]*\d+[.)]\s+.+", rewritten_query, re.MULTILINE)

    count = len(bullets) + len(numbers)
    return max(count, 1)  # At least 1 question


def is_ambiguous(query: str) -> bool:
    """Check if query is ambiguous and needs clarification."""
    for pattern in AMBIGUOUS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def is_database_meta_question(query: str) -> bool:
    """Check if query is about the database itself (requires database_stats tool)."""
    for pattern in DATABASE_META_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def evaluate_complexity(
    query: str,
    routing: Optional[RoutingDecision] = None,
    rewritten_query: Optional[str] = None,
) -> ComplexityScore:
    """
    Evaluate query complexity to determine routing path.

    Args:
        query: Original user query
        routing: RoutingDecision from QueryOrchestrator
        rewritten_query: Rewritten query with sub-questions

    Returns:
        ComplexityScore with score, level, and reasoning
    """
    factors = {}
    reasoning_parts = []

    # 1. Query type factor
    query_type_score = 0.0
    if routing:
        if routing.query_type in AGENT_QUERY_TYPES:
            query_type_score = 1.0
            reasoning_parts.append(f"Query type {routing.query_type.value} requires analysis")
        elif routing.query_type in MODERATE_QUERY_TYPES:
            query_type_score = 0.5
            reasoning_parts.append(f"Query type {routing.query_type.value} may need multi-step")
        else:
            query_type_score = 0.0
    factors["query_type"] = query_type_score

    # 2. Entity count factor
    entity_count = len(routing.entities_detected) if routing else 0
    if entity_count >= MIN_ENTITIES_FOR_COMPLEX:
        entity_score = 1.0
        reasoning_parts.append(f"{entity_count} entities detected (multi-entity query)")
    elif entity_count >= 2:
        entity_score = 0.5
        reasoning_parts.append(f"{entity_count} entities detected")
    else:
        entity_score = 0.0
    factors["entity_count"] = entity_score

    # 3. Sub-question count factor
    subq_count = count_subquestions(rewritten_query)
    if subq_count >= MIN_SUBQUESTIONS_FOR_COMPLEX:
        subq_score = 1.0
        reasoning_parts.append(f"{subq_count} sub-questions identified")
    elif subq_count >= 2:
        subq_score = 0.5
        reasoning_parts.append(f"{subq_count} sub-questions identified")
    else:
        subq_score = 0.0
    factors["subquestion_count"] = subq_score

    # 4. Complex keywords factor
    keyword_count, matched_keywords = count_complex_keywords(query)
    if keyword_count >= 3:
        keyword_score = 1.0
        reasoning_parts.append(f"Multiple complex keywords: {keyword_count} matches")
    elif keyword_count >= 1:
        keyword_score = 0.5
        reasoning_parts.append(f"Complex keyword detected")
    else:
        keyword_score = 0.0
    factors["keyword_matches"] = keyword_score

    # 5. Low confidence factor
    confidence = routing.confidence if routing else 0.5
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        confidence_score = 1.0
        reasoning_parts.append(f"Low routing confidence ({confidence:.2f})")
    elif confidence < 0.6:
        confidence_score = 0.5
    else:
        confidence_score = 0.0
    factors["low_confidence"] = confidence_score

    # Calculate weighted score
    total_score = sum(
        factors[k] * COMPLEXITY_WEIGHTS[k]
        for k in COMPLEXITY_WEIGHTS
    )

    # Check for database meta-questions (force agent path)
    if is_database_meta_question(query):
        level = ComplexityLevel.COMPLEX
        reasoning_parts.append("Database meta-question requires database_stats tool")
        total_score = max(total_score, AGENT_THRESHOLD + 0.2)
    # Check for ambiguity (overrides other factors)
    elif is_ambiguous(query):
        level = ComplexityLevel.AMBIGUOUS
        reasoning_parts.append("Query appears ambiguous, clarification needed")
        total_score = max(total_score, AGENT_THRESHOLD + 0.1)
    # Determine level
    elif total_score >= AGENT_THRESHOLD:
        level = ComplexityLevel.COMPLEX
    elif total_score >= MODERATE_THRESHOLD:
        level = ComplexityLevel.MODERATE
    else:
        level = ComplexityLevel.SIMPLE

    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Simple factual query"

    logger.debug(
        f"Complexity evaluation: score={total_score:.2f}, level={level.value}, "
        f"factors={factors}"
    )

    return ComplexityScore(
        score=total_score,
        level=level,
        factors=factors,
        reasoning=reasoning,
    )


def should_use_agent(
    query: str,
    routing: Optional[RoutingDecision] = None,
    rewritten_query: Optional[str] = None,
    threshold: float = AGENT_THRESHOLD,
) -> Tuple[bool, ComplexityScore]:
    """
    Convenience function to check if agent path should be used.

    Args:
        query: Original user query
        routing: RoutingDecision from QueryOrchestrator
        rewritten_query: Rewritten query with sub-questions
        threshold: Custom threshold (default: AGENT_THRESHOLD)

    Returns:
        Tuple of (should_use_agent: bool, complexity_score: ComplexityScore)
    """
    complexity = evaluate_complexity(query, routing, rewritten_query)
    use_agent = complexity.score >= threshold

    if use_agent:
        logger.info(
            f"Agent path triggered: score={complexity.score:.2f}, "
            f"reason={complexity.reasoning}"
        )

    return use_agent, complexity


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ComplexityConfig:
    """Configuration for complexity evaluation."""
    agent_threshold: float = AGENT_THRESHOLD
    moderate_threshold: float = MODERATE_THRESHOLD
    min_entities_for_complex: int = MIN_ENTITIES_FOR_COMPLEX
    min_subquestions_for_complex: int = MIN_SUBQUESTIONS_FOR_COMPLEX
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    weights: dict = field(default_factory=lambda: COMPLEXITY_WEIGHTS.copy())


__all__ = [
    "ComplexityLevel",
    "ComplexityScore",
    "ComplexityConfig",
    "evaluate_complexity",
    "should_use_agent",
    "count_complex_keywords",
    "count_subquestions",
    "is_ambiguous",
    "is_database_meta_question",
    "AGENT_THRESHOLD",
]
