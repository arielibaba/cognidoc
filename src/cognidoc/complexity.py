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
from typing import Dict, List, Optional, Tuple

from .query_orchestrator import QueryType, RoutingDecision
from .utils.logger import logger


class ComplexityLevel(Enum):
    """Complexity levels for routing decisions."""

    SIMPLE = "simple"  # Fast path: standard RAG
    MODERATE = "moderate"  # Fast path with enhanced retrieval
    COMPLEX = "complex"  # Agent path: multi-step reasoning
    AMBIGUOUS = "ambiguous"  # Agent path: needs clarification


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

# Semantic complexity categories — reference phrases for embedding-based matching
COMPLEXITY_CATEGORIES: Dict[str, List[str]] = {
    "causal": [
        "why does this happen, what is the cause, what are the consequences and effects",
        "pourquoi, quelle est la cause, quelles sont les conséquences et effets",
    ],
    "analytical": [
        "analyze and evaluate, explain and justify the reasoning",
        "analyser et évaluer, expliquer et justifier le raisonnement",
    ],
    "comparative": [
        "compare the advantages and disadvantages, what are the differences",
        "comparer les avantages et inconvénients, quelles sont les différences",
    ],
    "multi_step": [
        "what are the steps in the process, first do this then do that",
        "quelles sont les étapes du processus, d'abord faire ceci puis cela",
    ],
    "synthesis": [
        "summarize and give an overview, provide a synthesis",
        "résumer et donner une vue d'ensemble, fournir une synthèse",
    ],
}

SEMANTIC_SIMILARITY_THRESHOLD = 0.45

# Regex fallback keywords (used when embedding provider is unavailable)
_COMPLEX_KEYWORDS_REGEX = [
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
    r"\?.*\?",  # Multiple question marks
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
    # French patterns - listing documents
    r"\bliste[rz]?\b.*\bdoc",  # "liste les documents", "lister les docs"
    r"\bnoms?\b.*\bdoc",  # "noms des documents", "nom des docs"
    r"\bdoc.*\bnoms?\b",  # "documents et leurs noms"
    r"\bcite[rz]?\b.*\bdoc",  # "cite les documents", "citer les docs"
    r"\b[eé]num[eè]re[rz]?\b.*\bdoc",  # "énumère les documents"
    r"\bquels?\b.*\bdoc",  # "quels documents", "quels sont les docs"
    r"\bdonne.*\bnoms?\b",  # "donne-moi les noms", "donne leurs noms"
    # Spanish patterns - document count
    r"\bcu[aá]ntos? doc",  # "cuántos documentos", "cuantos docs"
    r"\bcu[aá]ntos?.{0,20}base\b",  # "cuántos hay en la base"
    r"\bn[uú]mero de doc",  # "número de documentos"
    r"\bbase.{0,15}contiene",  # "la base contiene"
    r"\bbase.{0,15}tiene",  # "la base tiene"
    r"\btama[ñn]o de (?:la |esta )?base\b",  # "tamaño de la base"
    r"\bestad[ií]sticas?\b.*\bbase\b",  # "estadísticas de la base"
    r"\besta base\b.*\bcu[aá]ntos?\b",  # "esta base...cuántos"
    r"\bla base\b.*\bcu[aá]ntos?\b",  # "la base...cuántos"
    # Spanish patterns - listing documents
    r"\blista[r]?\b.*\bdoc",  # "lista los documentos", "listar docs"
    r"\bnombres?\b.*\bdoc",  # "nombres de los documentos"
    r"\bdoc.*\bnombres?\b",  # "documentos y sus nombres"
    r"\bmenciona[r]?\b.*\bdoc",  # "menciona los documentos"
    r"\benumera[r]?\b.*\bdoc",  # "enumera los documentos"
    r"\bqu[eé] doc",  # "qué documentos hay"
    r"\bcu[aá]les?\b.*\bdoc",  # "cuáles documentos"
    r"\bdame.*\bnombres?\b",  # "dame los nombres"
    # German patterns - document count
    r"\bwie viele? doc",  # "wie viele Dokumente"
    r"\bwie viele? dok",  # "wie viele Dokumente" (German spelling)
    r"\bwie viele?.{0,20}(?:datenbank|basis)\b",  # "wie viele in der Datenbank"
    r"\banzahl (?:der |von )?doc",  # "Anzahl der Dokumente"
    r"\banzahl (?:der |von )?dok",  # "Anzahl der Dokumente"
    r"\b(?:datenbank|basis).{0,15}enth[aä]lt",  # "die Datenbank enthält"
    r"\b(?:datenbank|basis).{0,15}hat",  # "die Basis hat"
    r"\bgr[oö][sß]e der (?:datenbank|basis)\b",  # "Größe der Datenbank"
    r"\bstatistik(?:en)?\b.*\b(?:datenbank|basis)\b",  # "Statistiken der Datenbank"
    r"\bdiese (?:datenbank|basis)\b.*\bwie viele?\b",  # "diese Datenbank...wie viele"
    # German patterns - listing documents
    r"\bliste[n]?\b.*\bdok",  # "liste die Dokumente", "listen"
    r"\bnamen?\b.*\bdok",  # "Namen der Dokumente"
    r"\bdok.*\bnamen?\b",  # "Dokumente und ihre Namen"
    r"\bnenne[n]?\b.*\bdok",  # "nenne die Dokumente"
    r"\baufz[aä]hl(?:en|e)?\b.*\bdok",  # "aufzählen die Dokumente"
    r"\bwelche\b.*\bdok",  # "welche Dokumente"
    r"\bwas f[uü]r\b.*\bdok",  # "was für Dokumente"
    r"\bgib mir.*\bnamen?\b",  # "gib mir die Namen"
    r"\bzeig(?:e|en)?\b.*\bdok",  # "zeige die Dokumente"
    # English patterns
    r"\bhow many doc",  # "how many documents/docs"
    r"\bdocument count\b",
    r"\bsize of (?:the )?(?:database|knowledge base)\b",
    r"\bstats?\b.*\b(?:database|knowledge base)\b",
    r"\btotal (?:de )?(?:documents?|fichiers?)\b",
    r"\btotal (?:documents?|files?)\b",
    r"\bnumber of doc",  # "number of documents"
    # English patterns - listing documents
    r"\blist\b.*\bdoc",  # "list the documents", "list all docs"
    r"\bnames?\b.*\bdoc",  # "names of documents"
    r"\bdoc.*\bnames?\b",  # "document names"
    r"\bwhat\b.*\bdoc",  # "what documents are there"
    r"\bwhich\b.*\bdoc",  # "which documents"
    # Generic database meta patterns (all languages)
    r"\b(?:database|base|datenbank|basis)\b.*\b(?:size|taille|tama[ñn]o|gr[oö][sß]e|count|nombre|n[uú]mero|anzahl)\b",
    r"\b(?:count|nombre|n[uú]mero|anzahl)\b.*\b(?:database|base|datenbank|basis)\b",
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


_category_embeddings: Optional[Dict[str, List[List[float]]]] = None


def _get_category_embeddings() -> Dict[str, List[List[float]]]:
    """Lazily compute and cache category reference embeddings."""
    global _category_embeddings
    if _category_embeddings is not None:
        return _category_embeddings

    from .utils.rag_utils import get_embedding

    _category_embeddings = {}
    for category, phrases in COMPLEXITY_CATEGORIES.items():
        _category_embeddings[category] = [get_embedding(p) for p in phrases]
    return _category_embeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _count_complex_keywords_regex(query: str) -> Tuple[int, List[str]]:
    """Regex fallback for keyword counting (when embeddings unavailable)."""
    query_lower = query.lower()
    matches = []
    for pattern in _COMPLEX_KEYWORDS_REGEX:
        if re.search(pattern, query_lower, re.IGNORECASE):
            matches.append(pattern)
    return len(matches), matches


def count_complex_keywords(query: str) -> Tuple[int, List[str]]:
    """
    Count complexity categories matched via semantic similarity.

    Embeds the query and computes cosine similarity against pre-computed
    category reference phrases. Falls back to regex matching if the
    embedding provider is unavailable.

    Returns:
        Tuple of (count, list of matched category names)
    """
    try:
        from .utils.rag_utils import get_embedding

        query_emb = get_embedding(query)
        cat_embs = _get_category_embeddings()

        matched = []
        for category, ref_embeddings in cat_embs.items():
            max_sim = max(_cosine_similarity(query_emb, ref) for ref in ref_embeddings)
            if max_sim >= SEMANTIC_SIMILARITY_THRESHOLD:
                matched.append(category)

        return len(matched), matched
    except Exception:
        return _count_complex_keywords_regex(query)


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

    # 2. Entity count factor (continuous: 0 → 0.0, 2 → 0.5, 4+ → 1.0)
    entity_count = len(routing.entities_detected) if routing else 0
    entity_score = min(entity_count / 4.0, 1.0)
    if entity_count >= 1:
        reasoning_parts.append(f"{entity_count} entities detected (score={entity_score:.2f})")
    factors["entity_count"] = entity_score

    # 3. Sub-question count factor (continuous: 1 → 0.0, 2 → 0.33, 4+ → 1.0)
    subq_count = count_subquestions(rewritten_query or query)
    subq_score = min((subq_count - 1) / 3.0, 1.0)
    if subq_count >= 2:
        reasoning_parts.append(f"{subq_count} sub-questions identified (score={subq_score:.2f})")
    factors["subquestion_count"] = subq_score

    # 4. Complex keywords factor (continuous: 0 → 0.0, 1 → 0.33, 3+ → 1.0)
    keyword_count, matched_keywords = count_complex_keywords(query)
    keyword_score = min(keyword_count / 3.0, 1.0)
    if keyword_count >= 1:
        reasoning_parts.append(f"{keyword_count} complex keywords (score={keyword_score:.2f})")
    factors["keyword_matches"] = keyword_score

    # 5. Low confidence factor (continuous: 0.7+ → 0.0, 0.2 → 1.0)
    confidence = routing.confidence if routing else 0.5
    confidence_score = max(0.0, min(1.0, (0.7 - confidence) / 0.5))
    if confidence_score > 0.1:
        reasoning_parts.append(
            f"Low routing confidence ({confidence:.2f}, score={confidence_score:.2f})"
        )
    factors["low_confidence"] = confidence_score

    # Calculate weighted score
    total_score = sum(factors[k] * COMPLEXITY_WEIGHTS[k] for k in COMPLEXITY_WEIGHTS)

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
            f"Agent path triggered: score={complexity.score:.2f}, " f"reason={complexity.reasoning}"
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
