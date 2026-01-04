"""
Hybrid Retriever module combining Vector RAG and GraphRAG.

Provides:
- Query analysis and routing
- Result fusion from multiple sources
- LLM-based reranking of combined results
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ollama

from .graph_config import get_graph_config, GraphConfig
from .graph_retrieval import GraphRetriever, GraphRetrievalResult
from .knowledge_graph import KnowledgeGraph
from .constants import (
    LLM,
    EMBED_MODEL,
    VECTOR_STORE_DIR,
    INDEX_DIR,
    TOP_K_RETRIEVED_CHILDREN,
    TOP_K_RERANKED_PARENTS,
)
from .utils.rag_utils import (
    VectorIndex,
    KeywordIndex,
    Document,
    NodeWithScore,
    rerank_documents,
)
from .utils.logger import logger


class QueryType(Enum):
    """Types of queries for routing."""
    FACTUAL = "factual"           # Simple fact lookup
    RELATIONAL = "relational"      # About relationships
    COMPARATIVE = "comparative"    # Comparing entities
    EXPLORATORY = "exploratory"    # Broad/global questions
    PROCEDURAL = "procedural"      # How-to questions
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query: str
    query_type: QueryType
    entities_mentioned: List[str] = field(default_factory=list)
    relationship_keywords: List[str] = field(default_factory=list)
    use_vector: bool = True
    use_graph: bool = True
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    confidence: float = 0.5


@dataclass
class HybridRetrievalResult:
    """Combined result from hybrid retrieval."""
    query: str
    query_analysis: QueryAnalysis
    vector_results: List[NodeWithScore] = field(default_factory=list)
    graph_results: Optional[GraphRetrievalResult] = None
    fused_context: str = ""
    source_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def analyze_query(
    query: str,
    config: Optional[GraphConfig] = None,
    model: str = None,
) -> QueryAnalysis:
    """
    Analyze a query to determine its type and routing strategy.

    Uses pattern matching and optionally LLM classification.
    """
    if config is None:
        config = get_graph_config()

    analysis = QueryAnalysis(
        query=query,
        query_type=QueryType.UNKNOWN,
        use_vector=True,
        use_graph=True,
    )

    query_lower = query.lower()

    # Check for relationship patterns
    relationship_patterns = config.routing.graph_query_patterns or [
        "relationship",
        "related to",
        "connected",
        "depends on",
        "affects",
        "impacts",
        "between .* and",
    ]

    for pattern in relationship_patterns:
        if re.search(pattern, query_lower):
            analysis.query_type = QueryType.RELATIONAL
            analysis.relationship_keywords.append(pattern)
            analysis.graph_weight = 0.7
            analysis.vector_weight = 0.3
            break

    # Check for comparative patterns
    comparative_patterns = [
        r"compare",
        r"difference between",
        r"vs\.?",
        r"versus",
        r"better than",
        r"similar to",
    ]

    for pattern in comparative_patterns:
        if re.search(pattern, query_lower):
            analysis.query_type = QueryType.COMPARATIVE
            analysis.graph_weight = 0.6
            analysis.vector_weight = 0.4
            break

    # Check for exploratory/global patterns
    exploratory_patterns = [
        r"what are all",
        r"list all",
        r"summarize",
        r"overview",
        r"main (?:topics|themes|concepts)",
        r"how many",
    ]

    for pattern in exploratory_patterns:
        if re.search(pattern, query_lower):
            analysis.query_type = QueryType.EXPLORATORY
            analysis.graph_weight = 0.8
            analysis.vector_weight = 0.2
            break

    # Check for procedural patterns
    procedural_patterns = [
        r"how (?:do|does|to|can)",
        r"steps to",
        r"process (?:for|of)",
        r"procedure",
        r"guide",
    ]

    for pattern in procedural_patterns:
        if re.search(pattern, query_lower):
            analysis.query_type = QueryType.PROCEDURAL
            analysis.graph_weight = 0.3
            analysis.vector_weight = 0.7
            break

    # Default to factual if no patterns matched
    if analysis.query_type == QueryType.UNKNOWN:
        # Check for question words
        if re.search(r"^(what|who|where|when|which)\b", query_lower):
            analysis.query_type = QueryType.FACTUAL
            analysis.graph_weight = 0.4
            analysis.vector_weight = 0.6
        else:
            analysis.query_type = QueryType.FACTUAL
            analysis.graph_weight = config.routing.graph_weight
            analysis.vector_weight = config.routing.vector_weight

    # Apply routing strategy from config
    if config.routing.strategy == "hybrid":
        analysis.use_vector = True
        analysis.use_graph = True
    elif config.routing.strategy == "classifier":
        # Use graph only for relational/exploratory queries
        analysis.use_graph = analysis.query_type in [
            QueryType.RELATIONAL,
            QueryType.COMPARATIVE,
            QueryType.EXPLORATORY,
        ]

    analysis.confidence = 0.8 if analysis.query_type != QueryType.UNKNOWN else 0.5

    logger.debug(
        f"Query analysis: type={analysis.query_type.value}, "
        f"vector={analysis.use_vector}, graph={analysis.use_graph}"
    )

    return analysis


def fuse_results(
    query: str,
    vector_results: List[NodeWithScore],
    graph_result: GraphRetrievalResult,
    analysis: QueryAnalysis,
    model: str = None,
) -> Tuple[str, List[str]]:
    """
    Fuse results from vector and graph retrieval.

    Returns:
        Tuple of (fused_context, source_chunks)
    """
    if model is None:
        model = LLM

    context_parts = []
    source_chunks = []

    # Add graph context if available and weighted
    if graph_result and graph_result.context and analysis.graph_weight > 0:
        context_parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
        context_parts.append(graph_result.context)
        context_parts.append("")

        # Track source chunks from graph
        for entity in graph_result.entities:
            source_chunks.extend(entity.source_chunks)

    # Add vector results if available and weighted
    if vector_results and analysis.vector_weight > 0:
        context_parts.append("=== DOCUMENT CONTEXT ===")
        for i, nws in enumerate(vector_results, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(nws.node.text)
            context_parts.append("")

            # Track source chunk
            if "name" in nws.node.metadata:
                source_chunks.append(nws.node.metadata["name"])

    # Deduplicate source chunks
    source_chunks = list(dict.fromkeys(source_chunks))

    fused_context = "\n".join(context_parts)

    return fused_context, source_chunks


class HybridRetriever:
    """
    Hybrid retriever combining Vector RAG and GraphRAG.

    Usage:
        retriever = HybridRetriever()
        retriever.load()
        result = retriever.retrieve("What is the relationship between X and Y?")
    """

    def __init__(
        self,
        config: Optional[GraphConfig] = None,
        vector_store_path: str = None,
        index_path: str = None,
        graph_path: str = None,
    ):
        self.config = config or get_graph_config()
        self.vector_store_path = vector_store_path or VECTOR_STORE_DIR
        self.index_path = index_path or INDEX_DIR
        self.graph_path = graph_path

        # Components (lazy-loaded)
        self._vector_index: Optional[VectorIndex] = None
        self._keyword_index: Optional[KeywordIndex] = None
        self._graph_retriever: Optional[GraphRetriever] = None

    def load(self) -> Dict[str, bool]:
        """
        Load all retrieval components.

        Returns dict with load status for each component.
        """
        status = {
            "vector_index": False,
            "keyword_index": False,
            "graph": False,
        }

        # Load vector index
        try:
            child_index_path = f"{self.index_path}/child_documents"
            self._vector_index = VectorIndex.load(child_index_path, self.vector_store_path)
            status["vector_index"] = True
            logger.info("Vector index loaded")
        except Exception as e:
            logger.warning(f"Failed to load vector index: {e}")

        # Load keyword index
        try:
            parent_index_path = f"{self.index_path}/parent_documents"
            self._keyword_index = KeywordIndex.load(parent_index_path)
            status["keyword_index"] = True
            logger.info("Keyword index loaded")
        except Exception as e:
            logger.warning(f"Failed to load keyword index: {e}")

        # Load graph retriever
        try:
            self._graph_retriever = GraphRetriever(self.graph_path, self.config)
            if self._graph_retriever.load():
                status["graph"] = True
                logger.info("Knowledge graph loaded")
            else:
                logger.warning("Knowledge graph is empty or not found")
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")

        return status

    def is_loaded(self) -> bool:
        """Check if at least vector index is loaded."""
        return self._vector_index is not None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        model: str = None,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            use_reranking: Whether to use LLM reranking
            model: LLM model to use

        Returns:
            HybridRetrievalResult with combined context
        """
        if model is None:
            model = LLM
        if top_k is None:
            top_k = TOP_K_RETRIEVED_CHILDREN

        # Analyze query
        analysis = analyze_query(query, self.config)

        # Initialize results
        vector_results = []
        graph_result = None

        # Vector retrieval
        if analysis.use_vector and self._vector_index:
            try:
                vector_results = self._vector_index.search(query, top_k=top_k)
                logger.debug(f"Vector retrieval: {len(vector_results)} results")

                # Get parent documents
                if self._keyword_index and vector_results:
                    parent_results = []
                    seen_parents = set()

                    for nws in vector_results:
                        parent_name = nws.node.metadata.get("parent")
                        if parent_name and parent_name not in seen_parents:
                            parents = self._keyword_index.search_by_metadata("name", parent_name)
                            for parent in parents:
                                if parent.metadata.get("name") not in seen_parents:
                                    parent_results.append(NodeWithScore(node=parent, score=nws.score))
                                    seen_parents.add(parent.metadata.get("name"))

                    # Rerank if enabled
                    if use_reranking and parent_results:
                        parent_results = rerank_documents(
                            parent_results,
                            query,
                            model,
                            top_n=TOP_K_RERANKED_PARENTS,
                        )

                    # Use parent results as main vector results
                    if parent_results:
                        vector_results = parent_results

            except Exception as e:
                logger.error(f"Vector retrieval failed: {e}")

        # Graph retrieval
        if analysis.use_graph and self._graph_retriever and self._graph_retriever.is_loaded():
            try:
                graph_result = self._graph_retriever.retrieve(query, model)
                logger.debug(
                    f"Graph retrieval: {len(graph_result.entities)} entities, "
                    f"confidence={graph_result.confidence:.2f}"
                )
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")

        # Fuse results
        fused_context, source_chunks = fuse_results(
            query,
            vector_results,
            graph_result,
            analysis,
            model,
        )

        return HybridRetrievalResult(
            query=query,
            query_analysis=analysis,
            vector_results=vector_results,
            graph_results=graph_result,
            fused_context=fused_context,
            source_chunks=source_chunks,
            metadata={
                "vector_count": len(vector_results),
                "graph_entities": len(graph_result.entities) if graph_result else 0,
                "query_type": analysis.query_type.value,
            },
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all components."""
        stats = {
            "vector_index": "loaded" if self._vector_index else "not_loaded",
            "keyword_index": "loaded" if self._keyword_index else "not_loaded",
            "graph": self._graph_retriever.get_statistics() if self._graph_retriever else "not_loaded",
        }
        return stats


# Global retriever instance (lazy-loaded)
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get the global hybrid retriever (lazy-loaded singleton)."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
        _hybrid_retriever.load()
    return _hybrid_retriever


def hybrid_retrieve(
    query: str,
    top_k: int = None,
    use_reranking: bool = True,
    model: str = None,
) -> HybridRetrievalResult:
    """
    Convenience function for hybrid retrieval.

    Uses the global retriever instance.
    """
    retriever = get_hybrid_retriever()
    return retriever.retrieve(query, top_k, use_reranking, model)
