"""
Hybrid Retriever module combining Vector RAG and GraphRAG.

Provides:
- Intelligent query orchestration with LLM classification
- BM25 + Dense hybrid search (#7)
- Cross-encoder reranking (#9)
- Lost-in-the-middle reordering (#14)
- Contextual compression (#13)
- Smart result fusion with deduplication
- Retrieval cache for identical queries
"""

import hashlib
import json
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama

from .graph_config import get_graph_config, GraphConfig
from .graph_retrieval import GraphRetriever, GraphRetrievalResult
from .knowledge_graph import KnowledgeGraph
from .query_orchestrator import (
    QueryOrchestrator,
    OrchestratorConfig,
    RoutingDecision,
    RetrievalMode,
    QueryType,
    route_query,
)
from .constants import (
    DEFAULT_LLM_MODEL,
    EMBED_MODEL,
    VECTOR_STORE_DIR,
    INDEX_DIR,
    TOP_K_RETRIEVED_CHILDREN,
    TOP_K_RERANKED_PARENTS,
    ENABLE_HYBRID_SEARCH,
    HYBRID_DENSE_WEIGHT,
    BM25_K1,
    BM25_B,
    BM25_INDEX_PATH,
    ENABLE_CROSS_ENCODER,
    CROSS_ENCODER_BATCH_SIZE,
    ENABLE_LOST_IN_MIDDLE_REORDER,
    ENABLE_CONTEXTUAL_COMPRESSION,
    COMPRESSION_MAX_TOKENS_PER_DOC,
    MAX_SOURCE_CHUNKS_PER_ENTITY,
    MAX_SOURCE_CHUNKS_FROM_GRAPH,
)
from .utils.rag_utils import (
    VectorIndex,
    KeywordIndex,
    Document,
    NodeWithScore,
    rerank_documents,
)
from .utils.advanced_rag import (
    BM25Index,
    hybrid_search_fusion,
    cross_encoder_rerank,
    reorder_lost_in_middle,
    compress_context,
    MetadataFilter,
    filter_by_metadata,
)
from .utils.logger import logger


# Re-export QueryType for backward compatibility
__all__ = ["QueryType", "HybridRetriever", "HybridRetrievalResult"]


# =============================================================================
# Retrieval Cache (for identical queries)
# =============================================================================

class RetrievalCache:
    """
    LRU cache for retrieval results.

    Avoids re-executing expensive retrieval for identical queries.
    Uses in-memory cache with TTL expiration.
    """

    def __init__(self, max_size: int = 50, ttl_seconds: int = 300):
        """
        Initialize retrieval cache.

        Args:
            max_size: Maximum number of cached results
            ttl_seconds: Time-to-live in seconds (default 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (result, timestamp)
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, top_k: int, use_reranking: bool) -> str:
        """Create cache key from query parameters."""
        key_data = f"{query}|{top_k}|{use_reranking}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, top_k: int, use_reranking: bool) -> Optional["HybridRetrievalResult"]:
        """Get cached result if valid."""
        key = self._make_key(query, top_k, use_reranking)

        if key in self._cache:
            result, timestamp = self._cache[key]
            elapsed = time.time() - timestamp

            if elapsed < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"Retrieval cache HIT (age={elapsed:.1f}s)")
                return result
            else:
                # Expired
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, query: str, top_k: int, use_reranking: bool, result: "HybridRetrievalResult") -> None:
        """Cache a retrieval result."""
        key = self._make_key(query, top_k, use_reranking)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }


# Global retrieval cache
_retrieval_cache = RetrievalCache()


@dataclass
class QueryAnalysis:
    """Result of query analysis (legacy, now uses RoutingDecision internally)."""
    query: str
    query_type: QueryType
    entities_mentioned: List[str] = field(default_factory=list)
    relationship_keywords: List[str] = field(default_factory=list)
    use_vector: bool = True
    use_graph: bool = True
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    confidence: float = 0.5

    @classmethod
    def from_routing_decision(cls, decision: RoutingDecision) -> "QueryAnalysis":
        """Convert RoutingDecision to QueryAnalysis for compatibility."""
        return cls(
            query=decision.query,
            query_type=decision.query_type,
            entities_mentioned=decision.entities_detected,
            relationship_keywords=[],
            use_vector=not decision.skip_vector,
            use_graph=not decision.skip_graph,
            vector_weight=decision.vector_weight,
            graph_weight=decision.graph_weight,
            confidence=decision.confidence,
        )


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

    Now uses the intelligent QueryOrchestrator for classification.
    """
    # Use the new orchestrator (now uses unified LLM client)
    decision = route_query(query)

    # Convert to QueryAnalysis for backward compatibility
    return QueryAnalysis.from_routing_decision(decision)


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
        model = DEFAULT_LLM_MODEL

    context_parts = []
    source_chunks = []

    # Add graph context if available and weighted
    if graph_result and graph_result.context and analysis.graph_weight > 0:
        context_parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
        context_parts.append(graph_result.context)
        context_parts.append("")

        # Track source chunks from graph (with limits to prevent LLM timeout)
        for entity in graph_result.entities:
            # Cap per-entity chunks
            entity_chunks = entity.source_chunks[:MAX_SOURCE_CHUNKS_PER_ENTITY]
            source_chunks.extend(entity_chunks)
            # Stop if total limit reached
            if len(source_chunks) >= MAX_SOURCE_CHUNKS_FROM_GRAPH:
                source_chunks = source_chunks[:MAX_SOURCE_CHUNKS_FROM_GRAPH]
                break

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

    Advanced features:
    - Intelligent query orchestration with LLM classification
    - BM25 + Dense hybrid search (#7)
    - Cross-encoder reranking (#9)
    - Lost-in-the-middle reordering (#14)
    - Contextual compression (#13)
    - Smart skip logic and confidence-based fallback

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
        bm25_index_path: str = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
    ):
        self.config = config or get_graph_config()
        self.vector_store_path = vector_store_path or VECTOR_STORE_DIR
        self.index_path = index_path or INDEX_DIR
        self.graph_path = graph_path
        self.bm25_index_path = bm25_index_path or BM25_INDEX_PATH

        # Query orchestrator for intelligent routing
        self._orchestrator = QueryOrchestrator(orchestrator_config or OrchestratorConfig())

        # Components (lazy-loaded)
        self._vector_index: Optional[VectorIndex] = None
        self._keyword_index: Optional[KeywordIndex] = None
        self._bm25_index: Optional[BM25Index] = None  # #7: BM25 for hybrid search
        self._bm25_load_attempted: bool = False  # Track lazy loading attempts
        self._graph_retriever: Optional[GraphRetriever] = None
        self._graph_load_attempted: bool = False  # Track lazy loading attempts

    def load(self) -> Dict[str, bool]:
        """
        Load all retrieval components.

        Returns dict with load status for each component.
        """
        status = {
            "vector_index": False,
            "keyword_index": False,
            "bm25_index": False,
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

        # BM25 index is lazy-loaded only when hybrid search is actually used
        # This saves memory and startup time when only dense vector search is used
        self._bm25_index = None
        self._bm25_load_attempted = False
        status["bm25_index"] = "lazy"
        logger.info("BM25 index set to lazy loading (will load on first hybrid search)")

        # Graph retriever is lazy-loaded only when needed
        # This saves memory and startup time when only vector search is used
        self._graph_retriever = None
        self._graph_load_attempted = False
        status["graph"] = "lazy"
        logger.info("Knowledge graph set to lazy loading (will load on first graph query)")

        return status

    def is_loaded(self) -> bool:
        """Check if at least vector index is loaded."""
        return self._vector_index is not None

    def _ensure_graph_loaded(self) -> bool:
        """
        Lazy load the knowledge graph on first use.

        Returns:
            True if graph is loaded and available
        """
        if self._graph_retriever is not None:
            return self._graph_retriever.is_loaded()

        if self._graph_load_attempted:
            return False

        self._graph_load_attempted = True
        try:
            logger.info("Lazy loading knowledge graph...")
            self._graph_retriever = GraphRetriever(self.graph_path, self.config)
            if self._graph_retriever.load():
                logger.info("Knowledge graph loaded successfully")
                return True
            else:
                logger.warning("Knowledge graph is empty or not found")
                return False
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")
            return False

    def _ensure_bm25_loaded(self) -> bool:
        """
        Lazy load the BM25 index on first hybrid search.

        Returns:
            True if BM25 index is loaded and available
        """
        if self._bm25_index is not None:
            return True

        if self._bm25_load_attempted:
            return False

        if not ENABLE_HYBRID_SEARCH:
            return False

        self._bm25_load_attempted = True
        try:
            if Path(self.bm25_index_path).exists():
                logger.info("Lazy loading BM25 index...")
                self._bm25_index = BM25Index.load(self.bm25_index_path)
                logger.info("BM25 index loaded successfully")
                return True
            else:
                # Build BM25 index from keyword index documents
                if self._keyword_index:
                    logger.info("Building BM25 index from keyword documents...")
                    self._bm25_index = BM25Index(k1=BM25_K1, b=BM25_B)
                    docs = self._keyword_index.get_all_documents()
                    if docs:
                        bm25_docs = [{"text": d.text, "metadata": d.metadata} for d in docs]
                        self._bm25_index.add_documents(bm25_docs)
                        self._bm25_index.save(self.bm25_index_path)
                        logger.info(f"Built and saved BM25 index with {len(docs)} documents")
                        return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load/build BM25 index: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        model: str = None,
        use_hybrid_search: bool = None,
        use_cross_encoder: bool = None,
        use_lost_in_middle: bool = None,
        use_compression: bool = None,
        metadata_filters: List[MetadataFilter] = None,
        use_smart_routing: bool = True,
        pre_computed_routing: Optional["RoutingDecision"] = None,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval with advanced RAG features.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            use_reranking: Whether to use reranking (cross-encoder or LLM)
            model: LLM model to use
            use_hybrid_search: Use BM25+dense hybrid (default from config)
            use_cross_encoder: Use cross-encoder reranking (default from config)
            use_lost_in_middle: Apply lost-in-middle reordering (default from config)
            use_compression: Apply contextual compression (default from config)
            metadata_filters: List of MetadataFilter for filtering results (#16)
            use_smart_routing: Use intelligent orchestrator for routing decisions
            pre_computed_routing: Pre-computed routing decision (skips LLM classification)

        Returns:
            HybridRetrievalResult with combined context
        """
        # Auto-load if not already loaded
        if not self.is_loaded():
            self.load()

        if model is None:
            model = DEFAULT_LLM_MODEL
        if top_k is None:
            top_k = TOP_K_RETRIEVED_CHILDREN

        # Check retrieval cache for identical queries (skip if metadata filters or pre-computed routing)
        use_cache = metadata_filters is None and pre_computed_routing is None
        if use_cache:
            cached_result = _retrieval_cache.get(query, top_k, use_reranking)
            if cached_result is not None:
                cached_result.metadata["from_cache"] = True
                return cached_result

        # Use config defaults if not specified
        if use_hybrid_search is None:
            # Lazy load BM25 only when hybrid search is enabled
            if ENABLE_HYBRID_SEARCH:
                self._ensure_bm25_loaded()
            use_hybrid_search = ENABLE_HYBRID_SEARCH and self._bm25_index is not None
        elif use_hybrid_search:
            # Explicitly requested hybrid search - ensure BM25 is loaded
            self._ensure_bm25_loaded()
            if not self._bm25_index:
                use_hybrid_search = False
                logger.warning("Hybrid search requested but BM25 index unavailable")
        if use_cross_encoder is None:
            use_cross_encoder = ENABLE_CROSS_ENCODER
        if use_lost_in_middle is None:
            use_lost_in_middle = ENABLE_LOST_IN_MIDDLE_REORDER
        if use_compression is None:
            use_compression = ENABLE_CONTEXTUAL_COMPRESSION

        t_total_start = time.perf_counter()

        # Smart routing with orchestrator
        if pre_computed_routing is not None:
            # Use pre-computed routing (from parallel rewrite+classify)
            routing = pre_computed_routing
            analysis = QueryAnalysis.from_routing_decision(routing)
            logger.info(
                f"Using pre-computed routing: mode={routing.mode.value}, "
                f"skip_vector={routing.skip_vector}, skip_graph={routing.skip_graph}"
            )
        elif use_smart_routing:
            routing = self._orchestrator.route(query)
            analysis = QueryAnalysis.from_routing_decision(routing)

            # Log routing decision
            logger.info(
                f"Smart routing: mode={routing.mode.value}, "
                f"skip_vector={routing.skip_vector}, skip_graph={routing.skip_graph}, "
                f"reasoning='{routing.reasoning}'"
            )
        else:
            # Legacy: use old analyze_query
            analysis = analyze_query(query, self.config)
            routing = None

        t_routing_end = time.perf_counter()

        # Determine what to skip
        skip_vector = routing.skip_vector if routing else False
        skip_graph = routing.skip_graph if routing else False

        # =======================================================================
        # PARALLEL RETRIEVAL: Vector and Graph run concurrently
        # This significantly reduces latency by overlapping slow operations
        # =======================================================================
        vector_results = []
        graph_result = None
        vector_confidence = 0.0
        graph_confidence = 0.0

        def _do_vector_retrieval():
            """Vector retrieval task (Dense + BM25 + Parent + Rerank)."""
            nonlocal vector_results, vector_confidence
            if not (analysis.use_vector and self._vector_index and not skip_vector):
                return

            t_start = time.perf_counter()
            try:
                # Dense vector search
                dense_results = self._vector_index.search(query, top_k=top_k)
                logger.debug(f"Dense retrieval: {len(dense_results)} results")

                # #7: Hybrid search with BM25
                if use_hybrid_search and self._bm25_index:
                    sparse_results = self._bm25_index.search(query, top_k=top_k)
                    logger.debug(f"BM25 retrieval: {len(sparse_results)} results")

                    # Convert to common format for fusion
                    dense_list = [(nws.node, nws.score) for nws in dense_results]
                    sparse_list = [
                        (Document(text=doc["text"], metadata=doc.get("metadata", {})), score)
                        for doc, score in sparse_results
                    ]

                    # Fuse results using RRF
                    fused = hybrid_search_fusion(
                        dense_list,
                        sparse_list,
                        alpha=HYBRID_DENSE_WEIGHT,
                        top_k=top_k,
                    )

                    # Convert back to NodeWithScore
                    vector_results = [
                        NodeWithScore(node=doc if isinstance(doc, Document) else Document(text=str(doc)), score=score)
                        for doc, score in fused
                    ]
                    logger.debug(f"Hybrid fusion: {len(vector_results)} results (alpha={HYBRID_DENSE_WEIGHT})")
                else:
                    vector_results = dense_results

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

                    # #16: Metadata filtering
                    if metadata_filters and parent_results:
                        filtered_docs = filter_by_metadata(
                            [nws.node for nws in parent_results],
                            metadata_filters,
                            match_all=True,
                        )
                        if filtered_docs:
                            parent_results = [
                                NodeWithScore(node=doc, score=0.5)
                                for doc in filtered_docs
                            ]
                            logger.debug(f"Metadata filtering: {len(parent_results)} results after filtering")
                        else:
                            logger.warning("Metadata filtering returned no results, using unfiltered")

                    # Reranking: #9 Cross-encoder or LLM-based
                    if use_reranking and parent_results:
                        if use_cross_encoder:
                            # #9: Cross-encoder reranking
                            reranked = cross_encoder_rerank(
                                query=query,
                                documents=[nws.node for nws in parent_results],
                                model=model,
                                top_n=TOP_K_RERANKED_PARENTS,
                                batch_size=CROSS_ENCODER_BATCH_SIZE,
                            )
                            parent_results = [
                                NodeWithScore(node=doc, score=score)
                                for doc, score in reranked
                            ]
                            logger.debug(f"Cross-encoder reranking: {len(parent_results)} results")
                        else:
                            # Fallback to LLM reranking (uses unified LLM client)
                            parent_results = rerank_documents(
                                parent_results,
                                query,
                                top_n=TOP_K_RERANKED_PARENTS,
                            )
                            logger.debug(f"LLM reranking: {len(parent_results)} results")

                    # #14: Lost-in-the-middle reordering
                    if use_lost_in_middle and len(parent_results) > 2:
                        docs_ordered = reorder_lost_in_middle([nws.node for nws in parent_results])
                        parent_results = [
                            NodeWithScore(node=doc, score=parent_results[i].score if i < len(parent_results) else 0)
                            for i, doc in enumerate(docs_ordered)
                        ]
                        logger.debug("Applied lost-in-the-middle reordering")

                    # Use parent results as main vector results
                    if parent_results:
                        vector_results = parent_results

                # Calculate confidence
                if vector_results:
                    avg_score = sum(nws.score for nws in vector_results) / len(vector_results)
                    vector_confidence = min(1.0, avg_score)

                t_elapsed = time.perf_counter() - t_start
                logger.debug(f"Vector retrieval: {len(vector_results)} results in {t_elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Vector retrieval failed: {e}")

        def _do_graph_retrieval():
            """Graph retrieval task (Entity matching + Community summaries)."""
            nonlocal graph_result, graph_confidence
            if skip_graph:
                logger.info("Graph retrieval skipped by smart routing")
                return
            if not analysis.use_graph:
                return

            # Lazy load graph on first use
            if not self._ensure_graph_loaded():
                logger.debug("Graph not available, skipping graph retrieval")
                return

            t_start = time.perf_counter()
            try:
                graph_result = self._graph_retriever.retrieve(query)
                graph_confidence = graph_result.confidence
                t_elapsed = time.perf_counter() - t_start
                logger.debug(
                    f"Graph retrieval: {len(graph_result.entities)} entities, "
                    f"confidence={graph_result.confidence:.2f}, time={t_elapsed:.2f}s"
                )
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")
                graph_confidence = 0.0

        # Run both retrievals in parallel if both are needed
        need_vector = analysis.use_vector and self._vector_index and not skip_vector
        need_graph = analysis.use_graph and not skip_graph

        if need_vector and need_graph:
            # PARALLEL execution - significant latency reduction
            logger.debug("Running vector and graph retrieval in parallel")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_do_vector_retrieval),
                    executor.submit(_do_graph_retrieval),
                ]
                # Wait for both to complete
                for future in as_completed(futures):
                    try:
                        future.result()  # Raises exception if task failed
                    except Exception as e:
                        logger.error(f"Parallel retrieval task failed: {e}")
        else:
            # Sequential execution - only one retrieval needed
            _do_vector_retrieval()
            _do_graph_retrieval()

        # Confidence-based fallback adjustment
        if routing and (vector_confidence > 0 or graph_confidence > 0):
            adjusted_routing = self._orchestrator.should_fallback(
                routing, vector_confidence, graph_confidence
            )
            if adjusted_routing.reasoning != routing.reasoning:
                logger.info(f"Fallback adjustment: {adjusted_routing.reasoning}")
                analysis.vector_weight = adjusted_routing.vector_weight
                analysis.graph_weight = adjusted_routing.graph_weight

        # Fuse results
        fused_context, source_chunks = fuse_results(
            query,
            vector_results,
            graph_result,
            analysis,
            model,
        )

        # #13: Contextual compression
        if use_compression and vector_results:
            try:
                compressed_texts = compress_context(
                    query=query,
                    documents=[nws.node for nws in vector_results],
                    model=model,
                    max_tokens_per_doc=COMPRESSION_MAX_TOKENS_PER_DOC,
                )
                # Update fused context with compressed content
                if compressed_texts:
                    compressed_context = "\n\n".join(compressed_texts)
                    fused_context = f"=== COMPRESSED DOCUMENT CONTEXT ===\n{compressed_context}"
                    if graph_result and graph_result.context:
                        fused_context = f"=== KNOWLEDGE GRAPH CONTEXT ===\n{graph_result.context}\n\n{fused_context}"
                    logger.debug(f"Applied contextual compression: {len(compressed_texts)} segments")
            except Exception as e:
                logger.warning(f"Contextual compression failed: {e}")

        t_total_end = time.perf_counter()
        logger.info(f"Hybrid retrieval completed in {t_total_end - t_total_start:.2f}s")

        result = HybridRetrievalResult(
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
                "hybrid_search": use_hybrid_search,
                "cross_encoder": use_cross_encoder,
                "lost_in_middle": use_lost_in_middle,
                "compression": use_compression,
                # Smart routing info
                "smart_routing": use_smart_routing,
                "routing_mode": routing.mode.value if routing else "legacy",
                "vector_skipped": skip_vector,
                "graph_skipped": skip_graph,
                "vector_confidence": vector_confidence,
                "graph_confidence": graph_confidence,
                "vector_weight": analysis.vector_weight,
                "graph_weight": analysis.graph_weight,
                "from_cache": False,
            },
        )

        # Cache result for future identical queries
        if use_cache:
            _retrieval_cache.put(query, top_k, use_reranking, result)

        return result

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


def get_retrieval_cache_stats() -> Dict[str, Any]:
    """Get retrieval cache statistics."""
    return _retrieval_cache.stats()


def clear_retrieval_cache() -> None:
    """Clear the retrieval cache."""
    _retrieval_cache.clear()
    logger.info("Retrieval cache cleared")
