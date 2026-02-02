"""
Advanced RAG utilities for improved retrieval quality.

Implements:
- BM25 sparse retrieval for hybrid search
- Cross-encoder reranking with caching
- Lost-in-the-middle reordering
- Contextual compression
- Citation verification
"""

import hashlib
import re
import math
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


# =============================================================================
# BM25 Tokenization Cache (module-level for reuse across instances)
# =============================================================================


@lru_cache(maxsize=1000)
def _cached_tokenize(text: str) -> tuple:
    """
    Cached tokenization for BM25 queries.

    Returns tuple (immutable) for caching compatibility.
    Same query always produces same tokens - safe to cache.
    """
    text_lower = text.lower()
    tokens = tuple(re.findall(r"\b\w+\b", text_lower))
    return tokens


from .logger import logger
from .llm_client import llm_chat


# =============================================================================
# Reranking Cache
# =============================================================================


class RerankingCache:
    """
    LRU cache for reranking results.

    Caches results based on (query, docs_hash) to avoid
    expensive reranking for identical query+documents.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize reranking cache.

        Args:
            max_size: Maximum cached results
            ttl_seconds: Time-to-live (default 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (result, timestamp)
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, docs_texts: List[str]) -> str:
        """Create cache key from query and document texts."""
        # Hash documents to create a stable key
        docs_hash = hashlib.md5("||".join(docs_texts).encode()).hexdigest()[:16]
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        return f"{query_hash}:{docs_hash}"

    def get(self, query: str, docs_texts: List[str]) -> Optional[List[Tuple[int, float]]]:
        """
        Get cached reranking result.

        Args:
            query: Search query
            docs_texts: List of document texts (for hashing)

        Returns:
            List of (doc_index, score) tuples or None
        """
        key = self._make_key(query, docs_texts)

        if key in self._cache:
            result, timestamp = self._cache[key]
            elapsed = time.time() - timestamp

            if elapsed < self.ttl_seconds:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"Reranking cache HIT (age={elapsed:.1f}s)")
                return result
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, query: str, docs_texts: List[str], result: List[Tuple[int, float]]) -> None:
        """Cache reranking result."""
        key = self._make_key(query, docs_texts)

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


# Global reranking cache
_reranking_cache = RerankingCache()


def get_reranking_cache_stats() -> Dict[str, Any]:
    """Get reranking cache statistics."""
    return _reranking_cache.stats()


def clear_reranking_cache() -> None:
    """Clear the reranking cache."""
    _reranking_cache.clear()
    logger.info("Reranking cache cleared")


# =============================================================================
# BM25 Sparse Index for Hybrid Search (#7)
# =============================================================================


class BM25Index:
    """
    BM25 sparse retrieval index.

    Implements Okapi BM25 algorithm for keyword-based retrieval.
    Combined with dense vector search for hybrid retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0.75 is standard)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> num docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # doc_idx -> {term: count}
        self.vocab: set = set()
        self.N: int = 0  # Total number of documents

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the index.

        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        for doc in documents:
            text = doc.get("text", "")
            tokens = self._tokenize(text)

            self.documents.append(doc)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies for this document
            term_freq = Counter(tokens)
            self.term_freqs.append(dict(term_freq))

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                self.vocab.add(term)

        self.N = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        logger.info(f"BM25 index built with {self.N} documents, {len(self.vocab)} terms")

    def _idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Standard BM25 IDF formula
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given query tokens."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.term_freqs[doc_idx]

        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document, score) tuples sorted by score descending
        """
        # Use cached tokenization for queries (same query = same tokens)
        query_tokens = list(_cached_tokenize(query))

        scores = []
        for idx in range(self.N):
            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k documents with scores
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.documents[idx], score))

        return results

    def save(self, path: str):
        """Save index to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
            "vocab": list(self.vocab),
            "N": self.N,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load index from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        index = cls(k1=data["k1"], b=data["b"])
        index.documents = data["documents"]
        index.doc_lengths = data["doc_lengths"]
        index.avg_doc_length = data["avg_doc_length"]
        index.doc_freqs = data["doc_freqs"]
        index.term_freqs = data["term_freqs"]
        index.vocab = set(data["vocab"])
        index.N = data["N"]

        logger.info(f"BM25 index loaded from {path}: {index.N} documents")
        return index


# =============================================================================
# Hybrid Search: Combine BM25 + Dense (#7)
# =============================================================================


def hybrid_search_fusion(
    dense_results: List[Tuple[Any, float]],
    sparse_results: List[Tuple[Any, float]],
    alpha: float = 0.5,
    top_k: int = 10,
) -> List[Tuple[Any, float]]:
    """
    Fuse dense (vector) and sparse (BM25) search results using RRF.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings.

    Args:
        dense_results: List of (doc, score) from vector search
        sparse_results: List of (doc, score) from BM25
        alpha: Weight for dense results (1-alpha for sparse)
        top_k: Number of results to return

    Returns:
        Fused list of (doc, score) tuples
    """
    k = 60  # RRF constant

    # Build score maps using document text as key
    def get_doc_key(doc):
        if hasattr(doc, "text"):
            return doc.text[:200]  # Use first 200 chars as key
        elif isinstance(doc, dict):
            return doc.get("text", str(doc))[:200]
        return str(doc)[:200]

    scores = {}
    doc_map = {}

    # Score from dense results
    for rank, (doc, _) in enumerate(dense_results):
        key = get_doc_key(doc)
        rrf_score = alpha * (1 / (k + rank + 1))
        scores[key] = scores.get(key, 0) + rrf_score
        doc_map[key] = doc

    # Score from sparse results
    for rank, (doc, _) in enumerate(sparse_results):
        key = get_doc_key(doc)
        rrf_score = (1 - alpha) * (1 / (k + rank + 1))
        scores[key] = scores.get(key, 0) + rrf_score
        if key not in doc_map:
            doc_map[key] = doc

    # Sort by fused score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        results.append((doc_map[key], scores[key]))

    return results


# =============================================================================
# Cross-Encoder Reranking (#9) - Using Qwen3-Reranker via Ollama
# =============================================================================

# Ollama API URL for reranking
try:
    from ..constants import OLLAMA_URL

    OLLAMA_API_URL = f"{OLLAMA_URL}/api/generate"
except ImportError:
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Default reranker model (can be overridden via CROSS_ENCODER_MODEL env var)
try:
    from ..constants import CROSS_ENCODER_MODEL
except ImportError:
    CROSS_ENCODER_MODEL = "dengcao/Qwen3-Reranker-0.6B:F16"


def _score_single_document(
    query: str,
    doc_text: str,
    model: str = None,
) -> float:
    """
    Score a single query-document pair using Qwen3-Reranker.

    Note: This uses binary yes/no scoring as Qwen3-Reranker is trained for that format.
    For continuous scoring with fine-grained ranking, use LLM-based reranking
    (set ENABLE_CROSS_ENCODER=false to use rerank_documents instead).

    Returns:
        Score between 0.0 and 1.0 (1.0 = relevant, 0.0 = not relevant)
    """
    import requests

    if model is None:
        model = CROSS_ENCODER_MODEL

    # Qwen3-Reranker prompt format (from HuggingFace documentation)
    prompt = f"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query.
<Query>: {query}
<Document>: {doc_text}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {"num_predict": 3, "temperature": 0},
            },
            timeout=30,
        )
        result = response.json()
        answer = result.get("response", "").strip().lower()

        # Convert yes/no to score
        if "yes" in answer:
            return 1.0
        elif "no" in answer:
            return 0.0
        else:
            return 0.5  # Uncertain

    except Exception as e:
        logger.warning(f"Reranker scoring failed: {e}")
        return 0.5  # Default score on error


def cross_encoder_rerank(
    query: str,
    documents: List[Any],
    model: str = None,
    top_n: int = 5,
    batch_size: int = 10,  # Now means max parallel requests
    use_cache: bool = True,
) -> List[Tuple[Any, float]]:
    """
    Rerank documents using Qwen3-Reranker cross-encoder via Ollama.

    Uses parallel requests for fast reranking (~0.1s per document).
    Much faster than LLM-based reranking (~5s per batch).
    Results are cached to avoid redundant reranking.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Reranker model (default: CROSS_ENCODER_MODEL)
        top_n: Number of top documents to return
        batch_size: Max parallel requests (default: 10)
        use_cache: Whether to use reranking cache (default: True)

    Returns:
        List of (document, score) tuples sorted by relevance
    """
    if not documents:
        return []

    if model is None:
        model = CROSS_ENCODER_MODEL

    # Extract document texts for cache key
    docs_texts = []
    for doc in documents:
        text = doc.text if hasattr(doc, "text") else str(doc)
        docs_texts.append(text[:500])  # Use first 500 chars for hash

    # Check cache
    if use_cache:
        cached = _reranking_cache.get(query, docs_texts)
        if cached is not None:
            # Reconstruct result from cached (index, score) pairs
            result = [(documents[idx], score) for idx, score in cached if idx < len(documents)]
            return result[:top_n]

    t_start = time.perf_counter()

    def _score_doc(idx_doc: Tuple[int, Any]) -> Tuple[int, Any, float]:
        """Score a single document and return (index, doc, score)."""
        idx, doc = idx_doc
        text = doc.text if hasattr(doc, "text") else str(doc)
        # Truncate for efficiency (Qwen3-Reranker supports 32k but shorter is faster)
        text = text[:2000]
        score = _score_single_document(query, text, model)
        return (idx, doc, score)

    # Score documents in parallel
    # Adaptive batch size: use min(batch_size, num_docs, cpu_count * 2)
    # No need for more workers than documents or more than 2x CPU cores
    import os

    cpu_count = os.cpu_count() or 4
    effective_batch_size = min(batch_size, len(documents), cpu_count * 2)
    scored_docs = []
    doc_items = list(enumerate(documents))

    with ThreadPoolExecutor(max_workers=effective_batch_size) as executor:
        futures = {executor.submit(_score_doc, item): item[0] for item in doc_items}

        for future in as_completed(futures):
            try:
                idx, doc, score = future.result()
                scored_docs.append((idx, doc, score))
            except Exception as e:
                logger.error(f"Reranker task failed: {e}")
                # Add with default score
                idx = futures[future]
                scored_docs.append((idx, documents[idx], 0.5))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[2], reverse=True)

    # Cache result as (index, score) pairs
    if use_cache:
        cache_data = [(idx, score) for idx, doc, score in scored_docs]
        _reranking_cache.put(query, docs_texts, cache_data)

    t_elapsed = time.perf_counter() - t_start
    logger.debug(
        f"Cross-encoder rerank: {len(documents)} docs in {t_elapsed:.2f}s "
        f"(Qwen3-Reranker, {batch_size} parallel)"
    )

    # Return as (doc, score) tuples
    return [(doc, score) for idx, doc, score in scored_docs[:top_n]]


# =============================================================================
# Lost-in-the-Middle Reordering (#14)
# =============================================================================


def reorder_lost_in_middle(documents: List[Any]) -> List[Any]:
    """
    Reorder documents to mitigate the "lost in the middle" problem.

    LLMs tend to pay more attention to the beginning and end of context.
    This reordering places the most relevant documents at the start and end,
    with less relevant ones in the middle.

    Args:
        documents: List of documents (assumed ordered by relevance, best first)

    Returns:
        Reordered list with best docs at start and end
    """
    if len(documents) <= 2:
        return documents

    # Split into two halves
    reordered = []
    left = []
    right = []

    for i, doc in enumerate(documents):
        if i % 2 == 0:
            left.append(doc)
        else:
            right.append(doc)

    # Interleave: best at start, second-best at end, etc.
    # Pattern: 0, 2, 4, ... (middle) ..., 5, 3, 1
    reordered = left + list(reversed(right))

    return reordered


# =============================================================================
# Contextual Compression (#13)
# =============================================================================


def compress_context(
    query: str,
    documents: List[Any],
    model: str = None,  # Deprecated, uses configured LLM provider
    max_tokens_per_doc: int = 200,
    max_workers: int = 4,
    skip_threshold: int = None,
) -> List[str]:
    """
    Compress documents to extract only query-relevant information.

    Reduces token usage while maintaining relevant context.
    Uses parallel LLM calls for improved performance.

    Args:
        query: The user's query
        documents: List of documents to compress
        model: Deprecated, uses configured LLM provider (Gemini by default)
        max_tokens_per_doc: Target max tokens per compressed doc
        max_workers: Maximum parallel LLM calls (default: 4)
        skip_threshold: Skip compression for docs under this word count (default from config)

    Returns:
        List of compressed document texts
    """
    if not documents:
        return []

    # Import skip threshold from config if not specified
    if skip_threshold is None:
        from ..constants import COMPRESSION_SKIP_THRESHOLD

        skip_threshold = COMPRESSION_SKIP_THRESHOLD

    t_start = time.perf_counter()

    def _compress_single(doc_idx: int, doc: Any) -> Tuple[int, Optional[str]]:
        """Compress a single document. Returns (index, compressed_text or None)."""
        text = doc.text if hasattr(doc, "text") else str(doc)

        # Skip if already short enough (avoids expensive LLM calls for small docs)
        # Estimate tokens as len(text) / 4 (approx 4 chars per token)
        estimated_tokens = len(text) // 4
        if estimated_tokens < skip_threshold:
            return (doc_idx, text)

        prompt = f"""Extract ONLY the parts of this document that are relevant to the query.
Keep it concise (under {max_tokens_per_doc} words). If nothing is relevant, respond with "NOT_RELEVANT".

Query: {query}

Document:
{text[:2000]}

Relevant extract:"""

        try:
            result = llm_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            ).strip()

            if result and result != "NOT_RELEVANT":
                return (doc_idx, result)
            return (doc_idx, None)  # Not relevant

        except Exception as e:
            logger.warning(f"Compression failed for document {doc_idx}: {e}")
            # Fallback: truncate
            words = text.split()[:max_tokens_per_doc]
            return (doc_idx, " ".join(words))

    # Run compression in parallel
    results: Dict[int, Optional[str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compress_single, idx, doc): idx for idx, doc in enumerate(documents)
        }

        for future in as_completed(futures):
            try:
                doc_idx, compressed_text = future.result()
                results[doc_idx] = compressed_text
            except Exception as e:
                logger.error(f"Compression task failed: {e}")

    # Rebuild list in original order, filtering out None (not relevant)
    compressed = [results[idx] for idx in sorted(results.keys()) if results[idx] is not None]

    t_elapsed = time.perf_counter() - t_start
    logger.debug(
        f"Contextual compression: {len(documents)} docs -> {len(compressed)} relevant "
        f"in {t_elapsed:.2f}s (parallel, {max_workers} workers)"
    )

    return compressed


# =============================================================================
# Citation Verification (#15)
# =============================================================================


@dataclass
class CitationResult:
    """Result of citation verification."""

    claim: str
    supported: bool
    supporting_doc_idx: Optional[int] = None
    confidence: float = 0.0
    explanation: str = ""


def verify_citations(
    answer: str,
    documents: List[Any],
    model: str = None,  # Deprecated, uses configured LLM provider
) -> Tuple[bool, List[CitationResult], str]:
    """
    Verify that claims in the answer are supported by the source documents.

    Helps reduce hallucinations by checking factual grounding.

    Args:
        answer: The generated answer to verify
        documents: Source documents used for generation
        model: Deprecated, uses configured LLM provider (Gemini by default)

    Returns:
        Tuple of (all_supported, claim_results, verification_summary)
    """
    if not documents or not answer:
        return True, [], "No verification needed"

    # Build context from documents
    context = ""
    for i, doc in enumerate(documents):
        text = doc.text if hasattr(doc, "text") else str(doc)
        context += f"\n[Doc {i+1}]: {text[:500]}\n"

    prompt = f"""Verify if the claims in the ANSWER are supported by the SOURCE DOCUMENTS.

SOURCE DOCUMENTS:
{context}

ANSWER TO VERIFY:
{answer}

For each factual claim in the answer, determine if it's supported by the documents.
Output JSON format:
{{
    "claims": [
        {{"claim": "...", "supported": true/false, "doc_idx": 1, "confidence": 0.9}},
        ...
    ],
    "overall_supported": true/false,
    "summary": "..."
}}

Verification:"""

    try:
        response_text = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).strip()

        # Try to parse JSON
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            result = json.loads(json_match.group())

            claims = []
            for c in result.get("claims", []):
                claims.append(
                    CitationResult(
                        claim=c.get("claim", ""),
                        supported=c.get("supported", False),
                        supporting_doc_idx=c.get("doc_idx"),
                        confidence=c.get("confidence", 0.0),
                    )
                )

            return (
                result.get("overall_supported", True),
                claims,
                result.get("summary", "Verification complete"),
            )

    except Exception as e:
        logger.warning(f"Citation verification failed: {e}")

    return True, [], "Verification could not be completed"


# =============================================================================
# Metadata Filtering (#16)
# =============================================================================


@dataclass
class MetadataFilter:
    """Filter for metadata-based document filtering."""

    field: str
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, contains, in


def filter_by_metadata(
    documents: List[Any],
    filters: List[MetadataFilter],
    match_all: bool = True,
) -> List[Any]:
    """
    Filter documents by metadata criteria.

    Args:
        documents: List of documents with metadata
        filters: List of MetadataFilter conditions
        match_all: If True, all filters must match; if False, any filter

    Returns:
        Filtered list of documents
    """
    if not filters:
        return documents

    def matches_filter(doc: Any, f: MetadataFilter) -> bool:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})

        value = metadata.get(f.field)
        if value is None:
            return False

        if f.operator == "eq":
            return str(value) == str(f.value)
        elif f.operator == "ne":
            return str(value) != str(f.value)
        elif f.operator == "contains":
            return str(f.value).lower() in str(value).lower()
        elif f.operator == "in":
            return value in f.value if isinstance(f.value, list) else value == f.value
        elif f.operator == "gt":
            return float(value) > float(f.value)
        elif f.operator == "lt":
            return float(value) < float(f.value)
        elif f.operator == "gte":
            return float(value) >= float(f.value)
        elif f.operator == "lte":
            return float(value) <= float(f.value)

        return False

    filtered = []
    for doc in documents:
        if match_all:
            if all(matches_filter(doc, f) for f in filters):
                filtered.append(doc)
        else:
            if any(matches_filter(doc, f) for f in filters):
                filtered.append(doc)

    return filtered


# =============================================================================
# Multi-Index Manager (#17)
# =============================================================================


@dataclass
class ContentTypeIndex:
    """Configuration for a content-type specific index."""

    content_type: str  # "text", "table", "image"
    index_name: str
    weight: float = 1.0  # Weight in combined search


class MultiIndexManager:
    """
    Manages multiple indexes for different content types.

    Provides specialized retrieval for text, tables, and images
    with weighted combination.
    """

    def __init__(self):
        self.indexes: Dict[str, Any] = {}
        self.bm25_indexes: Dict[str, BM25Index] = {}
        self.weights: Dict[str, float] = {}

    def register_index(
        self,
        content_type: str,
        vector_index: Any,
        bm25_index: Optional[BM25Index] = None,
        weight: float = 1.0,
    ):
        """Register an index for a content type."""
        self.indexes[content_type] = vector_index
        if bm25_index:
            self.bm25_indexes[content_type] = bm25_index
        self.weights[content_type] = weight
        logger.info(f"Registered index for content type: {content_type}")

    def search(
        self,
        query: str,
        content_types: Optional[List[str]] = None,
        top_k: int = 10,
        use_hybrid: bool = True,
    ) -> List[Tuple[Any, float, str]]:
        """
        Search across multiple indexes.

        Args:
            query: Search query
            content_types: Types to search (None = all)
            top_k: Results per index
            use_hybrid: Whether to use BM25+dense hybrid

        Returns:
            List of (document, score, content_type) tuples
        """
        if content_types is None:
            content_types = list(self.indexes.keys())

        all_results = []

        for ctype in content_types:
            if ctype not in self.indexes:
                continue

            vector_index = self.indexes[ctype]
            weight = self.weights.get(ctype, 1.0)

            # Dense search
            dense_results = vector_index.search(query, top_k=top_k)

            # Hybrid search if BM25 available
            if use_hybrid and ctype in self.bm25_indexes:
                bm25_index = self.bm25_indexes[ctype]
                sparse_results = bm25_index.search(query, top_k=top_k)

                # Convert to common format
                dense_list = [
                    (r.node if hasattr(r, "node") else r, r.score if hasattr(r, "score") else 0)
                    for r in dense_results
                ]
                sparse_list = sparse_results

                fused = hybrid_search_fusion(dense_list, sparse_list, alpha=0.6, top_k=top_k)

                for doc, score in fused:
                    all_results.append((doc, score * weight, ctype))
            else:
                for r in dense_results:
                    doc = r.node if hasattr(r, "node") else r
                    score = r.score if hasattr(r, "score") else 0
                    all_results.append((doc, score * weight, ctype))

        # Sort by weighted score
        all_results.sort(key=lambda x: x[1], reverse=True)

        return all_results[:top_k]


# =============================================================================
# Overlapping Chunks Utility (#2)
# =============================================================================


def create_overlapping_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Create overlapping text chunks.

    Args:
        text: Text to chunk
        chunk_size: Target size in words
        overlap: Number of words to overlap between chunks

    Returns:
        List of overlapping chunks
    """
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks
