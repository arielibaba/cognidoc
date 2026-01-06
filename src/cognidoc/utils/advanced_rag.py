"""
Advanced RAG utilities for improved retrieval quality.

Implements:
- BM25 sparse retrieval for hybrid search
- Cross-encoder reranking
- Lost-in-the-middle reordering
- Contextual compression
- Citation verification
"""

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

import ollama

from .logger import logger


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
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the index.

        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        for doc in documents:
            text = doc.get('text', '')
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
        query_tokens = self._tokenize(query)

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
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'doc_freqs': self.doc_freqs,
            'term_freqs': self.term_freqs,
            'vocab': list(self.vocab),
            'N': self.N,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BM25Index':
        """Load index from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        index = cls(k1=data['k1'], b=data['b'])
        index.documents = data['documents']
        index.doc_lengths = data['doc_lengths']
        index.avg_doc_length = data['avg_doc_length']
        index.doc_freqs = data['doc_freqs']
        index.term_freqs = data['term_freqs']
        index.vocab = set(data['vocab'])
        index.N = data['N']

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
        if hasattr(doc, 'text'):
            return doc.text[:200]  # Use first 200 chars as key
        elif isinstance(doc, dict):
            return doc.get('text', str(doc))[:200]
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
# Cross-Encoder Reranking (#9)
# =============================================================================

def cross_encoder_rerank(
    query: str,
    documents: List[Any],
    model: str = "granite3.3:8b",
    top_n: int = 5,
    batch_size: int = 5,
) -> List[Tuple[Any, float]]:
    """
    Rerank documents using a cross-encoder approach with LLM.

    More accurate than bi-encoder but slower. Uses LLM to score
    query-document pairs directly.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: LLM model for scoring
        top_n: Number of top documents to return
        batch_size: Documents to score per LLM call

    Returns:
        List of (document, score) tuples sorted by relevance
    """
    if not documents:
        return []

    client = ollama.Client()
    scored_docs = []

    # Score documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Build prompt for batch scoring
        docs_text = ""
        for j, doc in enumerate(batch):
            text = doc.text if hasattr(doc, 'text') else str(doc)
            text = text[:500]  # Truncate for efficiency
            docs_text += f"\n[{j+1}] {text}\n"

        prompt = f"""Score the relevance of each document to the query on a scale of 0-10.
Output ONLY a JSON array of scores, e.g., [8, 5, 9, 3, 7]

Query: {query}

Documents:{docs_text}

Scores (JSON array only):"""

        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )

            # Parse scores from response
            response_text = response['message']['content'].strip()
            # Extract JSON array
            match = re.search(r'\[[\d\s,\.]+\]', response_text)
            if match:
                scores = json.loads(match.group())
                for j, doc in enumerate(batch):
                    score = scores[j] if j < len(scores) else 0
                    scored_docs.append((doc, float(score)))
            else:
                # Fallback: assign decreasing scores
                for j, doc in enumerate(batch):
                    scored_docs.append((doc, 5.0 - j * 0.1))

        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            # Fallback scores
            for j, doc in enumerate(batch):
                scored_docs.append((doc, 5.0))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_n]


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
    model: str = "granite3.3:8b",
    max_tokens_per_doc: int = 200,
) -> List[str]:
    """
    Compress documents to extract only query-relevant information.

    Reduces token usage while maintaining relevant context.

    Args:
        query: The user's query
        documents: List of documents to compress
        model: LLM model for compression
        max_tokens_per_doc: Target max tokens per compressed doc

    Returns:
        List of compressed document texts
    """
    if not documents:
        return []

    client = ollama.Client()
    compressed = []

    for doc in documents:
        text = doc.text if hasattr(doc, 'text') else str(doc)

        # Skip if already short
        if len(text.split()) < max_tokens_per_doc:
            compressed.append(text)
            continue

        prompt = f"""Extract ONLY the parts of this document that are relevant to the query.
Keep it concise (under {max_tokens_per_doc} words). If nothing is relevant, respond with "NOT_RELEVANT".

Query: {query}

Document:
{text[:2000]}

Relevant extract:"""

        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": max_tokens_per_doc * 2},
            )

            result = response['message']['content'].strip()
            if result and result != "NOT_RELEVANT":
                compressed.append(result)
            # Skip if not relevant

        except Exception as e:
            logger.warning(f"Compression failed for document: {e}")
            # Fallback: truncate
            words = text.split()[:max_tokens_per_doc]
            compressed.append(" ".join(words))

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
    model: str = "granite3.3:8b",
) -> Tuple[bool, List[CitationResult], str]:
    """
    Verify that claims in the answer are supported by the source documents.

    Helps reduce hallucinations by checking factual grounding.

    Args:
        answer: The generated answer to verify
        documents: Source documents used for generation
        model: LLM model for verification

    Returns:
        Tuple of (all_supported, claim_results, verification_summary)
    """
    if not documents or not answer:
        return True, [], "No verification needed"

    client = ollama.Client()

    # Build context from documents
    context = ""
    for i, doc in enumerate(documents):
        text = doc.text if hasattr(doc, 'text') else str(doc)
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
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )

        response_text = response['message']['content'].strip()

        # Try to parse JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())

            claims = []
            for c in result.get('claims', []):
                claims.append(CitationResult(
                    claim=c.get('claim', ''),
                    supported=c.get('supported', False),
                    supporting_doc_idx=c.get('doc_idx'),
                    confidence=c.get('confidence', 0.0),
                ))

            return (
                result.get('overall_supported', True),
                claims,
                result.get('summary', 'Verification complete'),
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
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        if isinstance(doc, dict):
            metadata = doc.get('metadata', {})

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
                dense_list = [(r.node if hasattr(r, 'node') else r, r.score if hasattr(r, 'score') else 0) for r in dense_results]
                sparse_list = sparse_results

                fused = hybrid_search_fusion(dense_list, sparse_list, alpha=0.6, top_k=top_k)

                for doc, score in fused:
                    all_results.append((doc, score * weight, ctype))
            else:
                for r in dense_results:
                    doc = r.node if hasattr(r, 'node') else r
                    score = r.score if hasattr(r, 'score') else 0
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
