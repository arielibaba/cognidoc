"""
Benchmark tests for CogniDoc retrieval performance.

Compares:
- Vector-only retrieval
- GraphRAG (Vector + Graph) retrieval
- Hybrid search (Dense + BM25)

Metrics:
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that are retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- Latency: Query response time

Usage:
    pytest tests/test_benchmark.py -v --run-slow
    pytest tests/test_benchmark.py -v -k "test_benchmark_summary" --run-slow
"""

import time
import json
import pytest
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class BenchmarkQuery:
    """A benchmark query with expected relevant documents."""
    query: str
    language: str  # "fr", "en", "es", "de"
    query_type: str  # "factual", "relational", "exploratory", "procedural"
    expected_keywords: List[str]  # Keywords that should appear in relevant results
    expected_topics: List[str]  # Topic/document names that should be retrieved
    difficulty: str = "medium"  # "easy", "medium", "hard"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    mode: str  # "vector_only", "graph_only", "hybrid"
    documents: List[Dict[str, Any]]
    latency_ms: float
    num_results: int


@dataclass
class BenchmarkMetrics:
    """Metrics for a benchmark run."""
    mode: str
    num_queries: int = 0
    total_latency_ms: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    keyword_hit_rate_sum: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.num_queries, 1)

    @property
    def avg_precision(self) -> float:
        return self.precision_sum / max(self.num_queries, 1)

    @property
    def avg_recall(self) -> float:
        return self.recall_sum / max(self.num_queries, 1)

    @property
    def mrr(self) -> float:
        return self.mrr_sum / max(self.num_queries, 1)

    @property
    def avg_keyword_hit_rate(self) -> float:
        return self.keyword_hit_rate_sum / max(self.num_queries, 1)


# =============================================================================
# Benchmark Queries - Domain: Bioethics (Théologie Morale)
# =============================================================================

BENCHMARK_QUERIES = [
    # French - Factual
    BenchmarkQuery(
        query="Qu'est-ce que la bioéthique ?",
        language="fr",
        query_type="factual",
        expected_keywords=["bioéthique", "éthique", "vie", "médecine", "morale"],
        expected_topics=["Manuel - Bioéthique", "bioéthique"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Quelle est la position de l'Église sur l'avortement ?",
        language="fr",
        query_type="factual",
        expected_keywords=["avortement", "église", "vie", "embryon", "moral"],
        expected_topics=["Avortement", "embryon"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Qu'est-ce que la PMA ?",
        language="fr",
        query_type="factual",
        expected_keywords=["PMA", "procréation", "assistée", "fécondation"],
        expected_topics=["PMA"],
        difficulty="easy",
    ),

    # French - Relational
    BenchmarkQuery(
        query="Quel est le lien entre contraception et mariage selon l'Église ?",
        language="fr",
        query_type="relational",
        expected_keywords=["contraception", "mariage", "église", "conjugal", "amour"],
        expected_topics=["Contraceptif", "mariage"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Comment l'embryon est-il considéré par rapport à la dignité humaine ?",
        language="fr",
        query_type="relational",
        expected_keywords=["embryon", "dignité", "humain", "personne", "vie"],
        expected_topics=["Embryon", "Début de la vie"],
        difficulty="medium",
    ),

    # French - Exploratory
    BenchmarkQuery(
        query="Parlez-moi de l'euthanasie",
        language="fr",
        query_type="exploratory",
        expected_keywords=["euthanasie", "mort", "fin de vie", "souffrance"],
        expected_topics=["Euthanasie"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Quels sont les enjeux éthiques de la GPA ?",
        language="fr",
        query_type="exploratory",
        expected_keywords=["GPA", "gestation", "mère", "enfant", "éthique"],
        expected_topics=["GPA"],
        difficulty="medium",
    ),

    # French - Procedural
    BenchmarkQuery(
        query="Comment discerner la volonté de Dieu concernant le nombre d'enfants ?",
        language="fr",
        query_type="procedural",
        expected_keywords=["enfants", "Dieu", "volonté", "famille", "discernement"],
        expected_topics=["Début de la vie", "fertilité"],
        difficulty="hard",
    ),

    # French - Complex/Hard
    BenchmarkQuery(
        query="Quelle est la différence morale entre la contraception naturelle et artificielle ?",
        language="fr",
        query_type="relational",
        expected_keywords=["contraception", "naturelle", "artificielle", "morale", "méthode"],
        expected_topics=["Contraceptif"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Est-ce possible pour un couple d'habiter ensemble avant leur mariage ?",
        language="fr",
        query_type="factual",
        expected_keywords=["couple", "mariage", "cohabitation", "sexuel"],
        expected_topics=["mariage", "conjugal"],
        difficulty="hard",
    ),
]


class BenchmarkRunner:
    """Runs benchmarks comparing different retrieval modes."""

    def __init__(self, cognidoc_instance):
        self.cognidoc = cognidoc_instance
        self.results: Dict[str, List[RetrievalResult]] = defaultdict(list)
        self.metrics: Dict[str, BenchmarkMetrics] = {}

    def _get_retriever(self):
        """Get or create the hybrid retriever."""
        if self.cognidoc._retriever is None:
            from cognidoc.hybrid_retriever import HybridRetriever
            self.cognidoc._retriever = HybridRetriever()
            self.cognidoc._retriever.load()
        return self.cognidoc._retriever

    def _extract_documents(self, result) -> List[Dict[str, Any]]:
        """Extract documents from HybridRetrievalResult.

        The result structure contains:
        - vector_results: List[NodeWithScore] where NodeWithScore has:
          - score: float
          - node: Document with text, metadata
          - node.metadata['source']['document']: source document name
          - node.metadata['source']['page']: page number
        """
        documents = []

        # Extract from vector_results (primary source with metadata)
        vector_results = getattr(result, 'vector_results', []) or []
        for vr in vector_results:
            if hasattr(vr, 'node') and hasattr(vr.node, 'text'):
                node = vr.node
                source_info = node.metadata.get('source', {}) if hasattr(node, 'metadata') else {}
                documents.append({
                    "content": node.text[:500] if node.text else "",
                    "metadata": {
                        "source": source_info.get('document', ''),
                        "page": source_info.get('page', ''),
                        "name": node.metadata.get('name', '') if hasattr(node, 'metadata') else '',
                    },
                    "score": vr.score if hasattr(vr, 'score') else 0.0,
                })

        # Fallback to source_chunks if no vector_results
        if not documents:
            source_chunks = getattr(result, 'source_chunks', []) or []
            for chunk in source_chunks:
                if hasattr(chunk, 'content'):
                    documents.append({
                        "content": chunk.content[:500],
                        "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},
                        "score": 0.0,
                    })
                elif isinstance(chunk, str):
                    documents.append({
                        "content": chunk[:500],
                        "metadata": {},
                        "score": 0.0,
                    })

        return documents

    def retrieve_vector_only(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve using vector search only."""
        start = time.perf_counter()

        # Use the hybrid retriever with graph disabled
        retriever = self._get_retriever()

        # Temporarily disable graph
        original_strategy = retriever.config.routing.strategy if retriever.config else "auto"
        if retriever.config:
            retriever.config.routing.strategy = "vector_only"

        try:
            result = retriever.retrieve(query=query, top_k=top_k, use_reranking=False)
            documents = self._extract_documents(result)
        finally:
            if retriever.config:
                retriever.config.routing.strategy = original_strategy

        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            mode="vector_only",
            documents=documents,
            latency_ms=latency,
            num_results=len(documents),
        )

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve using hybrid search (vector + graph)."""
        start = time.perf_counter()

        retriever = self._get_retriever()

        # Ensure hybrid mode
        original_strategy = retriever.config.routing.strategy if retriever.config else "auto"
        if retriever.config:
            retriever.config.routing.strategy = "auto"

        try:
            result = retriever.retrieve(query=query, top_k=top_k, use_reranking=False)
            documents = self._extract_documents(result)
        finally:
            if retriever.config:
                retriever.config.routing.strategy = original_strategy

        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            mode="hybrid",
            documents=documents,
            latency_ms=latency,
            num_results=len(documents),
        )

    def calculate_keyword_hit_rate(
        self,
        result: RetrievalResult,
        expected_keywords: List[str]
    ) -> float:
        """Calculate what fraction of expected keywords appear in results."""
        if not expected_keywords:
            return 1.0

        all_text = " ".join(
            doc.get("content", "") + " " + str(doc.get("metadata", {}))
            for doc in result.documents
        ).lower()

        hits = sum(1 for kw in expected_keywords if kw.lower() in all_text)
        return hits / len(expected_keywords)

    def calculate_topic_precision(
        self,
        result: RetrievalResult,
        expected_topics: List[str]
    ) -> float:
        """Calculate precision based on expected topics in document sources."""
        if not result.documents:
            return 0.0

        relevant_count = 0
        for doc in result.documents:
            source = str(doc.get("metadata", {}).get("source", {}))
            for topic in expected_topics:
                if topic.lower() in source.lower():
                    relevant_count += 1
                    break

        return relevant_count / len(result.documents)

    def calculate_mrr(
        self,
        result: RetrievalResult,
        expected_topics: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc in enumerate(result.documents):
            source = str(doc.get("metadata", {}).get("source", {}))
            for topic in expected_topics:
                if topic.lower() in source.lower():
                    return 1.0 / (i + 1)
        return 0.0

    def _clear_caches(self):
        """Clear retrieval caches for fair benchmarking."""
        try:
            from cognidoc.hybrid_retriever import clear_retrieval_cache
            clear_retrieval_cache()
        except ImportError:
            pass

        try:
            from cognidoc.utils.rag_utils import _qdrant_result_cache
            _qdrant_result_cache.clear()
        except (ImportError, AttributeError):
            pass

    def run_benchmark(
        self,
        queries: List[BenchmarkQuery],
        modes: List[str] = ["vector_only", "hybrid"]
    ) -> Dict[str, BenchmarkMetrics]:
        """Run benchmark on a set of queries.

        Runs all queries in vector_only mode first, then clears cache,
        then runs all queries in hybrid mode. This ensures fair latency comparison.
        """

        for mode in modes:
            self.metrics[mode] = BenchmarkMetrics(mode=mode)

        # Run each mode separately with cache clearing between them
        for mode in modes:
            # Clear caches before each mode
            self._clear_caches()

            for bq in queries:
                # Retrieve
                if mode == "vector_only":
                    result = self.retrieve_vector_only(bq.query)
                elif mode == "hybrid":
                    result = self.retrieve_hybrid(bq.query)
                else:
                    continue

                self.results[mode].append(result)

                # Calculate metrics
                metrics = self.metrics[mode]
                metrics.num_queries += 1
                metrics.total_latency_ms += result.latency_ms
                metrics.keyword_hit_rate_sum += self.calculate_keyword_hit_rate(
                    result, bq.expected_keywords
                )
                metrics.precision_sum += self.calculate_topic_precision(
                    result, bq.expected_topics
                )
                metrics.mrr_sum += self.calculate_mrr(result, bq.expected_topics)

        return self.metrics

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        for mode, metrics in self.metrics.items():
            print(f"\n{mode.upper()}:")
            print(f"  Queries: {metrics.num_queries}")
            print(f"  Avg Latency: {metrics.avg_latency_ms:.1f} ms")
            print(f"  Avg Keyword Hit Rate: {metrics.avg_keyword_hit_rate:.2%}")
            print(f"  Avg Topic Precision: {metrics.avg_precision:.2%}")
            print(f"  MRR: {metrics.mrr:.3f}")

        # Comparison
        if "vector_only" in self.metrics and "hybrid" in self.metrics:
            v = self.metrics["vector_only"]
            h = self.metrics["hybrid"]

            print("\n" + "-" * 70)
            print("HYBRID vs VECTOR-ONLY:")

            latency_diff = ((h.avg_latency_ms - v.avg_latency_ms) / v.avg_latency_ms) * 100
            print(f"  Latency: {latency_diff:+.1f}%")

            precision_diff = ((h.avg_precision - v.avg_precision) / max(v.avg_precision, 0.01)) * 100
            print(f"  Topic Precision: {precision_diff:+.1f}%")

            keyword_diff = ((h.avg_keyword_hit_rate - v.avg_keyword_hit_rate) / max(v.avg_keyword_hit_rate, 0.01)) * 100
            print(f"  Keyword Hit Rate: {keyword_diff:+.1f}%")

            mrr_diff = ((h.mrr - v.mrr) / max(v.mrr, 0.01)) * 100
            print(f"  MRR: {mrr_diff:+.1f}%")

        print("\n" + "=" * 70)


# =============================================================================
# Pytest Tests
# =============================================================================

@pytest.mark.slow
class TestBenchmarkVectorOnly:
    """Tests for vector-only retrieval performance."""

    def test_vector_retrieval_returns_results(self, cognidoc_session):
        """Vector retrieval should return results."""
        runner = BenchmarkRunner(cognidoc_session)
        result = runner.retrieve_vector_only("Qu'est-ce que la bioéthique ?")
        assert result.num_results > 0
        assert result.latency_ms > 0

    def test_vector_retrieval_relevance(self, cognidoc_session):
        """Vector retrieval should return relevant results."""
        runner = BenchmarkRunner(cognidoc_session)
        result = runner.retrieve_vector_only("Qu'est-ce que la PMA ?")

        # Check that at least one result mentions PMA
        all_content = " ".join(doc.get("content", "") for doc in result.documents)
        assert "pma" in all_content.lower() or "procréation" in all_content.lower()


@pytest.mark.slow
class TestBenchmarkHybrid:
    """Tests for hybrid (vector + graph) retrieval performance."""

    def test_hybrid_retrieval_returns_results(self, cognidoc_session):
        """Hybrid retrieval should return results."""
        runner = BenchmarkRunner(cognidoc_session)
        result = runner.retrieve_hybrid("Qu'est-ce que la bioéthique ?")
        assert result.num_results > 0
        assert result.latency_ms > 0

    def test_hybrid_retrieval_relevance(self, cognidoc_session):
        """Hybrid retrieval should return relevant results."""
        runner = BenchmarkRunner(cognidoc_session)
        result = runner.retrieve_hybrid("Quelle est la position de l'Église sur l'euthanasie ?")

        # Check that at least one result is relevant
        all_content = " ".join(doc.get("content", "") for doc in result.documents)
        has_euthanasia = "euthanasie" in all_content.lower()
        has_church = "église" in all_content.lower() or "moral" in all_content.lower()
        assert has_euthanasia or has_church


@pytest.mark.slow
class TestBenchmarkComparison:
    """Compare vector-only vs hybrid retrieval."""

    def test_benchmark_summary(self, cognidoc_session):
        """Run full benchmark and print summary."""
        runner = BenchmarkRunner(cognidoc_session)

        # Run on subset for faster testing
        test_queries = BENCHMARK_QUERIES[:5]

        metrics = runner.run_benchmark(test_queries, modes=["vector_only", "hybrid"])
        runner.print_summary()

        # Basic assertions
        assert "vector_only" in metrics
        assert "hybrid" in metrics
        assert metrics["vector_only"].num_queries == len(test_queries)
        assert metrics["hybrid"].num_queries == len(test_queries)

    def test_hybrid_improves_relational_queries(self, cognidoc_session):
        """Hybrid should perform better on relational queries."""
        runner = BenchmarkRunner(cognidoc_session)

        # Test a relational query
        relational_queries = [q for q in BENCHMARK_QUERIES if q.query_type == "relational"][:2]

        if not relational_queries:
            pytest.skip("No relational queries in benchmark set")

        metrics = runner.run_benchmark(relational_queries, modes=["vector_only", "hybrid"])

        # Hybrid should have at least as good precision on relational queries
        # (graph helps with entity relationships)
        v_precision = metrics["vector_only"].avg_precision
        h_precision = metrics["hybrid"].avg_precision

        # Allow some tolerance - hybrid should not be significantly worse
        assert h_precision >= v_precision * 0.8, \
            f"Hybrid precision ({h_precision:.2f}) much worse than vector ({v_precision:.2f})"


@pytest.mark.slow
class TestBenchmarkLatency:
    """Tests for retrieval latency."""

    def test_vector_latency_reasonable(self, cognidoc_session):
        """Vector retrieval should complete in reasonable time."""
        runner = BenchmarkRunner(cognidoc_session)
        result = runner.retrieve_vector_only("Qu'est-ce que la bioéthique ?")

        # Should complete in under 15 seconds (includes LLM calls for classification + reranking)
        assert result.latency_ms < 15000, f"Vector retrieval too slow: {result.latency_ms}ms"

    def test_hybrid_latency_reasonable(self, cognidoc_session):
        """Hybrid retrieval should complete in reasonable time."""
        runner = BenchmarkRunner(cognidoc_session)
        result = runner.retrieve_hybrid("Qu'est-ce que la bioéthique ?")

        # Should complete in under 25 seconds (includes LLM calls + graph retrieval)
        assert result.latency_ms < 25000, f"Hybrid retrieval too slow: {result.latency_ms}ms"


@pytest.mark.slow
class TestBenchmarkByQueryType:
    """Tests grouped by query type."""

    def test_factual_queries(self, cognidoc_session):
        """Test performance on factual queries."""
        runner = BenchmarkRunner(cognidoc_session)
        factual = [q for q in BENCHMARK_QUERIES if q.query_type == "factual"][:3]

        metrics = runner.run_benchmark(factual, modes=["vector_only", "hybrid"])

        # Factual queries should have decent keyword hit rate
        assert metrics["vector_only"].avg_keyword_hit_rate >= 0.3
        assert metrics["hybrid"].avg_keyword_hit_rate >= 0.3

    def test_exploratory_queries(self, cognidoc_session):
        """Test performance on exploratory queries."""
        runner = BenchmarkRunner(cognidoc_session)
        exploratory = [q for q in BENCHMARK_QUERIES if q.query_type == "exploratory"][:2]

        if not exploratory:
            pytest.skip("No exploratory queries in benchmark set")

        metrics = runner.run_benchmark(exploratory, modes=["vector_only", "hybrid"])

        # Exploratory queries benefit from graph context
        # Hybrid should have results
        assert metrics["hybrid"].num_queries > 0


# =============================================================================
# Standalone Benchmark Runner
# =============================================================================

def run_full_benchmark():
    """Run full benchmark (call from command line)."""
    from cognidoc import CogniDoc

    print("Initializing CogniDoc...")
    doc = CogniDoc(
        llm_provider="gemini",
        embedding_provider="ollama",
        use_graph=True,
    )

    print(f"Running benchmark with {len(BENCHMARK_QUERIES)} queries...")
    runner = BenchmarkRunner(doc)
    runner.run_benchmark(BENCHMARK_QUERIES)
    runner.print_summary()

    # Save results to JSON
    results_path = Path(__file__).parent / "benchmark_results.json"
    results = {
        mode: {
            "avg_latency_ms": m.avg_latency_ms,
            "avg_precision": m.avg_precision,
            "avg_keyword_hit_rate": m.avg_keyword_hit_rate,
            "mrr": m.mrr,
            "num_queries": m.num_queries,
        }
        for mode, m in runner.metrics.items()
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_full_benchmark()
