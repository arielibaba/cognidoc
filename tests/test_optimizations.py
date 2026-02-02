"""
Unit tests for pipeline optimizations.

Tests for:
1. Entity extraction concurrency (adaptive semaphore, optimal concurrency)
2. Thread pool size for parallel retrieval
3. Parallel extraction with asyncio.as_completed
4. PDF batch conversion with ProcessPoolExecutor
5. YOLO parallel image loading with ThreadPoolExecutor
6. Embeddings connection pooling with shared httpx client
"""

import asyncio
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time

import pytest


# =============================================================================
# 1. Entity Extraction Concurrency Tests
# =============================================================================


class TestEntityExtractionConcurrency:
    """Tests for entity extraction concurrency optimization."""

    def test_get_optimal_concurrency_returns_valid_range(self):
        """Test that get_optimal_concurrency returns value between 2 and 8."""
        from cognidoc.extract_entities import get_optimal_concurrency

        result = get_optimal_concurrency()

        assert isinstance(result, int)
        assert 2 <= result <= 8

    def test_get_optimal_concurrency_based_on_cpu_count(self):
        """Test that optimal concurrency is based on CPU count."""
        from cognidoc.extract_entities import get_optimal_concurrency

        # Mock different CPU counts
        with patch("os.cpu_count", return_value=4):
            result = get_optimal_concurrency()
            # With 4 cores: min(8, 4-1) = 3, max(2, 3) = 3
            assert result == 3

        with patch("os.cpu_count", return_value=2):
            result = get_optimal_concurrency()
            # With 2 cores: min(8, 2-1) = 1, max(2, 1) = 2
            assert result == 2

        with patch("os.cpu_count", return_value=16):
            result = get_optimal_concurrency()
            # With 16 cores: min(8, 16-1) = 8, max(2, 8) = 8
            assert result == 8

    def test_get_optimal_concurrency_handles_none_cpu_count(self):
        """Test fallback when cpu_count returns None."""
        from cognidoc.extract_entities import get_optimal_concurrency

        with patch("os.cpu_count", return_value=None):
            result = get_optimal_concurrency()
            # Should use fallback of 4 cores: min(8, 4-1) = 3
            assert result == 3

    def test_get_optimal_concurrency_handles_exception(self):
        """Test fallback when cpu_count raises exception."""
        from cognidoc.extract_entities import get_optimal_concurrency

        with patch("os.cpu_count", side_effect=Exception("Test error")):
            result = get_optimal_concurrency()
            # Should return safe default of 4
            assert result == 4

    def test_semaphore_limits_concurrent_tasks(self):
        """Test that asyncio.Semaphore properly limits concurrent execution."""
        max_concurrent = 3
        concurrent_count = 0
        max_observed_concurrent = 0

        async def task(semaphore):
            nonlocal concurrent_count, max_observed_concurrent
            async with semaphore:
                concurrent_count += 1
                max_observed_concurrent = max(max_observed_concurrent, concurrent_count)
                await asyncio.sleep(0.01)  # Simulate work
                concurrent_count -= 1

        async def run_test():
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [asyncio.create_task(task(semaphore)) for _ in range(10)]
            await asyncio.gather(*tasks)

        asyncio.run(run_test())

        assert max_observed_concurrent <= max_concurrent


# =============================================================================
# 2. Thread Pool Size Tests (Parallel Retrieval)
# =============================================================================


class TestThreadPoolSizeOptimization:
    """Tests for thread pool size in parallel retrieval."""

    def test_thread_pool_executor_parallel_execution(self):
        """Test that ThreadPoolExecutor runs tasks in parallel."""
        start_time = time.time()
        results = []

        def slow_task(task_id, sleep_time=0.1):
            time.sleep(sleep_time)
            return task_id

        # Run 4 tasks with 2 workers - should take ~0.2s (not 0.4s)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(slow_task, i) for i in range(4)]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.time() - start_time

        assert len(results) == 4
        # With 2 workers, 4 tasks of 0.1s each should take ~0.2s
        # Allow some overhead, but should be much less than sequential (0.4s)
        assert elapsed < 0.35

    def test_parallel_retrieval_pattern(self):
        """Test the parallel retrieval pattern used in hybrid_retriever."""
        vector_result = None
        graph_result = None

        def vector_retrieval():
            nonlocal vector_result
            time.sleep(0.05)
            vector_result = "vector_done"

        def graph_retrieval():
            nonlocal graph_result
            time.sleep(0.05)
            graph_result = "graph_done"

        start_time = time.time()

        # Simulate parallel execution as in HybridRetriever.retrieve()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(vector_retrieval),
                executor.submit(graph_retrieval),
            ]
            for future in as_completed(futures):
                future.result()  # Raises exception if task failed

        elapsed = time.time() - start_time

        assert vector_result == "vector_done"
        assert graph_result == "graph_done"
        # Both tasks ran in parallel - total time should be ~0.05s, not 0.1s
        assert elapsed < 0.1


# =============================================================================
# 3. Parallel Extraction Tests (asyncio.as_completed)
# =============================================================================


class TestParallelExtraction:
    """Tests for parallel extraction using asyncio.as_completed."""

    def test_as_completed_processes_all_tasks(self):
        """Test that asyncio.as_completed processes all tasks."""
        results = []

        async def task(task_id):
            await asyncio.sleep(0.01)
            return task_id

        async def run_test():
            tasks = [asyncio.create_task(task(i)) for i in range(5)]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)

        asyncio.run(run_test())

        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    def test_as_completed_early_stopping(self):
        """Test early stopping pattern with as_completed."""
        processed = []
        should_stop = False

        async def task(task_id):
            await asyncio.sleep(0.01 * task_id)  # Variable delay
            return task_id

        async def run_test():
            nonlocal should_stop
            tasks = {asyncio.create_task(task(i)): i for i in range(10)}
            pending = set(tasks.keys())

            for coro in asyncio.as_completed(pending):
                if should_stop:
                    # Cancel remaining tasks
                    for t in pending:
                        if not t.done():
                            t.cancel()
                    break

                try:
                    result = await coro
                    processed.append(result)
                    # Stop after 3 results
                    if len(processed) >= 3:
                        should_stop = True
                except asyncio.CancelledError:
                    pass

        asyncio.run(run_test())

        # Should have processed at least 3 before stopping
        assert len(processed) >= 3

    def test_concurrent_limit_with_semaphore(self):
        """Test that semaphore limits concurrent async tasks."""
        active_count = 0
        max_active = 0

        async def limited_task(semaphore, task_id):
            nonlocal active_count, max_active
            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1
            return task_id

        async def run_test():
            max_concurrent = 4
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [asyncio.create_task(limited_task(semaphore, i)) for i in range(20)]
            await asyncio.gather(*tasks)

        asyncio.run(run_test())

        assert max_active <= 4


# =============================================================================
# 4. PDF Batch Conversion Tests
# =============================================================================


# Module-level function for ProcessPoolExecutor (can't pickle local functions)
def _cpu_bound_task(n):
    """Simple CPU-bound task for testing ProcessPoolExecutor."""
    result = 0
    for i in range(n):
        result += i
    return result


class TestPDFBatchConversion:
    """Tests for PDF batch conversion with ProcessPoolExecutor."""

    def test_process_pool_executor_parallel_execution(self):
        """Test that ProcessPoolExecutor runs tasks in parallel processes."""
        start_time = time.time()
        results = []

        # Run tasks with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_cpu_bound_task, 100000) for _ in range(4)]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.time() - start_time

        assert len(results) == 4
        # All results should be the same (sum of 0 to 99999)
        expected = sum(range(100000))
        assert all(r == expected for r in results)

    def test_batch_processing_pattern(self):
        """Test the batch processing pattern for PDF conversion."""
        from cognidoc.convert_pdf_to_image import DEFAULT_PAGE_BATCH_SIZE

        # Verify default batch size is set
        assert DEFAULT_PAGE_BATCH_SIZE == 5

    def test_conversion_result_dataclass(self):
        """Test ConversionResult dataclass."""
        from cognidoc.convert_pdf_to_image import ConversionResult

        result = ConversionResult(
            pdf_path="/path/to/file.pdf",
            success=True,
            pages_converted=10,
        )
        assert result.pdf_path == "/path/to/file.pdf"
        assert result.success is True
        assert result.pages_converted == 10
        assert result.error is None

        # Test with error
        result_error = ConversionResult(
            pdf_path="/path/to/file.pdf",
            success=False,
            pages_converted=0,
            error="Test error",
        )
        assert result_error.success is False
        assert result_error.error == "Test error"

    def test_relative_path_prefix_encoding(self):
        """Test path encoding for subdirectory PDFs."""
        from cognidoc.convert_pdf_to_image import get_relative_path_prefix, PATH_SEPARATOR

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)

            # Create nested directory structure
            subdir = base_dir / "projet_A"
            subdir.mkdir()
            pdf_path = subdir / "doc.pdf"
            pdf_path.touch()

            prefix = get_relative_path_prefix(pdf_path, base_dir)
            # Should encode path: projet_A/doc -> projet_A__doc
            assert prefix == f"projet_A{PATH_SEPARATOR}doc"

    def test_parallel_flag_controls_execution_mode(self):
        """Test that parallel flag controls execution mode."""
        from cognidoc.convert_pdf_to_image import DEFAULT_MAX_WORKERS

        # Verify default workers
        assert DEFAULT_MAX_WORKERS == 2


# =============================================================================
# 5. YOLO Parallel Loading Tests
# =============================================================================


class TestYOLOParallelLoading:
    """Tests for YOLO parallel image loading with ThreadPoolExecutor."""

    def test_thread_pool_parallel_image_loading(self):
        """Test parallel image loading pattern used in YOLO processing."""
        loaded_images = []

        def load_image(path):
            time.sleep(0.02)  # Simulate I/O
            return (path, f"data_{path}")

        image_paths = [f"image_{i}.png" for i in range(8)]

        start_time = time.time()

        # Use ThreadPoolExecutor as in DetectionProcessor.process_images_batch
        with ThreadPoolExecutor(max_workers=min(len(image_paths), 4)) as executor:
            loaded = list(executor.map(load_image, image_paths))

        elapsed = time.time() - start_time

        assert len(loaded) == 8
        # With 4 workers, 8 tasks of 0.02s should take ~0.04s
        # Sequential would be 0.16s
        assert elapsed < 0.1

    def test_batch_processing_mode(self):
        """Test batch processing configuration."""
        from cognidoc.extract_objects_from_image import extract_objects_from_image

        # Function should accept batch_size and use_batching parameters
        import inspect

        sig = inspect.signature(extract_objects_from_image)
        params = list(sig.parameters.keys())

        assert "batch_size" in params
        assert "use_batching" in params

    def test_yolo_availability_check(self):
        """Test YOLO availability detection."""
        from cognidoc.extract_objects_from_image import is_yolo_available, is_yolo_model_available

        # is_yolo_available checks package availability
        result = is_yolo_available()
        assert isinstance(result, bool)

        # is_yolo_model_available checks both package and model file
        result = is_yolo_model_available("/nonexistent/path.pt")
        assert result is False


# =============================================================================
# 6. Embeddings Connection Pooling Tests
# =============================================================================


class TestEmbeddingsConnectionPooling:
    """Tests for embeddings connection pooling with shared httpx client."""

    def test_shared_async_client_singleton(self):
        """Test that OllamaEmbeddingProvider uses a shared async client."""
        from cognidoc.utils.embedding_providers import (
            OllamaEmbeddingProvider,
            EmbeddingConfig,
            EmbeddingProvider,
        )

        # Reset shared client
        OllamaEmbeddingProvider._shared_async_client = None

        with patch("ollama.Client"):
            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="test-model",
            )
            provider1 = OllamaEmbeddingProvider(config)
            provider2 = OllamaEmbeddingProvider(config)

        # Get shared client (this should create it)
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            client1 = provider1._get_async_client(60.0)
            client2 = provider2._get_async_client(60.0)

            # Should be the same instance (singleton)
            assert client1 is client2
            # AsyncClient should only be created once
            assert mock_client.call_count == 1

    def test_connection_pooling_limits(self):
        """Test that connection pooling uses correct limits."""
        from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

        # Reset shared client
        OllamaEmbeddingProvider._shared_async_client = None

        with patch("httpx.AsyncClient") as mock_client:
            with patch("httpx.Limits") as mock_limits:
                mock_limits.return_value = MagicMock()
                mock_client.return_value = MagicMock()

                OllamaEmbeddingProvider._get_async_client(60.0)

                # Verify Limits was called with correct parameters
                mock_limits.assert_called_once_with(
                    max_keepalive_connections=10,
                    max_connections=20,
                )

    def test_embed_async_uses_connection_pooling(self):
        """Test that embed_async uses the shared client."""
        from cognidoc.utils.embedding_providers import (
            OllamaEmbeddingProvider,
            EmbeddingConfig,
            EmbeddingProvider,
        )

        # Reset shared client
        OllamaEmbeddingProvider._shared_async_client = None

        async def run_test():
            with patch("ollama.Client"):
                config = EmbeddingConfig(
                    provider=EmbeddingProvider.OLLAMA,
                    model="test-model",
                )
                provider = OllamaEmbeddingProvider(config)

                # Mock the shared client
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
                mock_client.post.return_value = mock_response

                with patch.object(
                    OllamaEmbeddingProvider, "_get_async_client", return_value=mock_client
                ):
                    texts = ["text1", "text2"]
                    results = await provider.embed_async(texts, max_concurrent=2)

                    # Should have called post twice
                    assert mock_client.post.call_count == 2
                    assert len(results) == 2

        asyncio.run(run_test())

    def test_batch_embedding_configuration(self):
        """Test batch embedding configuration in create_embeddings."""
        from cognidoc.create_embeddings import DEFAULT_BATCH_SIZE, MAX_CONCURRENT_REQUESTS

        # Verify default values
        assert DEFAULT_BATCH_SIZE == 32
        assert MAX_CONCURRENT_REQUESTS == 4

    def test_embed_batch_async_uses_connection_pooling(self):
        """Test that embed_batch_async uses httpx connection pooling."""
        from cognidoc.create_embeddings import embed_batch_async, ChunkToEmbed

        async def run_test():
            with tempfile.TemporaryDirectory() as tmp_dir:
                embeddings_path = Path(tmp_dir)

                chunks = [
                    ChunkToEmbed(
                        file_path=Path("test1.txt"),
                        text="Test text 1",
                        metadata={"child": "test1.txt"},
                    ),
                    ChunkToEmbed(
                        file_path=Path("test2.txt"),
                        text="Test text 2",
                        metadata={"child": "test2.txt"},
                    ),
                ]

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"embedding": [0.1, 0.2]}
                    mock_client.post.return_value = mock_response
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock()
                    mock_client_class.return_value = mock_client

                    with patch("cognidoc.create_embeddings.get_embedding_cache", return_value=None):
                        success, errors = await embed_batch_async(
                            chunks,
                            embeddings_path,
                            "test-model",
                            use_cache=False,
                            max_concurrent=2,
                        )

                        # Verify AsyncClient was created with limits
                        mock_client_class.assert_called_once()
                        call_kwargs = mock_client_class.call_args.kwargs
                        assert "limits" in call_kwargs

        asyncio.run(run_test())


# =============================================================================
# Integration Tests
# =============================================================================


class TestOptimizationsIntegration:
    """Integration tests for multiple optimizations working together."""

    def test_concurrent_patterns_compatibility(self):
        """Test that async and threading patterns work together."""
        thread_results = []
        async_results = []

        def thread_task(task_id):
            time.sleep(0.01)
            return f"thread_{task_id}"

        async def async_task(task_id):
            await asyncio.sleep(0.01)
            return f"async_{task_id}"

        async def run_async_tasks():
            tasks = [asyncio.create_task(async_task(i)) for i in range(5)]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                async_results.append(result)

        # Run thread tasks
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(thread_task, i) for i in range(5)]
            for future in as_completed(futures):
                thread_results.append(future.result())

        # Run async tasks
        asyncio.run(run_async_tasks())

        assert len(thread_results) == 5
        assert len(async_results) == 5
        assert all(r.startswith("thread_") for r in thread_results)
        assert all(r.startswith("async_") for r in async_results)

    def test_error_handling_in_parallel_execution(self):
        """Test error handling in parallel execution patterns."""
        results = []
        errors = []

        def task_with_possible_error(task_id):
            if task_id == 2:
                raise ValueError(f"Error in task {task_id}")
            return task_id

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(task_with_possible_error, i): i for i in range(5)}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except ValueError as e:
                    errors.append(str(e))

        assert len(results) == 4
        assert len(errors) == 1
        assert "Error in task 2" in errors[0]


# =============================================================================
# Performance Benchmark Tests (optional, skipped by default)
# =============================================================================


@pytest.mark.skip(reason="Performance benchmarks - run manually")
class TestOptimizationBenchmarks:
    """Performance benchmark tests for optimizations."""

    def test_parallel_vs_sequential_speedup(self):
        """Benchmark parallel vs sequential execution speedup."""
        import statistics

        def work(n):
            return sum(range(n))

        # Sequential
        seq_times = []
        for _ in range(3):
            start = time.time()
            results = [work(100000) for _ in range(8)]
            seq_times.append(time.time() - start)

        # Parallel
        par_times = []
        for _ in range(3):
            start = time.time()
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(work, [100000] * 8))
            par_times.append(time.time() - start)

        seq_avg = statistics.mean(seq_times)
        par_avg = statistics.mean(par_times)
        speedup = seq_avg / par_avg

        print(f"\nSequential avg: {seq_avg:.3f}s")
        print(f"Parallel avg: {par_avg:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Expect at least 1.5x speedup with 4 workers
        assert speedup > 1.5


# =============================================================================
# 7. Reranking Parser Tests
# =============================================================================


class TestRerankingParser:
    """Tests for LLM reranking response parsing robustness."""

    def _make_docs(self, n=5):
        """Create n mock documents for reranking."""
        from cognidoc.utils.rag_utils import Document, NodeWithScore

        docs = []
        for i in range(n):
            node = Document(text=f"Document content {i+1}", metadata={})
            docs.append(NodeWithScore(node=node, score=0.5))
        return docs

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_standard_format(self, mock_llm):
        """Test parsing of standard 'Document N (score: X): summary' format."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "Document 2 (score: 9): Very relevant\n"
            "Document 1 (score: 7): Somewhat relevant\n"
            "Document 4 (score: 3): Less relevant"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 3
        assert result[0].score == 9.0
        assert result[1].score == 7.0
        assert result[2].score == 3.0

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_score_with_equals(self, mock_llm):
        """Test parsing with 'score = X' format."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "Document 3 (score = 8): good match\n" "Document 1 (score = 5): ok match"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 2
        assert result[0].score == 8.0
        assert result[1].score == 5.0

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_no_score_defaults_to_five(self, mock_llm):
        """Test that missing score defaults to 5.0."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "Document 1: very relevant content\n" "Document 3: somewhat relevant content"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 2
        assert all(r.score == 5.0 for r in result)

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_markdown_bold_format(self, mock_llm):
        """Test parsing when LLM wraps 'Document' in markdown bold."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "**Document 2** (score: 9): relevant\n" "**Document 1** (score: 6): ok"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 2
        assert result[0].score == 9.0

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_numbered_list_format(self, mock_llm):
        """Test parsing when LLM prefixes with numbered list."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "1. Document 3 (score: 10)\n" "2. Document 1 (score: 7)\n" "3. Document 5 (score: 4)"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 3
        assert result[0].score == 10.0

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_lowercase_document(self, mock_llm):
        """Test parsing with lowercase 'document'."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = "document 2 (score: 8): relevant\n" "document 4 (score: 5): ok"
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 2

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_decimal_scores(self, mock_llm):
        """Test parsing with decimal scores like 8.5."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = "Document 1 (score: 8.5): great\n" "Document 3 (score: 6.2): ok"
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 2
        assert result[0].score == 8.5
        assert result[1].score == 6.2

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_out_of_range_doc_ignored(self, mock_llm):
        """Test that doc numbers outside valid range are ignored."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "Document 0 (score: 9): invalid\n"
            "Document 1 (score: 8): valid\n"
            "Document 99 (score: 7): invalid"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 1
        assert result[0].score == 8.0

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_empty_response_falls_back(self, mock_llm):
        """Test that unparseable response returns original order."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = "I cannot rank these documents."
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 3
        # Should be original order (first 3 docs)
        assert result[0].node.text == "Document content 1"

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_llm_exception_falls_back(self, mock_llm):
        """Test that LLM exception returns original order."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.side_effect = Exception("API error")
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        assert len(result) == 3

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_sorts_by_score_descending(self, mock_llm):
        """Test that results are sorted by score descending."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = (
            "Document 3 (score: 3)\n" "Document 1 (score: 9)\n" "Document 2 (score: 6)"
        )
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=3)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @patch("cognidoc.utils.llm_client.llm_chat")
    def test_top_n_limits_output(self, mock_llm):
        """Test that top_n limits the number of returned docs."""
        from cognidoc.utils.rag_utils import rerank_documents

        mock_llm.return_value = "\n".join(f"Document {i} (score: {10-i})" for i in range(1, 6))
        docs = self._make_docs(5)
        result = rerank_documents(docs, "test query", top_n=2)
        assert len(result) == 2


# =============================================================================
# 8. Cache Key Normalization Tests
# =============================================================================


def _make_temp_cache(tmp_path, **kwargs):
    """Create a fresh RetrievalCache with a temp db, resetting singleton state."""
    from cognidoc.hybrid_retriever import RetrievalCache

    RetrievalCache._instance = None
    RetrievalCache._initialized = False
    db_path = str(tmp_path / "test_retrieval_cache.db")
    return RetrievalCache(db_path=db_path, **kwargs)


class TestCacheNormalization:
    """Tests for retrieval cache key normalization."""

    def test_cache_key_case_insensitive(self, tmp_path):
        """Same query with different casing should produce the same cache key."""
        cache = _make_temp_cache(tmp_path)
        key1 = cache._make_key("What is X?", 10, True)
        key2 = cache._make_key("what is x?", 10, True)
        assert key1 == key2

    def test_cache_key_strips_whitespace(self, tmp_path):
        """Extra whitespace should be collapsed."""
        cache = _make_temp_cache(tmp_path)
        key1 = cache._make_key("  hello  world  ", 10, True)
        key2 = cache._make_key("hello world", 10, True)
        assert key1 == key2

    def test_cache_key_strips_trailing_punctuation(self, tmp_path):
        """Trailing punctuation should be stripped."""
        cache = _make_temp_cache(tmp_path)
        key1 = cache._make_key("hello?", 10, True)
        key2 = cache._make_key("hello", 10, True)
        assert key1 == key2

    def test_cache_key_different_queries_differ(self, tmp_path):
        """Different queries should produce different keys."""
        cache = _make_temp_cache(tmp_path)
        key1 = cache._make_key("what is X", 10, True)
        key2 = cache._make_key("what is Y", 10, True)
        assert key1 != key2


# =============================================================================
# 9. Proportional Confidence Adjustment Tests
# =============================================================================


class TestProportionalConfidence:
    """Tests for proportional confidence shift in should_fallback."""

    def _make_orchestrator(self, threshold=0.3):
        from cognidoc.query_orchestrator import QueryOrchestrator

        config = Mock()
        config.confidence_threshold = threshold
        config.routing = Mock()
        config.routing.strategy = "auto"
        orch = QueryOrchestrator.__new__(QueryOrchestrator)
        orch.config = config
        return orch

    def test_shift_zero_at_threshold(self):
        """Confidence at threshold should produce zero shift."""
        orch = self._make_orchestrator(threshold=0.3)
        assert orch._confidence_shift(0.3) == 0.0

    def test_shift_max_at_zero(self):
        """Zero confidence should produce max shift of 0.3."""
        orch = self._make_orchestrator(threshold=0.3)
        assert abs(orch._confidence_shift(0.0) - 0.3) < 1e-9

    def test_shift_proportional(self):
        """Half-threshold confidence should produce half max shift."""
        orch = self._make_orchestrator(threshold=0.3)
        assert abs(orch._confidence_shift(0.15) - 0.15) < 1e-9

    def test_should_fallback_proportional(self):
        """Integration: verify weights adjusted proportionally."""
        from cognidoc.query_orchestrator import RoutingDecision, QueryType, RetrievalMode

        orch = self._make_orchestrator(threshold=0.3)
        decision = RoutingDecision(
            query="test",
            query_type=QueryType.FACTUAL,
            mode=RetrievalMode.HYBRID,
            vector_weight=0.7,
            graph_weight=0.3,
            skip_vector=False,
            skip_graph=False,
            confidence=0.8,
            reasoning="test",
        )
        # Vector confidence barely below threshold (0.29) → small shift
        adjusted = orch.should_fallback(decision, vector_confidence=0.29, graph_confidence=0.8)
        shift = 0.3 * (1.0 - 0.29 / 0.3)
        assert abs(adjusted.graph_weight - (0.3 + shift)) < 1e-9
        assert abs(adjusted.vector_weight - (0.7 - shift)) < 1e-9


# =============================================================================
# 10. Proportional Fusion Tests
# =============================================================================


class TestProportionalFusion:
    """Tests for proportional result capping in fuse_results."""

    def _make_vector_results(self, n):
        results = []
        for i in range(n):
            node = Mock()
            node.text = f"Document {i} content"
            node.metadata = {"name": f"doc_{i}.pdf"}
            nws = Mock()
            nws.node = node
            results.append(nws)
        return results

    def _make_graph_result(self, n_entities, chunks_per=3):
        graph = Mock()
        entities = []
        for i in range(n_entities):
            entity = Mock()
            entity.source_chunks = [f"chunk_{i}_{j}" for j in range(chunks_per)]
            entities.append(entity)
        graph.entities = entities
        graph.context = "Graph context text"
        return graph

    def _make_analysis(self, vector_weight, graph_weight):
        analysis = Mock()
        analysis.vector_weight = vector_weight
        analysis.graph_weight = graph_weight
        return analysis

    def test_fusion_scales_vector_by_weight(self):
        """Weight=0.2 with 10 results should use ~2 results."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(10)
        graph = self._make_graph_result(0)
        graph.context = ""
        graph.entities = []
        analysis = self._make_analysis(0.2, 0.8)

        _, sources = fuse_results("test", vector, graph, analysis)
        # max_vector = max(1, round(10 * 0.2)) = 2
        assert len([s for s in sources if s.startswith("doc_")]) == 2

    def test_fusion_scales_graph_by_weight(self):
        """Weight=0.3 with 10 entities should use ~3 entities."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(0)
        graph = self._make_graph_result(10, chunks_per=1)
        analysis = self._make_analysis(0.0, 0.3)

        _, sources = fuse_results("test", vector, graph, analysis)
        # max_graph_entities = max(1, round(10 * 0.3)) = 3, each with 1 chunk
        assert len(sources) == 3

    def test_fusion_minimum_one(self):
        """Weight=0.1 with 5 results should still use at least 1."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = self._make_graph_result(0)
        graph.context = ""
        graph.entities = []
        analysis = self._make_analysis(0.1, 0.9)

        _, sources = fuse_results("test", vector, graph, analysis)
        # max_vector = max(1, round(5 * 0.1)) = max(1, 1) = 1
        assert len([s for s in sources if s.startswith("doc_")]) >= 1

    def test_fusion_zero_weight_excludes(self):
        """Weight=0 should exclude results entirely."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = self._make_graph_result(0)
        graph.context = ""
        graph.entities = []
        analysis = self._make_analysis(0.0, 1.0)

        context, sources = fuse_results("test", vector, graph, analysis)
        assert "DOCUMENT CONTEXT" not in context
        assert len([s for s in sources if s.startswith("doc_")]) == 0


# =============================================================================
# 11. Multilingual Query Pattern Tests
# =============================================================================


class TestMultilingualPatterns:
    """Tests for multilingual query classification patterns."""

    def test_french_relational(self):
        """French relational query should be classified correctly."""
        from cognidoc.query_orchestrator import classify_query_rules, QueryType

        qtype, conf, _ = classify_query_rules("quelle est la relation entre X et Y")
        assert qtype == QueryType.RELATIONAL

    def test_french_question_word(self):
        """French question word should classify as FACTUAL."""
        from cognidoc.query_orchestrator import classify_query_rules, QueryType

        qtype, conf, _ = classify_query_rules("quel est le rôle de X")
        assert qtype == QueryType.FACTUAL

    def test_spanish_procedural(self):
        """Spanish procedural query should be classified correctly."""
        from cognidoc.query_orchestrator import classify_query_rules, QueryType

        qtype, conf, _ = classify_query_rules("cómo hacer una presentación")
        assert qtype == QueryType.PROCEDURAL

    def test_spanish_question_word(self):
        """Spanish question word should classify as FACTUAL."""
        from cognidoc.query_orchestrator import classify_query_rules, QueryType

        qtype, conf, _ = classify_query_rules("qué es la inteligencia artificial")
        assert qtype == QueryType.FACTUAL

    def test_german_exploratory(self):
        """German summary query should classify as EXPLORATORY."""
        from cognidoc.query_orchestrator import classify_query_rules, QueryType

        qtype, conf, _ = classify_query_rules("zusammenfassung von kapitel 3")
        assert qtype == QueryType.EXPLORATORY

    def test_german_question_word(self):
        """German question word should classify as FACTUAL."""
        from cognidoc.query_orchestrator import classify_query_rules, QueryType

        qtype, conf, _ = classify_query_rules("was ist maschinelles lernen")
        assert qtype == QueryType.FACTUAL


# =============================================================================
# 12. MODERATE Path Differentiation Tests
# =============================================================================


class TestModeratePathDifferentiation:
    """Tests for MODERATE path top_k increase."""

    def test_moderate_top_k_multiplied(self):
        """Score 0.45 should multiply top_k by 1.5."""
        from cognidoc.constants import TOP_K_RETRIEVED_CHILDREN

        score = 0.45
        effective_top_k = TOP_K_RETRIEVED_CHILDREN
        if 0.35 <= score < 0.55:
            effective_top_k = round(TOP_K_RETRIEVED_CHILDREN * 1.5)
        assert effective_top_k == round(TOP_K_RETRIEVED_CHILDREN * 1.5)

    def test_fast_top_k_unchanged(self):
        """Score 0.2 should keep original top_k."""
        from cognidoc.constants import TOP_K_RETRIEVED_CHILDREN

        score = 0.2
        effective_top_k = TOP_K_RETRIEVED_CHILDREN
        if 0.35 <= score < 0.55:
            effective_top_k = round(TOP_K_RETRIEVED_CHILDREN * 1.5)
        assert effective_top_k == TOP_K_RETRIEVED_CHILDREN

    def test_agent_path_not_affected(self):
        """Score >= 0.55 (agent path) should keep original top_k."""
        from cognidoc.constants import TOP_K_RETRIEVED_CHILDREN

        score = 0.6
        effective_top_k = TOP_K_RETRIEVED_CHILDREN
        if 0.35 <= score < 0.55:
            effective_top_k = round(TOP_K_RETRIEVED_CHILDREN * 1.5)
        assert effective_top_k == TOP_K_RETRIEVED_CHILDREN


# =============================================================================
# 13. Cache Invalidation After Ingestion Tests
# =============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation after ingestion."""

    def test_clear_retrieval_cache_exists(self):
        """clear_retrieval_cache function should exist and be callable."""
        from cognidoc.hybrid_retriever import clear_retrieval_cache

        # Should not raise
        clear_retrieval_cache()

    def test_clear_query_embedding_cache_exists(self):
        """clear_query_embedding_cache function should exist and be callable."""
        from cognidoc.utils.rag_utils import clear_query_embedding_cache

        # Should not raise
        clear_query_embedding_cache()

    def test_clear_reranking_cache_exists(self):
        """clear_reranking_cache function should exist and be callable."""
        from cognidoc.utils.advanced_rag import clear_reranking_cache

        # Should not raise
        clear_reranking_cache()


# =============================================================================
# 14. Lost-in-the-Middle Fusion Tests
# =============================================================================


class TestLostInTheMiddleFusion:
    """Tests for lost-in-the-middle reordering at the fusion level."""

    def _make_vector_results(self, n):
        results = []
        for i in range(n):
            node = Mock()
            node.text = f"Vector doc {i} content"
            node.metadata = {"name": f"doc_{i}.pdf"}
            nws = Mock()
            nws.node = node
            nws.score = 1.0 - i * 0.1
            results.append(nws)
        return results

    def _make_graph_result(self, n_entities=2):
        graph = Mock()
        entities = []
        for i in range(n_entities):
            entity = Mock()
            entity.source_chunks = [f"graph_chunk_{i}"]
            entities.append(entity)
        graph.entities = entities
        graph.context = "Graph context text"
        return graph

    def _make_analysis(self, vector_weight, graph_weight):
        analysis = Mock()
        analysis.vector_weight = vector_weight
        analysis.graph_weight = graph_weight
        return analysis

    def test_sandwich_when_vector_dominant(self):
        """Graph context sandwiched between vector docs when vector_weight >= graph_weight."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = self._make_graph_result()
        analysis = self._make_analysis(0.7, 0.3)

        context, _ = fuse_results("test", vector, graph, analysis)

        # Vector docs should appear before AND after graph context
        graph_pos = context.index("KNOWLEDGE GRAPH CONTEXT")
        first_doc_pos = context.index("Vector doc")
        last_doc_pos = context.rindex("Vector doc")

        assert first_doc_pos < graph_pos, "First vector doc should precede graph context"
        assert last_doc_pos > graph_pos, "Last vector doc should follow graph context"

    def test_best_chunks_at_extremities(self):
        """Most relevant vector chunks should occupy start and end positions."""
        from cognidoc.hybrid_retriever import fuse_results

        # Input order: doc0 (best) → doc4 (worst) by relevance
        vector = self._make_vector_results(5)
        graph = self._make_graph_result()
        analysis = self._make_analysis(1.0, 0.3)

        context, _ = fuse_results("test", vector, graph, analysis)

        # After reorder_lost_in_middle: [0,2,4,3,1] then sandwich split
        # doc0 (best) should be first vector doc in context
        # doc1 (2nd best) should be last vector doc in context
        lines = context.split("\n")
        vector_doc_lines = [l for l in lines if l.startswith("Vector doc")]
        assert vector_doc_lines[0] == "Vector doc 0 content", "Best chunk should be first"
        assert vector_doc_lines[-1] == "Vector doc 1 content", "2nd best chunk should be last"

    def test_graph_first_when_graph_dominant(self):
        """Graph context comes first when graph_weight > vector_weight."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = self._make_graph_result()
        analysis = self._make_analysis(0.2, 0.8)

        context, _ = fuse_results("test", vector, graph, analysis)

        graph_pos = context.index("KNOWLEDGE GRAPH CONTEXT")
        first_doc_pos = context.index("Vector doc")

        assert graph_pos < first_doc_pos, "Graph should come first when graph-dominant"

    def test_no_sandwich_with_few_vector_results(self):
        """No sandwich when fewer than 3 vector results (not enough to split)."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(2)
        graph = self._make_graph_result()
        analysis = self._make_analysis(0.7, 0.3)

        context, _ = fuse_results("test", vector, graph, analysis)

        # With only 2 vector docs, graph should come first (default order)
        graph_pos = context.index("KNOWLEDGE GRAPH CONTEXT")
        first_doc_pos = context.index("Vector doc")
        assert graph_pos < first_doc_pos

    def test_sandwich_preserves_all_documents(self):
        """All capped vector documents and graph context present after sandwich reorder."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = self._make_graph_result()
        # vector_weight=1.0 so max_vector = round(5*1.0) = 5 (no capping)
        analysis = self._make_analysis(1.0, 0.3)

        context, sources = fuse_results("test", vector, graph, analysis)

        for i in range(5):
            assert f"Vector doc {i} content" in context, f"Doc {i} missing from context"
        assert "Graph context text" in context
        doc_sources = [s for s in sources if s.startswith("doc_")]
        assert len(doc_sources) == 5

    def test_vector_only_no_sandwich(self):
        """No sandwich when graph has no context (vector-only retrieval)."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = Mock()
        graph.context = ""
        graph.entities = []
        analysis = self._make_analysis(1.0, 0.0)

        context, _ = fuse_results("test", vector, graph, analysis)

        assert "KNOWLEDGE GRAPH CONTEXT" not in context
        assert "DOCUMENT CONTEXT" in context

    def test_litm_disabled_preserves_relevance_order(self):
        """When use_lost_in_middle=False, vector docs keep relevance order."""
        from cognidoc.hybrid_retriever import fuse_results

        vector = self._make_vector_results(5)
        graph = Mock()
        graph.context = ""
        graph.entities = []
        analysis = self._make_analysis(1.0, 0.0)

        context, _ = fuse_results("test", vector, graph, analysis, use_lost_in_middle=False)

        lines = context.split("\n")
        vector_doc_lines = [l for l in lines if l.startswith("Vector doc")]
        # Should be in original relevance order: 0, 1, 2, 3, 4
        for i, line in enumerate(vector_doc_lines):
            assert line == f"Vector doc {i} content"


# =============================================================================
# 15. Persistent Retrieval Cache Tests
# =============================================================================


@dataclass
class _FakeCacheResult:
    """Picklable stand-in for HybridRetrievalResult in cache tests."""

    query: str = ""
    fused_context: str = ""
    vector_results: list = field(default_factory=list)
    graph_results: object = None
    source_chunks: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TestPersistentRetrievalCache:
    """Tests for SQLite-backed retrieval cache with semantic similarity."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        """Reset the RetrievalCache singleton before and after each test."""
        from cognidoc.hybrid_retriever import RetrievalCache

        old_instance = RetrievalCache._instance
        old_initialized = RetrievalCache._initialized
        RetrievalCache._instance = None
        RetrievalCache._initialized = False
        yield
        RetrievalCache._instance = old_instance
        RetrievalCache._initialized = old_initialized

    def _make_cache(self, tmp_path, **kwargs):
        from cognidoc.hybrid_retriever import RetrievalCache

        db_path = str(tmp_path / "test_cache.db")
        return RetrievalCache(db_path=db_path, **kwargs)

    def _make_result(self, query="test"):
        """Create a minimal picklable HybridRetrievalResult-like object."""
        return _FakeCacheResult(query=query, fused_context=f"Context for {query}")

    @patch("cognidoc.hybrid_retriever.RetrievalCache._get_query_embedding", return_value=None)
    def test_exact_match_hit(self, mock_emb, tmp_path):
        """Same query with same params returns cached result."""
        cache = self._make_cache(tmp_path)
        result = self._make_result()
        cache.put("What is RAG?", 10, True, result)

        cached = cache.get("What is RAG?", 10, True)
        assert cached is not None
        assert cached.fused_context == "Context for test"

    @patch("cognidoc.hybrid_retriever.RetrievalCache._get_query_embedding", return_value=None)
    def test_exact_match_miss_different_params(self, mock_emb, tmp_path):
        """Same query but different top_k/reranking is a miss."""
        cache = self._make_cache(tmp_path)
        result = self._make_result()
        cache.put("What is RAG?", 10, True, result)

        # Different top_k
        assert cache.get("What is RAG?", 5, True) is None
        # Different reranking
        assert cache.get("What is RAG?", 10, False) is None

    def test_semantic_match_hit(self, tmp_path):
        """Semantically similar query returns cached result via embedding match."""
        cache = self._make_cache(tmp_path)
        result = self._make_result()

        # Embeddings: near-identical vectors (cosine sim > 0.92)
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.98, 0.1, 0.05]  # very close to emb_a

        with patch.object(cache, "_get_query_embedding", side_effect=[emb_a, emb_b]):
            cache.put("What is RAG?", 10, True, result)

        with patch.object(cache, "_get_query_embedding", return_value=emb_b):
            cached = cache.get("Tell me about RAG", 10, True)

        assert cached is not None
        assert cached.fused_context == "Context for test"

    def test_semantic_match_below_threshold(self, tmp_path):
        """Dissimilar embedding below threshold returns None."""
        cache = self._make_cache(tmp_path)
        result = self._make_result()

        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]  # orthogonal, sim = 0.0

        with patch.object(cache, "_get_query_embedding", return_value=emb_a):
            cache.put("What is RAG?", 10, True, result)

        with patch.object(cache, "_get_query_embedding", return_value=emb_b):
            cached = cache.get("How does chunking work?", 10, True)

        assert cached is None

    @patch("cognidoc.hybrid_retriever.RetrievalCache._get_query_embedding", return_value=None)
    def test_ttl_expiration(self, mock_emb, tmp_path):
        """Entry older than TTL is not returned."""
        cache = self._make_cache(tmp_path, ttl_seconds=1)
        result = self._make_result()
        cache.put("What is RAG?", 10, True, result)

        time.sleep(1.1)

        assert cache.get("What is RAG?", 10, True) is None

    @patch("cognidoc.hybrid_retriever.RetrievalCache._get_query_embedding", return_value=None)
    def test_eviction_at_max_size(self, mock_emb, tmp_path):
        """Oldest entry evicted when cache is full."""
        cache = self._make_cache(tmp_path, max_size=3)

        for i in range(4):
            cache.put(f"query {i}", 10, True, self._make_result(f"q{i}"))

        # First entry should be evicted
        assert cache.get("query 0", 10, True) is None
        # Latest entries should exist
        assert cache.get("query 3", 10, True) is not None

    def test_persistence_across_instances(self, tmp_path):
        """Cache survives singleton reset (simulates process restart)."""
        from cognidoc.hybrid_retriever import RetrievalCache

        db_path = str(tmp_path / "test_cache.db")
        cache1 = RetrievalCache(db_path=db_path)
        result = self._make_result()

        with patch.object(cache1, "_get_query_embedding", return_value=None):
            cache1.put("What is RAG?", 10, True, result)

        # Reset singleton
        RetrievalCache._instance = None
        RetrievalCache._initialized = False

        cache2 = RetrievalCache(db_path=db_path)
        with patch.object(cache2, "_get_query_embedding", return_value=None):
            cached = cache2.get("What is RAG?", 10, True)

        assert cached is not None
        assert cached.fused_context == "Context for test"

    def test_embedding_failure_fallback(self, tmp_path):
        """When embedding fails, only exact match works."""
        cache = self._make_cache(tmp_path)
        result = self._make_result()

        # Store with embedding
        with patch.object(cache, "_get_query_embedding", return_value=[1.0, 0.0]):
            cache.put("What is RAG?", 10, True, result)

        # Retrieve with embedding failure — semantic match skipped
        with patch.object(cache, "_get_query_embedding", return_value=None):
            # Exact match still works
            assert cache.get("What is RAG?", 10, True) is not None
            # Semantic match can't work
            assert cache.get("Tell me about RAG", 10, True) is None

    @patch("cognidoc.hybrid_retriever.RetrievalCache._get_query_embedding", return_value=None)
    def test_clear(self, mock_emb, tmp_path):
        """Clear empties the database."""
        cache = self._make_cache(tmp_path)
        cache.put("q1", 10, True, self._make_result())
        cache.put("q2", 10, True, self._make_result())

        cache.clear()

        assert cache.get("q1", 10, True) is None
        assert cache.stats()["size"] == 0

    @patch("cognidoc.hybrid_retriever.RetrievalCache._get_query_embedding", return_value=None)
    def test_stats(self, mock_emb, tmp_path):
        """Stats returns correct hit/miss counters and size."""
        cache = self._make_cache(tmp_path)
        cache.put("q1", 10, True, self._make_result())

        cache.get("q1", 10, True)  # hit
        cache.get("q2", 10, True)  # miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
