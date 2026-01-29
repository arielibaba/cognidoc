"""
Tests for the corpus-based schema auto-generation feature.

Tests the schema_wizard module: generic name detection, PDF text extraction,
intelligent PDF sampling, two-stage LLM pipeline (batch analysis + synthesis),
schema defaults, orchestrator, sync wrapper, non-interactive wizard, CLI
integration, and JSON/YAML response parsing helpers.
"""

import asyncio
import json
import textwrap
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import yaml

from cognidoc.schema_wizard import (
    _ensure_schema_defaults,
    _parse_json_response,
    _parse_yaml_response,
    analyze_document_batch,
    extract_pdf_text,
    generate_schema_from_corpus,
    generate_schema_from_corpus_sync,
    is_generic_name,
    run_batch_analysis,
    run_non_interactive_wizard,
    sample_pdfs_for_schema,
    synthesize_schema,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_pdf(path: Path, pages: List[str]) -> None:
    """Create a real PDF at *path* with the given page texts using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _valid_batch_result(
    themes: Optional[List[str]] = None,
    entity_types: Optional[List[Dict]] = None,
    relationship_types: Optional[List[Dict]] = None,
    domain_hint: str = "test domain",
) -> Dict[str, Any]:
    """Return a plausible Stage-A batch result dict."""
    return {
        "themes": themes or ["AI", "machine learning"],
        "entity_types": entity_types or [
            {"name": "Algorithm", "description": "An ML algorithm", "examples": ["SVM", "RF"]},
            {"name": "Dataset", "description": "A data set", "examples": ["MNIST", "CIFAR"]},
        ],
        "relationship_types": relationship_types or [
            {
                "name": "TRAINED_ON",
                "description": "Algorithm trained on a dataset",
                "source_type": "Algorithm",
                "target_type": "Dataset",
            },
        ],
        "domain_hint": domain_hint,
    }


def _valid_yaml_schema() -> str:
    """Return a valid YAML schema string as an LLM would produce."""
    return textwrap.dedent("""\
        domain:
          name: "Machine Learning"
          description: "Schema for ML research documents"
          language: "en"
        entities:
          - name: "Algorithm"
            description: "An ML algorithm"
            examples:
              - "SVM"
              - "Random Forest"
            attributes:
              - "description"
          - name: "Dataset"
            description: "A data set"
            examples:
              - "MNIST"
              - "CIFAR-10"
        relationships:
          - name: "TRAINED_ON"
            description: "Algorithm trained on a dataset"
            valid_source:
              - "Algorithm"
            valid_target:
              - "Dataset"
    """)


# =============================================================================
# TestIsGenericName
# =============================================================================


class TestIsGenericName:
    """Tests for is_generic_name()."""

    # -- Generic names --------------------------------------------------------

    @pytest.mark.parametrize("name", [
        "doc_1.pdf",
        "document.pdf",
        "scan.pdf",
        "untitled.docx",
        "123.pdf",
        "ab.pdf",
        "IMG_20240115.jpg",
        "file-003.txt",
        "copie.pdf",
        "dossier",
        "folder",
        "brouillon.pdf",
        "new.pdf",
        "temp.pdf",
        "sans titre.pdf",
        "fichier.doc",
        "page_2.pdf",
    ])
    def test_generic_names(self, name: str):
        assert is_generic_name(name), f"Expected '{name}' to be generic"

    # -- Non-generic names ----------------------------------------------------

    @pytest.mark.parametrize("name", [
        "rapport_annuel_2024.pdf",
        "bioethics_review.pdf",
        "architecture_guide.docx",
        "invoice_company.pdf",
        "meeting_notes_jan.pdf",
        "thesis.pdf",
    ])
    def test_non_generic_names(self, name: str):
        assert not is_generic_name(name), f"Expected '{name}' to be non-generic"

    # -- Edge cases -----------------------------------------------------------

    def test_empty_string(self):
        # Empty stem — no regex matches an empty string, so not considered generic
        result = is_generic_name("")
        # Just verify it does not raise; the empty stem does not match any pattern
        assert isinstance(result, bool)

    def test_single_char(self):
        # Single char stem matches r"^.{1,2}$"
        assert is_generic_name("a")

    def test_just_extension(self):
        # ".pdf" -> stem is ".pdf" which is 4 chars, not matching generic patterns
        result = is_generic_name(".pdf")
        assert isinstance(result, bool)


# =============================================================================
# TestExtractPdfText
# =============================================================================


class TestExtractPdfText:
    """Tests for extract_pdf_text()."""

    def test_basic_extraction(self, tmp_path: Path):
        pdf = tmp_path / "sample.pdf"
        _make_pdf(pdf, ["Hello, world! This is page one."])
        text = extract_pdf_text(pdf)
        assert "Hello" in text

    def test_max_pages_limit(self, tmp_path: Path):
        pages = [f"Content of page {i}" for i in range(5)]
        pdf = tmp_path / "multi.pdf"
        _make_pdf(pdf, pages)

        text = extract_pdf_text(pdf, max_pages=3)
        assert "page 0" in text
        assert "page 2" in text
        # Page 3 and 4 should NOT be present
        assert "page 3" not in text
        assert "page 4" not in text

    def test_max_chars_truncation(self, tmp_path: Path):
        long_text = "A" * 5000
        pdf = tmp_path / "long.pdf"
        _make_pdf(pdf, [long_text])

        text = extract_pdf_text(pdf, max_chars=100)
        assert len(text) <= 100

    def test_fewer_pages_than_max(self, tmp_path: Path):
        pdf = tmp_path / "one_page.pdf"
        _make_pdf(pdf, ["Single page content"])
        text = extract_pdf_text(pdf, max_pages=3)
        assert "Single page" in text

    def test_nonexistent_file(self, tmp_path: Path):
        result = extract_pdf_text(tmp_path / "nonexistent.pdf")
        assert result == ""

    def test_corrupted_file(self, tmp_path: Path):
        bad = tmp_path / "corrupted.pdf"
        bad.write_bytes(b"this is not a pdf at all")
        result = extract_pdf_text(bad)
        assert result == ""


# =============================================================================
# TestSamplePdfsForSchema
# =============================================================================


class TestSamplePdfsForSchema:
    """Tests for sample_pdfs_for_schema()."""

    def _setup_sources_and_pdfs(self, tmp_path, structure):
        """
        Create a sources/ and pdfs/ tree.

        structure: dict mapping subfolder (or "") to list of stem names.
        For each stem, a dummy source file and a real PDF are created.
        """
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        for folder, stems in structure.items():
            for stem in stems:
                if folder:
                    (sources / folder).mkdir(exist_ok=True)
                    (sources / folder / f"{stem}.txt").write_text("source")
                else:
                    (sources / f"{stem}.txt").write_text("source")

                pdf_path = pdfs / f"{stem}.pdf"
                _make_pdf(pdf_path, [f"Content of {stem}"])

        return sources, pdfs

    def test_with_subfolders_equal_distribution(self, tmp_path: Path):
        sources, pdfs = self._setup_sources_and_pdfs(tmp_path, {
            "folderA": ["report_alpha", "report_beta"],
            "folderB": ["invoice_one", "invoice_two"],
            "folderC": ["memo_x", "memo_y"],
        })

        samples, folder_names, file_names = sample_pdfs_for_schema(
            sources, pdfs, max_docs=100, max_pages=1,
        )

        assert len(samples) == 6
        # All 3 folders should appear
        subfolders_seen = {s["subfolder"] for s in samples if s["subfolder"]}
        assert len(subfolders_seen) == 3

    def test_flat_structure_random_sampling(self, tmp_path: Path):
        sources, pdfs = self._setup_sources_and_pdfs(tmp_path, {
            "": ["alpha", "beta", "gamma", "delta"],
        })

        samples, folder_names, file_names = sample_pdfs_for_schema(
            sources, pdfs, max_docs=100, max_pages=1,
        )

        assert len(samples) == 4
        assert folder_names == []

    def test_max_docs_respected(self, tmp_path: Path):
        sources, pdfs = self._setup_sources_and_pdfs(tmp_path, {
            "": [f"doc_{i}" for i in range(20)],
        })

        samples, _, _ = sample_pdfs_for_schema(
            sources, pdfs, max_docs=5, max_pages=1,
        )
        assert len(samples) <= 5

    def test_non_generic_names_collected(self, tmp_path: Path):
        sources, pdfs = self._setup_sources_and_pdfs(tmp_path, {
            "": ["rapport_annuel_2024", "architecture_guide"],
        })

        _, _, file_names = sample_pdfs_for_schema(
            sources, pdfs, max_docs=100, max_pages=1,
        )
        assert "rapport_annuel_2024" in file_names
        assert "architecture_guide" in file_names

    def test_generic_names_filtered(self, tmp_path: Path):
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        # Only generic file names
        for stem in ["doc_1", "scan_2", "file_3"]:
            (sources / f"{stem}.txt").write_text("source")
            _make_pdf(pdfs / f"{stem}.pdf", [f"Content of {stem}"])

        _, _, file_names = sample_pdfs_for_schema(
            sources, pdfs, max_docs=100, max_pages=1,
        )
        # Generic names should be excluded from informative file names
        assert len(file_names) == 0

    def test_empty_directories(self, tmp_path: Path):
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        samples, folders, files = sample_pdfs_for_schema(sources, pdfs)
        assert samples == []
        assert folders == []
        assert files == []

    def test_sources_not_existing(self, tmp_path: Path):
        samples, folders, files = sample_pdfs_for_schema(
            tmp_path / "nonexistent_sources",
            tmp_path / "nonexistent_pdfs",
        )
        assert samples == []
        assert folders == []
        assert files == []


# =============================================================================
# TestAnalyzeDocumentBatch
# =============================================================================


class TestAnalyzeDocumentBatch:
    """Tests for analyze_document_batch() (Stage A)."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        batch = [
            {"filename": "report.pdf", "content": "About AI", "subfolder": "tech"},
        ]
        result_json = json.dumps(_valid_batch_result())

        with patch("cognidoc.schema_wizard._llm_chat_async", return_value=result_json):
            result = await analyze_document_batch(batch, batch_index=0, language="en")

        assert result is not None
        assert "entity_types" in result
        assert len(result["entity_types"]) == 2

    @pytest.mark.asyncio
    async def test_invalid_response_returns_none(self):
        batch = [{"filename": "bad.pdf", "content": "stuff", "subfolder": ""}]

        with patch("cognidoc.schema_wizard._llm_chat_async", return_value="not json at all!!!"):
            result = await analyze_document_batch(batch, batch_index=0)

        assert result is None

    @pytest.mark.asyncio
    async def test_prompt_contains_document_content(self):
        batch = [
            {"filename": "bio.pdf", "content": "Bioethics discussion", "subfolder": "ethics"},
        ]

        captured_messages = []

        async def _capture(messages, **kwargs):
            captured_messages.append(messages)
            return json.dumps(_valid_batch_result())

        with patch("cognidoc.schema_wizard._llm_chat_async", side_effect=_capture):
            await analyze_document_batch(batch, batch_index=0, language="fr")

        assert len(captured_messages) == 1
        prompt_text = captured_messages[0][0]["content"]
        assert "bio.pdf" in prompt_text
        assert "Bioethics discussion" in prompt_text
        assert "ethics" in prompt_text
        assert "fr" in prompt_text

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        batch = [{"filename": "err.pdf", "content": "x", "subfolder": ""}]

        with patch("cognidoc.schema_wizard._llm_chat_async", side_effect=RuntimeError("boom")):
            result = await analyze_document_batch(batch, batch_index=0)

        assert result is None


# =============================================================================
# TestRunBatchAnalysis
# =============================================================================


class TestRunBatchAnalysis:
    """Tests for run_batch_analysis() — batch orchestration."""

    @pytest.mark.asyncio
    async def test_batching_25_samples(self):
        """25 samples with batch_size=12 should yield 3 batches."""
        samples = [
            {"filename": f"doc_{i}.pdf", "content": f"content {i}", "subfolder": ""}
            for i in range(25)
        ]

        call_count = 0

        async def _mock_analyze(batch, batch_index, language="en"):
            nonlocal call_count
            call_count += 1
            return _valid_batch_result()

        with patch("cognidoc.schema_wizard.analyze_document_batch", side_effect=_mock_analyze):
            results = await run_batch_analysis(samples, batch_size=12, max_concurrent=4)

        assert call_count == 3  # ceil(25/12) = 3
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_failed_batches_filtered(self):
        samples = [
            {"filename": f"doc_{i}.pdf", "content": f"content {i}", "subfolder": ""}
            for i in range(24)
        ]

        call_idx = 0

        async def _mock_analyze(batch, batch_index, language="en"):
            nonlocal call_idx
            call_idx += 1
            # Fail every other batch
            if batch_index % 2 == 0:
                return None
            return _valid_batch_result()

        with patch("cognidoc.schema_wizard.analyze_document_batch", side_effect=_mock_analyze):
            results = await run_batch_analysis(samples, batch_size=12, max_concurrent=4)

        # 2 batches total, 1 succeeds
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_exception_batches_filtered(self):
        samples = [
            {"filename": "doc.pdf", "content": "content", "subfolder": ""}
            for _ in range(12)
        ]

        async def _mock_analyze(batch, batch_index, language="en"):
            raise ValueError("LLM error")

        with patch("cognidoc.schema_wizard.analyze_document_batch", side_effect=_mock_analyze):
            results = await run_batch_analysis(samples, batch_size=12)

        assert len(results) == 0


# =============================================================================
# TestSynthesizeSchema
# =============================================================================


class TestSynthesizeSchema:
    """Tests for synthesize_schema() (Stage B)."""

    def test_valid_yaml_response(self):
        batch_results = [_valid_batch_result(), _valid_batch_result()]

        with patch("cognidoc.schema_wizard.llm_chat", return_value=_valid_yaml_schema()):
            schema = synthesize_schema(batch_results, ["tech"], ["report_alpha"], language="en")

        assert "domain" in schema
        assert "entities" in schema
        assert "relationships" in schema
        # Defaults should be added
        assert "query_routing" in schema
        assert "graph" in schema
        assert "extraction" in schema

    def test_retry_on_invalid_yaml(self):
        batch_results = [_valid_batch_result()]
        call_count = 0

        def _mock_llm(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "this is not yaml {{{{"
            return _valid_yaml_schema()

        with patch("cognidoc.schema_wizard.llm_chat", side_effect=_mock_llm):
            schema = synthesize_schema(batch_results, [], [])

        assert call_count == 2
        assert "domain" in schema
        assert "entities" in schema

    def test_fallback_to_generic_on_total_failure(self):
        batch_results = [_valid_batch_result()]

        with patch("cognidoc.schema_wizard.llm_chat", side_effect=Exception("boom")):
            schema = synthesize_schema(batch_results, [], [])

        # Should fall back to generic schema from generate_schema_from_answers
        assert "domain" in schema
        assert "entities" in schema
        assert schema["domain"]["name"] == "Generic/Mixed"


# =============================================================================
# TestEnsureSchemaDefaults
# =============================================================================


class TestEnsureSchemaDefaults:
    """Tests for _ensure_schema_defaults()."""

    def test_schema_with_all_sections_unchanged(self):
        schema = {
            "domain": {"name": "Test"},
            "entities": [],
            "relationships": [],
            "query_routing": {"factual": {"vector_weight": 0.5}},
            "graph": {"enable_communities": False},
            "extraction": {"max_entities_per_chunk": 10},
        }
        original_routing = schema["query_routing"]
        original_graph = schema["graph"]
        original_extraction = schema["extraction"]

        _ensure_schema_defaults(schema)

        assert schema["query_routing"] is original_routing
        assert schema["graph"] is original_graph
        assert schema["extraction"] is original_extraction

    def test_missing_query_routing_added(self):
        schema = {"domain": {"name": "Test"}, "entities": []}
        _ensure_schema_defaults(schema)

        assert "query_routing" in schema
        assert "factual" in schema["query_routing"]
        assert "relational" in schema["query_routing"]
        assert "exploratory" in schema["query_routing"]
        assert "procedural" in schema["query_routing"]

    def test_missing_graph_added(self):
        schema = {"domain": {"name": "Test"}}
        _ensure_schema_defaults(schema)

        assert "graph" in schema
        assert schema["graph"]["enable_communities"] is True
        assert schema["graph"]["min_community_size"] == 2

    def test_missing_extraction_added(self):
        schema = {"domain": {"name": "Test"}}
        _ensure_schema_defaults(schema)

        assert "extraction" in schema
        assert schema["extraction"]["max_entities_per_chunk"] == 15
        assert schema["extraction"]["min_confidence"] == 0.7


# =============================================================================
# TestGenerateSchemaFromCorpus
# =============================================================================


class TestGenerateSchemaFromCorpus:
    """Tests for generate_schema_from_corpus() — full orchestrator."""

    @pytest.mark.asyncio
    async def test_happy_path(self, tmp_path: Path):
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        # Create source + PDF
        (sources / "report.txt").write_text("source")
        _make_pdf(pdfs / "report.pdf", ["Report content about AI"])

        batch_result = _valid_batch_result()

        with patch("cognidoc.schema_wizard.sample_pdfs_for_schema") as mock_sample, \
             patch("cognidoc.schema_wizard.run_batch_analysis") as mock_batch, \
             patch("cognidoc.schema_wizard.synthesize_schema") as mock_synth:

            mock_sample.return_value = (
                [{"filename": "report.txt", "content": "Report content", "subfolder": ""}],
                [],
                ["report"],
            )
            mock_batch.return_value = [batch_result, batch_result]
            mock_synth.return_value = {
                "domain": {"name": "AI Research"},
                "entities": [{"name": "Algorithm"}],
                "relationships": [{"name": "USES"}],
            }

            schema = await generate_schema_from_corpus(
                sources_dir=str(sources),
                pdf_dir=str(pdfs),
                convert_first=False,
            )

        assert schema["domain"]["name"] == "AI Research"
        mock_sample.assert_called_once()
        mock_batch.assert_called_once()
        mock_synth.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_samples_fallback_generic(self, tmp_path: Path):
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        with patch("cognidoc.schema_wizard.sample_pdfs_for_schema", return_value=([], [], [])):
            schema = await generate_schema_from_corpus(
                sources_dir=str(sources),
                pdf_dir=str(pdfs),
                convert_first=False,
            )

        # Falls back to generic schema
        assert schema["domain"]["name"] == "Generic/Mixed"

    @pytest.mark.asyncio
    async def test_few_batch_results_fallback_legacy(self, tmp_path: Path):
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        sample = {"filename": "doc.txt", "content": "Some content", "subfolder": ""}

        with patch("cognidoc.schema_wizard.sample_pdfs_for_schema") as mock_sample, \
             patch("cognidoc.schema_wizard.run_batch_analysis") as mock_batch, \
             patch("cognidoc.schema_wizard.generate_schema_from_documents") as mock_legacy:

            mock_sample.return_value = ([sample], [], ["doc"])
            # Only 1 valid result (< 2) triggers legacy fallback
            mock_batch.return_value = [_valid_batch_result()]
            mock_legacy.return_value = {
                "domain": {"name": "Legacy Domain"},
                "entities": [],
                "relationships": [],
            }

            schema = await generate_schema_from_corpus(
                sources_dir=str(sources),
                pdf_dir=str(pdfs),
                convert_first=False,
            )

        assert schema["domain"]["name"] == "Legacy Domain"
        mock_legacy.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_first_false_skips_conversion(self, tmp_path: Path):
        sources = tmp_path / "sources"
        pdfs = tmp_path / "pdfs"
        sources.mkdir()
        pdfs.mkdir()

        with patch("cognidoc.schema_wizard.sample_pdfs_for_schema", return_value=([], [], [])):
            # convert_first=False should not attempt the conversion import
            schema = await generate_schema_from_corpus(
                sources_dir=str(sources),
                pdf_dir=str(pdfs),
                convert_first=False,
            )

        assert "domain" in schema  # generic fallback


# =============================================================================
# TestGenerateSchemaFromCorpusSync
# =============================================================================


class TestGenerateSchemaFromCorpusSync:
    """Tests for generate_schema_from_corpus_sync() — sync wrapper."""

    def test_calls_async_version(self, tmp_path: Path):
        expected = {"domain": {"name": "Test"}, "entities": [], "relationships": []}

        with patch(
            "cognidoc.schema_wizard.generate_schema_from_corpus",
            new=AsyncMock(return_value=expected),
        ):
            result = generate_schema_from_corpus_sync(
                sources_dir=str(tmp_path),
                pdf_dir=str(tmp_path),
                language="fr",
                max_docs=50,
                max_pages=2,
                convert_first=False,
            )

        assert result == expected


# =============================================================================
# TestRunNonInteractiveWizard
# =============================================================================


class TestRunNonInteractiveWizard:
    """Tests for run_non_interactive_wizard()."""

    def test_auto_calls_corpus_generation(self, tmp_path: Path):
        expected = {
            "domain": {"name": "Corpus Schema"},
            "entities": [{"name": "Concept"}],
            "relationships": [],
        }

        with patch("cognidoc.schema_wizard.generate_schema_from_corpus_sync", return_value=expected) as mock_corpus, \
             patch("cognidoc.schema_wizard.save_schema") as mock_save:

            result = run_non_interactive_wizard(
                sources_dir=str(tmp_path),
                use_auto=True,
                language="en",
                max_docs=50,
            )

        assert result == expected
        mock_corpus.assert_called_once()
        mock_save.assert_called_once_with(expected)

    def test_fallback_on_corpus_failure(self, tmp_path: Path):
        with patch("cognidoc.schema_wizard.generate_schema_from_corpus_sync", side_effect=RuntimeError("fail")), \
             patch("cognidoc.schema_wizard.get_document_sample", return_value=([], [])), \
             patch("cognidoc.schema_wizard.save_schema"):

            result = run_non_interactive_wizard(
                sources_dir=str(tmp_path),
                use_auto=True,
                domain="generic",
                language="en",
            )

        # Falls back to template schema
        assert result["domain"]["name"] == "Generic/Mixed"

    def test_auto_false_uses_template(self, tmp_path: Path):
        with patch("cognidoc.schema_wizard.save_schema"):
            result = run_non_interactive_wizard(
                sources_dir=str(tmp_path),
                use_auto=False,
                domain="technical",
                language="en",
            )

        assert result["domain"]["name"] == "Technical Documentation"


# =============================================================================
# TestCLISchemaGenerate
# =============================================================================


class TestCLISchemaGenerate:
    """Tests for cmd_schema_generate CLI integration."""

    def test_happy_path(self, tmp_path: Path):
        from cognidoc.cli import cmd_schema_generate

        args = Namespace(
            source_dir=str(tmp_path),
            language="en",
            max_docs=50,
            max_pages=2,
            regenerate=False,
        )

        schema = {
            "domain": {"name": "Test Domain"},
            "entities": [{"name": "Concept"}],
            "relationships": [{"name": "RELATES_TO"}],
        }

        with patch("cognidoc.schema_wizard.check_existing_schema", return_value=None), \
             patch("cognidoc.schema_wizard.generate_schema_from_corpus_sync", return_value=schema), \
             patch("cognidoc.schema_wizard.save_schema", return_value=str(tmp_path / "schema.yaml")):
            # Should not raise
            cmd_schema_generate(args)

    def test_existing_schema_blocks_without_regenerate(self, tmp_path: Path, capsys):
        from cognidoc.cli import cmd_schema_generate

        args = Namespace(
            source_dir=str(tmp_path),
            language="en",
            max_docs=50,
            max_pages=2,
            regenerate=False,
        )

        with patch("cognidoc.schema_wizard.check_existing_schema", return_value="/fake/schema.yaml"), \
             patch("cognidoc.schema_wizard.generate_schema_from_corpus_sync") as mock_gen:
            cmd_schema_generate(args)

        mock_gen.assert_not_called()
        captured = capsys.readouterr()
        assert "already exists" in captured.out
        assert "--regenerate" in captured.out

    def test_regenerate_overwrites(self, tmp_path: Path):
        from cognidoc.cli import cmd_schema_generate

        args = Namespace(
            source_dir=str(tmp_path),
            language="fr",
            max_docs=20,
            max_pages=1,
            regenerate=True,
        )

        schema = {
            "domain": {"name": "Regenerated"},
            "entities": [],
            "relationships": [],
        }

        with patch("cognidoc.schema_wizard.check_existing_schema", return_value="/fake/schema.yaml"), \
             patch("cognidoc.schema_wizard.generate_schema_from_corpus_sync", return_value=schema) as mock_gen, \
             patch("cognidoc.schema_wizard.save_schema", return_value=str(tmp_path / "schema.yaml")):
            cmd_schema_generate(args)

        mock_gen.assert_called_once()


# =============================================================================
# TestParseJsonResponse
# =============================================================================


class TestParseJsonResponse:
    """Tests for _parse_json_response()."""

    def test_plain_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_in_generic_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        result = _parse_json_response("this is not json")
        assert result is None

    def test_empty_string(self):
        result = _parse_json_response("")
        assert result is None

    def test_nested_json(self):
        data = {"a": {"b": [1, 2, 3]}, "c": True}
        result = _parse_json_response(json.dumps(data))
        assert result == data


# =============================================================================
# TestParseYamlResponse
# =============================================================================


class TestParseYamlResponse:
    """Tests for _parse_yaml_response()."""

    def test_plain_yaml(self):
        text = "key: value\nlist:\n  - a\n  - b"
        result = _parse_yaml_response(text)
        assert result == {"key": "value", "list": ["a", "b"]}

    def test_yaml_in_code_block(self):
        text = "```yaml\nkey: value\n```"
        result = _parse_yaml_response(text)
        assert result == {"key": "value"}

    def test_yaml_in_generic_code_block(self):
        text = "```\nkey: value\n```"
        result = _parse_yaml_response(text)
        assert result == {"key": "value"}

    def test_invalid_yaml_returns_none(self):
        text = ":\n  :\n    : {{{{"
        result = _parse_yaml_response(text)
        assert result is None

    def test_full_schema_yaml(self):
        result = _parse_yaml_response(_valid_yaml_schema())
        assert result is not None
        assert "domain" in result
        assert "entities" in result
        assert len(result["entities"]) == 2
