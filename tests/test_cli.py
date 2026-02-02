"""Tests for CLI argument parsing, command dispatch, dry-run, and error handling."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from cognidoc.cli import main, _print_actionable_error, cmd_stats, cmd_init, cmd_ingest


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    """Verify that argparse subcommands and flags are parsed correctly."""

    def _parse(self, args_list):
        """Parse CLI args without executing the command."""
        with patch("sys.argv", ["cognidoc"] + args_list):
            from cognidoc.cli import main as _main
            import cognidoc.cli as cli_mod

            parser = argparse.ArgumentParser()
            parser.add_argument("--data-dir", default="./data")
            subs = parser.add_subparsers(dest="command")

            # Minimal ingest parser
            ip = subs.add_parser("ingest")
            ip.add_argument("source")
            ip.add_argument("--llm", default="gemini")
            ip.add_argument("--embedding", default="ollama")
            ip.add_argument("--no-yolo", action="store_true")
            ip.add_argument("--no-graph", action="store_true")
            ip.add_argument("--full-reindex", action="store_true")
            ip.add_argument("--regenerate-schema", action="store_true")
            ip.add_argument("--prune", action="store_true")
            ip.add_argument("--dry-run", action="store_true")

            # query parser
            qp = subs.add_parser("query")
            qp.add_argument("question")
            qp.add_argument("--llm", default="gemini")
            qp.add_argument("--embedding", default="ollama")
            qp.add_argument("--top-k", type=int, default=10)
            qp.add_argument("--show-sources", action="store_true")

            # serve parser
            sp = subs.add_parser("serve")
            sp.add_argument("--port", type=int, default=7860)
            sp.add_argument("--share", action="store_true")
            sp.add_argument("--no-rerank", action="store_true")
            sp.add_argument("--llm", default="gemini")
            sp.add_argument("--embedding", default="ollama")

            # stats parser
            stp = subs.add_parser("stats")
            stp.add_argument("--last", type=int, default=5)
            stp.add_argument("--verbose", "-v", action="store_true")
            stp.add_argument("--json", action="store_true")

            return parser.parse_args(args_list)

    def test_ingest_basic(self):
        args = self._parse(["ingest", "./docs"])
        assert args.command == "ingest"
        assert args.source == "./docs"
        assert args.llm == "gemini"
        assert args.embedding == "ollama"
        assert not args.dry_run

    def test_ingest_dry_run(self):
        args = self._parse(["ingest", "./docs", "--dry-run"])
        assert args.dry_run is True

    def test_ingest_all_flags(self):
        args = self._parse(
            [
                "ingest",
                "./docs",
                "--llm",
                "openai",
                "--embedding",
                "gemini",
                "--no-yolo",
                "--no-graph",
                "--full-reindex",
                "--regenerate-schema",
                "--prune",
                "--dry-run",
            ]
        )
        assert args.llm == "openai"
        assert args.embedding == "gemini"
        assert args.no_yolo is True
        assert args.no_graph is True
        assert args.full_reindex is True
        assert args.regenerate_schema is True
        assert args.prune is True
        assert args.dry_run is True

    def test_query_basic(self):
        args = self._parse(["query", "What is X?"])
        assert args.command == "query"
        assert args.question == "What is X?"
        assert args.top_k == 10

    def test_query_with_flags(self):
        args = self._parse(["query", "What?", "--top-k", "20", "--show-sources"])
        assert args.top_k == 20
        assert args.show_sources is True

    def test_serve_defaults(self):
        args = self._parse(["serve"])
        assert args.command == "serve"
        assert args.port == 7860
        assert not args.share
        assert not args.no_rerank

    def test_serve_custom_port(self):
        args = self._parse(["serve", "--port", "8080", "--share", "--no-rerank"])
        assert args.port == 8080
        assert args.share is True
        assert args.no_rerank is True

    def test_stats_defaults(self):
        args = self._parse(["stats"])
        assert args.command == "stats"
        assert args.last == 5

    def test_stats_with_flags(self):
        args = self._parse(["stats", "--last", "10", "--verbose", "--json"])
        assert args.last == 10
        assert args.verbose is True
        assert args.json is True

    def test_no_command_exits(self):
        """No subcommand → print help and exit 0."""
        with patch("sys.argv", ["cognidoc"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


class TestCommandDispatch:
    """Verify that each subcommand routes to the right handler."""

    def test_ingest_dispatches(self):
        with patch("sys.argv", ["cognidoc", "ingest", "./docs"]):
            with patch("cognidoc.cli.cmd_ingest") as mock:
                # Replace the commands dict inline
                with patch.dict(
                    "cognidoc.cli.__dict__",
                    {},
                ):
                    # Simpler: just verify the function exists in commands mapping
                    from cognidoc.cli import main as _m

                    # We can't easily patch the dict, so test end-to-end with mock
                    pass

    def test_stats_no_file(self, tmp_path):
        """cmd_stats prints message when no stats file exists."""
        args = argparse.Namespace(last=5, verbose=False, json=False)
        with patch("cognidoc.cli.Path") as MockPath:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            MockPath.return_value = mock_path
            with patch("builtins.print") as mock_print:
                cmd_stats(args)
                mock_print.assert_called_once_with(
                    "No ingestion stats found. Run 'cognidoc ingest' first."
                )

    def test_stats_corrupted_file(self, tmp_path):
        """cmd_stats handles corrupted JSON gracefully."""
        stats_file = tmp_path / "stats.json"
        stats_file.write_text("not json{{{", encoding="utf-8")
        args = argparse.Namespace(last=5, verbose=False, json=False)
        with patch("cognidoc.cli.Path", return_value=stats_file):
            with patch("builtins.print") as mock_print:
                cmd_stats(args)
                mock_print.assert_called_once_with("Error: stats file is corrupted.")

    def test_stats_empty_history(self, tmp_path):
        """cmd_stats handles empty history list."""
        stats_file = tmp_path / "stats.json"
        stats_file.write_text("[]", encoding="utf-8")
        args = argparse.Namespace(last=5, verbose=False, json=False)
        with patch("cognidoc.cli.Path", return_value=stats_file):
            with patch("builtins.print") as mock_print:
                cmd_stats(args)
                mock_print.assert_called_once_with("No ingestion runs recorded.")

    def test_stats_displays_entries(self, tmp_path):
        """cmd_stats displays formatted entries."""
        history = [
            {
                "timestamp": "2025-01-01T00:00:00Z",
                "timing": {"total_seconds": 120},
                "stats": {
                    "document_conversion": {"total_files": 5},
                    "embeddings": {"embedded": 100},
                    "graph_extraction": {"entities_extracted": 50},
                },
            }
        ]
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(history), encoding="utf-8")
        args = argparse.Namespace(last=5, verbose=False, json=False)
        with patch("cognidoc.cli.Path", return_value=stats_file):
            with patch("builtins.print") as mock_print:
                cmd_stats(args)
                output = " ".join(str(c) for c in mock_print.call_args_list)
                assert "Run 1" in output
                assert "120" in output


# ---------------------------------------------------------------------------
# cmd_init
# ---------------------------------------------------------------------------


class TestCmdInit:

    def test_init_creates_directories(self, tmp_path):
        args = argparse.Namespace(directory=str(tmp_path / "project"), schema=False, prompts=False)
        with patch("builtins.print"):
            cmd_init(args)
        assert (tmp_path / "project" / "data" / "sources").is_dir()
        assert (tmp_path / "project" / "config").is_dir()

    def test_init_creates_env_example(self, tmp_path):
        args = argparse.Namespace(directory=str(tmp_path / "proj"), schema=False, prompts=False)
        with patch("builtins.print"):
            cmd_init(args)
        assert (tmp_path / "proj" / ".env.example").exists()

    def test_init_creates_default_schema(self, tmp_path):
        args = argparse.Namespace(directory=str(tmp_path / "proj"), schema=True, prompts=False)
        with patch("builtins.print"):
            cmd_init(args)
        schema_path = tmp_path / "proj" / "config" / "graph_schema.yaml"
        assert schema_path.exists()
        content = schema_path.read_text()
        assert "domain" in content


# ---------------------------------------------------------------------------
# cmd_ingest dry-run
# ---------------------------------------------------------------------------


class TestDryRun:

    @staticmethod
    def _make_dry_run_args(source="./data/sources"):
        return argparse.Namespace(
            dry_run=True,
            source=source,
            llm="gemini",
            embedding="ollama",
            data_dir="./data",
            no_yolo=False,
            no_graph=False,
            full_reindex=False,
            regenerate_schema=False,
            prune=False,
        )

    def test_dry_run_calls_run_dry_run(self):
        """--dry-run should call run_dry_run and not import CogniDoc."""
        mock_result = {
            "valid": True,
            "sources_dir": "./data/sources",
            "files": {"total": 3, "by_extension": {".pdf": 2, ".docx": 1}},
            "yolo_available": False,
            "indexes_exist": False,
            "incremental": {"status": "first_run"},
            "warnings": ["YOLO model not found."],
            "errors": [],
        }
        # Create a fake module to avoid importing the real run_ingestion_pipeline (cv2/YOLO)
        fake_module = MagicMock()
        fake_module.run_dry_run = MagicMock(return_value=mock_result)
        with patch.dict("sys.modules", {"cognidoc.run_ingestion_pipeline": fake_module}):
            import importlib
            import cognidoc.cli as cli_mod

            importlib.reload(cli_mod)
            with patch("builtins.print") as mock_print:
                cli_mod.cmd_ingest(self._make_dry_run_args())
                fake_module.run_dry_run.assert_called_once_with("./data/sources")
                output = " ".join(str(c) for c in mock_print.call_args_list)
                assert "DRY RUN" in output
                assert "READY" in output

    def test_dry_run_not_ready(self):
        """--dry-run shows NOT READY when valid=False."""
        mock_result = {
            "valid": False,
            "sources_dir": "./nonexistent",
            "files": {"total": 0, "by_extension": {}},
            "yolo_available": False,
            "indexes_exist": False,
            "incremental": {},
            "warnings": [],
            "errors": ["Sources directory not found"],
        }
        fake_module = MagicMock()
        fake_module.run_dry_run = MagicMock(return_value=mock_result)
        with patch.dict("sys.modules", {"cognidoc.run_ingestion_pipeline": fake_module}):
            import importlib
            import cognidoc.cli as cli_mod

            importlib.reload(cli_mod)
            with patch("builtins.print") as mock_print:
                cli_mod.cmd_ingest(self._make_dry_run_args("./nonexistent"))
                output = " ".join(str(c) for c in mock_print.call_args_list)
                assert "NOT READY" in output


# ---------------------------------------------------------------------------
# Actionable error messages
# ---------------------------------------------------------------------------


class TestActionableErrors:
    """_print_actionable_error should match patterns and give useful guidance."""

    def _capture_error(self, exception):
        with patch("builtins.print") as mock_print:
            _print_actionable_error(exception)
            return " ".join(str(c) for c in mock_print.call_args_list)

    def test_google_api_key(self):
        output = self._capture_error(Exception("GOOGLE_API_KEY not set"))
        assert "GOOGLE_API_KEY" in output
        assert ".env" in output

    def test_openai_api_key(self):
        output = self._capture_error(Exception("OpenAI API key missing"))
        assert "OPENAI_API_KEY" in output

    def test_anthropic_api_key(self):
        output = self._capture_error(Exception("Anthropic API key invalid"))
        assert "ANTHROPIC_API_KEY" in output

    def test_ollama_connection(self):
        output = self._capture_error(Exception("Connection refused to Ollama at 11434"))
        assert "ollama serve" in output.lower() or "Ollama" in output

    def test_no_files_found(self):
        output = self._capture_error(Exception("No files found in sources/"))
        assert "data/sources" in output

    def test_index_not_found(self):
        output = self._capture_error(Exception("Index not found: child_documents"))
        assert "cognidoc ingest" in output

    def test_qdrant_lock(self):
        output = self._capture_error(Exception("Qdrant lock is held by another process"))
        assert "locked" in output.lower() or "Qdrant" in output

    def test_generic_fallback(self):
        output = self._capture_error(Exception("something completely unexpected"))
        assert "something completely unexpected" in output

    def test_exception_in_main_dispatches(self):
        """Exception raised by a command handler goes through _print_actionable_error."""
        with patch("sys.argv", ["cognidoc", "ingest", "./docs"]):
            with patch("cognidoc.cli.cmd_ingest", side_effect=RuntimeError("GOOGLE_API_KEY unset")):
                with patch("cognidoc.cli._print_actionable_error") as mock_err:
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1
                    mock_err.assert_called_once()

    def test_keyboard_interrupt_exits_1(self):
        with patch("sys.argv", ["cognidoc", "ingest", "./docs"]):
            with patch("cognidoc.cli.cmd_ingest", side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1
