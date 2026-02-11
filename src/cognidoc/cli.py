"""
CogniDoc CLI - Command-line interface for the Hybrid RAG document assistant.

Commands:
- cognidoc ingest: Ingest documents into the knowledge base
- cognidoc query: Query the knowledge base
- cognidoc serve: Launch the Gradio web interface
- cognidoc init: Initialize a new CogniDoc project
- cognidoc info: Show configuration information
- cognidoc schema-generate: Auto-generate graph schema from corpus
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List


def validate_environment(
    llm_provider: str = "gemini",
    embedding_provider: str = "ollama",
    check_data_dir: bool = False,
    data_dir: str = "./data",
) -> List[str]:
    """
    Validate that the runtime environment is properly configured.

    Returns a list of warning/error strings.  An empty list means everything
    looks good.  Errors are prefixed with ``[ERROR]``, warnings with
    ``[WARN]``.
    """
    issues: List[str] = []

    # --- API keys ---
    key_map = {
        "gemini": ("GOOGLE_API_KEY", "Set GOOGLE_API_KEY or create a .env file"),
        "openai": ("OPENAI_API_KEY", "Set OPENAI_API_KEY in your environment"),
        "anthropic": ("ANTHROPIC_API_KEY", "Set ANTHROPIC_API_KEY in your environment"),
    }
    for provider in (llm_provider, embedding_provider):
        if provider in key_map:
            env_var, hint = key_map[provider]
            if not os.environ.get(env_var):
                issues.append(f"[ERROR] {env_var} not set (required by {provider}). {hint}")

    # --- Ollama reachability ---
    if "ollama" in (llm_provider, embedding_provider):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            import urllib.request

            urllib.request.urlopen(host, timeout=2)
        except Exception:
            issues.append(
                f"[WARN] Cannot reach Ollama at {host}. " "Make sure it is running (ollama serve)."
            )

    # --- Data directory ---
    if check_data_dir:
        dp = Path(data_dir)
        if not dp.exists():
            issues.append(
                f"[WARN] Data directory {dp} does not exist. It will be created on first ingestion."
            )

    return issues


def cmd_ingest(args) -> None:
    """Handle the ingest command."""
    # Handle dry-run mode
    if args.dry_run:
        from .run_ingestion_pipeline import run_dry_run

        results = run_dry_run(args.source)
        print("\n" + "=" * 50)
        print("  DRY RUN — Pipeline Validation")
        print("=" * 50)
        print(f"\nSources: {results['sources_dir']}")
        files = results.get("files", {})
        print(f"Files found: {files.get('total', 0)}")
        for ext, count in files.get("by_extension", {}).items():
            print(f"  {ext}: {count}")
        inc = results.get("incremental", {})
        if inc.get("status") == "first_run":
            print("\nMode: First ingestion (full pipeline)")
        elif "new_files" in inc:
            print(
                f"\nMode: Incremental ({inc['new_files']} new, "
                f"{inc['modified_files']} modified)"
            )
        print(f"\nYOLO available: {results['yolo_available']}")
        print(f"Existing indexes: {results['indexes_exist']}")
        for w in results.get("warnings", []):
            print(f"\nWARNING: {w}")
        for e in results.get("errors", []):
            print(f"\nERROR: {e}")
        status = "READY" if results["valid"] else "NOT READY"
        print(f"\nStatus: {status}")
        return

    # Validate paths: reject directory traversal attempts
    source_path = Path(args.source).resolve()
    data_path = Path(args.data_dir).resolve() if args.data_dir else None
    if ".." in str(args.source) or (args.data_dir and ".." in str(args.data_dir)):
        print("[ERROR] Path contains '..', which is not allowed.", file=sys.stderr)
        sys.exit(1)

    # Validate environment before heavy imports
    issues = validate_environment(
        args.llm, args.embedding, check_data_dir=True, data_dir=args.data_dir
    )
    for issue in issues:
        print(issue, file=sys.stderr)
    if any(i.startswith("[ERROR]") for i in issues):
        sys.exit(1)

    from .api import CogniDoc

    print(f"Initializing CogniDoc with LLM={args.llm}, Embedding={args.embedding}")

    doc = CogniDoc(
        llm_provider=args.llm,
        embedding_provider=args.embedding,
        data_dir=args.data_dir,
        use_yolo=not args.no_yolo,
        use_graph=not args.no_graph,
    )

    print(f"Ingesting documents from: {args.source}")
    result = doc.ingest(
        source=args.source,
        skip_yolo=args.no_yolo,
        skip_graph=args.no_graph,
        full_reindex=args.full_reindex,
        regenerate_schema=args.regenerate_schema,
        prune=args.prune,
    )

    print(f"\nIngestion complete:")
    print(f"  Documents processed: {result.documents_processed}")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Entities extracted: {result.entities_extracted}")
    print(f"  Relationships extracted: {result.relationships_extracted}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"  - {error}")


def cmd_query(args) -> None:
    """Handle the query command."""
    issues = validate_environment(
        args.llm, args.embedding, check_data_dir=True, data_dir=args.data_dir
    )
    for issue in issues:
        print(issue, file=sys.stderr)
    if any(i.startswith("[ERROR]") for i in issues):
        sys.exit(1)

    from .api import CogniDoc

    doc = CogniDoc(
        llm_provider=args.llm,
        embedding_provider=args.embedding,
        data_dir=args.data_dir,
    )

    result = doc.query(args.question, top_k=args.top_k)

    print(f"\nAnswer:\n{result.answer}")

    if args.show_sources and result.sources:
        print(f"\nSources ({len(result.sources)}):")
        for i, source in enumerate(result.sources, 1):
            page = f" (p.{source['page']})" if source.get("page") else ""
            print(f"  [{i}] {source['source']}{page}")


def cmd_serve(args) -> None:
    """Handle the serve command."""
    issues = validate_environment(args.llm, args.embedding)
    for issue in issues:
        print(issue, file=sys.stderr)
    if any(i.startswith("[ERROR]") for i in issues):
        sys.exit(1)

    from .api import CogniDoc

    doc = CogniDoc(
        llm_provider=args.llm,
        embedding_provider=args.embedding,
        data_dir=args.data_dir,
        use_reranking=not args.no_rerank,
    )

    print(f"Launching CogniDoc UI on port {args.port}")
    if args.share:
        print("Creating public shareable link...")

    doc.launch_ui(
        port=args.port,
        share=args.share,
        no_rerank=args.no_rerank,
    )


def cmd_init(args) -> None:
    """Handle the init command."""
    import shutil

    target_dir = Path(args.directory or ".")

    print(f"Initializing CogniDoc project in {target_dir.absolute()}")

    # Create directories
    (target_dir / "data" / "sources").mkdir(parents=True, exist_ok=True)
    (target_dir / "config").mkdir(parents=True, exist_ok=True)

    # Copy schema template if requested
    if args.schema:
        schema_src = Path(__file__).parent.parent.parent / "config" / "graph_schema.yaml"
        schema_dst = target_dir / "config" / "graph_schema.yaml"

        if schema_src.exists():
            shutil.copy(schema_src, schema_dst)
            print(f"  Created: config/graph_schema.yaml")
        else:
            print(f"  Schema template not found, creating default...")
            _create_default_schema(schema_dst)

    # Copy prompts if requested
    if args.prompts:
        prompts_src = Path(__file__).parent / "prompts"
        prompts_dst = target_dir / "prompts"

        if prompts_src.exists():
            shutil.copytree(prompts_src, prompts_dst, dirs_exist_ok=True)
            print(f"  Created: prompts/")
        else:
            print(f"  Prompts directory not found")

    # Create .env template
    env_path = target_dir / ".env.example"
    if not env_path.exists():
        _create_env_template(env_path)
        print(f"  Created: .env.example")

    print("\nProject initialized! Next steps:")
    print("  1. Copy your documents to data/sources/")
    print("  2. Copy .env.example to .env and add your API keys")
    print("  3. Run: cognidoc ingest ./data/sources")
    print("  4. Run: cognidoc serve")


def _create_default_schema(path: Path) -> None:
    """Create a default graph schema file."""
    schema = """# CogniDoc Graph Schema
# Customize entity and relationship types for your domain

domain:
  name: "general"
  description: "General knowledge domain"

entity_types:
  - name: "Concept"
    description: "Abstract ideas and principles"
    examples: ["machine learning", "data processing"]

  - name: "Person"
    description: "Named individuals"
    examples: ["John Smith", "Marie Curie"]

  - name: "Organization"
    description: "Companies, institutions, groups"
    examples: ["OpenAI", "University of Paris"]

relationship_types:
  - name: "RELATED_TO"
    description: "General relationship"

  - name: "PART_OF"
    description: "Hierarchical relationship"

  - name: "CREATED_BY"
    description: "Authorship or creation"

extraction:
  confidence_threshold: 0.7
  max_entities_per_chunk: 15
  max_relationships_per_chunk: 20

graph_settings:
  entity_merge_threshold: 0.85
  community_resolution: 1.0
  max_traversal_depth: 3
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema)


def _create_env_template(path: Path) -> None:
    """Create a .env template file."""
    template = """# CogniDoc Configuration

# API Keys (add your keys here)
GOOGLE_API_KEY=

# LLM Configuration
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-3-flash-preview

# Embedding Configuration
COGNIDOC_EMBEDDING_PROVIDER=ollama
COGNIDOC_EMBEDDING_MODEL=qwen3-embedding:0.6b

# Ollama (if using local inference)
OLLAMA_HOST=http://localhost:11434

# Generation settings
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.85
"""
    path.write_text(template)


def cmd_schema_generate(args) -> None:
    """Handle the schema-generate command."""
    from .schema_wizard import (
        check_existing_schema,
        generate_schema_from_corpus_sync,
        save_schema,
    )
    from .constants import SOURCES_DIR, PDF_DIR

    sources_dir = args.source_dir or str(SOURCES_DIR)
    pdf_dir = str(PDF_DIR)

    # Check existing
    existing = check_existing_schema()
    if existing and not args.regenerate:
        print(f"Schema already exists at: {existing}")
        print("Use --regenerate to overwrite.")
        return

    print(f"Generating schema from corpus in: {sources_dir}")
    print(f"  Language: {args.language}")
    print(f"  Max documents: {args.max_docs}")

    schema = generate_schema_from_corpus_sync(
        sources_dir=sources_dir,
        pdf_dir=pdf_dir,
        language=args.language,
        max_docs=args.max_docs,
        convert_first=True,
    )

    output_path = save_schema(schema)
    print(f"\nSchema generated and saved to: {output_path}")
    print(f"  Domain: {schema.get('domain', {}).get('name', 'Unknown')}")
    print(f"  Entity types: {len(schema.get('entities', []))}")
    print(f"  Relationship types: {len(schema.get('relationships', []))}")


def cmd_migrate_graph(args) -> None:
    """Handle the migrate-graph command — convert NetworkX graph to Kùzu."""
    from .knowledge_graph import KnowledgeGraph, GRAPH_DIR, has_valid_knowledge_graph
    from .constants import KUZU_DB_DIR

    graph_path = args.graph_path or str(GRAPH_DIR)

    if not has_valid_knowledge_graph(graph_path):
        print(f"No valid knowledge graph found at: {graph_path}")
        print("Run 'cognidoc ingest' first to create a graph.")
        sys.exit(1)

    try:
        from cognidoc.graph_backend_kuzu import KUZU_AVAILABLE

        if not KUZU_AVAILABLE:
            print("Error: kuzu is not installed.")
            print("  Install it with: pip install 'cognidoc[kuzu]'")
            sys.exit(1)
    except ImportError:
        print("Error: kuzu is not installed.")
        print("  Install it with: pip install 'cognidoc[kuzu]'")
        sys.exit(1)

    print(f"Loading NetworkX graph from: {graph_path}")
    kg = KnowledgeGraph.load(graph_path)
    print(f"  Nodes: {len(kg.nodes)}")
    print(f"  Edges: {kg.number_of_edges()}")
    print(f"  Communities: {len(kg.communities)}")

    # Create Kùzu backend and import
    from .graph_backend_kuzu import KuzuBackend

    db_path = args.kuzu_path or KUZU_DB_DIR
    print(f"\nMigrating to Kùzu database at: {db_path}")

    kuzu_backend = KuzuBackend(db_path=str(db_path))

    # Export current graph as node_link and import into Kùzu
    graph_data = kg._backend.to_node_link_data()
    kuzu_backend.from_node_link_data(graph_data)

    print(f"  Migrated nodes: {kuzu_backend.number_of_nodes()}")
    print(f"  Migrated edges: {kuzu_backend.number_of_edges()}")

    print(f"\nMigration complete!")
    print(f"To use Kùzu as your graph backend, set:")
    print(f"  GRAPH_BACKEND=kuzu")
    if db_path != KUZU_DB_DIR:
        print(f"  KUZU_DB_DIR={db_path}")


def cmd_stats(args) -> None:
    """Handle the stats command - show ingestion history."""
    import json

    from .constants import INGESTION_STATS_PATH

    path = Path(INGESTION_STATS_PATH)
    if not path.exists():
        print("No ingestion stats found. Run 'cognidoc ingest' first.")
        return

    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError):
        print("Error: stats file is corrupted.")
        return

    if not history:
        print("No ingestion runs recorded.")
        return

    entries = history[-args.last :]

    print(f"Ingestion History (last {len(entries)} of {len(history)} runs)")
    print("=" * 70)

    for i, entry in enumerate(entries):
        ts = entry.get("timestamp", "?")
        timing = entry.get("timing", {})
        stats = entry.get("stats", {})

        total_time = timing.get("total_seconds", "?")
        docs = stats.get("document_conversion", {})
        embeds = stats.get("embeddings", {})
        graph = stats.get("graph_extraction", {})

        run_num = len(history) - len(entries) + i + 1
        print(f"\nRun {run_num}: {ts}")
        print(f"  Duration:  {total_time}s")
        print(f"  Docs:      {docs.get('total_files', docs.get('converted', 0))}")
        print(f"  Embedded:  {embeds.get('embedded', 0)}")
        print(f"  Entities:  {graph.get('entities_extracted', 0)}")

        if args.verbose:
            stages = timing.get("stages", {})
            if stages:
                print("  Stages:")
                for stage, dur in stages.items():
                    print(f"    {stage}: {dur}s")

    if args.json:
        print(f"\nRaw JSON at: {path}")


def cmd_info(args) -> None:
    """Handle the info command."""
    from .api import CogniDoc
    from .utils.embedding_providers import get_available_embedding_providers

    print("CogniDoc Configuration")
    print("=" * 40)

    # Show available providers
    available_embedding = get_available_embedding_providers()
    print(f"\nAvailable embedding providers:")
    for provider in available_embedding:
        print(f"  - {provider.value}")

    if not available_embedding:
        print("  (none - install cognidoc[ollama] or set API keys)")

    # Show current config if data dir exists
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        print(f"\nData directory: {data_dir.absolute()}")

        # Count documents
        pdfs = list((data_dir / "pdfs").glob("**/*.pdf")) if (data_dir / "pdfs").exists() else []
        print(f"  PDFs: {len(pdfs)}")

        # Check indexes
        indexes_dir = data_dir / "indexes"
        if indexes_dir.exists():
            print(f"  Indexes: found")
        else:
            print(f"  Indexes: not built (run cognidoc ingest)")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CogniDoc - Hybrid RAG Document Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cognidoc init --schema --prompts           Initialize a new project
  cognidoc schema-generate ./data/sources   Auto-generate graph schema
  cognidoc ingest ./documents               Ingest documents
  cognidoc query "What is the main topic?"
  cognidoc serve --port 8080                Launch web UI

For more information: https://github.com/arielibaba/cognidoc
        """,
    )

    # Global options
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Data directory (default: ./data)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest command ---
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the knowledge base",
    )
    ingest_parser.add_argument(
        "source",
        help="Path to document(s) or directory",
    )
    ingest_parser.add_argument(
        "--llm",
        default="gemini",
        help="LLM provider (gemini, openai, anthropic, ollama)",
    )
    ingest_parser.add_argument(
        "--embedding",
        default="ollama",
        help="Embedding provider (ollama, openai, gemini)",
    )
    ingest_parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO detection (use simple extraction)",
    )
    ingest_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable GraphRAG processing",
    )
    ingest_parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="Force full re-ingestion of all documents (ignore incremental manifest)",
    )
    ingest_parser.add_argument(
        "--regenerate-schema",
        action="store_true",
        help="Force graph schema regeneration even if one already exists",
    )
    ingest_parser.add_argument(
        "--prune",
        action="store_true",
        help="Detect and remove deleted source files from indexes",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup and show what would be processed without ingesting",
    )

    # --- query command ---
    query_parser = subparsers.add_parser(
        "query",
        help="Query the knowledge base",
    )
    query_parser.add_argument(
        "question",
        help="Question to ask",
    )
    query_parser.add_argument(
        "--llm",
        default="gemini",
        help="LLM provider",
    )
    query_parser.add_argument(
        "--embedding",
        default="ollama",
        help="Embedding provider",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve",
    )
    query_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source documents",
    )

    # --- serve command ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the Gradio web interface",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    serve_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    serve_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking (faster)",
    )
    serve_parser.add_argument(
        "--llm",
        default="gemini",
        help="LLM provider",
    )
    serve_parser.add_argument(
        "--embedding",
        default="ollama",
        help="Embedding provider",
    )

    # --- init command ---
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new CogniDoc project",
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to initialize (default: current)",
    )
    init_parser.add_argument(
        "--schema",
        action="store_true",
        help="Copy graph schema template",
    )
    init_parser.add_argument(
        "--prompts",
        action="store_true",
        help="Copy prompt templates",
    )

    # --- schema-generate command ---
    schema_parser = subparsers.add_parser(
        "schema-generate",
        help="Auto-generate graph schema from document corpus",
    )
    schema_parser.add_argument(
        "source_dir",
        nargs="?",
        default=None,
        help="Path to source documents (default: data/sources/)",
    )
    schema_parser.add_argument(
        "--language",
        default="en",
        help="Schema language code (default: en)",
    )
    schema_parser.add_argument(
        "--max-docs",
        type=int,
        default=100,
        help="Maximum documents to sample (default: 100)",
    )
    # --max-pages removed: character budget per document is now computed
    # adaptively based on corpus size (fewer docs → more text per doc).
    schema_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate even if schema already exists",
    )

    # --- stats command ---
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show ingestion history and statistics",
    )
    stats_parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="Number of recent runs to show (default: 5)",
    )
    stats_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-stage timing breakdown",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Show path to raw JSON file",
    )

    # --- migrate-graph command ---
    migrate_parser = subparsers.add_parser(
        "migrate-graph",
        help="Migrate knowledge graph from NetworkX to Kùzu",
    )
    migrate_parser.add_argument(
        "--graph-path",
        default=None,
        help="Path to existing knowledge graph directory (default: auto-detect)",
    )
    migrate_parser.add_argument(
        "--kuzu-path",
        default=None,
        help="Path for Kùzu database (default: data/indexes/kuzu_db)",
    )

    # --- info command ---
    info_parser = subparsers.add_parser(
        "info",
        help="Show configuration information",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to command handler
    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "serve": cmd_serve,
        "init": cmd_init,
        "info": cmd_info,
        "stats": cmd_stats,
        "schema-generate": cmd_schema_generate,
        "migrate-graph": cmd_migrate_graph,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        _print_actionable_error(e)
        sys.exit(1)


def _print_actionable_error(e: Exception) -> None:
    """Print actionable error messages for common failure modes."""
    msg = str(e)
    msg_lower = msg.lower()

    # Missing API key
    if "google_api_key" in msg_lower or "api_key" in msg_lower and "google" in msg_lower:
        print(
            "Error: GOOGLE_API_KEY not found.\n"
            "  Set it in your environment or create a .env file:\n"
            "    echo 'GOOGLE_API_KEY=your-key-here' > .env",
            file=sys.stderr,
        )
    elif "openai" in msg_lower and ("api" in msg_lower or "key" in msg_lower):
        print(
            "Error: OpenAI API key not configured.\n"
            "  Set OPENAI_API_KEY in your environment or .env file.",
            file=sys.stderr,
        )
    elif "anthropic" in msg_lower and ("api" in msg_lower or "key" in msg_lower):
        print(
            "Error: Anthropic API key not configured.\n"
            "  Set ANTHROPIC_API_KEY in your environment or .env file.",
            file=sys.stderr,
        )
    # Ollama connection issues
    elif "connection" in msg_lower and "ollama" in msg_lower or "11434" in msg:
        print(
            "Error: Cannot connect to Ollama.\n"
            "  Make sure Ollama is running: ollama serve\n"
            "  Default address: http://localhost:11434",
            file=sys.stderr,
        )
    # Missing source files
    elif "no files found" in msg_lower or "sources" in msg_lower and "not found" in msg_lower:
        print(
            "Error: No documents found to process.\n"
            "  Copy your documents to data/sources/ and try again:\n"
            "    mkdir -p data/sources && cp your-docs/* data/sources/",
            file=sys.stderr,
        )
    # Missing indexes (query before ingest)
    elif "index" in msg_lower and ("not found" in msg_lower or "not loaded" in msg_lower):
        print(
            "Error: Knowledge base not found.\n"
            "  Run ingestion first: cognidoc ingest ./data/sources",
            file=sys.stderr,
        )
    # Qdrant lock conflict
    elif "qdrant" in msg_lower and "lock" in msg_lower:
        print(
            "Error: Qdrant database is locked by another process.\n"
            "  Close any running CogniDoc instances and try again.",
            file=sys.stderr,
        )
    # Generic fallback with the original error
    else:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
