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
import sys
from pathlib import Path


def cmd_ingest(args):
    """Handle the ingest command."""
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


def cmd_query(args):
    """Handle the query command."""
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


def cmd_serve(args):
    """Handle the serve command."""
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


def cmd_init(args):
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


def _create_default_schema(path: Path):
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


def _create_env_template(path: Path):
    """Create a .env template file."""
    template = """# CogniDoc Configuration

# API Keys (add your keys here)
GOOGLE_API_KEY=

# LLM Configuration
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash

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


def cmd_schema_generate(args):
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
    print(f"  Max pages per document: {args.max_pages}")

    schema = generate_schema_from_corpus_sync(
        sources_dir=sources_dir,
        pdf_dir=pdf_dir,
        language=args.language,
        max_docs=args.max_docs,
        max_pages=args.max_pages,
        convert_first=True,
    )

    output_path = save_schema(schema)
    print(f"\nSchema generated and saved to: {output_path}")
    print(f"  Domain: {schema.get('domain', {}).get('name', 'Unknown')}")
    print(f"  Entity types: {len(schema.get('entities', []))}")
    print(f"  Relationship types: {len(schema.get('relationships', []))}")


def cmd_info(args):
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


def main():
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
    schema_parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Maximum pages per document to extract (default: 3)",
    )
    schema_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate even if schema already exists",
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
        "schema-generate": cmd_schema_generate,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
