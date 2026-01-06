"""
CogniDoc - Hybrid RAG Document Assistant with Vector + GraphRAG.

A document processing and retrieval pipeline that combines Vector RAG
and GraphRAG for intelligent document querying.

Example:
    from cognidoc import CogniDoc

    # Simple usage (Gemini LLM + Ollama embeddings)
    doc = CogniDoc()
    doc.ingest("./documents/")
    result = doc.query("What is the main topic?")
    print(result.answer)

    # Full cloud mode
    doc = CogniDoc(
        llm_provider="openai",
        embedding_provider="openai",
    )

    # Launch web interface
    doc.launch_ui(port=7860)
"""

__version__ = "0.1.0"

from .api import (
    CogniDoc,
    CogniDocConfig,
    QueryResult,
    IngestionResult,
)

__all__ = [
    "CogniDoc",
    "CogniDocConfig",
    "QueryResult",
    "IngestionResult",
    "__version__",
]
