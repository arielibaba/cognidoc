"""
Build vector and keyword indexes for RAG retrieval.

Uses direct Qdrant operations instead of LlamaIndex.
"""

import json
from pathlib import Path

from .constants import (
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    VECTOR_STORE_DIR,
    INDEX_DIR,
    CHILD_DOCUMENTS_INDEX,
    PARENT_DOCUMENTS_INDEX,
    EMBED_MODEL,
)
from typing import List, Tuple

from .utils.rag_utils import Document, VectorIndex, KeywordIndex
from .utils.logger import logger


def load_embeddings_with_documents(
    embeddings_dir: str,
    chunks_dir: str,
    docs_dir: str,
) -> Tuple[List[List[float]], List[Document], List[Document]]:
    """
    Load embeddings and associated documents from disk.

    Returns:
        Tuple of (embeddings, child_documents, parent_documents)
    """
    embeddings_path = Path(embeddings_dir)
    chunks_path = Path(chunks_dir)
    docs_path = Path(docs_dir)

    embeddings = []
    child_docs = []
    parent_docs = []

    logger.info(f"Loading embeddings from {embeddings_path}")

    for embedding_file in sorted(embeddings_path.rglob("*.json")):
        logger.debug(f"Loading {embedding_file.name}")

        with open(embedding_file, "r", encoding="utf-8") as f:
            embedding_json = json.load(f)

        # Get embedding vector
        embeddings.append(embedding_json["embedding"])

        # Get child document
        child_name = embedding_json["metadata"]["child"]
        child_doc_path = chunks_path / child_name
        with open(child_doc_path, "r", encoding="utf-8") as f:
            child_text = f.read()

        child_docs.append(
            Document(
                text=child_text,
                metadata={
                    "name": child_name,
                    "parent": embedding_json["metadata"]["parent"],
                    "source": embedding_json["metadata"]["source"],
                },
            )
        )

        # Get parent document
        parent_name = embedding_json["metadata"]["parent"]
        if "_parent_chunk" in parent_name:
            parent_doc_path = chunks_path / parent_name
        else:
            parent_doc_path = docs_path / parent_name

        with open(parent_doc_path, "r", encoding="utf-8") as f:
            parent_text = f.read()

        parent_docs.append(
            Document(
                text=parent_text,
                metadata={
                    "name": parent_name,
                    "source": embedding_json["metadata"]["source"],
                },
            )
        )

    logger.info(f"Loaded {len(embeddings)} embeddings with documents")
    return embeddings, child_docs, parent_docs


def build_indexes(recreate: bool = False) -> None:
    """
    Build child (vector) and parent (keyword) indexes.

    Args:
        recreate: If True, recreate collections even if they exist
    """
    # Ensure directories exist
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    # Load embeddings and documents
    logger.info("Loading embeddings and documents...")
    embeddings, child_docs, parent_docs = load_embeddings_with_documents(
        embeddings_dir=str(EMBEDDINGS_DIR),
        chunks_dir=str(CHUNKS_DIR),
        docs_dir=str(PROCESSED_DIR),
    )

    # Create child documents vector index
    logger.info("Creating child documents vector index...")
    child_index = VectorIndex.create(
        qdrant_path=str(VECTOR_STORE_DIR),
        collection_name="child_documents",
        embed_model=EMBED_MODEL,
        recreate=recreate,
    )
    child_index.add_documents(child_docs, embeddings=embeddings)

    # Save child index metadata
    child_index_path = Path(INDEX_DIR) / CHILD_DOCUMENTS_INDEX
    child_index.save(str(child_index_path))

    # Close the Qdrant client to release the storage folder lock
    # so that subsequent VectorIndex.load() calls can access it
    child_index.close()

    # Create parent documents keyword index
    logger.info("Creating parent documents keyword index...")
    parent_index = KeywordIndex()
    parent_index.add_documents(parent_docs)

    # Save parent index
    parent_index_path = Path(INDEX_DIR) / PARENT_DOCUMENTS_INDEX
    parent_index.save(str(parent_index_path))

    logger.info(f"Indexes built successfully and saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_indexes(recreate=True)
