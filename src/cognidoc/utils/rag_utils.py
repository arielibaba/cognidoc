"""
RAG utilities - simple replacements for LlamaIndex abstractions.

Provides:
- Document and NodeWithScore dataclasses
- Direct Qdrant operations for vector search
- Keyword index using simple dict storage
- LLM reranking with direct Ollama calls
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .logger import logger
from .embedding_providers import get_embedding_provider


@dataclass
class Document:
    """Simple document with text and metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class NodeWithScore:
    """Document with relevance score."""
    node: Document
    score: float = 0.0


# =============================================================================
# Embedding utilities
# =============================================================================

def get_embedding(text: str, model: str = None) -> List[float]:
    """
    Get embedding vector for text.

    Uses the configured embedding provider (Ollama by default, but can be
    OpenAI or Gemini based on configuration).

    Args:
        text: Text to embed
        model: Model name (optional, uses provider default if not specified)
               Note: model parameter is kept for backwards compatibility but
               the actual model is determined by the provider configuration.

    Returns:
        Embedding vector as list of floats
    """
    provider = get_embedding_provider()
    return provider.embed_single(text)


def get_embedding_dimension(model: str = None) -> int:
    """
    Get embedding dimension.

    Uses the configured embedding provider.

    Args:
        model: Model name (optional, for backwards compatibility)

    Returns:
        Dimension of embedding vectors
    """
    provider = get_embedding_provider()
    return provider.dimension


# =============================================================================
# Vector Index (replaces VectorStoreIndex)
# =============================================================================

class VectorIndex:
    """
    Simple vector index using Qdrant directly.

    Replaces LlamaIndex's VectorStoreIndex with direct Qdrant operations.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embed_model: str,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embed_model = embed_model
        self._documents: Dict[str, Document] = {}

    @classmethod
    def create(
        cls,
        qdrant_path: str,
        collection_name: str,
        embed_model: str,
        recreate: bool = False,
    ) -> "VectorIndex":
        """Create a new vector index with Qdrant collection."""
        client = QdrantClient(path=qdrant_path)

        # Get embedding dimension
        embed_dim = get_embedding_dimension(embed_model)

        # Check if collection exists
        existing = [c.name for c in client.get_collections().collections]

        if collection_name in existing:
            if recreate:
                client.delete_collection(collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
                )
                logger.info(f"Recreated collection: {collection_name}")
            else:
                logger.info(f"Using existing collection: {collection_name}")
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {collection_name} (dim={embed_dim})")

        return cls(client, collection_name, embed_model)

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Add documents to the index.

        Args:
            documents: List of documents to add
            embeddings: Pre-computed embeddings (optional, will compute if not provided)
        """
        points = []

        for i, doc in enumerate(documents):
            # Use pre-computed embedding or compute new one
            if embeddings and i < len(embeddings):
                vector = embeddings[i]
            else:
                vector = get_embedding(doc.text, self.embed_model)

            # Store document reference
            self._documents[doc.id] = doc

            # Create Qdrant point
            points.append(PointStruct(
                id=doc.id,
                vector=vector,
                payload={
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
            ))

        # Batch upsert
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(f"Added {len(points)} documents to {self.collection_name}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[NodeWithScore]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with scores
        """
        # Get query embedding
        query_vector = get_embedding(query, self.embed_model)

        # Search Qdrant using query_points (newer API)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        # Convert to NodeWithScore
        nodes = []
        for result in results.points:
            doc = Document(
                text=result.payload.get("text", ""),
                metadata=result.payload.get("metadata", {}),
                id=str(result.id),
            )
            nodes.append(NodeWithScore(node=doc, score=result.score))

        return nodes

    def save(self, path: str) -> None:
        """Save index metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save documents mapping
        docs_data = {
            doc_id: {"text": doc.text, "metadata": doc.metadata}
            for doc_id, doc in self._documents.items()
        }

        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # Save index config
        config = {
            "collection_name": self.collection_name,
            "embed_model": self.embed_model,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved index to {path}")

    @classmethod
    def load(cls, path: str, qdrant_path: str) -> "VectorIndex":
        """Load index from disk."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create index
        client = QdrantClient(path=qdrant_path)
        index = cls(client, config["collection_name"], config["embed_model"])

        # Load documents
        docs_path = path / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
            for doc_id, data in docs_data.items():
                index._documents[doc_id] = Document(
                    text=data["text"],
                    metadata=data["metadata"],
                    id=doc_id,
                )

        logger.info(f"Loaded index from {path}")
        return index


# =============================================================================
# Keyword Index (replaces SimpleKeywordTableIndex)
# =============================================================================

class KeywordIndex:
    """
    Simple keyword-based document index.

    Stores documents and allows lookup by metadata fields.
    Replaces LlamaIndex's SimpleKeywordTableIndex.
    """

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._metadata_index: Dict[str, Dict[str, List[str]]] = {}

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        for doc in documents:
            self._documents[doc.id] = doc

            # Index by metadata fields
            for key, value in doc.metadata.items():
                if key not in self._metadata_index:
                    self._metadata_index[key] = {}

                str_value = str(value)
                if str_value not in self._metadata_index[key]:
                    self._metadata_index[key][str_value] = []

                self._metadata_index[key][str_value].append(doc.id)

        logger.info(f"Added {len(documents)} documents to keyword index")

    def search_by_metadata(self, key: str, value: Any) -> List[Document]:
        """Find documents where metadata[key] == value."""
        str_value = str(value)

        if key not in self._metadata_index:
            return []

        doc_ids = self._metadata_index[key].get(str_value, [])
        return [self._documents[doc_id] for doc_id in doc_ids if doc_id in self._documents]

    def get_all_documents(self) -> List[Document]:
        """Get all documents in the index."""
        return list(self._documents.values())

    def save(self, path: str) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save documents
        docs_data = {
            doc_id: {"text": doc.text, "metadata": doc.metadata}
            for doc_id, doc in self._documents.items()
        }

        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved keyword index to {path}")

    @classmethod
    def load(cls, path: str) -> "KeywordIndex":
        """Load index from disk."""
        path = Path(path)
        index = cls()

        docs_path = path / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)

            documents = [
                Document(text=data["text"], metadata=data["metadata"], id=doc_id)
                for doc_id, data in docs_data.items()
            ]
            index.add_documents(documents)

        logger.info(f"Loaded keyword index from {path}")
        return index


# =============================================================================
# LLM Reranking (replaces LLMRerank)
# =============================================================================

RERANK_PROMPT = """A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided.

Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score as a number from 1-10.
Do not include any documents that are not relevant to the question.

Example format:
Document 1 (score: 8): <summary of document 1>
Document 2 (score: 6): <summary of document 2>
...

Question: {query}

Documents:
{context}

Answer:"""


def rerank_documents(
    documents: List[NodeWithScore],
    query: str,
    top_n: int = 5,
    temperature: float = 0.1,
) -> List[NodeWithScore]:
    """
    Rerank documents using LLM.

    Uses the unified LLM client (Gemini by default).

    Args:
        documents: List of documents with initial scores
        query: User query for relevance assessment
        top_n: Number of top documents to return
        temperature: LLM temperature

    Returns:
        Reranked list of documents
    """
    from .llm_client import llm_chat

    if not documents:
        return []

    # Build context string
    context_parts = []
    for i, nws in enumerate(documents, 1):
        # Truncate text for context
        text = nws.node.text[:500] + "..." if len(nws.node.text) > 500 else nws.node.text
        context_parts.append(f"Document {i}: {text}")

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = RERANK_PROMPT.format(query=query, context=context)

    # Call LLM
    try:
        result_text = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        # Parse response to extract document numbers and scores
        reranked = []
        for line in result_text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("Document"):
                continue

            try:
                # Extract document number
                doc_num = int(line.split(":")[0].split()[1].strip("()"))

                # Extract score if present
                score = 5.0  # default
                if "score:" in line.lower():
                    score_part = line.lower().split("score:")[1]
                    score = float(score_part.split(")")[0].strip())

                if 1 <= doc_num <= len(documents):
                    original = documents[doc_num - 1]
                    reranked.append(NodeWithScore(node=original.node, score=score))
            except (ValueError, IndexError):
                continue

        # If parsing failed, return original order
        if not reranked:
            logger.warning("Reranking parsing failed, using original order")
            return documents[:top_n]

        # Sort by score descending and return top_n
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n]

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return documents[:top_n]


# =============================================================================
# Streaming LLM response
# =============================================================================

def stream_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
):
    """
    Stream chat response using the unified LLM client.

    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Generation temperature

    Yields:
        Accumulated response text
    """
    from .llm_client import llm_stream

    yield from llm_stream(messages, temperature)
