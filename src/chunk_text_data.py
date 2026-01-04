"""
Semantic text chunking module.

Provides semantic chunking of text documents using embeddings to detect
topic boundaries. No LangChain dependencies - uses direct Ollama calls.
"""

import re
import numpy as np
from pathlib import Path
from collections import deque
from typing import List

import ollama

from .helpers import get_token_count, clean_up_text
from .constants import (
    PROCESSED_DIR,
    CHUNKS_DIR,
    EMBED_MODEL,
    MAX_CHUNK_SIZE,
    BUFFER_SIZE,
    BREAKPOINT_THRESHOLD_TYPE,
    BREAKPOINT_THRESHOLD_AMOUNT,
    SENTENCE_SPLIT_REGEX,
    OLLAMA_URL
)


# =============================================================================
# Embedding utilities
# =============================================================================

def get_embedding(text: str, model: str) -> List[float]:
    """Get embedding vector for text using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def get_embeddings_batch(texts: List[str], model: str) -> List[List[float]]:
    """Get embeddings for multiple texts."""
    embeddings = []
    for text in texts:
        if text.strip():
            embeddings.append(get_embedding(text, model))
        else:
            # Return zero vector for empty text (will get actual dimension from first real embedding)
            if embeddings:
                embeddings.append([0.0] * len(embeddings[0]))
            else:
                embeddings.append([0.0] * 896)  # Fallback dimension
    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


# =============================================================================
# Semantic Chunker (replaces LangChain SemanticChunker)
# =============================================================================

class SemanticChunker:
    """
    Semantic text chunker using embeddings to detect topic boundaries.

    This is a from-scratch implementation replacing LangChain's SemanticChunker.

    Algorithm:
    1. Split text into sentences
    2. Create sentence groups with buffer (context window)
    3. Compute embeddings for each group
    4. Calculate cosine similarity between adjacent groups
    5. Find breakpoints where similarity drops below threshold
    6. Split text at breakpoints
    """

    def __init__(
        self,
        embed_model: str,
        buffer_size: int = 1,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 0.95,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
    ):
        """
        Initialize the semantic chunker.

        Args:
            embed_model: Ollama model name for embeddings
            buffer_size: Number of sentences to include before/after for context
            breakpoint_threshold_type: "percentile" or "standard_deviation"
            breakpoint_threshold_amount: Threshold value (0.0-1.0 for percentile)
            sentence_split_regex: Regex pattern for splitting sentences
        """
        self.embed_model = embed_model
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.sentence_split_regex = sentence_split_regex

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # First try the provided regex
        sentences = re.split(self.sentence_split_regex, text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # If regex didn't work well, fall back to simple splitting
        if len(sentences) <= 1 and len(text) > 200:
            # Try splitting on common sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_sentence_groups(self, sentences: List[str]) -> List[str]:
        """Create sentence groups with buffer context."""
        if len(sentences) <= 1:
            return sentences

        groups = []
        for i in range(len(sentences)):
            # Get buffer sentences before and after
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)

            group = " ".join(sentences[start:end])
            groups.append(group)

        return groups

    def _calculate_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Find breakpoint indices based on similarity drops.

        Returns indices where the text should be split.
        """
        if not similarities:
            return []

        similarities_arr = np.array(similarities)

        # Calculate threshold based on type
        if self.breakpoint_threshold_type == "percentile":
            # Lower similarities indicate topic changes
            # We want to find similarities below the threshold percentile
            threshold = np.percentile(
                similarities_arr,
                (1 - self.breakpoint_threshold_amount) * 100
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(similarities_arr)
            std = np.std(similarities_arr)
            threshold = mean - (self.breakpoint_threshold_amount * std)
        else:
            # Default: use percentile
            threshold = np.percentile(similarities_arr, 5)

        # Find indices where similarity is below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # +1 because we split after this sentence

        return breakpoints

    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Create sentence groups with buffer
        groups = self._create_sentence_groups(sentences)

        # Get embeddings for all groups
        try:
            embeddings = get_embeddings_batch(groups, self.embed_model)
        except Exception as e:
            print(f"Embedding failed: {e}, returning original text")
            return [text]

        # Calculate similarities between adjacent groups
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints
        breakpoints = self._calculate_breakpoints(similarities)

        # If no breakpoints found, return as single chunk
        if not breakpoints:
            return [text]

        # Split sentences at breakpoints
        chunks = []
        start = 0
        for bp in breakpoints:
            if bp > start and bp <= len(sentences):
                chunk = " ".join(sentences[start:bp])
                if chunk.strip():
                    chunks.append(chunk)
                start = bp

        # Add remaining sentences
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]


# =============================================================================
# Text splitting utilities
# =============================================================================

def hard_split(
    text: str,
    max_chunk_size: int,
    overlap: int = 0
) -> List[str]:
    """
    Fallback simple splitter that breaks a text into chunks
    of at most max_chunk_size tokens, using whitespace as delimiter.

    Args:
        text: The text to split.
        max_chunk_size: Maximum allowed size for each chunk in tokens.
        overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of text segments that are within the token limit.
    """
    words = text.split()
    chunks: List[str] = []
    current_words: List[str] = []

    for word in words:
        candidate = current_words + [word]
        if get_token_count(" ".join(candidate)) > max_chunk_size:
            chunks.append(" ".join(current_words))
            if overlap > 0:
                carry = current_words[-overlap:]
            else:
                carry = []
            current_words = carry + [word]
        else:
            current_words = candidate

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def process_semantic_split(
    texts: List[str],
    semantic_splitter: SemanticChunker,
    max_chunk_size: int
) -> List[str]:
    """
    Splits a list of texts semantically into chunks within the token limit.

    Args:
        texts: List of text segments to process.
        semantic_splitter: The semantic splitter instance.
        max_chunk_size: Maximum allowed size for each chunk in tokens.

    Returns:
        List of text segments that are within the token limit.
    """
    queue = deque(texts)
    result = []

    while queue:
        segment = queue.popleft()

        if not segment.strip():
            continue

        token_count = get_token_count(segment)
        if token_count == 0:
            continue

        if token_count > max_chunk_size:
            sub_segments = semantic_splitter.split_text(segment)

            if not sub_segments or sub_segments == [segment]:
                fallback_chunks = hard_split(
                    segment, max_chunk_size, overlap=int(max_chunk_size * 0.1)
                )
                for fc in fallback_chunks:
                    result.append(fc)
                continue

            for sub in reversed(sub_segments):
                queue.appendleft(sub)
            continue

        result.append(segment)

    return result


def semantic_chunk_text_file(
    file_path: Path,
    embed_model_name: str,
    max_chunk_size: int,
    buffer_size: int,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
    sentence_split_regex: str,
    verbose: bool,
) -> List[str]:
    """
    Processes a single text file and returns semantic chunks.

    Args:
        file_path: Path to the text file to be processed.
        embed_model_name: Name of the embedding model to use.
        max_chunk_size: Maximum allowed size for each chunk in tokens.
        buffer_size: Buffer size for the semantic chunker.
        breakpoint_threshold_type: Type of breakpoint threshold for chunking.
        breakpoint_threshold_amount: Amount for the breakpoint threshold.
        sentence_split_regex: Regular expression for splitting sentences.
        verbose: If True, prints detailed processing information.

    Returns:
        List of semantic chunks extracted from the file.
    """
    semantic_splitter = SemanticChunker(
        embed_model=embed_model_name,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        sentence_split_regex=sentence_split_regex,
    )

    with open(file_path, "r", encoding="utf-8") as f:
        input_text = f.read()

    if verbose:
        print(f"\nReading file: {file_path}")

    semantic_chunks = process_semantic_split(
        [input_text], semantic_splitter, max_chunk_size
    )

    return semantic_chunks


def chunk_text_data(
    documents_dir: str,
    embed_model_name: str,
    parent_chunk_size: int,
    child_chunk_size: int,
    documents_chunks_dir: str,
    buffer_size: int,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
    sentence_split_regex: str,
    verbose: bool,
):
    """
    Performs txt documents chunking using semantic chunking.
    Stores the resulting chunks into a dedicated folder.

    Args:
        documents_dir: Path to the directory containing text files to be chunked.
        embed_model_name: Name of the embedding model.
        parent_chunk_size: Maximum allowed size for each parent chunk in tokens.
        child_chunk_size: Maximum allowed size for each child chunk in tokens.
        documents_chunks_dir: Path to the directory where chunks will be saved.
        buffer_size: Buffer size for the semantic chunker.
        breakpoint_threshold_type: Type of breakpoint threshold for chunking.
        breakpoint_threshold_amount: Amount for the breakpoint threshold.
        sentence_split_regex: Regular expression for splitting sentences.
        verbose: If True, prints detailed processing information.
    """
    documents_path = Path(documents_dir)
    documents_chunks_path = Path(documents_chunks_dir)

    documents_path.mkdir(parents=True, exist_ok=True)
    documents_chunks_path.mkdir(parents=True, exist_ok=True)

    if child_chunk_size is None or not child_chunk_size < parent_chunk_size:
        child_chunk_size = parent_chunk_size // 8

    if verbose:
        print(f"\nProcessing the files in {documents_path}...\n")

    # Process original text files
    for file_path in documents_path.rglob("*_Text.md"):
        if file_path.is_file():
            parent_chunks = semantic_chunk_text_file(
                file_path,
                embed_model_name,
                parent_chunk_size,
                buffer_size,
                breakpoint_threshold_type,
                breakpoint_threshold_amount,
                sentence_split_regex,
                verbose,
            )
            for idx, chunk in enumerate(parent_chunks, 1):
                chunk_file_name = (
                    documents_chunks_path / f"{file_path.stem}_parent_chunk_{idx}.txt"
                )
                with open(chunk_file_name, "w", encoding="utf-8") as file:
                    file.write(chunk)
                if verbose:
                    print(f"Saved chunk to: {chunk_file_name}")

                # Create child chunks
                child_chunks = hard_split(chunk, child_chunk_size)
                for child_idx, child_chunk in enumerate(child_chunks, 1):
                    child_chunk_file_name = (
                        documents_chunks_path
                        / f"{file_path.stem}_parent_chunk_{idx}_child_chunk_{child_idx}.txt"
                    )
                    with open(child_chunk_file_name, "w", encoding="utf-8") as file:
                        file.write(child_chunk)
                    if verbose:
                        print(f"Saved child chunk to: {child_chunk_file_name}")

    # Process image description files
    for file_path in documents_path.rglob("*_description.txt"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                description = f.read()

            description = clean_up_text(description)
            chunks = hard_split(description, parent_chunk_size)

            for idx, chunk in enumerate(chunks, 1):
                chunk_file_name = (
                    documents_chunks_path
                    / f"{file_path.stem}_chunk_{idx}.txt"
                )
                with open(chunk_file_name, "w", encoding="utf-8") as file:
                    file.write(chunk)
                if verbose:
                    print(f"Saved child chunk to: {chunk_file_name}")

    if verbose:
        print(
            f"\nAll files have been processed.\n"
            f"Data chunks were saved in TXT to: {documents_chunks_dir}.\n"
        )
