"""
Helper utilities for RAG pipeline.

Contains text processing, prompt loading, and chat utilities.
No LlamaIndex dependencies - uses direct Ollama calls.
"""

import json
import re
import torch
import tiktoken
from PIL import Image, ImageStat
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ollama
import markdown
from bs4 import BeautifulSoup
from httpx import ReadTimeout

from .constants import (
    MEMORY_WINDOW,
    SYSTEM_PROMPT_REWRITE_QUERY,
    USER_PROMPT_REWRITE_QUERY,
    SYSTEM_PROMPT_EXPAND_QUERY,
    USER_PROMPT_EXPAND_QUERY,
    LLM,
)
from .utils.rag_utils import Document, KeywordIndex, stream_chat


def clear_pytorch_cache():
    """Clears MPS cache in PyTorch if available."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("\nCache cleared.\n")


def load_prompt(filepath):
    """Load a prompt from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def is_relevant_image(
    image_path: Path,
    image_pixel_threshold: int,
    image_pixel_variance_threshold: float
) -> bool:
    """
    Determines if an image is relevant based on non-white pixel count and color variance.
    """
    try:
        with Image.open(image_path) as img:
            grayscale = img.convert("L")
            histogram = grayscale.histogram()
            non_white = sum(histogram[:-1])

            if non_white <= image_pixel_threshold:
                return False

            stat = ImageStat.Stat(img)
            avg_variance = sum(stat.var) / len(stat.var)

            if avg_variance < image_pixel_variance_threshold:
                return False

            return True
    except (IOError, OSError) as e:
        print(f"Error processing {image_path}: {e}")
        return False


def markdown_to_plain_text(markdown_text: str) -> str:
    """Converts Markdown text to plain text."""
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def ask_LLM_with_JSON(
    prompt: str,
    ollama_client: ollama.Client,
    model: str,
    model_options: Dict[str, Any]
) -> str:
    """Asks the LLM to generate a JSON response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Output JSON."},
        {"role": "user", "content": prompt}
    ]
    response = ollama_client.chat(
        model=model,
        messages=messages,
        options=model_options
    )
    return response["message"]["content"]


def recover_json(json_str: str, verbose: bool = False) -> Any:
    """Attempts to recover a JSON object from a potentially malformed string."""
    if '{' not in json_str:
        return json_str

    json_str = extract_json(json_str)

    try:
        return json.loads(json_str)
    except Exception:
        try:
            return json.loads(json_str.replace("'", '"'))
        except Exception:
            print(f"JSON recovery failed for {json_str}")
            return json_str


def extract_json(s: str) -> str:
    """Extracts JSON from a markdown code block if present."""
    code = re.search(r"```json(.*?)```", s, re.DOTALL)
    return code.group(1) if code else s


def extract_markdown_tables(text: str) -> List[str]:
    """Finds all Markdown tables in the text."""
    table_pattern = (
        r'(?:^[ \t]*\|.*\r?\n)'
        r'(?:^[ \t]*\|[-\s|:]+\r?\n)'
        r'(?:^[ \t]*\|.*\r?\n?)+'
    )
    return re.findall(table_pattern, text, flags=re.MULTILINE)


def remove_markdown_tables(text: str) -> str:
    """Removes all Markdown tables from the text."""
    table_pattern = (
        r'(?:^[ \t]*\|.*\r?\n)'
        r'(?:^[ \t]*\|[-\s|:]+\r?\n)'
        r'(?:^[ \t]*\|.*\r?\n?)+'
    )
    return re.sub(table_pattern, '', text, flags=re.MULTILINE)


def remove_code_blocks(text: str) -> str:
    """Strips out any fenced code block."""
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)


def remove_mermaid_blocks(text: str) -> str:
    """Strips out any mermaid block."""
    return re.sub(r'```mermaid.*?```', '', text, flags=re.DOTALL)


def remove_extracted_text_blocks(text: str) -> str:
    """Removes any EXTRACTED TEXT sections."""
    return re.sub(r'```EXTRACTED TEXT.*?```', '', text, flags=re.DOTALL)


def clean_up_text(text: str) -> str:
    """Cleans up text by removing code blocks, mermaid diagrams, and tables."""
    text = remove_code_blocks(text)
    text = remove_mermaid_blocks(text)
    text = remove_markdown_tables(text)
    return text


def get_token_count(input_text: str) -> int:
    """Returns the number of tokens for the input text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(input_text))


def load_embeddings_with_associated_documents(
    embeddings_dir: str,
    chunks_dir: str,
    docs_dir: str
) -> Tuple[List, List[Dict], List[Dict]]:
    """
    Loads embeddings and associated documents from directories.

    Returns:
        Tuple of (embeddings, child_documents, parent_documents)
    """
    embeddings_path = Path(embeddings_dir)
    chunks_path = Path(chunks_dir)
    docs_path = Path(docs_dir)

    embeddings, child_docs, parent_docs = [], [], []

    print(f"Loading embeddings from directory: {embeddings_path} ...\n")

    for embedding_file in embeddings_path.rglob("*.json"):
        print(f"\nLoading the embedding file : {embedding_file.name}...")
        with open(embedding_file, "r", encoding="utf-8") as f:
            embedding_json = json.load(f)

        embeddings.append(embedding_json["embedding"])

        # Get child document
        child_doc_path = chunks_path / f"{embedding_json['metadata']['child']}"
        with open(child_doc_path, "r", encoding="utf-8") as f:
            child_doc_text = f.read()
        child_doc = {
            "text": child_doc_text,
            "metadata": {
                "name": embedding_json["metadata"]["child"],
                "parent": embedding_json["metadata"]["parent"],
                "source": embedding_json["metadata"]["source"]
            }
        }
        child_docs.append(child_doc)

        # Get parent document
        parent_doc_name = embedding_json["metadata"]["parent"]
        if "_parent_chunk" in parent_doc_name:
            parent_doc_path = chunks_path / f"{parent_doc_name}"
        else:
            parent_doc_path = docs_path / f"{parent_doc_name}"

        with open(parent_doc_path, "r", encoding="utf-8") as f:
            parent_doc_text = f.read()
        parent_doc = {
            "text": parent_doc_text,
            "metadata": {
                "name": parent_doc_name,
                "source": embedding_json["metadata"]["source"]
            }
        }
        parent_docs.append(parent_doc)

    return embeddings, child_docs, parent_docs


def retrieve_from_keyword_index(index: KeywordIndex, key: str, value: Any) -> List[Document]:
    """Return documents from keyword index where metadata[key] == value."""
    return index.search_by_metadata(key, value)


def limit_chat_history(history: List[dict], max_tokens: int = MEMORY_WINDOW) -> List[dict]:
    """Truncates chat history to fit within a maximum token limit."""
    total_tokens = 0
    truncated = []
    for msg in reversed(history):
        tokens = get_token_count(msg["content"])
        if total_tokens + tokens > max_tokens:
            break
        truncated.append(msg)
        total_tokens += tokens
    return list(reversed(truncated))


def run_streaming(model: str, messages: List[Dict[str, str]], temperature: float = 0.7, top_p: float = 0.85):
    """
    Run LLM in streaming mode using direct Ollama calls.

    Args:
        model: Ollama model name
        messages: List of message dicts with 'role' and 'content'
        temperature: Generation temperature
        top_p: Top-p sampling

    Yields:
        Accumulated response text
    """
    # Convert ChatMessage objects to dicts if needed
    msg_list = []
    for m in messages:
        if hasattr(m, 'role') and hasattr(m, 'content'):
            msg_list.append({"role": str(m.role.value) if hasattr(m.role, 'value') else str(m.role), "content": m.content})
        else:
            msg_list.append(m)

    yield from stream_chat(model, msg_list, temperature, top_p)


def rewrite_query(model: str, user_query: str, conversation_history_str: str = "") -> str:
    """
    Rewrites the user query based on conversation history.

    Args:
        model: Ollama model name
        user_query: The new question to rewrite
        conversation_history_str: The conversation history as a string

    Returns:
        The rewritten question
    """
    with open(SYSTEM_PROMPT_REWRITE_QUERY, "r", encoding="utf-8") as s_prompt:
        system_message = s_prompt.read()

    with open(USER_PROMPT_REWRITE_QUERY, "r", encoding="utf-8") as u_prompt:
        user_message = u_prompt.read()

    user_prompt = user_message.format(
        conversation_history=conversation_history_str,
        question=user_query
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
    )

    rewritten = response["message"]["content"].strip()
    return rewritten or f"- {user_query}"


def parse_rewritten_query(text: str) -> List[str]:
    """Extracts bullet-pointed questions from the rewritten query."""
    lines = text.splitlines()
    bullets = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            bullets.append(stripped[2:].strip())
    return bullets


def expand_query(model: str, user_query: str) -> List[str]:
    """
    Expands the user query by identifying synonyms or related terms.

    Args:
        model: Ollama model name
        user_query: The user query to expand

    Returns:
        A list of expanded queries
    """
    try:
        sub_questions = [l.strip("-* \t") for l in user_query.splitlines() if l.strip()]
        if not sub_questions:
            sub_questions = [user_query]

        all_expanded = []
        with open(SYSTEM_PROMPT_EXPAND_QUERY, "r", encoding="utf-8") as s_prompt:
            system_message = s_prompt.read()
        with open(USER_PROMPT_EXPAND_QUERY, "r", encoding="utf-8") as u_prompt:
            user_message = u_prompt.read()

        for sq in sub_questions:
            user_prompt = user_message.format(subq=sq)
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
            )
            cands = parse_expanded_queries(response["message"]["content"])
            all_expanded.extend(cands or [sq])

        return all_expanded
    except ReadTimeout:
        print("Query expansion timed out – using raw query")
        return [user_query]
    except Exception as e:
        print(f"Error during query expansion: {e}")
        return [user_query]


def parse_expanded_queries(response_text: str, max_queries: int = 5) -> List[str]:
    """Parses expanded queries from LLM response."""
    marker = "Step 3"
    start = response_text.find(marker)
    if start == -1:
        return []
    lines = response_text[start:].splitlines()
    queries = []
    capture = False
    for line in lines:
        if "Expanded Queries:" in line:
            capture = True
            continue
        if capture and line.strip().startswith("-"):
            queries.append(line.strip().lstrip("- "))
    return queries[:max_queries]


def convert_history_to_tuples(history: List[dict]) -> List[dict]:
    """Converts chat history to Gradio messages format."""
    # New Gradio expects list of dicts with 'role' and 'content'
    return [{"role": msg["role"], "content": msg["content"]} for msg in history]


def reset_conversation():
    """Resets the conversation history."""
    return [], ""
