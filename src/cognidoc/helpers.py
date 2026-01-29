"""
Helper utilities for RAG pipeline.

Contains text processing, prompt loading, and chat utilities.
No LlamaIndex dependencies - uses direct Ollama calls.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Optional

import markdown
from bs4 import BeautifulSoup
from httpx import ReadTimeout

# Lazy tiktoken loading with caching
_tiktoken_encoding: Optional[Any] = None
_tiktoken_available: Optional[bool] = None

if TYPE_CHECKING:
    import ollama

from .constants import (
    MEMORY_WINDOW,
    SYSTEM_PROMPT_REWRITE_QUERY,
    USER_PROMPT_REWRITE_QUERY,
    SYSTEM_PROMPT_EXPAND_QUERY,
    USER_PROMPT_EXPAND_QUERY,
)
from .utils.rag_utils import Document, KeywordIndex
from .utils.llm_client import llm_chat, llm_stream, get_llm_client
from .utils.logger import logger


def get_memory_window() -> int:
    """
    Get the memory window size based on the current LLM's context window.

    Returns 50% of the model's context_window, or falls back to MEMORY_WINDOW constant.
    """
    try:
        client = get_llm_client()
        if client.config.context_window:
            # Use 50% of the model's context window for conversation memory
            return int(client.config.context_window * 0.5)
    except Exception:
        pass
    # Fallback to static constant
    return MEMORY_WINDOW


def clear_pytorch_cache():
    """Clears MPS cache in PyTorch if available."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.debug("PyTorch cache cleared")
    except ImportError:
        pass


def load_prompt(filepath):
    """Load a prompt from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def is_relevant_image(
    image_path: Path, image_pixel_threshold: int, image_pixel_variance_threshold: float
) -> bool:
    """
    Determines if an image is relevant based on non-white pixel count and color variance.
    """
    try:
        from PIL import Image, ImageStat

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
        logger.warning(f"Error processing image {image_path}: {e}")
        return False


def markdown_to_plain_text(markdown_text: str) -> str:
    """Converts Markdown text to plain text."""
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def ask_LLM_with_JSON(
    prompt: str, ollama_client: "ollama.Client", model: str, model_options: Dict[str, Any]
) -> str:
    """Asks the LLM to generate a JSON response using a specific Ollama client."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Output JSON."},
        {"role": "user", "content": prompt},
    ]
    response = ollama_client.chat(model=model, messages=messages, options=model_options)
    return response["message"]["content"]


def ask_llm_json_unified(prompt: str, temperature: float = 0.7) -> str:
    """
    Asks the LLM to generate a JSON response using the unified LLM client.

    Uses the default LLM provider (Gemini by default) configured via environment.
    This replaces the Ollama-specific ask_LLM_with_JSON for better quality.

    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation (default 0.7)

    Returns:
        The LLM response text
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    return llm_chat(messages, temperature=temperature, json_mode=True)


def recover_json(json_str: str, verbose: bool = False) -> Any:
    """Attempts to recover a JSON object from a potentially malformed string."""
    if "{" not in json_str:
        return json_str

    json_str = extract_json(json_str)

    try:
        return json.loads(json_str)
    except Exception:
        try:
            return json.loads(json_str.replace("'", '"'))
        except Exception:
            logger.warning(f"JSON recovery failed for: {json_str[:100]}...")
            return json_str


def extract_json(s: str) -> str:
    """Extracts JSON from a markdown code block if present."""
    code = re.search(r"```json(.*?)```", s, re.DOTALL)
    return code.group(1) if code else s


def extract_markdown_tables(text: str) -> List[str]:
    """Finds all Markdown tables in the text."""
    table_pattern = r"(?:^[ \t]*\|.*\r?\n)" r"(?:^[ \t]*\|[-\s|:]+\r?\n)" r"(?:^[ \t]*\|.*\r?\n?)+"
    return re.findall(table_pattern, text, flags=re.MULTILINE)


def remove_markdown_tables(text: str) -> str:
    """Removes all Markdown tables from the text."""
    table_pattern = r"(?:^[ \t]*\|.*\r?\n)" r"(?:^[ \t]*\|[-\s|:]+\r?\n)" r"(?:^[ \t]*\|.*\r?\n?)+"
    return re.sub(table_pattern, "", text, flags=re.MULTILINE)


def remove_code_blocks(text: str) -> str:
    """Strips out any fenced code block."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def remove_mermaid_blocks(text: str) -> str:
    """Strips out any mermaid block."""
    return re.sub(r"```mermaid.*?```", "", text, flags=re.DOTALL)


def remove_extracted_text_blocks(text: str) -> str:
    """Removes any EXTRACTED TEXT sections."""
    return re.sub(r"```EXTRACTED TEXT.*?```", "", text, flags=re.DOTALL)


def clean_up_text(text: str) -> str:
    """Cleans up text by removing code blocks, mermaid diagrams, and tables."""
    text = remove_code_blocks(text)
    text = remove_mermaid_blocks(text)
    text = remove_markdown_tables(text)
    return text


def get_token_count(input_text: str) -> int:
    """
    Returns the number of tokens for the input text.

    Uses tiktoken with cl100k_base encoding. Falls back to word-based
    estimation if tiktoken is unavailable (approximately 1.3 tokens per word).
    """
    global _tiktoken_encoding, _tiktoken_available

    # Try tiktoken first
    if _tiktoken_available is None:
        try:
            import tiktoken

            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            _tiktoken_available = True
        except Exception as e:
            logger.warning(f"tiktoken unavailable, using word-based estimation: {e}")
            _tiktoken_available = False

    if _tiktoken_available and _tiktoken_encoding is not None:
        return len(_tiktoken_encoding.encode(input_text))

    # Fallback: approximate token count based on words (1.3 tokens per word average)
    word_count = len(input_text.split())
    return int(word_count * 1.3)


def load_embeddings_with_associated_documents(
    embeddings_dir: str, chunks_dir: str, docs_dir: str
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

    logger.info(f"Loading embeddings from directory: {embeddings_path}")

    for embedding_file in embeddings_path.rglob("*.json"):
        logger.debug(f"Loading embedding file: {embedding_file.name}")
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
                "source": embedding_json["metadata"]["source"],
            },
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
            "metadata": {"name": parent_doc_name, "source": embedding_json["metadata"]["source"]},
        }
        parent_docs.append(parent_doc)

    return embeddings, child_docs, parent_docs


def retrieve_from_keyword_index(index: KeywordIndex, key: str, value: Any) -> List[Document]:
    """Return documents from keyword index where metadata[key] == value."""
    return index.search_by_metadata(key, value)


def limit_chat_history(history: List[dict], max_tokens: int = None) -> List[dict]:
    """Truncates chat history to fit within a maximum token limit.

    Uses the current LLM's context_window to calculate the limit dynamically.
    Falls back to MEMORY_WINDOW constant if no LLM is configured.
    """
    if max_tokens is None:
        max_tokens = get_memory_window()
    total_tokens = 0
    truncated = []
    for msg in reversed(history):
        content = msg.get("content", "")
        # Handle Gradio multimodal format (list) or None
        if content is None:
            content = ""
        elif isinstance(content, list):
            # Extract text from multimodal content
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            content = " ".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)

        tokens = get_token_count(content)
        if total_tokens + tokens > max_tokens:
            break
        truncated.append(msg)
        total_tokens += tokens
    return list(reversed(truncated))


def run_streaming(messages: List[Dict[str, str]], temperature: float = 0.7):
    """
    Run LLM in streaming mode using the unified LLM client.

    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Generation temperature

    Yields:
        Accumulated response text
    """
    # Convert ChatMessage objects to dicts if needed
    msg_list = []
    for m in messages:
        if hasattr(m, "role") and hasattr(m, "content"):
            msg_list.append(
                {
                    "role": str(m.role.value) if hasattr(m.role, "value") else str(m.role),
                    "content": m.content,
                }
            )
        else:
            msg_list.append(m)

    yield from llm_stream(msg_list, temperature)


def rewrite_query(user_query: str, conversation_history_str: str = "") -> str:
    """
    Rewrites the user query based on conversation history.

    Uses the unified LLM client (Gemini by default).

    Args:
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
        conversation_history=conversation_history_str, question=user_query
    )

    response = llm_chat(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    rewritten = response.strip()
    return rewritten or f"- {user_query}"


def parse_rewritten_query(text: str) -> List[str]:
    """Extracts bullet-pointed questions from the rewritten query."""
    lines = text.splitlines()
    bullets = []
    for line in lines:
        stripped = line.strip()
        # Handle both - and * bullet styles
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
        elif stripped.startswith("* "):
            bullets.append(stripped[2:].strip())
    return bullets


def expand_query(user_query: str) -> List[str]:
    """
    Expands the user query by identifying synonyms or related terms.

    Uses the unified LLM client (Gemini by default).

    Args:
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
            response = llm_chat(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            cands = parse_expanded_queries(response)
            all_expanded.extend(cands or [sq])

        return all_expanded
    except ReadTimeout:
        logger.warning("Query expansion timed out, using raw query")
        return [user_query]
    except Exception as e:
        logger.error(f"Error during query expansion: {e}")
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


def parallel_rewrite_and_classify(
    user_query: str,
    conversation_history_str: str = "",
) -> Tuple[str, Any]:
    """
    Run query rewriting and classification in parallel.

    This significantly reduces latency by overlapping two LLM calls
    instead of running them sequentially.

    Args:
        user_query: The user's question
        conversation_history_str: Formatted conversation history

    Returns:
        Tuple of (rewritten_query, routing_decision)
    """
    from .utils.llm_client import run_parallel_sync
    from .query_orchestrator import (
        CLASSIFIER_PROMPT,
        QueryType,
        RoutingDecision,
        RetrievalMode,
        WEIGHT_CONFIG,
        classify_query_rules,
    )

    # Build rewrite messages
    with open(SYSTEM_PROMPT_REWRITE_QUERY, "r", encoding="utf-8") as s_prompt:
        rewrite_system = s_prompt.read()
    with open(USER_PROMPT_REWRITE_QUERY, "r", encoding="utf-8") as u_prompt:
        rewrite_user_template = u_prompt.read()

    rewrite_user = rewrite_user_template.format(
        conversation_history=conversation_history_str, question=user_query
    )

    rewrite_messages = [
        {"role": "system", "content": rewrite_system},
        {"role": "user", "content": rewrite_user},
    ]

    # Build classify messages
    classify_messages = [{"role": "user", "content": CLASSIFIER_PROMPT.format(query=user_query)}]

    # Run both in parallel
    try:
        results = run_parallel_sync(
            [
                ("rewrite", rewrite_messages, 0.3),
                ("classify", classify_messages, 0.1),
            ]
        )

        # Parse rewrite result
        rewritten = results.get("rewrite", "").strip()
        if not rewritten:
            rewritten = f"- {user_query}"

        # Parse classify result
        classify_result = results.get("classify", "")
        query_type = QueryType.UNKNOWN
        confidence = 0.5
        reasoning = ""
        entities = []

        for line in classify_result.split("\n"):
            line = line.strip()
            if line.startswith("TYPE:"):
                type_str = line.split(":", 1)[1].strip().upper()
                try:
                    query_type = QueryType[type_str]
                except KeyError:
                    query_type = QueryType.UNKNOWN
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("ENTITIES:"):
                ent_str = line.split(":", 1)[1].strip()
                if ent_str.lower() != "none":
                    entities = [e.strip() for e in ent_str.split(",")]
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Build routing decision
        weight_cfg = WEIGHT_CONFIG.get(query_type, WEIGHT_CONFIG[QueryType.UNKNOWN])
        vector_weight = weight_cfg["vector"]
        graph_weight = weight_cfg["graph"]
        mode = weight_cfg["mode"]

        # Never skip vector completely - we need documents for references
        # Only skip if weight is very low AND it's not needed for context
        skip_vector = False  # Always include vector for document references
        skip_graph = graph_weight < 0.15

        # For exploratory queries, use hybrid mode but favor graph
        if query_type == QueryType.EXPLORATORY:
            mode = RetrievalMode.HYBRID
            vector_weight = 0.3  # Still get some vector results for references
            graph_weight = 0.7

        decision = RoutingDecision(
            query=user_query,
            query_type=query_type,
            mode=mode,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            skip_vector=skip_vector,
            skip_graph=skip_graph,
            confidence=confidence,
            reasoning=reasoning,
            entities_detected=entities,
        )

        logger.info(
            f"Parallel rewrite+classify: type={query_type.value}, "
            f"mode={mode.value}, rewritten_parts={len(rewritten.split(chr(10)))}"
        )

        return rewritten, decision

    except Exception as e:
        logger.warning(f"Parallel execution failed: {e}, falling back to sequential")
        # Fall back to sequential execution
        rewritten = rewrite_query(user_query, conversation_history_str)
        from .query_orchestrator import route_query

        decision = route_query(user_query)
        return rewritten, decision
