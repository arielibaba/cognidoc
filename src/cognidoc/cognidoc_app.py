"""
CogniDoc - Intelligent Document Assistant powered by Hybrid RAG.

Features:
- Hybrid RAG: combines Vector RAG + GraphRAG
- Multi-step retrieval: query rewriting, vector search, graph traversal, reranking
- Streaming responses
- Professional UI with retrieval mode controls
- Performance profiling

No LlamaIndex dependencies - uses direct Qdrant and Ollama calls.
"""

import argparse
import time
import unicodedata
import urllib.parse
import warnings

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .constants import (
    INDEX_DIR,
    VECTOR_STORE_DIR,
    CHILD_DOCUMENTS_INDEX,
    PARENT_DOCUMENTS_INDEX,
    TOP_K_RETRIEVED_CHILDREN,
    TOP_K_RERANKED_PARENTS,
    TOP_K_REFS,
    DEFAULT_LLM_MODEL,
    EMBED_MODEL,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    SYSTEM_PROMPT_GENERATE_FINAL_ANSWER,
    USER_PROMPT_GENERATE_FINAL_ANSWER,
    ENABLE_RERANKING,
    ENABLE_CITATION_VERIFICATION,
    PDF_DIR,
)
from .helpers import (
    clear_pytorch_cache,
    limit_chat_history,
    run_streaming,
    rewrite_query,
    parse_rewritten_query,
    convert_history_to_tuples,
    reset_conversation,
    parallel_rewrite_and_classify,
)
from .utils.rag_utils import (
    VectorIndex,
    KeywordIndex,
    NodeWithScore,
    rerank_documents,
)
from .utils.advanced_rag import verify_citations
from .hybrid_retriever import HybridRetriever, HybridRetrievalResult
from .utils.logger import logger, retrieval_metrics
from .complexity import evaluate_complexity, should_use_agent, AGENT_THRESHOLD
from .agent import CogniDocAgent, AgentState, create_agent
from .utils.metrics import QueryMetrics, get_performance_metrics

# Suppress warnings
warnings.filterwarnings("ignore")


def warmup_models_and_indexes():
    """
    Pre-load models and indexes at startup for faster first query.

    This reduces first-query latency by ~3s by:
    - Initializing the LLM client
    - Loading the embedding provider
    - Loading the hybrid retriever (vector + keyword indexes)
    """
    import time

    t_start = time.perf_counter()
    logger.info("Starting warm-up...")

    # 1. Initialize LLM client (triggers model loading)
    try:
        from .utils.llm_client import get_llm_client

        client = get_llm_client()
        logger.info(f"  LLM client ready: {client.config.provider.value}/{client.config.model}")
    except Exception as e:
        logger.warning(f"  LLM client warm-up failed: {e}")

    # 2. Initialize embedding provider
    try:
        from .utils.embedding_providers import get_embedding_provider

        provider = get_embedding_provider()
        # Trigger dimension computation (embeds "test")
        dim = provider.dimension
        logger.info(f"  Embedding provider ready: dim={dim}")
    except Exception as e:
        logger.warning(f"  Embedding provider warm-up failed: {e}")

    # 3. Load hybrid retriever (indexes)
    try:
        from .hybrid_retriever import get_hybrid_retriever

        retriever = get_hybrid_retriever()
        stats = retriever.get_statistics()
        logger.info(f"  Hybrid retriever ready: {stats}")
    except Exception as e:
        logger.warning(f"  Hybrid retriever warm-up failed: {e}")

    # 4. Warm up reranker model (Ollama keeps model in memory after first call)
    try:
        from .constants import CROSS_ENCODER_MODEL
        import httpx

        # Single dummy rerank call to load model into Ollama's memory
        httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": CROSS_ENCODER_MODEL, "prompt": "warmup"},
            timeout=30.0,
        )
        logger.info(f"  Reranker model ready: {CROSS_ENCODER_MODEL}")
    except Exception as e:
        logger.warning(f"  Reranker warm-up failed: {e}")

    t_elapsed = time.perf_counter() - t_start
    logger.info(f"Warm-up completed in {t_elapsed:.2f}s")


def detect_query_language(query: str) -> str:
    """
    Heuristic to detect query language.
    Returns 'es' for Spanish, 'de' for German, 'fr' for French, 'en' for English (default).

    Detection uses indicator counting - highest count wins (threshold: 2+).
    """
    # Spanish indicators
    spanish_indicators = [
        " es ",
        " son ",
        " que ",
        " qué ",
        " cómo ",
        " cuánto ",
        " cuántos ",
        " dónde ",
        " quién ",
        " cuál ",
        " cuáles ",
        " por qué ",
        " para ",
        " los ",
        " las ",
        " una ",
        " del ",
        " esta ",
        " este ",
        " estos ",
        " tengo ",
        " tiene ",
        " tienen ",
        " puedo ",
        " puede ",
        " hay ",
        " está ",
        " están ",
        "¿",
        " muy ",
        " también ",
        " pero ",
        " porque ",
        " esto ",
        " eso ",
        " aquí ",
        " ahí ",
        " ahora ",
        " hacer ",
        " ser ",
        " como ",
        " cuando ",
        " donde ",
        " quien ",
        " cual ",
        "ícame ",
        "ácame ",
        "éame ",
        "íame ",  # imperative + pronoun (explícame, dígame)
        "ámelo",
        "émelo",
        "ímelo",  # imperative + pronoun endings
        " documentos ",
        " datos ",
        " lista ",
        " archivo ",  # common nouns
    ]

    # German indicators
    german_indicators = [
        " ist ",
        " sind ",
        " das ",
        " der ",
        " die ",
        " den ",
        " dem ",
        " ein ",
        " eine ",
        " einer ",
        " und ",
        " oder ",
        " aber ",
        " nicht ",
        " auch ",
        " sehr ",
        " wie ",
        " was ",
        " wer ",
        " wo ",
        " wann ",
        " warum ",
        " welche ",
        " welcher ",
        " können ",
        " haben ",
        " werden ",
        " gibt ",
        " muss ",
        " kann ",
        " soll ",
        " wird ",
        " habe ",
        " hat ",
        "ß",
        "ä",
        "ö",
        "ü",
        " auf ",
        " aus ",
        " bei ",
        " mit ",
        " nach ",
        " von ",
        " zu ",
        " mir ",
        " dir ",
        " uns ",
        " euch ",
        " ihr ",
        " ihm ",
        " ihr ",
        " dokumente ",
        " datenbank ",
    ]

    # French indicators
    french_indicators = [
        " est ",
        " sont ",
        " que ",
        " qui ",
        " dans ",
        " pour ",
        " avec ",
        " les ",
        " des ",
        " une ",
        " sur ",
        " pas ",
        " plus ",
        " cette ",
        " ces ",
        " vous ",
        " nous ",
        " leur ",
        " quoi ",
        " comment ",
        " pourquoi ",
        " combien ",
        " quand ",
        "qu'",
        "d'",
        "l'",
        "n'",
        "-tu ",
        "-vous ",
        "-moi ",
        "-elle ",
        "-il ",
        " je ",
        " tu ",
    ]

    query_lower = query.lower()

    # Count indicators for each language
    spanish_count = sum(1 for ind in spanish_indicators if ind in query_lower)
    german_count = sum(1 for ind in german_indicators if ind in query_lower)
    french_count = sum(1 for ind in french_indicators if ind in query_lower)

    # Boost counts for language-specific question patterns
    spanish_questions = ["¿qué", "¿cómo", "¿cuánto", "¿dónde", "¿por qué", "¿cuál"]
    german_questions = ["wie viele", "was ist", "wo ist", "warum", "welche"]
    french_questions = ["est-ce", "qu'est", "combien", "pourquoi", "comment"]

    if any(q in query_lower for q in spanish_questions):
        spanish_count += 2
    if any(q in query_lower for q in german_questions):
        german_count += 2
    if any(q in query_lower for q in french_questions):
        french_count += 2

    # Sort by count descending, return language with highest count if >= 2
    counts = [("es", spanish_count), ("de", german_count), ("fr", french_count)]
    counts.sort(key=lambda x: x[1], reverse=True)

    if counts[0][1] >= 2:
        return counts[0][0]

    return "en"


def get_clarification_prefix(lang: str) -> str:
    """Get the clarification prefix in the appropriate language."""
    prefixes = {
        "fr": "**Clarification requise :**",
        "es": "**Se necesita aclaración:**",
        "de": "**Klärung erforderlich:**",
    }
    return prefixes.get(lang, "**Clarification needed:**")


def get_no_info_message(lang: str) -> str:
    """Get the 'no information available' message in the appropriate language."""
    messages = {
        "fr": "Je n'ai pas trouvé d'informations pertinentes dans la base documentaire pour répondre à cette question.",
        "es": "No he encontrado información relevante en la base documental para responder a esta pregunta.",
        "de": "Ich habe keine relevanten Informationen in der Dokumentenbasis gefunden, um diese Frage zu beantworten.",
    }
    return messages.get(
        lang, "I could not find relevant information in the document base to answer this question."
    )


# Prefixes used by the LLM to indicate "no information found" (for reference detection)
NO_INFO_PREFIXES = (
    "je n'ai pas trouvé d'informations",
    "i could not find relevant information",
    "no he encontrado información",
    "ich habe keine relevanten informationen",
    "no relevant details",
)


def format_sources_html(sources: list[dict]) -> str:
    """
    Format sources as a clean numbered list (Perplexity/ChatGPT style).

    Args:
        sources: List of dicts with keys: url, title, folder, page_display

    Returns:
        HTML string for the sources section
    """
    if not sources:
        return ""

    # Build numbered source list
    items_html = []
    for i, src in enumerate(sources, 1):
        url = src.get("url", "#")
        title = src.get("title", "Document")
        folder = src.get("folder", "")
        page_display = src.get("page_display", "")

        # Build display: folder/title or just title
        if folder:
            display_path = f"{folder}/{title}"
        else:
            display_path = title

        # Truncate if too long
        if len(display_path) > 80:
            display_path = display_path[:77] + "..."

        # Add page info
        page_suffix = f" · {page_display}" if page_display else ""

        item = f'<div class="source-item-wrapper"><a href="{url}" target="_blank" rel="noopener" class="source-item">[{i}] {display_path}{page_suffix}</a></div>'
        items_html.append(item)

    # Join with newlines to ensure one source per line
    sources_list = "\n".join(items_html)

    # Use a spacer div with explicit height for guaranteed visual separation
    # This ensures the gap before sources is always larger than paragraph gaps
    html = f"""<div class="sources-spacer"></div>
<div class="sources-section">
<div class="sources-divider"></div>
<div class="sources-title">Sources</div>
<div class="sources-list">
{sources_list}
</div>
</div>"""

    return html


# Custom CSS for professional styling
CUSTOM_CSS = """
/* Global styles */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.header-title {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    color: #a0aec0;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

.header-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Chat container */
.chat-container {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}

/* Settings panel */
.settings-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
}

.settings-title {
    color: #1a202c;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.settings-divider {
    border-top: 1px solid #e2e8f0;
    margin: 1rem 0;
}

/* Toggle cards */
.toggle-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.toggle-card:hover {
    border-color: #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
}

/* Info cards */
.info-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 100%);
    border: 1px solid #c3dafe;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}

.info-card-title {
    color: #3730a3;
    font-weight: 600;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
}

.info-card-text {
    color: #4a5568;
    font-size: 0.8rem;
    line-height: 1.5;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}

.secondary-btn:hover {
    background: #e2e8f0 !important;
}

/* Status indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}

.status-active {
    background: #dcfce7;
    color: #166534;
}

.status-inactive {
    background: #fef3c7;
    color: #92400e;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1rem;
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 1rem;
}

/* Chatbot styling */
.chatbot {
    border-radius: 12px !important;
}

/* Enhanced message styling */
.chatbot .message {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

.chatbot .message.bot {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border: 1px solid #e2e8f0 !important;
}

.chatbot .message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* References section styling */
.chatbot .message hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1rem 0;
}

.chatbot .message strong {
    color: #1a202c;
    font-weight: 600;
}

.chatbot .message a {
    color: #4f46e5 !important;
    text-decoration: none !important;
    font-weight: 500;
    transition: all 0.2s ease;
}

.chatbot .message a:hover {
    color: #7c3aed !important;
    text-decoration: underline !important;
}

/* Code and markdown enhancements */
.chatbot .message code {
    background: #f1f5f9;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
    color: #e11d48;
}

.chatbot .message pre {
    background: #1e293b;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
}

.chatbot .message pre code {
    background: transparent;
    color: inherit;
    padding: 0;
}

/* List styling in responses */
.chatbot .message ul, .chatbot .message ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

.chatbot .message li {
    margin: 0.25rem 0;
}

/* Typing indicator */
.chatbot .typing-indicator {
    display: flex;
    gap: 4px;
    padding: 8px 12px;
}

.chatbot .typing-indicator span {
    width: 8px;
    height: 8px;
    background: #94a3b8;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
}

@keyframes typing {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .header-title { font-size: 1.5rem; }
    .header-badge { display: none; }
}

/* Dashboard styles */
.metrics-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.metrics-card-title {
    color: #1a202c;
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

.metrics-value {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
}

.metrics-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.25rem;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.dashboard-section {
    margin-bottom: 2rem;
}

.dashboard-section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a202c;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #667eea;
}

/* =====================================================
   DARK MODE STYLES
   ===================================================== */

/* Dark mode toggle button */
.theme-toggle {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e2e8f0;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.theme-toggle:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.3);
}

.theme-toggle .icon {
    font-size: 1.1rem;
}

/* Dark mode overrides */
body.dark-mode {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #334155;
}

body.dark-mode .gradio-container {
    background: var(--bg-primary) !important;
}

body.dark-mode .gr-panel,
body.dark-mode .gr-box,
body.dark-mode .gr-form {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

body.dark-mode .settings-title {
    color: var(--text-primary) !important;
}

body.dark-mode .toggle-card {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

body.dark-mode .toggle-card:hover {
    border-color: #667eea !important;
}

body.dark-mode .info-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border-color: #334155 !important;
}

body.dark-mode .info-card-title {
    color: #a5b4fc !important;
}

body.dark-mode .info-card-text {
    color: var(--text-secondary) !important;
}

body.dark-mode .secondary-btn {
    background: var(--bg-tertiary) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

body.dark-mode .secondary-btn:hover {
    background: #475569 !important;
}

body.dark-mode .chat-container {
    border-color: var(--border-color) !important;
}

body.dark-mode .chatbot {
    background: var(--bg-secondary) !important;
}

body.dark-mode .chatbot .message.bot {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

body.dark-mode .chatbot .message strong {
    color: var(--text-primary) !important;
}

body.dark-mode .chatbot .message hr {
    border-color: var(--border-color) !important;
}

body.dark-mode .chatbot .message code {
    background: var(--bg-tertiary) !important;
    color: #f472b6 !important;
}

body.dark-mode .chatbot .message a {
    color: #818cf8 !important;
}

body.dark-mode .chatbot .message a:hover {
    color: #a5b4fc !important;
}

body.dark-mode input,
body.dark-mode textarea {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

body.dark-mode input::placeholder,
body.dark-mode textarea::placeholder {
    color: var(--text-secondary) !important;
}

body.dark-mode .metrics-card {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

body.dark-mode .metrics-card-title {
    color: var(--text-primary) !important;
}

body.dark-mode .metrics-label {
    color: var(--text-secondary) !important;
}

body.dark-mode .dashboard-section-title {
    color: var(--text-primary) !important;
}

body.dark-mode .footer {
    color: var(--text-secondary) !important;
}

body.dark-mode label,
body.dark-mode .gr-check-radio label {
    color: var(--text-primary) !important;
}

body.dark-mode .gr-check-radio {
    color: var(--text-secondary) !important;
}

/* Tab styling for dark mode */
body.dark-mode .tabs {
    background: var(--bg-secondary) !important;
}

body.dark-mode .tab-nav button {
    color: var(--text-secondary) !important;
}

body.dark-mode .tab-nav button.selected {
    color: var(--text-primary) !important;
    background: var(--bg-tertiary) !important;
}

/* Dropdown and select styling */
body.dark-mode select,
body.dark-mode .gr-dropdown {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* Table styling for dark mode */
body.dark-mode table {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

body.dark-mode th {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

body.dark-mode td {
    border-color: var(--border-color) !important;
}

/* Sources section styling - Clean numbered list */
/* Spacer to guarantee larger gap before sources than between paragraphs */
/* Paragraphs typically have ~1em margin, so we use 3em+ for clear separation */
.sources-spacer {
    display: block;
    height: 3em;
    min-height: 3em;
}

.sources-section {
    margin-top: 0;
    padding-top: 1em;
}

.sources-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #e2e8f0 20%, #e2e8f0 80%, transparent);
    margin-bottom: 1rem;
}

.sources-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.sources-list {
    display: block;
}

.source-item-wrapper {
    display: block;
    margin-bottom: 0.35rem;
}

.source-item {
    display: inline;
    font-size: 0.85rem;
    color: #475569;
    text-decoration: none;
    transition: color 0.15s ease;
    line-height: 1.6;
}

.source-item:hover {
    color: #667eea;
    text-decoration: none;
}

/* Dark mode for sources */
body.dark-mode .sources-divider {
    background: linear-gradient(to right, transparent, var(--border-color) 20%, var(--border-color) 80%, transparent);
}

body.dark-mode .sources-title {
    color: #94a3b8;
}

body.dark-mode .source-item {
    color: #94a3b8;
}

body.dark-mode .source-item:hover {
    color: #a5b4fc;
}
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CogniDoc - Intelligent Document Assistant")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking (faster but may reduce quality)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Disable agentic RAG for complex queries (faster, simpler)",
    )
    return parser.parse_args()


# Clear GPU cache
clear_pytorch_cache()

# Load hybrid retriever (includes vector index, keyword index, and knowledge graph)
logger.info("Loading hybrid retriever...")
hybrid_retriever = HybridRetriever()
hybrid_status = hybrid_retriever.load()
logger.info(f"Hybrid retriever status: {hybrid_status}")

# Get references to indexes from hybrid retriever for fallback
child_index = hybrid_retriever._vector_index
parent_index = hybrid_retriever._keyword_index

if child_index:
    logger.info("Indexes loaded successfully via hybrid retriever")
else:
    logger.warning("Vector index not available - some features may be limited")

# Initialize agentic RAG (for complex queries)
cognidoc_agent: CogniDocAgent = None
ENABLE_AGENTIC_RAG = True  # Can be disabled via CLI or env


def get_agent() -> CogniDocAgent:
    """Lazy initialization of the agent."""
    global cognidoc_agent
    if cognidoc_agent is None and ENABLE_AGENTIC_RAG:
        cognidoc_agent = create_agent(
            retriever=hybrid_retriever,
            max_steps=7,
            temperature=0.3,
        )
        logger.info("CogniDocAgent initialized for complex queries")
    return cognidoc_agent


def chat_conversation(
    user_message: str,
    history: list,
    enable_reranking: bool = True,
    enable_graph: bool = True,
    enable_agent: bool = True,
):
    """
    Main chat conversation handler with hybrid RAG.

    Supports two paths:
    - Fast path: Standard RAG pipeline for simple queries
    - Agent path: Multi-step reasoning for complex queries

    Args:
        user_message: User's input message
        history: Conversation history
        enable_reranking: Whether to use LLM reranking
        enable_graph: Whether to use GraphRAG (knowledge graph)
        enable_agent: Whether to use agentic RAG for complex queries

    Yields:
        Updated conversation history
    """
    t0 = time.perf_counter()

    # Normalize history
    if history is None:
        history = []
    elif history and isinstance(history[0], (list, tuple)):
        flat = []
        for u, a in history:
            flat += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        history = flat

    history.append({"role": "user", "content": user_message})
    history = limit_chat_history(
        history
    )  # Uses dynamic memory window based on LLM's context_window
    conv_history = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in history)

    # Parallel query rewriting and classification (uses unified LLM client)
    try:
        t1 = time.perf_counter()
        rewritten, routing_decision = parallel_rewrite_and_classify(user_message, conv_history)
        t2 = time.perf_counter()
        logger.info(f"Rewritten query ({len(rewritten.split(chr(10)))} parts):\n{rewritten}")
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        history.append(
            {
                "role": "assistant",
                "content": "I apologize, but the service is temporarily unavailable. Please try again in a moment.",
            }
        )
        yield convert_history_to_tuples(history)
        return

    candidates = parse_rewritten_query(rewritten)
    if not candidates:
        candidates = [user_message]

    # ==========================================================================
    # AGENTIC PATH: Evaluate complexity and route to agent if needed
    # ==========================================================================
    if enable_agent and ENABLE_AGENTIC_RAG:
        use_agent, complexity = should_use_agent(
            query=user_message,
            routing=routing_decision,
            rewritten_query=rewritten,
        )

        if use_agent:
            logger.info(
                f"Agent path triggered: complexity={complexity.score:.2f}, "
                f"level={complexity.level.value}, reason={complexity.reasoning}"
            )

            agent = get_agent()
            query_lang = detect_query_language(user_message)

            if agent:
                try:
                    # Run agent with streaming feedback
                    history.append({"role": "assistant", "content": ""})

                    # Use the rewritten query which includes conversation context
                    # This allows the agent to understand references like "cite-les-moi"
                    # when the previous message mentioned documents
                    agent_query = candidates[0] if candidates else user_message

                    # Stream agent progress and capture result
                    # Note: run_streaming is a generator that returns AgentResult
                    # We need to capture the return value via StopIteration.value
                    result = None
                    streaming_gen = agent.run_streaming(agent_query, complexity)
                    progress_lines = []

                    try:
                        while True:
                            state, message = next(streaming_gen)
                            if state == AgentState.FINISHED:
                                # Final answer will come from result
                                progress_lines.append("*Finalizing answer...*")
                            elif state == AgentState.NEEDS_CLARIFICATION:
                                # Agent needs clarification - use language-appropriate prefix
                                prefix = get_clarification_prefix(query_lang)
                                history[-1]["content"] = f"{prefix} {message}"
                                yield convert_history_to_tuples(history)
                                return
                            elif state == AgentState.ERROR:
                                logger.error(f"Agent error: {message}")
                                # Fall through to standard path
                                break
                            else:
                                # Progress update - show in UI
                                state_emoji = {
                                    AgentState.THINKING: "🤔",
                                    AgentState.ACTING: "⚡",
                                    AgentState.OBSERVING: "👁️",
                                    AgentState.REFLECTING: "💭",
                                }.get(state, "•")
                                progress_lines.append(f"{state_emoji} {message}")
                                # Show progress (last 5 lines) in UI
                                progress_display = "\n".join(progress_lines[-5:])
                                history[-1][
                                    "content"
                                ] = f"*Processing query...*\n\n{progress_display}"
                                yield convert_history_to_tuples(history)
                                logger.debug(f"Agent [{state.value}]: {message[:100]}")
                    except StopIteration as e:
                        # Generator returned - capture the AgentResult
                        result = e.value

                    if result and result.success:
                        t_end = time.perf_counter()
                        total_time = t_end - t0

                        # Format answer with agent metadata
                        answer = result.answer
                        if result.metadata.get("forced_conclusion"):
                            answer += "\n\n*Note: Response based on available information.*"

                        history[-1]["content"] = answer

                        logger.info(
                            f"Agent completed: {len(result.steps)} steps, "
                            f"tools={result.metadata.get('tools_used', [])}, "
                            f"total_time={total_time:.2f}s"
                        )

                        # Log metrics for agent path
                        cache_stats = get_performance_metrics().get_session_cache_stats()
                        agent_metrics = QueryMetrics(
                            path="agent",
                            query_type=(
                                str(routing_decision.query_type)
                                if routing_decision and routing_decision.query_type
                                else None
                            ),
                            complexity_score=complexity.score if complexity else None,
                            total_time_ms=total_time * 1000,
                            rewrite_time_ms=(t2 - t1) * 1000,
                            cache_hits=cache_stats["hits"],
                            cache_misses=cache_stats["misses"],
                            agent_steps=len(result.steps),
                            tools_used=result.metadata.get("tools_used", []),
                        )
                        get_performance_metrics().log_query(user_message, agent_metrics)
                        get_performance_metrics().reset_session_cache_stats()

                        yield convert_history_to_tuples(history)
                        return
                    else:
                        # Agent didn't return successful result - fall through to standard path
                        logger.warning(
                            f"Agent returned no result or failed, falling back to standard path"
                        )
                        history.pop()

                except Exception as e:
                    logger.error(f"Agent execution failed: {e}, falling back to standard path")
                    # Remove the empty assistant message and fall through
                    history.pop()

    # ==========================================================================
    # FAST PATH: Standard RAG pipeline
    # ==========================================================================

    # Show processing indicator immediately for better perceived performance
    # Extended streaming prefetch: show retrieval mode for better user feedback
    if enable_graph and hybrid_retriever.is_loaded():
        search_mode = (
            "vector + graph" if routing_decision and not routing_decision.skip_graph else "vector"
        )
        history.append(
            {"role": "assistant", "content": f"*🔍 Searching knowledge base ({search_mode})...*"}
        )
    else:
        history.append({"role": "assistant", "content": "*🔍 Searching knowledge base...*"})
    yield convert_history_to_tuples(history)

    # Retrieval (hybrid or vector-only)
    t3 = time.perf_counter()
    graph_context = ""
    graph_time = 0

    if enable_graph and hybrid_retriever.is_loaded():
        # Use hybrid retrieval (vector + graph)
        logger.info("Using hybrid retrieval (Vector + Graph)")
        combo_query = user_message + " | " + " | ".join(candidates)

        # Temporarily disable graph in config if not enabled
        if not enable_graph:
            hybrid_retriever.config.routing.strategy = "vector_only"

        # MODERATE path: increase retrieval depth for mid-complexity queries
        effective_top_k = TOP_K_RETRIEVED_CHILDREN
        if complexity and 0.35 <= complexity.score < 0.55:
            effective_top_k = round(TOP_K_RETRIEVED_CHILDREN * 1.5)
            logger.info(f"MODERATE path: top_k {TOP_K_RETRIEVED_CHILDREN} → {effective_top_k}")

        hybrid_result = hybrid_retriever.retrieve(
            query=combo_query,
            top_k=effective_top_k,
            use_reranking=enable_reranking,
            model=DEFAULT_LLM_MODEL,
            pre_computed_routing=routing_decision,  # Pass pre-computed routing
        )

        # Extract results from hybrid retrieval
        reranked = hybrid_result.vector_results
        graph_context = hybrid_result.graph_results.context if hybrid_result.graph_results else ""

        t4 = time.perf_counter()
        retrieval_time = t4 - t3
        rerank_time = 0  # Already included in hybrid retrieval

        logger.info(
            f"Hybrid retrieval: {len(reranked)} documents, "
            f"graph entities: {hybrid_result.metadata.get('graph_entities', 0)}, "
            f"query type: {hybrid_result.metadata.get('query_type', 'unknown')}"
        )

    else:
        # Fallback to vector-only retrieval
        logger.info("Using vector-only retrieval")
        effective_top_k = TOP_K_RETRIEVED_CHILDREN
        if complexity and 0.35 <= complexity.score < 0.55:
            effective_top_k = round(TOP_K_RETRIEVED_CHILDREN * 1.5)
            logger.info(f"MODERATE path: top_k {TOP_K_RETRIEVED_CHILDREN} → {effective_top_k}")

        retrieved = []
        for q in candidates:
            results = child_index.search(q, top_k=effective_top_k)
            retrieved.extend(results)

        # Get parent documents
        parents = []
        for nws in retrieved:
            parent_name = nws.node.metadata.get("parent")
            parent_docs = parent_index.search_by_metadata("name", parent_name)
            if parent_docs:
                parents.append(parent_docs[0])

        # Deduplicate parents
        seen = set()
        unique = []
        for p in parents:
            name = p.metadata.get("name")
            if name not in seen:
                seen.add(name)
                unique.append(p)

        t4 = time.perf_counter()
        retrieval_time = t4 - t3

        # Reranking (optional)
        rerank_time = 0
        if enable_reranking and unique:
            t_rerank_start = time.perf_counter()
            nws_list = [NodeWithScore(node=p, score=0.0) for p in unique]
            combo = user_message + " | " + " | ".join(candidates)
            reranked = rerank_documents(
                documents=nws_list,
                query=combo,
                top_n=TOP_K_RERANKED_PARENTS,
            )
            t_rerank_end = time.perf_counter()
            rerank_time = t_rerank_end - t_rerank_start
            logger.info(
                f"Reranking: {len(unique)} -> {len(reranked)} parents in {rerank_time:.2f}s"
            )
        else:
            # No reranking: use top-k from unique
            reranked = [NodeWithScore(node=p, score=0.0) for p in unique[:TOP_K_RERANKED_PARENTS]]
            if not enable_reranking:
                logger.info(f"Reranking disabled, using top-{len(reranked)} parents")

    # Log retrieval metrics
    retrieval_metrics.log_retrieval(
        query=user_message,
        num_retrieved=len(reranked),
        num_after_rerank=len(reranked),
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
    )

    # Update progress indicator - retrieval done, now generating
    history[-1]["content"] = f"*✨ Found {len(reranked)} relevant sources, generating answer...*"
    yield convert_history_to_tuples(history)

    # Build context (combine graph context with document context)
    doc_context = "\n".join(n.node.text for n in reranked)
    if graph_context and enable_graph:
        context = f"{graph_context}\n\n{doc_context}"
    else:
        context = doc_context

    # Group pages by document for cleaner references
    from collections import OrderedDict

    doc_pages = OrderedDict()  # Preserve order of first appearance
    for nws in reranked:
        source = nws.node.metadata.get("source", {})
        if isinstance(source, dict):
            doc = source.get("document", "Unknown")
            page = source.get("page", "?")
        else:
            doc = str(source)
            page = "?"
        if doc not in doc_pages:
            doc_pages[doc] = set()
        if page != "?":
            try:
                doc_pages[doc].add(int(page))
            except (ValueError, TypeError):
                # Page might be a string like "5 2" - extract first number
                import re

                match = re.search(r"\d+", str(page))
                if match:
                    doc_pages[doc].add(int(match.group()))

    # Format references with consolidated page numbers for styled display
    sources = []
    for doc, pages in doc_pages.items():
        # Create PDF file path - URL encoded filename
        # doc may contain path separators for subdirectories (e.g., "projet_A/doc")
        # Normalize to NFC to ensure consistent Unicode representation
        # (macOS often uses NFD, but URLs and most systems expect NFC)
        doc_normalized = unicodedata.normalize("NFC", doc)
        pdf_filename = f"{doc_normalized}.pdf"
        # Use safe='/' to preserve path separators in the URL
        encoded_filename = urllib.parse.quote(pdf_filename, safe="/")
        # Use /pdfs/ path for FastAPI static file serving
        base_url = f"/pdfs/{encoded_filename}"

        # Extract folder and filename from doc path (use normalized version)
        if "/" in doc_normalized:
            parts = doc_normalized.rsplit("/", 1)
            folder = parts[0]
            title = parts[1]
        else:
            folder = ""
            title = doc_normalized

        # Build page display and URL with page anchor
        if not pages:
            page_display = ""
            url = base_url
        elif len(pages) == 1:
            page_num = list(pages)[0]
            page_display = f"p. {page_num}"
            url = f"{base_url}#page={page_num}"
        else:
            sorted_pages = sorted(p for p in pages if isinstance(p, int))
            if sorted_pages:
                first_page = min(sorted_pages)
                if sorted_pages == list(range(first_page, max(sorted_pages) + 1)):
                    page_display = f"pp. {first_page}-{max(sorted_pages)}"
                else:
                    page_display = f"pp. {', '.join(map(str, sorted_pages[:3]))}" + (
                        "..." if len(sorted_pages) > 3 else ""
                    )
                url = f"{base_url}#page={first_page}"
            else:
                page_display = f"pp. {', '.join(map(str, list(pages)[:3]))}"
                url = base_url

        sources.append({"url": url, "title": title, "folder": folder, "page_display": page_display})

        if len(sources) >= TOP_K_REFS:
            break

    # Load prompts
    with open(SYSTEM_PROMPT_GENERATE_FINAL_ANSWER, "r", encoding="utf-8") as f:
        system_msg = f.read()

    with open(USER_PROMPT_GENERATE_FINAL_ANSWER, "r", encoding="utf-8") as f:
        user_msg_template = f.read()

    user_prompt = user_msg_template.format(
        conversation_history=conv_history, user_question=user_message, refined_context=context
    )

    msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}]

    t5 = time.perf_counter()

    # Stream response (reuse the existing assistant message from progress indicator)
    for chunk in run_streaming(msgs, TEMPERATURE_GENERATION):
        history[-1]["content"] = chunk
        yield convert_history_to_tuples(history)

    t6 = time.perf_counter()

    # Add references (only if the response is not a "no info" message)
    final = history[-1]["content"].strip()
    is_no_info = any(final.lower().startswith(prefix) for prefix in NO_INFO_PREFIXES)
    if not is_no_info and sources:
        confidence = hybrid_result.metadata.get("answer_confidence", 0)
        confidence_html = ""
        if confidence > 0:
            confidence_pct = round(confidence * 100)
            if confidence_pct >= 70:
                conf_color = "#4caf50"
            elif confidence_pct >= 40:
                conf_color = "#ff9800"
            else:
                conf_color = "#f44336"
            confidence_html = (
                f'<div style="font-size:0.85em;color:{conf_color};margin-top:4px;">'
                f"Confidence: {confidence_pct}%</div>"
            )
        final += "\n\n" + format_sources_html(sources) + confidence_html

    # #15: Citation verification (optional)
    if ENABLE_CITATION_VERIFICATION and reranked:
        try:
            all_supported, claim_results, summary = verify_citations(
                answer=final,
                documents=[nws.node for nws in reranked],
                model=DEFAULT_LLM_MODEL,
            )
            if not all_supported:
                unsupported = [c.claim for c in claim_results if not c.supported]
                if unsupported:
                    final += "\n\n---\n**Note:** Some claims may not be fully supported by the source documents."
                    logger.warning(f"Citation verification: {len(unsupported)} unsupported claims")
            else:
                logger.debug("Citation verification: all claims supported")
        except Exception as e:
            logger.warning(f"Citation verification failed: {e}")

    history[-1]["content"] = final

    # Log performance
    logger.info(
        f"""
    Performance Stats:
    - Query Rewriting:      {t2 - t1:.2f}s
    - Retrieval:            {retrieval_time:.2f}s
    - Reranking:            {rerank_time:.2f}s ({"enabled" if enable_reranking else "disabled"})
    - Prompt Construction:  {t5 - t4:.2f}s
    - LLM Inference:        {t6 - t5:.2f}s
    - TOTAL:                {t6 - t0:.2f}s
    """
    )

    # Log metrics for fast/enhanced path
    path_type = "enhanced" if complexity and complexity.score >= 0.35 else "fast"
    fast_metrics = QueryMetrics(
        path=path_type,
        query_type=(
            str(routing_decision.query_type)
            if routing_decision and routing_decision.query_type
            else None
        ),
        complexity_score=complexity.score if complexity else None,
        total_time_ms=(t6 - t0) * 1000,
        rewrite_time_ms=(t2 - t1) * 1000,
        retrieval_time_ms=retrieval_time * 1000,
        rerank_time_ms=rerank_time * 1000,
        llm_time_ms=(t6 - t5) * 1000,
    )
    get_performance_metrics().log_query(user_message, fast_metrics)

    yield convert_history_to_tuples(history)


# =============================================================================
# Dashboard Helper Functions
# =============================================================================


def create_latency_by_path_chart():
    """Create bar chart of average latency by path."""
    stats = get_performance_metrics().get_global_stats()
    path_stats = stats.get("path_distribution", {})

    if not path_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        fig.update_layout(height=300)
        return fig

    paths = list(path_stats.keys())
    latencies = [path_stats[p]["avg_latency_ms"] for p in paths]
    counts = [path_stats[p]["count"] for p in paths]

    colors = {"fast": "#22c55e", "enhanced": "#eab308", "agent": "#ef4444"}
    bar_colors = [colors.get(p, "#667eea") for p in paths]

    fig = go.Figure(
        data=[
            go.Bar(
                x=paths,
                y=latencies,
                text=[f"{l:.0f}ms" for l in latencies],
                textposition="outside",
                marker_color=bar_colors,
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.0f}ms<br>Count: %{customdata}<extra></extra>",
                customdata=counts,
            )
        ]
    )
    fig.update_layout(
        title="Average Latency by Path",
        xaxis_title="Path",
        yaxis_title="Latency (ms)",
        height=300,
        margin=dict(t=50, b=50, l=50, r=20),
    )
    return fig


def create_latency_over_time_chart():
    """Create line chart of latency over time."""
    data = get_performance_metrics().get_latency_over_time(limit=50)

    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        fig.update_layout(height=300)
        return fig

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    colors = {"fast": "#22c55e", "enhanced": "#eab308", "agent": "#ef4444"}

    fig = go.Figure()
    for path in df["path"].unique():
        path_df = df[df["path"] == path]
        fig.add_trace(
            go.Scatter(
                x=path_df["timestamp"],
                y=path_df["total_time_ms"],
                mode="lines+markers",
                name=path,
                line=dict(color=colors.get(path, "#667eea")),
                hovertemplate="<b>%{x}</b><br>%{y:.0f}ms<extra></extra>",
            )
        )

    fig.update_layout(
        title="Latency Over Time",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        height=300,
        margin=dict(t=50, b=50, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_path_distribution_chart():
    """Create pie chart of query path distribution."""
    distribution = get_performance_metrics().get_path_distribution()

    if not distribution:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        fig.update_layout(height=300)
        return fig

    colors = {"fast": "#22c55e", "enhanced": "#eab308", "agent": "#ef4444"}

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(distribution.keys()),
                values=list(distribution.values()),
                marker_colors=[colors.get(p, "#667eea") for p in distribution.keys()],
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Query Distribution by Path",
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def get_recent_queries_dataframe():
    """Get recent queries as a DataFrame for display."""
    queries = get_performance_metrics().get_recent_queries(limit=20)

    if not queries:
        return pd.DataFrame(columns=["Time", "Path", "Type", "Latency (ms)", "Cache Hits", "Steps"])

    df = pd.DataFrame(queries)
    df = df.rename(
        columns={
            "timestamp": "Time",
            "path": "Path",
            "query_type": "Type",
            "latency_ms": "Latency (ms)",
            "cache_hits": "Cache Hits",
            "agent_steps": "Steps",
        }
    )
    return df[["Time", "Path", "Type", "Latency (ms)", "Cache Hits", "Steps"]]


def get_global_stats_html():
    """Generate HTML for global stats display."""
    stats = get_performance_metrics().get_global_stats()

    return f"""
    <div class="metrics-grid">
        <div class="metrics-card">
            <div class="metrics-value">{stats.get('total_queries', 0)}</div>
            <div class="metrics-label">Total Queries</div>
        </div>
        <div class="metrics-card">
            <div class="metrics-value">{stats.get('avg_latency_ms', 0):.0f}ms</div>
            <div class="metrics-label">Avg Latency</div>
        </div>
        <div class="metrics-card">
            <div class="metrics-value">{stats.get('cache_hit_rate', 0):.1f}%</div>
            <div class="metrics-label">Cache Hit Rate</div>
        </div>
        <div class="metrics-card">
            <div class="metrics-value">{stats.get('avg_agent_steps', 0):.1f}</div>
            <div class="metrics-label">Avg Agent Steps</div>
        </div>
    </div>
    """


def create_fastapi_app(demo: gr.Blocks) -> "FastAPI":
    """
    Wrap a Gradio app in a FastAPI app with PDF serving endpoint.

    Registers /pdfs/{path} BEFORE Gradio's catch-all at "/" so that
    source document links work correctly.
    """
    from pathlib import Path as _Path
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    app = FastAPI()

    _pdf_dir = _Path(PDF_DIR).resolve()
    logger.info(f"PDF serving endpoint registered at /pdfs from {_pdf_dir}")

    @app.get("/pdfs/{file_path:path}")
    async def serve_pdf(file_path: str):
        decoded = urllib.parse.unquote(file_path)
        # Normalize Unicode (macOS uses NFD, URLs typically use NFC)
        decoded_nfc = unicodedata.normalize("NFC", decoded)
        decoded_nfd = unicodedata.normalize("NFD", decoded)
        for candidate in (decoded, decoded_nfc, decoded_nfd):
            full_path = (_pdf_dir / candidate).resolve()
            # Path traversal protection
            if not str(full_path).startswith(str(_pdf_dir)):
                raise HTTPException(status_code=403)
            if full_path.is_file():
                return FileResponse(full_path, media_type="application/pdf")
        raise HTTPException(status_code=404, detail=f"PDF not found: {decoded}")

    # Mount Gradio AFTER explicit routes
    app = gr.mount_gradio_app(app, demo, path="/")
    return app


def create_gradio_app(default_reranking: bool = True):
    """
    Create the Gradio application with professional UI.

    Args:
        default_reranking: Default value for reranking toggle

    Returns:
        Gradio Blocks app
    """
    with gr.Blocks(
        title="CogniDoc - Intelligent Document Assistant",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # Header with dark mode toggle
        gr.HTML(
            """
        <div class="header-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="header-title">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                            <path d="M2 17l10 5 10-5"/>
                            <path d="M2 12l10 5 10-5"/>
                        </svg>
                        CogniDoc
                    </h1>
                    <p class="header-subtitle">Intelligent Document Assistant powered by Hybrid RAG</p>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="header-badge">Vector + GraphRAG</span>
                    <button class="theme-toggle" onclick="toggleDarkMode()" id="theme-toggle-btn">
                        <span class="icon" id="theme-icon">🌙</span>
                        <span id="theme-label">Dark</span>
                    </button>
                </div>
            </div>
        </div>
        <script>
            // Dark mode toggle functionality
            window.toggleDarkMode = function() {
                const body = document.body;
                const icon = document.getElementById('theme-icon');
                const label = document.getElementById('theme-label');

                if (!icon || !label) {
                    console.error('Dark mode toggle elements not found');
                    return;
                }

                body.classList.toggle('dark-mode');
                const isDark = body.classList.contains('dark-mode');

                // Update button
                icon.textContent = isDark ? '☀️' : '🌙';
                label.textContent = isDark ? 'Light' : 'Dark';

                // Save preference
                localStorage.setItem('cognidoc-dark-mode', isDark ? 'true' : 'false');

                console.log('Dark mode toggled:', isDark);
            };

            // Load saved preference on page load
            function initDarkMode() {
                const savedPref = localStorage.getItem('cognidoc-dark-mode');
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

                // Use saved preference, or fall back to system preference
                const shouldBeDark = savedPref !== null ? savedPref === 'true' : prefersDark;

                const icon = document.getElementById('theme-icon');
                const label = document.getElementById('theme-label');

                if (shouldBeDark) {
                    document.body.classList.add('dark-mode');
                    if (icon) icon.textContent = '☀️';
                    if (label) label.textContent = 'Light';
                } else {
                    document.body.classList.remove('dark-mode');
                    if (icon) icon.textContent = '🌙';
                    if (label) label.textContent = 'Dark';
                }

                console.log('Dark mode initialized:', shouldBeDark);
            }

            // Run on DOMContentLoaded and also after a delay for Gradio
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initDarkMode);
            } else {
                initDarkMode();
            }
            // Also run after Gradio finishes rendering
            setTimeout(initDarkMode, 500);
        </script>
        """
        )

        with gr.Tabs():
            # =====================================================================
            # Chat Tab
            # =====================================================================
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    # Main chat area
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            height=550,
                            label="",
                            show_label=False,
                            elem_classes=["chat-container"],
                        )

                        with gr.Row():
                            user_input = gr.Textbox(
                                label="",
                                show_label=False,
                                placeholder="Ask me anything about your documents...",
                                lines=1,
                                max_lines=3,
                                scale=6,
                                container=False,
                            )
                            submit_btn = gr.Button(
                                "Send",
                                variant="primary",
                                scale=1,
                                min_width=100,
                                elem_classes=["primary-btn"],
                            )

                        with gr.Row():
                            reset_btn = gr.Button(
                                "🗑️ Clear Conversation",
                                variant="secondary",
                                size="sm",
                                elem_classes=["secondary-btn"],
                            )
                            gr.HTML(
                                """
                                <div style="flex: 1; text-align: right; color: #94a3b8; font-size: 0.8rem; padding: 8px;">
                                    Press Enter to send • Shift+Enter for new line
                                </div>
                            """
                            )

                    # Settings panel
                    with gr.Column(scale=1, min_width=280):
                        gr.HTML(
                            """
                        <div class="settings-title">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="3"/>
                                <path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                            </svg>
                            Retrieval Settings
                        </div>
                        """
                        )

                        rerank_toggle = gr.Checkbox(
                            label="🎯 Smart Reranking",
                            value=default_reranking,
                            info="LLM-based relevance scoring (+2-5s)",
                            elem_classes=["toggle-card"],
                        )

                        graph_toggle = gr.Checkbox(
                            label="🔗 Knowledge Graph",
                            value=True,
                            info="Entity relationships & connections",
                            elem_classes=["toggle-card"],
                        )

                        gr.HTML('<div class="settings-divider"></div>')

                        gr.HTML(
                            """
                        <div class="info-card">
                            <div class="info-card-title">💡 How it works</div>
                            <div class="info-card-text">
                                <strong>Smart Reranking</strong> uses AI to re-score
                                retrieved documents for better relevance.
                                <br><br>
                                <strong>Knowledge Graph</strong> understands entity
                                relationships for complex queries like
                                "How is X related to Y?"
                            </div>
                        </div>
                        """
                        )

                        gr.HTML('<div class="settings-divider"></div>')

                        # System status
                        graph_status = "active" if hybrid_status.get("graph", False) else "inactive"
                        vector_status = (
                            "active"
                            if hybrid_status.get("vector_index", False)
                            or hybrid_status.get("keyword_index", False)
                            else "inactive"
                        )

                        gr.HTML(
                            f"""
                        <div style="font-size: 0.8rem; color: #64748b;">
                            <div style="margin-bottom: 8px; font-weight: 600;">System Status</div>
                            <div style="display: flex; flex-direction: column; gap: 6px;">
                                <div class="status-indicator status-{vector_status}">
                                    <span>●</span> Vector Index
                                </div>
                                <div class="status-indicator status-{graph_status}">
                                    <span>●</span> Knowledge Graph
                                </div>
                            </div>
                        </div>
                        """
                        )

            # =====================================================================
            # Metrics Tab
            # =====================================================================
            with gr.Tab("📊 Metrics"):
                gr.HTML("<h3 class='dashboard-section-title'>Global Statistics</h3>")
                stats_display = gr.HTML(get_global_stats_html())

                gr.HTML("<h3 class='dashboard-section-title'>Performance Charts</h3>")
                with gr.Row():
                    latency_chart = gr.Plot(value=create_latency_by_path_chart())
                    distribution_chart = gr.Plot(value=create_path_distribution_chart())

                with gr.Row():
                    timeline_chart = gr.Plot(value=create_latency_over_time_chart())

                gr.HTML("<h3 class='dashboard-section-title'>Recent Queries</h3>")
                queries_table = gr.Dataframe(
                    value=get_recent_queries_dataframe(),
                    headers=["Time", "Path", "Type", "Latency (ms)", "Cache Hits", "Steps"],
                    interactive=False,
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                    export_csv_btn = gr.Button("📥 Export CSV", variant="secondary")
                    export_json_btn = gr.Button("📥 Export JSON", variant="secondary")

                export_file = gr.File(label="Download", visible=False)

                def refresh_metrics():
                    return (
                        get_global_stats_html(),
                        create_latency_by_path_chart(),
                        create_path_distribution_chart(),
                        create_latency_over_time_chart(),
                        get_recent_queries_dataframe(),
                    )

                def export_csv():
                    """Export metrics to CSV file."""
                    import tempfile
                    from datetime import datetime

                    metrics = get_performance_metrics()
                    csv_content = metrics.export_to_csv()
                    if not csv_content:
                        return gr.update(visible=False)
                    # Write to temp file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = tempfile.gettempdir() + f"/cognidoc_metrics_{timestamp}.csv"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(csv_content)
                    return gr.update(value=filepath, visible=True)

                def export_json():
                    """Export metrics to JSON file."""
                    import tempfile
                    from datetime import datetime

                    metrics = get_performance_metrics()
                    json_content = metrics.export_to_json()
                    if (
                        not json_content
                        or json_content
                        == '{"exported_at": "", "total_records": 0, "global_stats": {}, "queries": []}'
                    ):
                        return gr.update(visible=False)
                    # Write to temp file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = tempfile.gettempdir() + f"/cognidoc_metrics_{timestamp}.json"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(json_content)
                    return gr.update(value=filepath, visible=True)

                refresh_btn.click(
                    refresh_metrics,
                    inputs=[],
                    outputs=[
                        stats_display,
                        latency_chart,
                        distribution_chart,
                        timeline_chart,
                        queries_table,
                    ],
                )

                export_csv_btn.click(
                    export_csv,
                    inputs=[],
                    outputs=[export_file],
                )

                export_json_btn.click(
                    export_json,
                    inputs=[],
                    outputs=[export_file],
                )

        # Footer
        gr.HTML(
            """
        <div class="footer">
            <span>Powered by Hybrid RAG</span>
            <span style="margin: 0 8px;">•</span>
            <span>Vector Search + GraphRAG</span>
            <span style="margin: 0 8px;">•</span>
            <span>Local LLM Inference</span>
        </div>
        """
        )

        # Event handlers
        def submit_handler(user_msg, history, rerank, use_graph):
            if not user_msg.strip():
                yield history, ""
                return
            for result in chat_conversation(user_msg, history, rerank, use_graph):
                yield result, ""

        submit_btn.click(
            submit_handler,
            inputs=[user_input, chatbot, rerank_toggle, graph_toggle],
            outputs=[chatbot, user_input],
            queue=True,
        )

        user_input.submit(
            submit_handler,
            inputs=[user_input, chatbot, rerank_toggle, graph_toggle],
            outputs=[chatbot, user_input],
            queue=True,
        )

        reset_btn.click(
            reset_conversation,
            inputs=[],
            outputs=[chatbot, user_input],
            queue=False,
        )

    return demo


def main():
    """Main entry point."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    args = parse_args()

    # Set reranking based on CLI argument
    default_reranking = not args.no_rerank

    if args.no_rerank:
        logger.info("Reranking disabled via CLI argument")
    else:
        logger.info("Reranking enabled by default")

    # Set agentic RAG based on CLI argument
    global ENABLE_AGENTIC_RAG
    if args.no_agent:
        ENABLE_AGENTIC_RAG = False
        logger.info("Agentic RAG disabled via CLI argument")
    else:
        logger.info("Agentic RAG enabled for complex queries")

    # Warm-up models and indexes for faster first query
    warmup_models_and_indexes()

    # Create Gradio app
    demo = create_gradio_app(default_reranking=default_reranking)

    # Create FastAPI app with PDF serving + Gradio
    app = create_fastapi_app(demo)

    logger.info(f"Launching CogniDoc on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
