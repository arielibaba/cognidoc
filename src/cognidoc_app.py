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
import urllib.parse
import warnings

import gradio as gr

from .constants import (
    INDEX_DIR,
    VECTOR_STORE_DIR,
    CHILD_DOCUMENTS_INDEX,
    PARENT_DOCUMENTS_INDEX,
    TOP_K_RETRIEVED_CHILDREN,
    TOP_K_RERANKED_PARENTS,
    TOP_K_REFS,
    LLM,
    EMBED_MODEL,
    TEMPERATURE_GENERATION,
    TOP_P_GENERATION,
    MEMORY_WINDOW,
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

# Suppress warnings
warnings.filterwarnings("ignore")

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
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CogniDoc - Intelligent Document Assistant")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking (faster but may reduce quality)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
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


def chat_conversation(
    user_message: str,
    history: list,
    enable_reranking: bool = True,
    enable_graph: bool = True,
):
    """
    Main chat conversation handler with hybrid RAG.

    Args:
        user_message: User's input message
        history: Conversation history
        enable_reranking: Whether to use LLM reranking
        enable_graph: Whether to use GraphRAG (knowledge graph)

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
    history = limit_chat_history(history, max_tokens=MEMORY_WINDOW)
    conv_history = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in history)

    # Parallel query rewriting and classification (uses unified LLM client)
    try:
        t1 = time.perf_counter()
        rewritten, routing_decision = parallel_rewrite_and_classify(user_message, conv_history)
        t2 = time.perf_counter()
        logger.info(f"Rewritten query ({len(rewritten.split(chr(10)))} parts):\n{rewritten}")
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        history.append({
            "role": "assistant",
            "content": "I apologize, but the service is temporarily unavailable. Please try again in a moment."
        })
        yield convert_history_to_tuples(history)
        return

    candidates = parse_rewritten_query(rewritten)
    if not candidates:
        candidates = [user_message]

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

        hybrid_result = hybrid_retriever.retrieve(
            query=combo_query,
            top_k=TOP_K_RETRIEVED_CHILDREN,
            use_reranking=enable_reranking,
            model=LLM,
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
        retrieved = []
        for q in candidates:
            results = child_index.search(q, top_k=TOP_K_RETRIEVED_CHILDREN)
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
            logger.info(f"Reranking: {len(unique)} -> {len(reranked)} parents in {rerank_time:.2f}s")
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
                doc_pages[doc].add(page)

    # Format references with consolidated page numbers and clickable links
    refs = []
    for i, (doc, pages) in enumerate(doc_pages.items(), 1):
        # Create PDF file path - URL encoded filename
        pdf_filename = f"{doc}.pdf"
        encoded_filename = urllib.parse.quote(pdf_filename)
        # Use /pdfs/ path for FastAPI static file serving
        base_url = f"/pdfs/{encoded_filename}"

        if not pages:
            # No page info - link to document
            refs.append(f'{i}. <a href="{base_url}" target="_blank" rel="noopener">{doc}</a>')
        elif len(pages) == 1:
            page_num = list(pages)[0]
            refs.append(f'{i}. <a href="{base_url}#page={page_num}" target="_blank" rel="noopener">{doc}</a> - Page {page_num}')
        else:
            # Sort pages and create range or list
            sorted_pages = sorted(p for p in pages if isinstance(p, int))
            if sorted_pages:
                first_page = min(sorted_pages)
                # Check if consecutive for range format
                if sorted_pages == list(range(first_page, max(sorted_pages) + 1)):
                    refs.append(f'{i}. <a href="{base_url}#page={first_page}" target="_blank" rel="noopener">{doc}</a> - Pages {first_page}-{max(sorted_pages)}')
                else:
                    refs.append(f'{i}. <a href="{base_url}#page={first_page}" target="_blank" rel="noopener">{doc}</a> - Pages {", ".join(map(str, sorted_pages))}')
            else:
                refs.append(f'{i}. <a href="{base_url}" target="_blank" rel="noopener">{doc}</a> - Pages {", ".join(map(str, pages))}')
        if len(refs) >= TOP_K_REFS:
            break

    # Load prompts
    with open(SYSTEM_PROMPT_GENERATE_FINAL_ANSWER, "r", encoding="utf-8") as f:
        system_msg = f.read()

    with open(USER_PROMPT_GENERATE_FINAL_ANSWER, "r", encoding="utf-8") as f:
        user_msg_template = f.read()

    user_prompt = user_msg_template.format(
        conversation_history=conv_history,
        user_question=user_message,
        refined_context=context
    )

    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ]

    t5 = time.perf_counter()

    # Stream response
    history.append({"role": "assistant", "content": ""})
    for chunk in run_streaming(msgs, TEMPERATURE_GENERATION):
        history[-1]["content"] = chunk
        yield convert_history_to_tuples(history)

    t6 = time.perf_counter()

    # Add references
    final = history[-1]["content"].strip()
    if not final.lower().startswith("no relevant details"):
        final += "\n\n---\n**References:**\n" + "\n".join(refs)

    # #15: Citation verification (optional)
    if ENABLE_CITATION_VERIFICATION and reranked:
        try:
            all_supported, claim_results, summary = verify_citations(
                answer=final,
                documents=[nws.node for nws in reranked],
                model=LLM,
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
    logger.info(f"""
    Performance Stats:
    - Query Rewriting:      {t2 - t1:.2f}s
    - Retrieval:            {retrieval_time:.2f}s
    - Reranking:            {rerank_time:.2f}s ({"enabled" if enable_reranking else "disabled"})
    - Prompt Construction:  {t5 - t4:.2f}s
    - LLM Inference:        {t6 - t5:.2f}s
    - TOTAL:                {t6 - t0:.2f}s
    """)

    yield convert_history_to_tuples(history)


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
        )
    ) as demo:

        # Header
        gr.HTML("""
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
                <div>
                    <span class="header-badge">Vector + GraphRAG</span>
                </div>
            </div>
        </div>
        """)

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
                    gr.HTML("""
                        <div style="flex: 1; text-align: right; color: #94a3b8; font-size: 0.8rem; padding: 8px;">
                            Press Enter to send • Shift+Enter for new line
                        </div>
                    """)

            # Settings panel
            with gr.Column(scale=1, min_width=280):
                gr.HTML("""
                <div class="settings-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"/>
                        <path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                    </svg>
                    Retrieval Settings
                </div>
                """)

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

                gr.HTML("""
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
                """)

                gr.HTML('<div class="settings-divider"></div>')

                # System status
                graph_status = "active" if hybrid_status.get("graph", False) else "inactive"
                vector_status = "active" if hybrid_status.get("vector_index", False) or hybrid_status.get("keyword_index", False) else "inactive"

                gr.HTML(f"""
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
                """)

        # Footer
        gr.HTML("""
        <div class="footer">
            <span>Powered by Hybrid RAG</span>
            <span style="margin: 0 8px;">•</span>
            <span>Vector Search + GraphRAG</span>
            <span style="margin: 0 8px;">•</span>
            <span>Local LLM Inference</span>
        </div>
        """)

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

    # Create Gradio app
    demo = create_gradio_app(default_reranking=default_reranking)

    # Create FastAPI app and mount Gradio + static files
    app = FastAPI()

    # Mount PDF directory for static file serving
    app.mount("/pdfs", StaticFiles(directory=PDF_DIR), name="pdfs")
    logger.info(f"Static PDF files mounted at /pdfs from {PDF_DIR}")

    # Mount Gradio app
    app = gr.mount_gradio_app(app, demo, path="/")

    logger.info(f"Launching CogniDoc on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
