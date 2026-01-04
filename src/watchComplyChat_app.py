"""
WatchComplyChat - Gradio chat application for RAG-based document querying.

Features:
- Multi-step RAG: query rewriting, retrieval, reranking, generation
- Streaming responses
- Reranking toggle (UI checkbox + CLI argument)
- Performance profiling

No LlamaIndex dependencies - uses direct Qdrant and Ollama calls.
"""

import argparse
import time
import warnings

import gradio as gr
import nest_asyncio

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
)
from .helpers import (
    clear_pytorch_cache,
    limit_chat_history,
    run_streaming,
    rewrite_query,
    parse_rewritten_query,
    convert_history_to_tuples,
    reset_conversation,
)
from .utils.rag_utils import (
    VectorIndex,
    KeywordIndex,
    NodeWithScore,
    rerank_documents,
)
from .utils.logger import logger, retrieval_metrics

# Suppress warnings and apply async patch
warnings.filterwarnings("ignore")
nest_asyncio.apply()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WatchComplyChat - RAG Chat Application")
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

# Load indexes
logger.info("Loading indexes...")

child_index = VectorIndex.load(
    path=f"{INDEX_DIR}/{CHILD_DOCUMENTS_INDEX}",
    qdrant_path=VECTOR_STORE_DIR,
)

parent_index = KeywordIndex.load(
    path=f"{INDEX_DIR}/{PARENT_DOCUMENTS_INDEX}",
)

logger.info("Indexes loaded successfully")


def chat_conversation(user_message: str, history: list, enable_reranking: bool = True):
    """
    Main chat conversation handler.

    Args:
        user_message: User's input message
        history: Conversation history
        enable_reranking: Whether to use LLM reranking

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

    # Query rewriting
    try:
        t1 = time.perf_counter()
        rewritten = rewrite_query(LLM, user_message, conv_history)
        t2 = time.perf_counter()
        logger.info(f"Rewritten query ({len(rewritten.split(chr(10)))} parts):\n{rewritten}")
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        history.append({
            "role": "assistant",
            "content": "My apologies, the service timed out. Please try again later."
        })
        yield convert_history_to_tuples(history)
        return

    candidates = parse_rewritten_query(rewritten)
    if not candidates:
        candidates = [user_message]

    # Retrieval
    t3 = time.perf_counter()
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
            model=LLM,
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
        num_retrieved=len(retrieved),
        num_after_rerank=len(reranked),
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
    )

    # Build context and references
    context = "\n".join(n.node.text for n in reranked)
    refs = []
    seen_pages = set()
    for i, nws in enumerate(reranked, 1):
        source = nws.node.metadata.get("source", {})
        if isinstance(source, dict):
            doc = source.get("document", "Unknown")
            page = source.get("page", "?")
        else:
            doc = str(source)
            page = "?"
        if (doc, page) not in seen_pages:
            seen_pages.add((doc, page))
            refs.append(f"{i}. {doc} - Page {page}")
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
    for chunk in run_streaming(LLM, msgs, TEMPERATURE_GENERATION, TOP_P_GENERATION):
        history[-1]["content"] = chunk
        yield convert_history_to_tuples(history)

    t6 = time.perf_counter()

    # Add references
    final = history[-1]["content"].strip()
    if not final.lower().startswith("no relevant details"):
        final += "\n\n---\n**References:**\n" + "\n".join(refs)
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
    Create the Gradio application.

    Args:
        default_reranking: Default value for reranking toggle

    Returns:
        Gradio Blocks app
    """
    with gr.Blocks(title="WatchComplyChat") as demo:
        gr.Markdown("""
        # WatchComplyChat

        A Gen AI-Powered Legal Analytics Solution that Analyzes All the Sanctions
        from French and European Regulators.

        ---
        """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=600,
                    label="Conversation",
                )
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your question here...",
                    lines=2,
                    max_lines=5,
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    reset_btn = gr.Button("Reset Conversation", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                rerank_toggle = gr.Checkbox(
                    label="Enable LLM Reranking",
                    value=default_reranking,
                    info="Improves quality but adds latency (~2-5s)"
                )
                gr.Markdown("""
                ---
                ### About Reranking

                When enabled, retrieved documents are re-scored
                by the LLM for relevance. This improves answer
                quality but adds processing time.

                Disable for faster responses when precision
                is less critical.
                """)

        # Event handlers
        def submit_handler(user_msg, history, rerank):
            for result in chat_conversation(user_msg, history, rerank):
                yield result, ""

        submit_btn.click(
            submit_handler,
            inputs=[user_input, chatbot, rerank_toggle],
            outputs=[chatbot, user_input],
            queue=True,
        )

        user_input.submit(
            submit_handler,
            inputs=[user_input, chatbot, rerank_toggle],
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
    args = parse_args()

    # Set reranking based on CLI argument
    default_reranking = not args.no_rerank

    if args.no_rerank:
        logger.info("Reranking disabled via CLI argument")
    else:
        logger.info("Reranking enabled by default")

    # Create and launch app
    demo = create_gradio_app(default_reranking=default_reranking)

    logger.info(f"Launching WatchComplyChat on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
