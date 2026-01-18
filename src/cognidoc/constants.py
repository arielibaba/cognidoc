"""
Configuration constants for the Advanced Hybrid RAG pipeline.

This module loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from current working directory
load_dotenv(Path.cwd() / ".env")

# Package directory for embedded resources (prompts, templates)
# This always points to the installed package location
PACKAGE_DIR = Path(__file__).resolve().parent

# Project directory - current working directory by default
# Can be overridden via COGNIDOC_PROJECT_DIR environment variable
_project_dir_env = os.getenv("COGNIDOC_PROJECT_DIR")
PROJECT_DIR = Path(_project_dir_env) if _project_dir_env else Path.cwd()

# Data directory - PROJECT_DIR/data by default
# Can be overridden via COGNIDOC_DATA_DIR environment variable
# If set, COGNIDOC_DATA_DIR should point to the data folder directly (containing sources/, pdfs/, etc.)
_data_dir_env = os.getenv("COGNIDOC_DATA_DIR")
DATA_DIR = Path(_data_dir_env) if _data_dir_env else PROJECT_DIR / "data"

# For backward compatibility
BASE_DIR = PROJECT_DIR

# =============================================================================
# Directory Paths (relative to DATA_DIR)
# =============================================================================

SOURCES_DIR = DATA_DIR / "sources"        # Input: documents to process
PDF_DIR = DATA_DIR / "pdfs"               # Output: PDFs (copied or converted)
IMAGE_DIR = DATA_DIR / "images"
DETECTION_DIR = DATA_DIR / "detections"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
INDEX_DIR = DATA_DIR / "indexes"
CACHE_DIR = DATA_DIR / "cache"
TOOL_CACHE_DB = CACHE_DIR / "tool_cache.db"
METRICS_DB = CACHE_DIR / "metrics.db"

# =============================================================================
# YOLO Model Configuration (relative to PROJECT_DIR, not DATA_DIR)
# =============================================================================

YOLO_MODEL_PATH = PROJECT_DIR / "models/YOLOv11/yolov11x_best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.2
YOLO_IOU_THRESHOLD = 0.8

# Image filtering parameters (for extracted images)
MIN_IMAGE_PIXELS = 10000            # Minimum pixel count to consider an image valid
MIN_IMAGE_VARIANCE = 500            # Minimum pixel variance to avoid blank/uniform images

# =============================================================================
# Docling Model Configuration (via Ollama)
# =============================================================================

DOCLING_MODEL = os.getenv("DOCLING_MODEL", "ibm/granite-docling:258m-bf16")

# =============================================================================
# LLM Provider Configuration
# =============================================================================

# Default providers (from env or defaults)
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
DEFAULT_VISION_PROVIDER = os.getenv("DEFAULT_VISION_PROVIDER", "gemini").lower()

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "180.0"))

# Model names by provider
GEMINI_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("DEFAULT_VISION_MODEL", "gemini-2.5-flash")

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "granite3.3:8b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b-instruct")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")

# Qwen3-Embedding task instruction for query embeddings (improves retrieval accuracy by ~1-5%)
# Format: "Instruct: {task}\nQuery:{query}" - documents don't need instructions
QWEN_EMBEDDING_TASK = os.getenv(
    "QWEN_EMBEDDING_TASK",
    "Given a web search query, retrieve relevant passages that answer the query"
)

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-20250514")

# Default Ollama model for local processing tasks (entity extraction, table summarization)
OLLAMA_DEFAULT_MODEL = OLLAMA_LLM_MODEL

# Default LLM model for the application (Gemini 2.5 Flash)
DEFAULT_LLM_MODEL = GEMINI_LLM_MODEL

# =============================================================================
# Model Specifications (Official Provider Values)
# =============================================================================
# Each model has its official context window, max output tokens, and defaults.
# Values sourced from official documentation (as of January 2026).

MODEL_SPECS = {
    # -------------------------------------------------------------------------
    # Gemini Models (Google)
    # -------------------------------------------------------------------------
    "gemini-2.5-flash": {
        "provider": "gemini",
        "context_window": 1_048_576,      # 1M tokens
        "max_output_tokens": 65_536,      # 65K with thinking, 8K standard
        "default_temperature": 1.0,
        "default_top_p": 0.95,
        "default_top_k": 40,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "context_window": 1_048_576,      # 1M tokens
        "max_output_tokens": 65_536,
        "default_temperature": 1.0,
        "default_top_p": 0.95,
        "default_top_k": 40,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "gemini-2.0-flash": {
        "provider": "gemini",
        "context_window": 1_048_576,      # 1M tokens
        "max_output_tokens": 8_192,
        "default_temperature": 1.0,
        "default_top_p": 0.95,
        "default_top_k": 40,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },

    # -------------------------------------------------------------------------
    # OpenAI Models
    # -------------------------------------------------------------------------
    "gpt-4o": {
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 16_384,
        "default_temperature": 1.0,
        "default_top_p": 1.0,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 16_384,
        "default_temperature": 1.0,
        "default_top_p": 1.0,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 4_096,
        "default_temperature": 1.0,
        "default_top_p": 1.0,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },

    # -------------------------------------------------------------------------
    # Anthropic Models (Claude)
    # -------------------------------------------------------------------------
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 64_000,      # Extended thinking mode
        "default_temperature": 1.0,
        "default_top_p": 0.999,
        "supports_vision": True,
        "supports_json_mode": False,      # Via prompt instruction only
        "supports_streaming": True,
    },
    "claude-opus-4-20250514": {
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        "default_temperature": 1.0,
        "default_top_p": 0.999,
        "supports_vision": True,
        "supports_json_mode": False,
        "supports_streaming": True,
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 8_192,
        "default_temperature": 1.0,
        "default_top_p": 0.999,
        "supports_vision": True,
        "supports_json_mode": False,
        "supports_streaming": True,
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 4_096,
        "default_temperature": 1.0,
        "default_top_p": 0.999,
        "supports_vision": True,
        "supports_json_mode": False,
        "supports_streaming": True,
    },

    # -------------------------------------------------------------------------
    # Ollama Models (Local)
    # -------------------------------------------------------------------------
    "granite3.3:8b": {
        "provider": "ollama",
        "context_window": 128_000,
        "max_output_tokens": 8_192,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "supports_vision": False,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "qwen3-vl:8b-instruct": {
        "provider": "ollama",
        "context_window": 32_768,
        "max_output_tokens": 4_096,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "supports_vision": True,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "llama3.3:70b": {
        "provider": "ollama",
        "context_window": 128_000,
        "max_output_tokens": 8_192,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "supports_vision": False,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
    "mistral:7b": {
        "provider": "ollama",
        "context_window": 32_768,
        "max_output_tokens": 4_096,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "supports_vision": False,
        "supports_json_mode": True,
        "supports_streaming": True,
    },
}

# Embedding model specifications
EMBEDDING_MODEL_SPECS = {
    "qwen3-embedding:0.6b": {
        "provider": "ollama",
        "context_window": 8_192,
        "embedding_dimension": 1_024,
    },
    "qwen3-embedding:4b": {
        "provider": "ollama",
        "context_window": 8_192,
        "embedding_dimension": 2_560,
    },
    "nomic-embed-text": {
        "provider": "ollama",
        "context_window": 8_192,
        "embedding_dimension": 768,
    },
    "text-embedding-3-small": {
        "provider": "openai",
        "context_window": 8_191,
        "embedding_dimension": 1_536,
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "context_window": 8_191,
        "embedding_dimension": 3_072,
    },
    "text-embedding-004": {
        "provider": "gemini",
        "context_window": 2_048,
        "embedding_dimension": 768,
    },
}

# Default specs for unknown models (fallback)
DEFAULT_MODEL_SPECS = {
    "context_window": 32_768,
    "max_output_tokens": 4_096,
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "supports_vision": False,
    "supports_json_mode": False,
    "supports_streaming": True,
}

DEFAULT_EMBEDDING_SPECS = {
    "context_window": 8_192,
    "embedding_dimension": 1_024,
}


def get_model_specs(model_name: str) -> dict:
    """
    Get specifications for a model.

    Args:
        model_name: Name of the model (e.g., "gemini-2.5-flash", "gpt-4o")

    Returns:
        Dict with model specifications. Falls back to DEFAULT_MODEL_SPECS if unknown.
    """
    # Try exact match first
    if model_name in MODEL_SPECS:
        return {**DEFAULT_MODEL_SPECS, **MODEL_SPECS[model_name]}

    # Try prefix match (e.g., "gemini-2.5-flash-latest" -> "gemini-2.5-flash")
    for known_model in MODEL_SPECS:
        if model_name.startswith(known_model) or known_model.startswith(model_name.split(":")[0]):
            return {**DEFAULT_MODEL_SPECS, **MODEL_SPECS[known_model]}

    return DEFAULT_MODEL_SPECS.copy()


def get_embedding_specs(model_name: str) -> dict:
    """
    Get specifications for an embedding model.

    Args:
        model_name: Name of the embedding model

    Returns:
        Dict with embedding model specifications.
    """
    if model_name in EMBEDDING_MODEL_SPECS:
        return {**DEFAULT_EMBEDDING_SPECS, **EMBEDDING_MODEL_SPECS[model_name]}

    # Try prefix match for Ollama models with tags
    for known_model in EMBEDDING_MODEL_SPECS:
        if model_name.startswith(known_model.split(":")[0]):
            return {**DEFAULT_EMBEDDING_SPECS, **EMBEDDING_MODEL_SPECS[known_model]}

    return DEFAULT_EMBEDDING_SPECS.copy()


# =============================================================================
# Generation Parameters
# =============================================================================

TEMPERATURE_IMAGE_DESC = float(os.getenv("VISION_TEMPERATURE", "0.2"))
TEMPERATURE_TEXT_EXTRACT = 0.1
TOP_P_IMAGE_DESC = float(os.getenv("VISION_TOP_P", "0.85"))

TEMPERATURE_GENERATION = float(os.getenv("LLM_TEMPERATURE", "0.7"))
TOP_P_GENERATION = float(os.getenv("LLM_TOP_P", "0.85"))

# Context Windows (now derived from model specs, but kept for backward compatibility)
# Use get_model_specs(model_name)["context_window"] for model-specific values
CONTEXT_WINDOW = int(float(os.getenv("CONTEXT_WINDOW", "128000")))
MEMORY_WINDOW = int(float(os.getenv("MEMORY_WINDOW", str(CONTEXT_WINDOW * 0.5))))

# Query Expansion
TEMPERATURE_QUERY_EXPANSION = 0.3

# =============================================================================
# Prompt Files (in package directory)
# =============================================================================

SYSTEM_PROMPT_IMAGE_DESC = PACKAGE_DIR / "prompts/system_prompt_for_image_description.md"
USER_PROMPT_IMAGE_DESC = PACKAGE_DIR / "prompts/user_prompt_for_image_description.md"

SYSTEM_PROMPT_TEXT_EXTRACT = PACKAGE_DIR / "prompts/system_prompt_for_text_extract.md"
USER_PROMPT_TEXT_EXTRACT = PACKAGE_DIR / "prompts/user_prompt_for_text_extract.md"

TABLE_SUMMARY_PROMPT_PATH = PACKAGE_DIR / "prompts/markdown_extract_header_and_summarize_prompt.md"

SYSTEM_PROMPT_REWRITE_QUERY = PACKAGE_DIR / "prompts/system_prompt_rewrite_query.md"
USER_PROMPT_REWRITE_QUERY = PACKAGE_DIR / "prompts/user_prompt_rewrite_query.md"

SYSTEM_PROMPT_EXPAND_QUERY = PACKAGE_DIR / "prompts/system_prompt_expand_query.md"
USER_PROMPT_EXPAND_QUERY = PACKAGE_DIR / "prompts/user_prompt_expand_query.md"

SYSTEM_PROMPT_GENERATE_FINAL_ANSWER = PACKAGE_DIR / "prompts/system_prompt_generate_final_answer.md"
USER_PROMPT_GENERATE_FINAL_ANSWER = PACKAGE_DIR / "prompts/user_prompt_generate_final_answer.md"

# =============================================================================
# Chunking Configuration
# =============================================================================

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "512"))
SEMANTIC_CHUNK_BUFFER = int(os.getenv("SEMANTIC_CHUNK_BUFFER", "5"))  # Sentences buffer for context
SEMANTIC_BREAKPOINT_METHOD = "percentile"   # Method for detecting semantic boundaries
SEMANTIC_BREAKPOINT_VALUE = 0.95            # Threshold value (95th percentile)
SENTENCE_SPLIT_REGEX = r"\n\n\n"

# Chunk overlap for context continuity
CHUNK_OVERLAP_PERCENTAGE = float(os.getenv("CHUNK_OVERLAP_PERCENTAGE", "0.1"))  # 10% overlap

# =============================================================================
# Indexing Configuration
# =============================================================================

CHILD_DOCUMENTS_INDEX = "child_documents"
PARENT_DOCUMENTS_INDEX = "parent_documents"

# =============================================================================
# Retrieval Parameters
# =============================================================================

TOP_K_RETRIEVED_CHILDREN = int(os.getenv("TOP_K_RETRIEVED_CHILDREN", "10"))
TOP_K_RERANKED_PARENTS = int(os.getenv("TOP_K_RERANKED_PARENTS", "5"))
# Number of references to display (defaults to TOP_K_RERANKED_PARENTS to show all sources)
TOP_K_REFS = int(os.getenv("TOP_K_REFS", str(TOP_K_RERANKED_PARENTS)))

# Reranking toggle (can be overridden at runtime)
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# Graph source chunks limits (prevents LLM timeout on entities with many chunks)
MAX_SOURCE_CHUNKS_PER_ENTITY = int(os.getenv("MAX_SOURCE_CHUNKS_PER_ENTITY", "20"))
MAX_SOURCE_CHUNKS_FROM_GRAPH = int(os.getenv("MAX_SOURCE_CHUNKS_FROM_GRAPH", "100"))

# =============================================================================
# Advanced RAG Configuration
# =============================================================================

# Hybrid Search: BM25 (sparse) + Dense vector fusion
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))  # Dense vector weight (0.0-1.0)
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))  # Term frequency saturation
BM25_B = float(os.getenv("BM25_B", "0.75"))   # Length normalization
BM25_INDEX_PATH = str((Path(INDEX_DIR) / "bm25_index.json").resolve())

# Cross-Encoder Reranking (uses Qwen3-Reranker via Ollama)
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "true").lower() == "true"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "dengcao/Qwen3-Reranker-0.6B:F16")
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "10"))

# Lost-in-the-Middle Reordering (places best results at start/end for LLM attention)
ENABLE_LOST_IN_MIDDLE_REORDER = os.getenv("ENABLE_LOST_IN_MIDDLE_REORDER", "true").lower() == "true"

# Contextual Compression (extracts query-relevant content)
# Disabled by default: pre-chunked text (512 tokens max) + reranking already filters noise effectively.
# Enable via ENABLE_CONTEXTUAL_COMPRESSION=true for high-noise domains or very long documents.
ENABLE_CONTEXTUAL_COMPRESSION = os.getenv("ENABLE_CONTEXTUAL_COMPRESSION", "false").lower() == "true"
COMPRESSION_MAX_TOKENS_PER_DOC = int(os.getenv("COMPRESSION_MAX_TOKENS_PER_DOC", "200"))
# Skip ratio: skip compression for docs < (MAX_CHUNK_SIZE * ratio) tokens
# Default 0.5 = skip docs under 50% of max chunk size (small docs not worth compressing)
COMPRESSION_SKIP_RATIO = float(os.getenv("COMPRESSION_SKIP_RATIO", "0.5"))
# Computed threshold in tokens (can be overridden directly via COMPRESSION_SKIP_THRESHOLD)
COMPRESSION_SKIP_THRESHOLD = int(os.getenv(
    "COMPRESSION_SKIP_THRESHOLD",
    str(int(MAX_CHUNK_SIZE * COMPRESSION_SKIP_RATIO))
))

# Citation Verification
ENABLE_CITATION_VERIFICATION = os.getenv("ENABLE_CITATION_VERIFICATION", "false").lower() == "true"

# Multi-Index by Content Type
ENABLE_MULTI_INDEX = os.getenv("ENABLE_MULTI_INDEX", "false").lower() == "true"
CONTENT_TYPE_WEIGHTS = {
    "text": 1.0,
    "table": 0.9,
    "image": 0.8,
}

# =============================================================================
# Convert Path objects to strings for external use
# =============================================================================

SOURCES_DIR = str(SOURCES_DIR.resolve())
PDF_DIR = str(PDF_DIR.resolve())
IMAGE_DIR = str(IMAGE_DIR.resolve())
DETECTION_DIR = str(DETECTION_DIR.resolve())
PROCESSED_DIR = str(PROCESSED_DIR.resolve())
YOLO_MODEL_PATH = str(YOLO_MODEL_PATH.resolve())
SYSTEM_PROMPT_IMAGE_DESC = str(SYSTEM_PROMPT_IMAGE_DESC.resolve())
SYSTEM_PROMPT_TEXT_EXTRACT = str(SYSTEM_PROMPT_TEXT_EXTRACT.resolve())
USER_PROMPT_IMAGE_DESC = str(USER_PROMPT_IMAGE_DESC.resolve())
USER_PROMPT_TEXT_EXTRACT = str(USER_PROMPT_TEXT_EXTRACT.resolve())
TABLE_SUMMARY_PROMPT_PATH = str(TABLE_SUMMARY_PROMPT_PATH.resolve())
CHUNKS_DIR = str(CHUNKS_DIR.resolve())
EMBEDDINGS_DIR = str(EMBEDDINGS_DIR.resolve())
VECTOR_STORE_DIR = str(VECTOR_STORE_DIR.resolve())
INDEX_DIR = str(INDEX_DIR.resolve())
CACHE_DIR = str(CACHE_DIR.resolve())
TOOL_CACHE_DB = str(TOOL_CACHE_DB.resolve())
METRICS_DB = str(METRICS_DB.resolve())

# Ensure directories exist
Path(SOURCES_DIR).mkdir(parents=True, exist_ok=True)
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
