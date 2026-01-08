"""
Configuration constants for the Advanced Hybrid RAG pipeline.

This module loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the directory of the current file (constants.py)
# After restructuring: src/cognidoc/constants.py -> project root is ../../
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Package directory for prompts and other package resources
PACKAGE_DIR = Path(__file__).resolve().parent

# =============================================================================
# Directory Paths
# =============================================================================

SOURCES_DIR = BASE_DIR / "data/sources"  # Input: documents to process
PDF_DIR = BASE_DIR / "data/pdfs"          # Output: PDFs (copied or converted)
IMAGE_DIR = BASE_DIR / "data/images"
DETECTION_DIR = BASE_DIR / "data/detections"
PROCESSED_DIR = BASE_DIR / "data/processed"
CHUNKS_DIR = BASE_DIR / "data/chunks"
EMBEDDINGS_DIR = BASE_DIR / "data/embeddings"
VECTOR_STORE_DIR = BASE_DIR / "data/vector_store"
INDEX_DIR = BASE_DIR / "data/indexes"
CACHE_DIR = BASE_DIR / "data/cache"

# =============================================================================
# YOLO Model Configuration
# =============================================================================

YOLO_MODEL_PATH = BASE_DIR / "models/YOLOv11/yolov11x_best.pt"
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
GEMINI_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")
GEMINI_VISION_MODEL = os.getenv("DEFAULT_VISION_MODEL", "gemini-2.0-flash")

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "granite3.3:8b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b-instruct")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-20250514")

# Default Ollama model for local processing tasks (entity extraction, table summarization)
OLLAMA_DEFAULT_MODEL = OLLAMA_LLM_MODEL

# Default LLM model for the application (Gemini 2.0 Flash)
DEFAULT_LLM_MODEL = GEMINI_LLM_MODEL

# =============================================================================
# Generation Parameters
# =============================================================================

TEMPERATURE_IMAGE_DESC = float(os.getenv("VISION_TEMPERATURE", "0.2"))
TEMPERATURE_TEXT_EXTRACT = 0.1
TOP_P_IMAGE_DESC = float(os.getenv("VISION_TOP_P", "0.85"))

TEMPERATURE_GENERATION = float(os.getenv("LLM_TEMPERATURE", "0.7"))
TOP_P_GENERATION = float(os.getenv("LLM_TOP_P", "0.85"))

# Context Windows
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
TOP_K_REFS = int(os.getenv("TOP_K_REFS", "3"))

# Reranking toggle (can be overridden at runtime)
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# =============================================================================
# Advanced RAG Configuration
# =============================================================================

# Hybrid Search: BM25 (sparse) + Dense vector fusion
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))  # Dense vector weight (0.0-1.0)
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))  # Term frequency saturation
BM25_B = float(os.getenv("BM25_B", "0.75"))   # Length normalization
BM25_INDEX_PATH = str((Path(INDEX_DIR) / "bm25_index.json").resolve())

# Cross-Encoder Reranking
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "true").lower() == "true"
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "5"))

# Lost-in-the-Middle Reordering (places best results at start/end for LLM attention)
ENABLE_LOST_IN_MIDDLE_REORDER = os.getenv("ENABLE_LOST_IN_MIDDLE_REORDER", "true").lower() == "true"

# Contextual Compression (extracts query-relevant content)
ENABLE_CONTEXTUAL_COMPRESSION = os.getenv("ENABLE_CONTEXTUAL_COMPRESSION", "true").lower() == "true"
COMPRESSION_MAX_TOKENS_PER_DOC = int(os.getenv("COMPRESSION_MAX_TOKENS_PER_DOC", "200"))

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

# Ensure directories exist
Path(SOURCES_DIR).mkdir(parents=True, exist_ok=True)
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
