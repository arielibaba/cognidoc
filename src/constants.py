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
BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# Directory Paths
# =============================================================================

PDF_DIR = BASE_DIR / "../data/pdfs"
PDF_CONVERTED_DIR = BASE_DIR / "../data/pdfs_converted"
IMAGE_DIR = BASE_DIR / "../data/images"
DETECTION_DIR = BASE_DIR / "../data/detections"
PROCESSED_DIR = BASE_DIR / "../data/processed"
CHUNKS_DIR = BASE_DIR / "../data/chunks"
EMBEDDINGS_DIR = BASE_DIR / "../data/embeddings"
VECTOR_STORE_DIR = BASE_DIR / "../data/vector_store"
INDEX_DIR = BASE_DIR / "../data/indexes"
CACHE_DIR = BASE_DIR / "../data/cache"

# Handled file extensions
FILE_EXTENSIONS = ['.doc', '.docx', '.html', '.htm', '.ppt', '.pptx']
FILE_EXTENSIONS.extend([ext.upper() for ext in FILE_EXTENSIONS])

# =============================================================================
# YOLO Model Configuration
# =============================================================================

YOLO_MODEL_PATH = BASE_DIR / "../models/YOLOv11/yolov11x_best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.2
YOLO_IOU_THRESHOLD = 0.8

# Parameters to Use for Filtering the Extracted Images
IMAGE_PIXEL_THRESHOLD = 10000
IMAGE_PIXEL_VARIANCE_THRESHOLD = 500

# =============================================================================
# SmolDocling Model Configuration
# =============================================================================

SMOLDOCLING_MODEL_DIR = BASE_DIR / "../models/ds4sd_SmolDocling-256M-preview-mlx-bf16"

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
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding-0.6b")

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-20250514")

# Legacy compatibility (maps to current provider settings)
VLLM = OLLAMA_VISION_MODEL  # Vision model for Ollama
LLM = OLLAMA_LLM_MODEL       # LLM model for Ollama

# =============================================================================
# Generation Parameters
# =============================================================================

TEMPERATURE_IMAGE_DESC = float(os.getenv("VISION_TEMPERATURE", "0.2"))
TEMPERATURE_TEXT_EXTRACT = 0.1
TOP_P_IMAGE_DESC = float(os.getenv("VISION_TOP_P", "0.85"))
TOP_P_IMAGE_TEXT_EXTRACT = 0.1

TEMPERATURE_GENERATION = float(os.getenv("LLM_TEMPERATURE", "0.7"))
TOP_P_GENERATION = float(os.getenv("LLM_TOP_P", "0.85"))

# Context Windows
CONTEXT_WINDOW = int(float(os.getenv("CONTEXT_WINDOW", "128000")))
MEMORY_WINDOW = int(float(os.getenv("MEMORY_WINDOW", str(CONTEXT_WINDOW * 0.5))))

# Query Expansion
TEMPERATURE_QUERY_EXPANSION = 0.3

# =============================================================================
# Prompt Files
# =============================================================================

SYSTEM_PROMPT_IMAGE_DESC = BASE_DIR / "prompts/system_prompt_for_image_description.md"
USER_PROMPT_IMAGE_DESC = BASE_DIR / "prompts/user_prompt_for_image_description.md"

SYSTEM_PROMPT_TEXT_EXTRACT = BASE_DIR / "prompts/system_prompt_for_text_extract.md"
USER_PROMPT_TEXT_EXTRACT = BASE_DIR / "prompts/user_prompt_for_text_extract.md"

SUMMARIZE_TABLE_PROMPT = BASE_DIR / "prompts/markdown_extract_header_and_summarize_prompt.md"

SYSTEM_PROMPT_REWRITE_QUERY = BASE_DIR / "prompts/system_prompt_rewrite_query.md"
USER_PROMPT_REWRITE_QUERY = BASE_DIR / "prompts/user_prompt_rewrite_query.md"

SYSTEM_PROMPT_EXPAND_QUERY = BASE_DIR / "prompts/system_prompt_expand_query.md"
USER_PROMPT_EXPAND_QUERY = BASE_DIR / "prompts/user_prompt_expand_query.md"

SYSTEM_PROMPT_GENERATE_FINAL_ANSWER = BASE_DIR / "prompts/system_prompt_generate_final_answer.md"
USER_PROMPT_GENERATE_FINAL_ANSWER = BASE_DIR / "prompts/user_prompt_generate_final_answer.md"

# =============================================================================
# Chunking Configuration
# =============================================================================

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding-0.6b")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "512"))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "5"))
BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_AMOUNT = 0.95
SENTENCE_SPLIT_REGEX = r"\n\n\n"

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
# Convert Path objects to strings for external use
# =============================================================================

PDF_DIR = str(PDF_DIR.resolve())
PDF_CONVERTED_DIR = str(PDF_CONVERTED_DIR.resolve())
IMAGE_DIR = str(IMAGE_DIR.resolve())
DETECTION_DIR = str(DETECTION_DIR.resolve())
PROCESSED_DIR = str(PROCESSED_DIR.resolve())
YOLO_MODEL_PATH = str(YOLO_MODEL_PATH.resolve())
SYSTEM_PROMPT_IMAGE_DESC = str(SYSTEM_PROMPT_IMAGE_DESC.resolve())
SYSTEM_PROMPT_TEXT_EXTRACT = str(SYSTEM_PROMPT_TEXT_EXTRACT.resolve())
USER_PROMPT_IMAGE_DESC = str(USER_PROMPT_IMAGE_DESC.resolve())
USER_PROMPT_TEXT_EXTRACT = str(USER_PROMPT_TEXT_EXTRACT.resolve())
SMOLDOCLING_MODEL_DIR = str(SMOLDOCLING_MODEL_DIR.resolve())
CHUNKS_DIR = str((BASE_DIR / "../data/chunks").resolve())
EMBEDDINGS_DIR = str(EMBEDDINGS_DIR.resolve())
VECTOR_STORE_DIR = str(VECTOR_STORE_DIR.resolve())
INDEX_DIR = str(INDEX_DIR.resolve())
CACHE_DIR = str((BASE_DIR / "../data/cache").resolve())

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
