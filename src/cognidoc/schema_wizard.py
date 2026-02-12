"""
Schema Wizard - Interactive and automatic graph schema generation.

Provides three modes:
1. Interactive: Ask user questions about their domain
2. Automatic (legacy): Analyze text-readable document samples to generate schema
3. Corpus-based: Convert to PDF, sample intelligently, two-stage LLM pipeline
"""

import asyncio
import json
import math
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

from .utils.logger import logger
from .utils.llm_client import llm_chat
from .constants import SOURCES_DIR, PDF_DIR, PROJECT_DIR

# ---------------------------------------------------------------------------
# Adaptive sampling constants
# ---------------------------------------------------------------------------

# Total text budget across all sampled documents (chars).
# With 100 docs this yields 3000 chars/doc (current behavior);
# with fewer docs, each gets proportionally more.
_TOTAL_TEXT_BUDGET = 300_000
_MIN_CHARS_PER_DOC = 3_000
_MAX_CHARS_PER_DOC = 30_000

# Pages with fewer characters than this are considered empty
# (cover pages, full-page images, etc.) and skipped.
_MIN_PAGE_CHARS = 100

# Target character count per LLM batch call (Stage A).
_TARGET_BATCH_CHARS = 40_000

# Check if questionary is available
_QUESTIONARY_AVAILABLE = False
try:
    import questionary
    from questionary import Style

    _QUESTIONARY_AVAILABLE = True
except ImportError:
    questionary = None
    Style = None


# Default wizard style
WIZARD_STYLE = None
if _QUESTIONARY_AVAILABLE:
    WIZARD_STYLE = Style(
        [
            ("qmark", "fg:cyan bold"),
            ("question", "fg:white bold"),
            ("answer", "fg:green bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan bold"),
            ("selected", "fg:green"),
        ]
    )


# Predefined domain templates
DOMAIN_TEMPLATES = {
    "technical": {
        "name": "Technical Documentation",
        "entities": [
            "Product",
            "Technology",
            "Feature",
            "Component",
            "Process",
            "API",
            "Configuration",
        ],
        "relationships": ["USES", "IMPLEMENTS", "PART_OF", "DEPENDS_ON", "PRODUCES", "CONFIGURES"],
    },
    "legal": {
        "name": "Legal Documents",
        "entities": ["Law", "Article", "Person", "Organization", "Contract", "Obligation", "Right"],
        "relationships": ["REFERENCES", "AMENDS", "APPLIES_TO", "OBLIGATES", "GRANTS", "SIGNED_BY"],
    },
    "medical": {
        "name": "Medical/Healthcare",
        "entities": [
            "Disease",
            "Symptom",
            "Treatment",
            "Medication",
            "Procedure",
            "Patient",
            "Doctor",
        ],
        "relationships": [
            "TREATS",
            "CAUSES",
            "INDICATES",
            "PRESCRIBES",
            "DIAGNOSES",
            "CONTRAINDICATED",
        ],
    },
    "scientific": {
        "name": "Scientific Research",
        "entities": [
            "Concept",
            "Theory",
            "Experiment",
            "Result",
            "Researcher",
            "Institution",
            "Publication",
        ],
        "relationships": [
            "SUPPORTS",
            "CONTRADICTS",
            "CITES",
            "CONDUCTED_BY",
            "PUBLISHED_IN",
            "PROVES",
        ],
    },
    "business": {
        "name": "Business/Corporate",
        "entities": [
            "Company",
            "Product",
            "Service",
            "Person",
            "Department",
            "Project",
            "Strategy",
        ],
        "relationships": [
            "OWNS",
            "MANAGES",
            "WORKS_FOR",
            "PARTNERS_WITH",
            "COMPETES_WITH",
            "PROVIDES",
        ],
    },
    "educational": {
        "name": "Educational Content",
        "entities": ["Topic", "Concept", "Course", "Lesson", "Instructor", "Student", "Assessment"],
        "relationships": [
            "TEACHES",
            "PREREQUISITE_FOR",
            "COVERS",
            "ASSESSES",
            "ENROLLED_IN",
            "PART_OF",
        ],
    },
    "generic": {
        "name": "Generic/Mixed",
        "entities": ["Entity", "Concept", "Person", "Organization", "Process", "Document", "Event"],
        "relationships": [
            "RELATED_TO",
            "PART_OF",
            "CREATED_BY",
            "REFERENCES",
            "OCCURS_IN",
            "INVOLVES",
        ],
    },
}

LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("Français", "fr"),
    ("Español", "es"),
    ("Deutsch", "de"),
    ("Italiano", "it"),
    ("Português", "pt"),
    ("Other", "other"),
]


# ---------------------------------------------------------------------------
# Generic name detection
# ---------------------------------------------------------------------------

GENERIC_NAME_PATTERNS = [
    r"^doc(ument)?s?$",
    r"^files?$",
    r"^fichiers?$",
    r"^scans?$",
    r"^untitled",
    r"^sans[_ ]titre",
    r"^copie",
    r"^copy",
    r"^new$",
    r"^nouveau",
    r"^temp$",
    r"^draft$",
    r"^brouillon$",
    r"^dossier",
    r"^folder",
    r"^\d+$",
    r"^.{1,2}$",
    r"^(doc|file|scan|img|page|image|photo)[_\-]?\d+$",
    r"^IMG_\d+$",
]

_GENERIC_RE = [re.compile(p, re.IGNORECASE) for p in GENERIC_NAME_PATTERNS]


def is_generic_name(name: str) -> bool:
    """Check if a filename or folder name is generic (non-informative)."""
    stem = Path(name).stem.strip()
    return any(pat.match(stem) for pat in _GENERIC_RE)


# ---------------------------------------------------------------------------
# PDF text extraction (PyMuPDF) with distributed page sampling
# ---------------------------------------------------------------------------


def _select_distributed_pages(total_pages: int, max_chars: int) -> List[int]:
    """
    Select candidate pages from beginning (40%), middle (30%), end (30%).

    Returns more candidates than strictly needed to account for empty pages
    that will be skipped during extraction.

    Args:
        total_pages: Total number of pages in the PDF
        max_chars: Character budget — used to estimate how many pages to read

    Returns:
        Ordered list of page indices to try
    """
    # Generous estimate: ~800 usable chars per page (accounts for headers, margins)
    estimated = max(3, max_chars // 800 + 2)

    if total_pages <= estimated:
        return list(range(total_pages))

    n_begin = max(2, int(estimated * 0.4))
    n_middle = max(1, int(estimated * 0.3))
    n_end = max(1, int(estimated * 0.3))

    third = max(1, total_pages // 3)

    # Beginning: sequential from start
    begin_pages = list(range(min(n_begin, third)))

    # Middle: evenly spaced in middle third
    mid_start, mid_end = third, 2 * third
    mid_range = mid_end - mid_start
    if mid_range <= 0 or n_middle >= mid_range:
        middle_pages = list(range(mid_start, mid_end))
    else:
        step = max(1, mid_range // n_middle)
        middle_pages = list(range(mid_start, mid_end, step))[:n_middle]

    # End: evenly spaced in last third
    end_start = 2 * third
    end_range = total_pages - end_start
    if end_range <= 0 or n_end >= end_range:
        end_pages = list(range(end_start, total_pages))
    else:
        step = max(1, end_range // n_end)
        end_pages = list(range(end_start, total_pages, step))[:n_end]

    return begin_pages + middle_pages + end_pages


def extract_pdf_text(pdf_path: Path, max_chars: int = 3000) -> str:
    """
    Extract text from distributed pages of a PDF using PyMuPDF.

    Samples pages from beginning (40%), middle (30%), and end (30%)
    of the document. Skips near-empty pages (< 100 chars of text).

    Args:
        pdf_path: Path to the PDF file
        max_chars: Maximum characters to return (default 3000)

    Returns:
        Extracted text, or empty string on failure
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF (pymupdf) not installed. Cannot extract PDF text.")
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return ""

        candidates = _select_distributed_pages(total_pages, max_chars)
        text_parts = []
        chars_collected = 0

        for page_num in candidates:
            if chars_collected >= max_chars:
                break
            page_text = doc[page_num].get_text()
            if len(page_text.strip()) < _MIN_PAGE_CHARS:
                continue  # skip near-empty pages
            remaining = max_chars - chars_collected
            chunk = page_text[:remaining]
            text_parts.append(chunk)
            chars_collected += len(chunk)

        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Could not extract text from {pdf_path.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Intelligent PDF sampling for schema generation
# ---------------------------------------------------------------------------


def sample_pdfs_for_schema(
    sources_dir: Path,
    pdf_dir: Path,
    max_docs: int = 100,
) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """
    Sample PDFs intelligently for schema generation.

    Scans sources_dir for structure (subfolders vs flat), then reads text from
    distributed pages of sampled PDFs in pdf_dir. The character budget per
    document is computed adaptively: fewer documents → more text per document.

    Args:
        sources_dir: Path to source documents directory
        pdf_dir: Path to converted PDFs directory
        max_docs: Maximum documents to sample (default 100)

    Returns:
        Tuple of:
        - List of {filename, content, subfolder} dicts
        - List of non-generic subfolder names
        - List of non-generic file names (from sample)
    """
    sources_path = Path(sources_dir)
    pdf_path = Path(pdf_dir)

    if not sources_path.exists():
        logger.warning(f"Sources directory not found: {sources_path}")
        return [], [], []

    if not pdf_path.exists():
        logger.warning(f"PDF directory not found: {pdf_path}")
        return [], [], []

    # Discover subfolders in sources
    subfolders = set()
    source_files_by_folder: Dict[str, List[Path]] = {}

    for item in sources_path.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(sources_path)
        if len(rel.parts) > 1:
            folder = rel.parts[0]
            subfolders.add(folder)
            source_files_by_folder.setdefault(folder, []).append(item)
        else:
            source_files_by_folder.setdefault("", []).append(item)

    # Build index of available PDFs by stem
    available_pdfs = {}
    for pdf_file in pdf_path.glob("*.pdf"):
        available_pdfs[pdf_file.stem.lower()] = pdf_file

    # Determine which source files to sample
    sampled_sources: List[Tuple[Path, str]] = []  # (source_path, subfolder)

    if subfolders:
        # Distribute equally across subfolders
        folders_with_files = [f for f in subfolders if source_files_by_folder.get(f)]
        if folders_with_files:
            per_folder = max(1, math.ceil(max_docs / len(folders_with_files)))
            for folder in folders_with_files:
                folder_files = source_files_by_folder[folder]
                n = min(per_folder, len(folder_files))
                chosen = random.sample(folder_files, n)
                sampled_sources.extend((f, folder) for f in chosen)

        # Also include root-level files if any
        root_files = source_files_by_folder.get("", [])
        if root_files:
            remaining = max(0, max_docs - len(sampled_sources))
            if remaining > 0:
                n = min(remaining, len(root_files))
                chosen = random.sample(root_files, n)
                sampled_sources.extend((f, "") for f in chosen)

        # Trim to max_docs
        if len(sampled_sources) > max_docs:
            sampled_sources = random.sample(sampled_sources, max_docs)
    else:
        # Flat structure: sample randomly
        all_files = source_files_by_folder.get("", [])
        n = min(max_docs, len(all_files))
        if all_files:
            chosen = random.sample(all_files, n)
            sampled_sources = [(f, "") for f in chosen]

    # Compute adaptive character budget per document
    num_sampled = max(1, len(sampled_sources))
    chars_per_doc = max(
        _MIN_CHARS_PER_DOC, min(_MAX_CHARS_PER_DOC, _TOTAL_TEXT_BUDGET // num_sampled)
    )
    logger.debug(f"Adaptive sampling: {num_sampled} docs, {chars_per_doc} chars/doc")

    # Extract text from corresponding PDFs
    samples = []
    informative_file_names = []
    informative_folder_names = []

    seen_folders = set()
    for source_file, subfolder in sampled_sources:
        stem = source_file.stem.lower()
        pdf_file = available_pdfs.get(stem)

        if pdf_file is None:
            # Try case-insensitive match with original stem
            pdf_file = available_pdfs.get(source_file.stem)
            if pdf_file is None:
                continue

        content = extract_pdf_text(pdf_file, max_chars=chars_per_doc)
        if not content.strip():
            continue

        samples.append(
            {
                "filename": source_file.name,
                "content": content,
                "subfolder": subfolder,
            }
        )

        # Collect non-generic names
        if not is_generic_name(source_file.name):
            informative_file_names.append(source_file.stem)

        if subfolder and subfolder not in seen_folders:
            seen_folders.add(subfolder)
            if not is_generic_name(subfolder):
                informative_folder_names.append(subfolder)

    logger.info(
        f"Sampled {len(samples)} PDFs for schema generation"
        f" ({len(informative_folder_names)} informative folders,"
        f" {len(informative_file_names)} informative file names)"
    )

    return samples, informative_folder_names, informative_file_names


# ---------------------------------------------------------------------------
# Two-stage LLM pipeline for schema generation
# ---------------------------------------------------------------------------


async def _llm_chat_async(messages: List[Dict[str, str]], **kwargs) -> str:
    """Run llm_chat in a thread pool for async usage."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: llm_chat(messages, **kwargs))


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from an LLM response."""
    text = response.strip()

    # Try extracting from code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("JSON decode failed, trying json_repair fallback")

    # Try json_repair if available
    try:
        from json_repair import repair_json

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception as e:
        logger.debug(f"json_repair fallback failed: {e}")

    return None


def _parse_yaml_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse YAML from an LLM response."""
    text = response.strip()
    if "```yaml" in text:
        text = text.split("```yaml")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


async def analyze_document_batch(
    batch: List[Dict[str, str]],
    batch_index: int,
    language: str = "en",
) -> Optional[Dict[str, Any]]:
    """
    Stage A: Analyze a batch of documents to identify themes, entities, relationships.

    Args:
        batch: List of {filename, content, subfolder} dicts
        batch_index: Batch number for logging
        language: Target language for descriptions

    Returns:
        Dict with themes, entity_types, relationship_types, domain_hint — or None on failure
    """
    # Build excerpts
    excerpts_parts = []
    file_names = []
    for doc in batch:
        header = f"--- {doc['filename']}"
        if doc.get("subfolder"):
            header += f" (folder: {doc['subfolder']})"
        header += " ---"
        excerpts_parts.append(f"{header}\n{doc['content']}")
        file_names.append(doc["filename"])

    excerpts = "\n\n".join(excerpts_parts)

    prompt = f"""Analyze these document excerpts and identify the main themes, entity types, and relationship types.

METADATA:
- File names: {', '.join(file_names)}
- Language for output: {language}

DOCUMENT EXCERPTS:
{excerpts}

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "themes": ["theme1", "theme2", "theme3"],
  "entity_types": [
    {{"name": "EntityTypeName", "description": "What this entity represents", "examples": ["ex1", "ex2"]}}
  ],
  "relationship_types": [
    {{"name": "RELATIONSHIP_NAME", "description": "What this relationship means", "source_type": "EntityType1", "target_type": "EntityType2"}}
  ],
  "domain_hint": "Brief description of what these documents are about"
}}

Rules:
- Entity type names in PascalCase
- Relationship names in UPPER_SNAKE_CASE
- 3-8 entity types, 3-6 relationship types
- Be specific to the content, not generic"""

    try:
        response = await _llm_chat_async(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = _parse_json_response(response)
        if result and "entity_types" in result:
            logger.debug(
                f"Batch {batch_index}: found {len(result.get('entity_types', []))} entity types"
            )
            return result
        else:
            logger.warning(f"Batch {batch_index}: invalid JSON response from LLM")
            return None
    except Exception as e:
        logger.warning(f"Batch {batch_index} analysis failed: {e}")
        return None


async def run_batch_analysis(
    samples: List[Dict[str, str]],
    language: str = "en",
    batch_size: int = 0,
    max_concurrent: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run Stage A: parallel batch analysis of document samples.

    Args:
        samples: All document samples
        language: Target language
        batch_size: Documents per batch (0 = auto-compute from content length)
        max_concurrent: Max parallel LLM calls (default 4)

    Returns:
        List of valid batch results
    """
    # Auto-compute batch_size to keep ~40K chars per LLM call
    if batch_size <= 0:
        avg_chars = sum(len(s.get("content", "")) for s in samples) / max(1, len(samples))
        batch_size = max(3, min(12, int(_TARGET_BATCH_CHARS / max(1, avg_chars))))
        logger.debug(f"Auto batch_size={batch_size} (avg {avg_chars:.0f} chars/doc)")

    # Split into batches
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i : i + batch_size])

    logger.info(f"Schema analysis: {len(samples)} documents in {len(batches)} batches")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_batch(batch, idx):
        async with semaphore:
            return await analyze_document_batch(batch, idx, language)

    tasks = [_process_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter valid results
    valid_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(f"Batch {i} raised exception: {r}")
        elif r is not None:
            valid_results.append(r)

    logger.info(f"Schema analysis: {len(valid_results)}/{len(batches)} batches succeeded")
    return valid_results


def synthesize_schema(
    batch_results: List[Dict[str, Any]],
    folder_names: List[str],
    file_names: List[str],
    language: str = "en",
) -> Dict[str, Any]:
    """
    Stage B: Synthesize a final YAML schema from batch analysis results.

    Aggregates all batch results + metadata signals into a single LLM call
    that produces the final schema.

    Args:
        batch_results: Results from Stage A batches
        folder_names: Non-generic subfolder names
        file_names: Non-generic file names (sample)
        language: Target language

    Returns:
        Complete schema dict ready for save_schema()
    """
    # Aggregate batch results
    all_themes = []
    all_entity_types = []
    all_relationship_types = []
    all_domain_hints = []

    for result in batch_results:
        all_themes.extend(result.get("themes", []))
        all_entity_types.extend(result.get("entity_types", []))
        all_relationship_types.extend(result.get("relationship_types", []))
        hint = result.get("domain_hint", "")
        if hint:
            all_domain_hints.append(hint)

    # Format aggregated data for synthesis
    themes_str = ", ".join(set(all_themes)) if all_themes else "not identified"

    entities_str = ""
    for et in all_entity_types:
        name = et.get("name", "Unknown")
        desc = et.get("description", "")
        examples = et.get("examples", [])
        entities_str += f"- {name}: {desc}"
        if examples:
            entities_str += f" (examples: {', '.join(examples[:3])})"
        entities_str += "\n"

    rels_str = ""
    for rt in all_relationship_types:
        name = rt.get("name", "UNKNOWN")
        desc = rt.get("description", "")
        src = rt.get("source_type", "?")
        tgt = rt.get("target_type", "?")
        rels_str += f"- {name}: {desc} ({src} -> {tgt})\n"

    domain_hints_str = "; ".join(all_domain_hints[:5]) if all_domain_hints else "mixed documents"

    metadata_parts = []
    if folder_names:
        metadata_parts.append(f"Subfolder names: {', '.join(folder_names[:20])}")
    if file_names:
        metadata_parts.append(f"Sample file names: {', '.join(file_names[:20])}")
    metadata_str = "\n".join(metadata_parts) if metadata_parts else "No additional metadata"

    prompt = f"""You are creating a knowledge graph schema for a document corpus.

ANALYSIS RESULTS FROM {len(batch_results)} DOCUMENT BATCHES:

Domain hints: {domain_hints_str}

Themes identified: {themes_str}

Entity types found:
{entities_str}

Relationship types found:
{rels_str}

CORPUS METADATA:
{metadata_str}

Create a unified, deduplicated schema. Merge similar entity/relationship types.
Use {language} for descriptions and examples.

Output ONLY valid YAML (no markdown fences, no explanation):

domain:
  name: "Descriptive Domain Name"
  description: "1-2 sentence description"
  language: "{language}"
entities:
  - name: "EntityType"
    description: "Clear description"
    examples:
      - "example1"
      - "example2"
    attributes:
      - "description"
relationships:
  - name: "RELATIONSHIP_TYPE"
    description: "What this relationship means"
    valid_source:
      - "EntityType1"
    valid_target:
      - "EntityType2"

Rules:
- 5-12 entity types (not too many, not too few)
- 5-10 relationship types
- Entity names in PascalCase
- Relationship names in UPPER_SNAKE_CASE
- valid_source and valid_target MUST reference entity names you defined
- Each entity needs at least 2 examples
- Deduplicate: merge similar types into one well-named type"""

    for attempt in range(2):
        try:
            response = llm_chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2 if attempt == 0 else 0.1,
            )
            schema = _parse_yaml_response(response)

            if schema and "entities" in schema and "domain" in schema:
                # Add default sections if missing
                _ensure_schema_defaults(schema)
                return schema
            else:
                logger.warning(f"Schema synthesis attempt {attempt + 1}: invalid YAML structure")
        except Exception as e:
            logger.warning(f"Schema synthesis attempt {attempt + 1} failed: {e}")

    # Final fallback
    logger.warning("Schema synthesis failed, falling back to generic schema")
    return generate_schema_from_answers("generic", language)


def _ensure_schema_defaults(schema: Dict[str, Any]):
    """Add default query_routing, graph, and extraction sections if missing."""
    if "query_routing" not in schema:
        schema["query_routing"] = {
            "factual": {"vector_weight": 0.7, "graph_weight": 0.3},
            "relational": {"vector_weight": 0.2, "graph_weight": 0.8},
            "exploratory": {"vector_weight": 0.0, "graph_weight": 1.0},
            "procedural": {"vector_weight": 0.8, "graph_weight": 0.2},
        }

    if "graph" not in schema:
        schema["graph"] = {
            "enable_communities": True,
            "generate_community_summaries": True,
            "min_community_size": 2,
            "resolution": 1.0,
        }

    if "extraction" not in schema:
        schema["extraction"] = {
            "max_entities_per_chunk": 15,
            "max_relationships_per_chunk": 20,
            "min_confidence": 0.7,
        }


# ---------------------------------------------------------------------------
# Corpus-based schema generation orchestrator
# ---------------------------------------------------------------------------


async def generate_schema_from_corpus(
    sources_dir: str = None,
    pdf_dir: str = None,
    language: str = "en",
    max_docs: int = 100,
    convert_first: bool = True,
) -> Dict[str, Any]:
    """
    Full corpus-based schema generation pipeline.

    1. Optionally convert source documents to PDF
    2. Sample PDFs intelligently (max_docs, distributed across subfolders)
    3. Run two-stage LLM analysis (batch analysis → synthesis)
    4. Return schema dict

    The character budget per document is computed adaptively based on the
    number of sampled documents (fewer docs → more text per doc). Pages are
    sampled from beginning, middle, and end of each document.

    Args:
        sources_dir: Path to source documents
        pdf_dir: Path to PDF output directory
        language: Language code (default "en")
        max_docs: Maximum documents to sample (default 100)
        convert_first: Whether to run PDF conversion first (default True)

    Returns:
        Generated schema dict
    """
    if sources_dir is None:
        sources_dir = str(SOURCES_DIR)
    if pdf_dir is None:
        pdf_dir = str(PDF_DIR)

    sources_path = Path(sources_dir)
    pdf_path = Path(pdf_dir)

    # Step 1: Convert sources to PDF if requested
    if convert_first:
        try:
            from .convert_to_pdf import process_source_documents

            logger.info("Converting source documents to PDF for schema analysis...")
            pdf_path.mkdir(parents=True, exist_ok=True)
            process_source_documents(
                sources_dir=str(sources_path),
                pdf_output_dir=str(pdf_path),
            )
        except Exception as e:
            logger.warning(f"PDF conversion failed: {e}. Trying with existing PDFs...")

    # Step 2: Sample PDFs
    samples, folder_names, file_names = sample_pdfs_for_schema(
        sources_path,
        pdf_path,
        max_docs=max_docs,
    )

    if not samples:
        logger.warning("No document samples could be extracted, using generic schema")
        return generate_schema_from_answers("generic", language)

    logger.info(f"Analyzing {len(samples)} document samples for schema generation...")

    # Step 3: Stage A — batch analysis
    batch_results = await run_batch_analysis(samples, language=language)

    if len(batch_results) < 2:
        logger.warning(
            f"Only {len(batch_results)} valid batch results, "
            "falling back to single-shot schema generation"
        )
        # Fall back to legacy single-shot approach with sampled content
        legacy_samples = [{"filename": s["filename"], "content": s["content"]} for s in samples[:5]]
        return generate_schema_from_documents(legacy_samples, language)

    # Step 4: Stage B — synthesis
    schema = synthesize_schema(batch_results, folder_names, file_names, language)

    logger.info(
        f"Schema generated: {len(schema.get('entities', []))} entity types, "
        f"{len(schema.get('relationships', []))} relationship types"
    )

    return schema


def generate_schema_from_corpus_sync(
    sources_dir: str = None,
    pdf_dir: str = None,
    language: str = "en",
    max_docs: int = 100,
    convert_first: bool = True,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for generate_schema_from_corpus.

    Args:
        sources_dir: Path to source documents
        pdf_dir: Path to PDF output directory
        language: Language code
        max_docs: Maximum documents to sample
        convert_first: Whether to run PDF conversion first

    Returns:
        Generated schema dict
    """
    from .utils.async_utils import run_coroutine

    return run_coroutine(
        generate_schema_from_corpus(
            sources_dir=sources_dir,
            pdf_dir=pdf_dir,
            language=language,
            max_docs=max_docs,
            convert_first=convert_first,
        )
    )


# ---------------------------------------------------------------------------
# Existing functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def is_wizard_available() -> bool:
    """Check if interactive wizard is available (questionary installed + terminal)."""
    import sys

    return _QUESTIONARY_AVAILABLE and sys.stdin.isatty()


def get_document_sample(
    sources_dir: str = None,
    max_files: int = 5,
    max_chars_per_file: int = 2000,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Get a sample of documents from the sources directory.

    Legacy function: only reads text-readable files (.txt, .md, .html, etc.).
    For PDF-based sampling, use sample_pdfs_for_schema() instead.

    Args:
        sources_dir: Path to sources directory
        max_files: Maximum number of files to sample
        max_chars_per_file: Maximum characters to read per file

    Returns:
        Tuple of (list of {filename, content} dicts, list of subfolders found)
    """
    if sources_dir is None:
        sources_dir = SOURCES_DIR

    sources_path = Path(sources_dir)
    if not sources_path.exists():
        return [], []

    # Collect files from all subfolders
    all_files = []
    subfolders = set()

    for item in sources_path.rglob("*"):
        if item.is_file():
            # Track subfolder
            rel_path = item.relative_to(sources_path)
            if len(rel_path.parts) > 1:
                subfolders.add(rel_path.parts[0])

            # Only include text-readable files
            suffix = item.suffix.lower()
            if suffix in [".txt", ".md", ".html", ".htm", ".xml", ".json", ".csv"]:
                all_files.append(item)

    # Sample files (try to get from different subfolders if they exist)
    sampled_files = []
    if subfolders and len(all_files) > max_files:
        # Sample from each subfolder
        files_per_folder = max(1, max_files // len(subfolders))
        for folder in subfolders:
            folder_files = [f for f in all_files if f.relative_to(sources_path).parts[0] == folder]
            sampled_files.extend(
                random.sample(folder_files, min(files_per_folder, len(folder_files)))
            )

        # Fill remaining slots
        remaining = [f for f in all_files if f not in sampled_files]
        if remaining and len(sampled_files) < max_files:
            sampled_files.extend(
                random.sample(remaining, min(max_files - len(sampled_files), len(remaining)))
            )
    else:
        sampled_files = random.sample(all_files, min(max_files, len(all_files)))

    # Read content
    samples = []
    for file_path in sampled_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file]
            samples.append(
                {
                    "filename": file_path.name,
                    "content": content,
                }
            )
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")

    return samples, list(subfolders)


def generate_schema_from_documents(
    samples: List[Dict[str, str]],
    language: str = "en",
) -> Dict[str, Any]:
    """
    Use LLM to analyze document samples and generate a schema.

    Legacy single-shot approach. For corpus-based generation,
    use generate_schema_from_corpus() instead.

    Args:
        samples: List of {filename, content} dicts
        language: Language code for the schema

    Returns:
        Generated schema dict
    """
    if not samples:
        logger.warning("No document samples provided, using generic schema")
        return generate_schema_from_answers("generic", language, [], [])

    # Prepare document excerpts for LLM
    excerpts = "\n\n".join([f"=== {s['filename']} ===\n{s['content'][:1500]}" for s in samples[:5]])

    prompt = f"""Analyze these document excerpts and identify the main entity types and relationship types present.

DOCUMENT EXCERPTS:
{excerpts}

Based on these documents, provide:
1. A domain name (short, descriptive)
2. A domain description (1-2 sentences)
3. 5-10 entity types that would be useful for knowledge extraction
4. 5-8 relationship types between entities

Respond in this exact YAML format:
```yaml
domain:
  name: "Domain Name"
  description: "Brief description of the domain"
  language: "{language}"

entities:
  - name: "EntityType1"
    description: "What this entity represents"
    examples:
      - "example1"
      - "example2"
  - name: "EntityType2"
    description: "What this entity represents"
    examples:
      - "example1"

relationships:
  - name: "RELATIONSHIP_TYPE"
    description: "What this relationship represents"
    valid_source: ["EntityType1"]
    valid_target: ["EntityType2"]
```

Only output the YAML, nothing else."""

    try:
        response = llm_chat([{"role": "user", "content": prompt}])

        schema = _parse_yaml_response(response)
        if schema is None:
            raise ValueError("Could not parse YAML from LLM response")

        _ensure_schema_defaults(schema)
        return schema

    except Exception as e:
        logger.error(f"Failed to generate schema from documents: {e}")
        return generate_schema_from_answers("generic", language, [], [])


def generate_schema_from_answers(
    domain: str,
    language: str,
    custom_entities: List[str] = None,
    custom_relationships: List[str] = None,
) -> Dict[str, Any]:
    """
    Generate a schema based on wizard answers.

    Args:
        domain: Domain template key
        language: Language code
        custom_entities: Additional entity types from user
        custom_relationships: Additional relationship types from user

    Returns:
        Generated schema dict
    """
    template = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["generic"])

    # Combine template entities with custom ones
    entities = template["entities"].copy()
    if custom_entities:
        entities.extend([e for e in custom_entities if e not in entities])

    relationships = template["relationships"].copy()
    if custom_relationships:
        relationships.extend([r for r in custom_relationships if r not in relationships])

    # Build schema
    schema = {
        "domain": {
            "name": template["name"],
            "description": f"Schema for {template['name'].lower()} documents",
            "language": language,
        },
        "entities": [
            {
                "name": entity,
                "description": f"A {entity.lower()} entity",
                "examples": [],
                "attributes": ["description"],
            }
            for entity in entities
        ],
        "relationships": [
            {
                "name": rel,
                "description": f"A {rel.lower().replace('_', ' ')} relationship",
                "valid_source": entities[:3],
                "valid_target": entities[:3],
            }
            for rel in relationships
        ],
        "query_routing": {
            "factual": {"vector_weight": 0.7, "graph_weight": 0.3},
            "relational": {"vector_weight": 0.2, "graph_weight": 0.8},
            "exploratory": {"vector_weight": 0.0, "graph_weight": 1.0},
            "procedural": {"vector_weight": 0.8, "graph_weight": 0.2},
        },
        "graph": {
            "enable_communities": True,
            "generate_community_summaries": True,
            "min_community_size": 2,
            "resolution": 1.0,
        },
    }

    return schema


def save_schema(schema: Dict[str, Any], output_path: str = None) -> str:
    """
    Save schema to YAML file.

    Args:
        schema: Schema dict to save
        output_path: Path to save to (default: config/graph_schema.yaml)

    Returns:
        Path where schema was saved
    """
    if output_path is None:
        # Save to config directory in the project root (not the installed package)
        config_dir = PROJECT_DIR / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        output_path = config_dir / "graph_schema.yaml"

    output_path = Path(output_path)

    # Add header comment
    header = """# =============================================================================
# Graph Schema Configuration
# Generated by CogniDoc Schema Wizard
# =============================================================================
# This schema defines entity types, relationships, and query routing for
# knowledge graph extraction and retrieval.
# =============================================================================

"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(schema, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Schema saved to {output_path}")
    return str(output_path)


def run_interactive_wizard(sources_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    Run the interactive schema wizard.

    Args:
        sources_dir: Path to sources directory for document sampling

    Returns:
        Generated schema dict, or None if cancelled
    """
    if not is_wizard_available():
        logger.error(
            "Interactive wizard requires 'questionary'. Install with: pip install cognidoc[wizard]"
        )
        return None

    print("\n" + "=" * 60)
    print("  CogniDoc Schema Wizard")
    print("=" * 60)
    print("\nThis wizard will help you configure the knowledge graph schema")
    print("for optimal entity and relationship extraction.\n")

    # Step 1: Language
    language_choices = [f"{name} ({code})" for name, code in LANGUAGE_OPTIONS]
    language_answer = questionary.select(
        "What is the primary language of your documents?",
        choices=language_choices,
        style=WIZARD_STYLE,
    ).ask()

    if language_answer is None:
        return None

    language = dict(LANGUAGE_OPTIONS).get(language_answer.split(" (")[0], "en")

    # Step 2: Domain
    domain_choices = [
        f"{key.capitalize()}: {template['name']}" for key, template in DOMAIN_TEMPLATES.items()
    ]
    domain_answer = questionary.select(
        "What best describes your document domain?",
        choices=domain_choices,
        style=WIZARD_STYLE,
    ).ask()

    if domain_answer is None:
        return None

    domain = domain_answer.split(":")[0].lower()

    # Step 3: Custom entities
    template = DOMAIN_TEMPLATES[domain]
    print(f"\nDefault entities for {template['name']}: {', '.join(template['entities'])}")

    add_entities = questionary.confirm(
        "Would you like to add custom entity types?",
        default=False,
        style=WIZARD_STYLE,
    ).ask()

    custom_entities = []
    if add_entities:
        entities_input = questionary.text(
            "Enter additional entity types (comma-separated):",
            style=WIZARD_STYLE,
        ).ask()
        if entities_input:
            custom_entities = [e.strip() for e in entities_input.split(",") if e.strip()]

    # Step 4: Custom relationships
    print(f"\nDefault relationships: {', '.join(template['relationships'])}")

    add_relationships = questionary.confirm(
        "Would you like to add custom relationship types?",
        default=False,
        style=WIZARD_STYLE,
    ).ask()

    custom_relationships = []
    if add_relationships:
        rel_input = questionary.text(
            "Enter additional relationship types (comma-separated, use UPPER_CASE):",
            style=WIZARD_STYLE,
        ).ask()
        if rel_input:
            custom_relationships = [
                r.strip().upper().replace(" ", "_") for r in rel_input.split(",") if r.strip()
            ]

    # Step 5: Offer automatic generation
    print("\n" + "-" * 60)
    use_auto = questionary.confirm(
        "Would you prefer automatic schema generation based on your documents?\n"
        "(This will analyze a sample of your documents to create an optimized schema)",
        default=False,
        style=WIZARD_STYLE,
    ).ask()

    if use_auto:
        print("\nAnalyzing documents...")
        samples, subfolders = get_document_sample(sources_dir)

        if samples:
            print(
                f"Found {len(samples)} documents to analyze"
                + (f" from {len(subfolders)} subfolders" if subfolders else "")
            )
            schema = generate_schema_from_documents(samples, language)
            print("Schema generated from document analysis!")
        else:
            print("No readable documents found. Using template-based schema.")
            schema = generate_schema_from_answers(
                domain, language, custom_entities, custom_relationships
            )
    else:
        schema = generate_schema_from_answers(
            domain, language, custom_entities, custom_relationships
        )

    # Step 6: Save
    save_path = save_schema(schema)

    print("\n" + "=" * 60)
    print(f"  Schema saved to: {save_path}")
    print("=" * 60 + "\n")

    return schema


def run_non_interactive_wizard(
    sources_dir: str = None,
    use_auto: bool = True,
    domain: str = "generic",
    language: str = "en",
    max_docs: int = 100,
) -> Dict[str, Any]:
    """
    Run schema generation without user interaction.

    When use_auto=True, uses the corpus-based two-stage LLM pipeline
    (convert to PDF → sample → batch analysis → synthesis).

    Args:
        sources_dir: Path to sources directory
        use_auto: Whether to use automatic document analysis
        domain: Domain template to use if not auto
        language: Language code
        max_docs: Maximum documents to sample for corpus-based generation

    Returns:
        Generated schema dict
    """
    if use_auto:
        try:
            schema = generate_schema_from_corpus_sync(
                sources_dir=sources_dir,
                pdf_dir=str(PDF_DIR),
                language=language,
                max_docs=max_docs,
                convert_first=True,
            )
        except Exception as e:
            logger.warning(f"Corpus-based schema generation failed: {e}")
            logger.info("Falling back to legacy document sampling...")
            samples, _ = get_document_sample(sources_dir)
            if samples:
                logger.info(f"Analyzing {len(samples)} documents for schema generation...")
                schema = generate_schema_from_documents(samples, language)
            else:
                logger.info("No documents found, using template schema")
                schema = generate_schema_from_answers(domain, language)
    else:
        schema = generate_schema_from_answers(domain, language)

    save_schema(schema)
    return schema


def check_existing_schema(config_dir: str = None) -> Optional[str]:
    """
    Check if a schema already exists.

    Args:
        config_dir: Path to config directory

    Returns:
        Path to existing schema, or None
    """
    if config_dir is None:
        config_dir = PROJECT_DIR / "config"

    schema_path = Path(config_dir) / "graph_schema.yaml"
    if schema_path.exists():
        return str(schema_path)

    return None


def prompt_schema_choice(existing_path: str) -> str:
    """
    Ask user whether to use existing schema or create new.

    Args:
        existing_path: Path to existing schema

    Returns:
        "use_existing", "create_new", or "skip"
    """
    if not is_wizard_available():
        logger.info(f"Using existing schema: {existing_path}")
        return "use_existing"

    print(f"\nExisting schema found: {existing_path}")

    choice = questionary.select(
        "What would you like to do?",
        choices=[
            "Use existing schema",
            "Create new schema (interactive wizard)",
            "Skip graph extraction",
        ],
        style=WIZARD_STYLE,
    ).ask()

    if choice is None or "Skip" in choice:
        return "skip"
    elif "existing" in choice:
        return "use_existing"
    else:
        return "create_new"


__all__ = [
    "is_wizard_available",
    "run_interactive_wizard",
    "run_non_interactive_wizard",
    "check_existing_schema",
    "prompt_schema_choice",
    "generate_schema_from_documents",
    "generate_schema_from_answers",
    "save_schema",
    "get_document_sample",
    # Corpus-based schema generation
    "generate_schema_from_corpus",
    "generate_schema_from_corpus_sync",
    "is_generic_name",
    "sample_pdfs_for_schema",
    "extract_pdf_text",
]
