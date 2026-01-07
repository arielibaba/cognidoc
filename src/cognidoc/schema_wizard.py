"""
Schema Wizard - Interactive and automatic graph schema generation.

Provides two modes:
1. Interactive: Ask user questions about their domain
2. Automatic: Analyze document samples to generate schema
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

from .utils.logger import logger
from .utils.llm_client import llm_chat
from .constants import SOURCES_DIR

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
    WIZARD_STYLE = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:green bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
    ])


# Predefined domain templates
DOMAIN_TEMPLATES = {
    "technical": {
        "name": "Technical Documentation",
        "entities": ["Product", "Technology", "Feature", "Component", "Process", "API", "Configuration"],
        "relationships": ["USES", "IMPLEMENTS", "PART_OF", "DEPENDS_ON", "PRODUCES", "CONFIGURES"],
    },
    "legal": {
        "name": "Legal Documents",
        "entities": ["Law", "Article", "Person", "Organization", "Contract", "Obligation", "Right"],
        "relationships": ["REFERENCES", "AMENDS", "APPLIES_TO", "OBLIGATES", "GRANTS", "SIGNED_BY"],
    },
    "medical": {
        "name": "Medical/Healthcare",
        "entities": ["Disease", "Symptom", "Treatment", "Medication", "Procedure", "Patient", "Doctor"],
        "relationships": ["TREATS", "CAUSES", "INDICATES", "PRESCRIBES", "DIAGNOSES", "CONTRAINDICATED"],
    },
    "scientific": {
        "name": "Scientific Research",
        "entities": ["Concept", "Theory", "Experiment", "Result", "Researcher", "Institution", "Publication"],
        "relationships": ["SUPPORTS", "CONTRADICTS", "CITES", "CONDUCTED_BY", "PUBLISHED_IN", "PROVES"],
    },
    "business": {
        "name": "Business/Corporate",
        "entities": ["Company", "Product", "Service", "Person", "Department", "Project", "Strategy"],
        "relationships": ["OWNS", "MANAGES", "WORKS_FOR", "PARTNERS_WITH", "COMPETES_WITH", "PROVIDES"],
    },
    "educational": {
        "name": "Educational Content",
        "entities": ["Topic", "Concept", "Course", "Lesson", "Instructor", "Student", "Assessment"],
        "relationships": ["TEACHES", "PREREQUISITE_FOR", "COVERS", "ASSESSES", "ENROLLED_IN", "PART_OF"],
    },
    "generic": {
        "name": "Generic/Mixed",
        "entities": ["Entity", "Concept", "Person", "Organization", "Process", "Document", "Event"],
        "relationships": ["RELATED_TO", "PART_OF", "CREATED_BY", "REFERENCES", "OCCURS_IN", "INVOLVES"],
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


def is_wizard_available() -> bool:
    """Check if interactive wizard is available (questionary installed)."""
    return _QUESTIONARY_AVAILABLE


def get_document_sample(
    sources_dir: str = None,
    max_files: int = 5,
    max_chars_per_file: int = 2000,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Get a sample of documents from the sources directory.

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
            sampled_files.extend(random.sample(folder_files, min(files_per_folder, len(folder_files))))

        # Fill remaining slots
        remaining = [f for f in all_files if f not in sampled_files]
        if remaining and len(sampled_files) < max_files:
            sampled_files.extend(random.sample(remaining, min(max_files - len(sampled_files), len(remaining))))
    else:
        sampled_files = random.sample(all_files, min(max_files, len(all_files)))

    # Read content
    samples = []
    for file_path in sampled_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file]
            samples.append({
                "filename": file_path.name,
                "content": content,
            })
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")

    return samples, list(subfolders)


def generate_schema_from_documents(
    samples: List[Dict[str, str]],
    language: str = "en",
) -> Dict[str, Any]:
    """
    Use LLM to analyze document samples and generate a schema.

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
    excerpts = "\n\n".join([
        f"=== {s['filename']} ===\n{s['content'][:1500]}"
        for s in samples[:5]
    ])

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

        # Extract YAML from response
        yaml_content = response
        if "```yaml" in response:
            yaml_content = response.split("```yaml")[1].split("```")[0]
        elif "```" in response:
            yaml_content = response.split("```")[1].split("```")[0]

        schema = yaml.safe_load(yaml_content)

        # Add query routing defaults
        schema["query_routing"] = {
            "factual": {"vector_weight": 0.7, "graph_weight": 0.3},
            "relational": {"vector_weight": 0.2, "graph_weight": 0.8},
            "exploratory": {"vector_weight": 0.0, "graph_weight": 1.0},
            "procedural": {"vector_weight": 0.8, "graph_weight": 0.2},
        }

        schema["graph"] = {
            "enable_communities": True,
            "generate_community_summaries": True,
            "min_community_size": 2,
            "resolution": 1.0,
        }

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
        # Save to config directory
        config_dir = Path(__file__).parent.parent.parent / "config"
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
        logger.error("Interactive wizard requires 'questionary'. Install with: pip install cognidoc[wizard]")
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
        f"{key.capitalize()}: {template['name']}"
        for key, template in DOMAIN_TEMPLATES.items()
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
            custom_relationships = [r.strip().upper().replace(" ", "_") for r in rel_input.split(",") if r.strip()]

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
            print(f"Found {len(samples)} documents to analyze" +
                  (f" from {len(subfolders)} subfolders" if subfolders else ""))
            schema = generate_schema_from_documents(samples, language)
            print("Schema generated from document analysis!")
        else:
            print("No readable documents found. Using template-based schema.")
            schema = generate_schema_from_answers(domain, language, custom_entities, custom_relationships)
    else:
        schema = generate_schema_from_answers(domain, language, custom_entities, custom_relationships)

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
) -> Dict[str, Any]:
    """
    Run schema generation without user interaction.

    Args:
        sources_dir: Path to sources directory
        use_auto: Whether to use automatic document analysis
        domain: Domain template to use if not auto
        language: Language code

    Returns:
        Generated schema dict
    """
    if use_auto:
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
        config_dir = Path(__file__).parent.parent.parent / "config"

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
]
