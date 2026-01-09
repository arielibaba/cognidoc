# Session CogniDoc - 9 janvier 2026

## Résumé

Corrections majeures pour le routage agent, la détection de langue, et les questions méta sur la base de données.

## Tâches complétées cette session

| Tâche | Fichier | Description |
|-------|---------|-------------|
| **Fix patterns meta-questions** | `complexity.py` | Patterns plus flexibles pour "combien de documents", typos inclus |
| **Fix language consistency** | `prompts/*.md` | Règles de langue dans tous les prompts (rewrite, final_answer, agent) |
| **DatabaseStatsTool** | `agent_tools.py` | Nouvel outil pour répondre aux méta-questions sur la base |
| **Language detection** | `cognidoc_app.py` | Détection automatique FR/EN avec préfixes de clarification |
| **Tests E2E** | `tests/test_e2e_language_and_count.py` | 10 nouveaux tests pour patterns et langue |
| **Fix Gemini SDK** | `pyproject.toml` | Ajout dépendance `google-genai` dans extras |
| **Fix helpers TypeError** | `helpers.py` | Gestion format multimodal Gradio (list/None) |
| **Fix reranking provider** | `advanced_rag.py` | Utilisation `llm_chat()` au lieu de `ollama.Client()` |

## Modifications clés

### 1. Patterns DATABASE_META_PATTERNS (`complexity.py`)

Patterns plus robustes pour détecter les questions sur la base :

```python
DATABASE_META_PATTERNS = [
    # French patterns - flexible matching
    r"\bcombien de doc",      # "combien de documents", typos
    r"\bcombien.{0,20}base\b", # "combien...base" avec 20 chars max
    r"\bbase.{0,15}comprend",  # "cette base comprend", "la base comprend-elle"
    r"\bbase.{0,15}contient",  # "la base contient"
    ...
]
```

### 2. DatabaseStatsTool (`agent_tools.py`)

Nouvel outil (9e outil) pour répondre aux questions sur la base :

```python
class DatabaseStatsTool(BaseTool):
    name = ToolName.DATABASE_STATS
    # Retourne: total_documents, total_chunks, graph_nodes, graph_edges
```

### 3. Détection de langue (`cognidoc_app.py`)

```python
def detect_query_language(query: str) -> str:
    """Détecte FR ou EN basé sur indicateurs linguistiques."""
    french_indicators = [" est ", " sont ", " que ", ...]
    ...

def get_clarification_prefix(lang: str) -> str:
    if lang == "fr":
        return "**Clarification requise :**"
    return "**Clarification needed:**"
```

### 4. Règles de langue dans les prompts

Tous les prompts incluent maintenant :

```markdown
## Language Rules
- ALWAYS respond in the SAME LANGUAGE as the user's question.
- If the user asks in French, respond in French.
- If the user asks in English, respond in English.
```

## Tests (43+ tests passent)

| Module | Tests |
|--------|-------|
| `test_agent_tools.py` | 33 |
| `test_e2e_language_and_count.py` | 10 |
| **Total validé** | **43+** |

## Commandes CLI

```bash
# Lancer l'app (avec agent activé)
uv run python -m cognidoc.cognidoc_app

# Sans reranking (plus rapide)
uv run python -m cognidoc.cognidoc_app --no-rerank

# Tests
uv run python -m pytest tests/ -v
```

## Configuration

```
LLM:       gemini-2.0-flash (Gemini)
Embedding: qwen3-embedding:0.6b (Ollama)
Agent:     Activé (seuil complexité: 0.55)
DatabaseStatsTool: Activé pour meta-questions
```

## Structure mise à jour

```
src/cognidoc/
├── complexity.py        # DATABASE_META_PATTERNS améliorés
├── agent_tools.py       # 9 outils (NEW: database_stats)
├── agent.py             # Règles de langue dans SYSTEM_PROMPT
├── cognidoc_app.py      # detect_query_language(), get_clarification_prefix()
├── helpers.py           # Fix TypeError format multimodal
└── prompts/
    ├── system_prompt_rewrite_query.md      # Language Preservation rules
    └── system_prompt_generate_final_answer.md # Language Rules

tests/
├── test_agent_tools.py              # 33 tests
└── test_e2e_language_and_count.py   # 10 tests (NEW)
```

## Bugs corrigés

1. **Agent path non déclenché** - Patterns trop restrictifs pour "combien de documents"
2. **Réponses en anglais** - Règles de langue manquantes dans prompts
3. **TypeError helpers.py** - Format multimodal Gradio non géré
4. **Reranking 404** - Utilisait ollama.Client() avec modèle Gemini
5. **Gemini SDK manquant** - google-genai non installé dans venv

## Améliorations futures

1. **Streaming agent** - Afficher les étapes de raisonnement en temps réel
2. **Caching agent** - Mettre en cache les résultats des outils
3. **Support langues additionnelles** - Espagnol, Allemand, etc.
