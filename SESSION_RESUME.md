# Session CogniDoc - 9 janvier 2026

## Résumé

Corrections majeures pour le routage agent, la détection de langue, les questions méta sur la base de données, et la **mémoire conversationnelle du chatbot**.

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
| **Fix agent response empty** | `cognidoc_app.py` | Capture correcte du retour du générateur `run_streaming()` |
| **Fix chatbot memory** | `agent.py`, `cognidoc_app.py`, `helpers.py` | Mémoire conversationnelle fonctionnelle |
| **Fix DatabaseStatsTool list_documents** | `agent_tools.py` | Retourne les noms des documents avec `list_documents=True` |

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

### 5. Mémoire conversationnelle (`cognidoc_app.py`, `agent.py`, `helpers.py`)

La mémoire du chatbot fonctionne maintenant correctement :

```
User: "Combien de documents cette base comprend-elle?"
Bot:  "Cette base de données comprend 5 documents."

User: "cite-les-moi"
Bot:  "Cette base de données comprend les 5 documents suivants: test_document, Rapport Sémantique, ..."
```

**Flux corrigé:**
1. Query rewriter transforme "cite-les-moi" → "Cite-moi les 5 documents que cette base comprend."
2. L'agent reçoit la query réécrite (pas le message brut)
3. DatabaseStatsTool retourne les noms des documents via `list_documents=True`

### 6. DatabaseStatsTool amélioré (`agent_tools.py`)

```python
class DatabaseStatsTool(BaseTool):
    parameters = {
        "list_documents": "Set to true to get the list of document names/titles"
    }

    def execute(self, list_documents: bool = False) -> ToolResult:
        # Utilise get_all_documents() au lieu de .documents
        docs = ki.get_all_documents()
        if list_documents:
            doc_names = [doc.metadata.get('source', {}).get('document') for doc in docs]
            stats["document_names"] = sorted(list(set(doc_names)))
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
LLM:       gemini-2.5-flash (Gemini)
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
6. **Réponse agent vide** - Le générateur `run_streaming()` n'était pas correctement consommé, puis `run()` était appelé une seconde fois inutilement. Fix: capture du retour via `StopIteration.value`
7. **Mémoire chatbot cassée** - "cite-les-moi" après "combien de documents" causait "que voulez-vous citer?"
   - **Cause racine**: `KeyError: '"answer"'` dans `agent.py` dû aux accolades non échappées dans SYSTEM_PROMPT
   - **Fix**: `{"answer": "..."}` → `{{"answer": "..."}}`
8. **Agent utilisant raw query** - L'agent recevait "cite-les-moi" au lieu de la query réécrite avec contexte
   - **Fix**: `agent.run_streaming(candidates[0])` au lieu de `user_message`
9. **parse_rewritten_query incomplet** - Ne gérait que `- ` pas `* ` comme style de bullet
   - **Fix**: Ajout `elif stripped.startswith('* '):`
10. **DatabaseStatsTool sans noms de documents** - Utilisait `.documents` qui n'existe pas
    - **Fix**: Utilisation de `get_all_documents()` + extraction des métadonnées `source.document`

## Améliorations implémentées (session 2)

### 1. Cache des résultats d'outils (`agent_tools.py`)

```python
class ToolCache:
    """TTL-based cache for tool results."""
    TTL_CONFIG = {
        "database_stats": 300,      # 5 minutes
        "retrieve_vector": 120,     # 2 minutes
        "retrieve_graph": 120,
        "lookup_entity": 300,
        "compare_entities": 180,
    }

    @classmethod
    def get(cls, tool_name: str, **kwargs) -> Optional[Any]:
        # Check cache with MD5 hash key
        ...

    @classmethod
    def set(cls, tool_name: str, result: Any, **kwargs) -> None:
        # Store with timestamp
        ...
```

**Avantages:**
- Réduit la latence pour les requêtes répétées
- TTL configurable par outil
- Log cache hit/miss pour debug
- Indicateur `[cached]` dans les résultats

### 2. Streaming granulaire dans l'UI (`cognidoc_app.py`)

```python
state_emoji = {
    AgentState.THINKING: "🤔",
    AgentState.ACTING: "⚡",
    AgentState.OBSERVING: "👁️",
    AgentState.REFLECTING: "💭",
}
progress_lines.append(f"{state_emoji} {message}")
history[-1]["content"] = f"*Processing query...*\n\n{progress_display}"
yield convert_history_to_tuples(history)
```

L'utilisateur voit maintenant en temps réel:
- 🤔 [Step 1/7] Analyzing query...
- 🤔 Thought: I need to search for...
- ⚡ Calling retrieve_vector(query=...)
- 👁️ Result [cached]: Found 5 documents...
- 💭 Analysis: The documents contain...

### 3. Prompts optimisés pour réduire les steps (`agent.py`)

**Avant:** 5-7 steps typiques
**Après:** 2-3 steps pour la plupart des requêtes

```python
SYSTEM_PROMPT = """You are an efficient research assistant. Your goal is to answer questions QUICKLY with MINIMAL steps.

## Efficiency Guidelines - CRITICAL
1. **One retrieval is usually enough.** After ONE successful retrieve_vector or retrieve_graph call, you likely have enough information. Proceed to final_answer.
2. **Skip synthesis for simple questions.** Use final_answer directly after getting relevant documents.
3. **Target: 2-3 steps max for most queries.** Complex comparisons may need 4 steps.
...
"""
```

**Changements clés:**
- SYSTEM_PROMPT plus directif et efficace
- THINK_PROMPT simplifié (encourage action immédiate)
- REFLECT_PROMPT focalisé sur "Can you answer NOW?"
- Instructions claires pour éviter redondances

## Tests (127 tests passent)

| Module | Tests |
|--------|-------|
| `test_agent.py` | 27 |
| `test_agent_tools.py` | 33 |
| `test_complexity.py` | 24 |
| `test_e2e_language_and_count.py` | 10 |
| `test_providers.py` | 33 |
| **Total validé** | **127** |

### 4. Fix langue dans le Fast Path (`user_prompt_generate_final_answer.md`)

Le fast path répondait parfois en anglais (ex: "No relevant details are available").

**Avant:**
```markdown
- If insufficient information is available, respond clearly with:
  **"No relevant details are available."**
```

**Après:**
```markdown
- If insufficient information is available, respond in the user's language:
  - French: **"Je n'ai pas trouvé d'informations pertinentes..."**
  - English: **"I could not find relevant information..."**
- CRITICAL: Deliver your ENTIRE response in the SAME LANGUAGE as the user's question.
```

## Commits de cette session

| Hash | Description |
|------|-------------|
| `a56ecdf` | Improve agent performance: caching, streaming, and optimized prompts |
| `0a05114` | Update SESSION_RESUME.md with performance improvements |
| `c68164f` | Fix language consistency in fast path responses |

## Améliorations implémentées (session 3)

### 1. Cache Persistant SQLite (`utils/tool_cache.py`)

Remplacement du cache mémoire par un cache SQLite persistant :

```python
class PersistentToolCache:
    """SQLite-based persistent cache for tool results."""
    TTL_CONFIG = {
        "database_stats": 300,      # 5 minutes
        "retrieve_vector": 120,     # 2 minutes
        "retrieve_graph": 120,
        "lookup_entity": 300,
        "compare_entities": 180,
    }

    @classmethod
    def get(cls, tool_name: str, **kwargs) -> Optional[Any]
    @classmethod
    def set(cls, tool_name: str, result: Any, **kwargs) -> None
    @classmethod
    def cleanup_expired(cls) -> int  # Nettoie les entrées expirées
    @classmethod
    def stats(cls) -> Dict[str, Any]
```

**Avantages:**
- Persiste entre les redémarrages de l'app
- Même API que l'ancien ToolCache (migration transparente)
- Cleanup automatique des entrées expirées
- Stockage dans `data/cache/tool_cache.db`

### 2. Métriques de Performance (`utils/metrics.py`)

Nouveau système de métriques avec stockage SQLite :

```python
@dataclass
class QueryMetrics:
    path: str                    # "fast", "enhanced", "agent"
    query_type: Optional[str]    # "FACTUAL", "EXPLORATORY", etc.
    complexity_score: Optional[float]
    total_time_ms: float
    rewrite_time_ms: Optional[float]
    retrieval_time_ms: Optional[float]
    rerank_time_ms: Optional[float]
    llm_time_ms: Optional[float]
    cache_hits: int = 0
    cache_misses: int = 0
    agent_steps: Optional[int] = None
    tools_used: Optional[List[str]] = None

class PerformanceMetrics:
    def log_query(self, metrics: QueryMetrics) -> None
    def get_global_stats(self) -> Dict[str, Any]
    def get_latency_by_path(self) -> List[Dict]
    def get_latency_over_time(self, hours: int = 24) -> List[Dict]
    def get_path_distribution(self) -> List[Dict]
    def get_recent_queries(self, limit: int = 20) -> List[Dict]
```

### 3. Dashboard Metrics (`cognidoc_app.py`)

Nouvel onglet "Metrics" dans l'interface Gradio avec :

| Composant | Description |
|-----------|-------------|
| **Stats globales** | Total queries, avg latency, cache hit rate, avg agent steps |
| **Latence par path** | Bar chart Plotly (agent vs fast vs enhanced) |
| **Distribution paths** | Pie chart Plotly |
| **Latence temporelle** | Line chart avec évolution sur 24h |
| **Table requêtes** | 20 dernières requêtes avec détails |

```python
# Fonctions dashboard
def create_latency_by_path_chart() -> go.Figure
def create_path_distribution_chart() -> go.Figure
def create_latency_over_time_chart() -> go.Figure
def get_recent_queries_dataframe() -> pd.DataFrame
def get_global_stats_html() -> str
```

### 4. Fichiers modifiés/créés

| Fichier | Action |
|---------|--------|
| `src/cognidoc/utils/tool_cache.py` | **NOUVEAU** - PersistentToolCache SQLite |
| `src/cognidoc/utils/metrics.py` | **NOUVEAU** - PerformanceMetrics + QueryMetrics |
| `src/cognidoc/agent_tools.py` | Import PersistentToolCache, tracking cache hits |
| `src/cognidoc/cognidoc_app.py` | Dashboard Metrics, logging QueryMetrics |
| `src/cognidoc/constants.py` | TOOL_CACHE_DB, METRICS_DB paths |
| `pyproject.toml` | Ajout `plotly>=5.0` aux dépendances UI |

### 5. Commits session 3

| Hash | Description |
|------|-------------|
| `c2521fa` | Add persistent SQLite cache and performance metrics dashboard |
| `0a8f73c` | Fix QueryType enum serialization for SQLite metrics |
| `6eddd30` | Add plotly to UI dependencies for metrics dashboard |

### 6. Tests vérifiés

```bash
# 127 tests passent
uv run python -m pytest tests/ -v
```

| Métrique | Valeur |
|----------|--------|
| Tests passés | 127 |
| Couverture cache | ✅ |
| Couverture metrics | ✅ |

### 7. Export Métriques CSV/JSON (`utils/metrics.py`, `cognidoc_app.py`)

Ajout de fonctions d'export et boutons dans le dashboard :

```python
class PerformanceMetrics:
    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export all metrics to CSV format."""
        ...

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all metrics to JSON format with global stats."""
        ...
```

**Boutons dans l'onglet Metrics:**
- 📥 Export CSV - Télécharge toutes les requêtes en CSV
- 📥 Export JSON - Télécharge avec stats globales incluses

**Format JSON:**
```json
{
  "exported_at": "2026-01-09T13:28:55",
  "total_records": 4,
  "global_stats": { ... },
  "queries": [ ... ]
}
```

### 8. Commits session 3 (complet)

| Hash | Description |
|------|-------------|
| `c2521fa` | Add persistent SQLite cache and performance metrics dashboard |
| `0a8f73c` | Fix QueryType enum serialization for SQLite metrics |
| `6eddd30` | Add plotly to UI dependencies for metrics dashboard |
| `1520d57` | Update SESSION_RESUME.md with session 3 changes |
| `721ebc8` | Add CSV and JSON export for metrics dashboard |

## Améliorations implémentées (session 4)

### 1. Batching Async des Embeddings (`create_embeddings.py`)

Refactoring complet pour traitement par lots avec requêtes concurrentes :

```python
# Nouveaux paramètres
create_embeddings(
    chunks_dir="...",
    embeddings_dir="...",
    embed_model="qwen3-embedding:0.6b",
    batch_size=32,        # Chunks par batch
    max_concurrent=4,     # Requêtes HTTP simultanées
)
```

**Architecture:**
```
Phase 1: Collecte
├── Scan des fichiers
├── Vérification cache
└── Filtrage (parent chunks, fichiers courts)
         ↓
Phase 2: Batching async
├── Batches de 32 chunks
├── 4 requêtes HTTP concurrentes (asyncio + httpx)
├── Progress bar tqdm
└── Cache + écriture fichiers
```

**Benchmark:** 8 textes
- Séquentiel: 1.01s (126ms/texte)
- Async batch: 0.19s (24ms/texte)
- **Speedup: 5.27x**

### 2. Parallélisation PDF→Images (`convert_pdf_to_image.py`)

Conversion parallèle avec ProcessPoolExecutor :

```python
convert_pdf_to_image(
    pdf_dir="data/pdfs",
    image_dir="data/images",
    dpi=600,
    max_workers=4,    # Processus parallèles
    parallel=True,
)
```

**Benchmark:** 7 PDFs, 12 pages, 150 DPI
- Séquentiel: 13.63s
- Parallèle (4 workers): 1.23s
- **Speedup: 11x**

### 3. Méthode embed_async() (`utils/embedding_providers.py`)

Nouvelle méthode async pour OllamaEmbeddingProvider :

```python
class OllamaEmbeddingProvider:
    async def embed_async(
        self,
        texts: List[str],
        max_concurrent: int = 4
    ) -> List[List[float]]:
        """Embed avec requêtes HTTP concurrentes."""
        ...
```

### 4. Fichiers modifiés

| Fichier | Action |
|---------|--------|
| `src/cognidoc/create_embeddings.py` | Refactoring complet avec batching async |
| `src/cognidoc/convert_pdf_to_image.py` | Ajout ProcessPoolExecutor + tqdm |
| `src/cognidoc/utils/embedding_providers.py` | Ajout `embed_async()` |

### 5. Fix asyncio conflict (`create_embeddings.py`)

Le pipeline async appelait `asyncio.run()` alors qu'il était déjà dans une event loop :

```python
# Détection du contexte async
try:
    loop = asyncio.get_running_loop()
    in_async_context = True
except RuntimeError:
    in_async_context = False

# Workaround: ThreadPoolExecutor pour isoler l'event loop
if in_async_context:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, embed_batch_async(...))
        success, errors = future.result()
else:
    success, errors = asyncio.run(embed_batch_async(...))
```

### 6. Commits session 4

| Hash | Description |
|------|-------------|
| `74f94ec` | Add batched async embedding generation for faster ingestion |
| `eed1313` | Add parallel PDF to image conversion for faster ingestion |
| `077202b` | Fix asyncio conflict in pipeline |

### 8. Optimisations pour M2 16GB

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `max_workers` (PDF) | 4 | Évite saturation mémoire unifiée |
| `max_concurrent` (embed) | 4 | Overlap I/O sans surcharger Ollama |
| `batch_size` | 32 | Bon équilibre mémoire/throughput |

### 9. Résultats tests pipeline

Pipeline complet exécuté avec succès :

| Étape | Temps | Détails |
|-------|-------|---------|
| **PDF → Images** | ~6s | 7 PDFs, 12 pages (parallèle 4 workers) |
| **YOLO Detection** | ~17s | 17 images, 17 text regions |
| **Embedding Generation** | 0.88s | 14 nouveaux + 10 du cache |
| **Index Building** | 0.64s | 29 documents indexés |
| **Graph Extraction** | 53.6s | 24 chunks, 64 entities, 40 relations |
| **Graph Building** | 9.7s | 25 nodes, 19 edges, 11 communities |

**Tests unitaires:** 127/127 passés

## Améliorations implémentées (session 5)

### 1. Fix comptage documents sources vs chunks (`agent_tools.py`)

**Problème:** `DatabaseStatsTool` retournait 29 (chunks) au lieu de 2 (documents sources uniques).

**Solution:** Extraction des noms de fichiers sources depuis les métadonnées de chaque chunk :

```python
def execute(self, list_documents: bool = False) -> ToolResult:
    """Get database statistics.

    Returns unique source document count (not chunk count).
    - total_documents: Number of unique source files (PDFs)
    - total_chunks: Number of chunks in the index
    - document_names: List of unique source document names
    """
    # Extract unique source documents from ALL chunks
    unique_sources = set()
    for doc in docs:
        source = doc.metadata.get('source', {})
        if isinstance(source, dict):
            name = source.get('document')
        # ...

    stats["total_documents"] = len(unique_sources)  # Sources uniques
    stats["total_chunks"] = len(docs)               # Chunks
```

### 2. Patterns pour listage documents (`complexity.py`)

Ajout de 12 nouveaux patterns pour détecter les questions de listage :

```python
DATABASE_META_PATTERNS = [
    # ... existing patterns ...

    # French patterns - listing documents (NEW)
    r"\bliste[rz]?\b.*\bdoc",     # "liste les documents"
    r"\bnoms?\b.*\bdoc",          # "noms des documents"
    r"\bcite[rz]?\b.*\bdoc",      # "cite les documents"
    r"\b[eé]num[eè]re[rz]?\b.*\bdoc",  # "énumère les documents"
    r"\bquels?\b.*\bdoc",         # "quels documents"
    r"\bdonne.*\bnoms?\b",        # "donne-moi les noms"

    # English patterns - listing documents (NEW)
    r"\blist\b.*\bdoc",           # "list the documents"
    r"\bnames?\b.*\bdoc",         # "names of documents"
    r"\bwhat\b.*\bdoc",           # "what documents"
    r"\bwhich\b.*\bdoc",          # "which documents"
]
```

### 3. Réécriture README.md

Le README a été entièrement réécrit pour plus de clarté :
- Quick Start en premier (3 étapes simples)
- Suppression des redondances
- Diagrammes simplifiés
- Enchaînement logique : installer → configurer → utiliser → comprendre
- 426 lignes supprimées, 181 ajoutées

### 4. Commits session 5

| Hash | Description |
|------|-------------|
| `a4ee039` | Fix document count to return unique sources instead of chunks |
| `65e1a61` | Update documentation with session 5 changes |
| `2a1c93f` | Rewrite README.md for clarity and logical flow |

### 5. Tests vérifiés

```bash
# 127 tests passent
.venv/bin/python -m pytest tests/ -v
```

### 6. État final

- **App:** Fonctionne correctement (testée avec questions sur documents)
- **DatabaseStatsTool:** Retourne 2 documents sources (pas 29 chunks)
- **Patterns listage:** 12 nouveaux patterns pour détecter "liste les docs", "quels documents", etc.
- **Documentation:** README.md réécrit, SESSION_RESUME.md et CLAUDE.md mis à jour

## Améliorations implémentées (session 6)

### 1. Upgrade Gemini 2.0 → 2.5 Flash

Remplacement du modèle LLM par défaut dans tout le codebase :

```python
# Avant
DEFAULT_LLM_MODEL = "gemini-2.0-flash"

# Après
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
```

**Fichiers modifiés (11):**
- `src/cognidoc/constants.py`
- `src/cognidoc/utils/llm_providers.py`
- `src/cognidoc/utils/llm_client.py`
- `src/cognidoc/api.py`
- `.env`
- `README.md`
- `CLAUDE.md`
- Et autres fichiers de configuration

**Avantages Gemini 2.5 Flash:**
- Meilleur raisonnement (thinking tokens)
- Qualité similaire avec `thinking=0`
- 20-30% moins de tokens utilisés

### 2. YOLO Batching (`extract_objects_from_image.py`)

Batch inference GPU pour améliorer l'utilisation mémoire sur M2/M3 :

```python
def process_images_batch(
    self,
    image_paths: List[Path]
) -> List[Tuple[Path, np.ndarray, List[Dict], List[Dict]]]:
    """Run batch inference on multiple images for improved GPU utilization."""
    with timer(f"YOLO batch inference ({len(valid_paths)} images)"):
        batch_results = self.model(
            [str(p) for p in valid_paths],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
```

**Nouveaux paramètres CLI:**
```bash
--yolo-batch-size 2      # Taille des batches (défaut: 2)
--no-yolo-batching       # Désactiver le batching
```

**Gain estimé:** 15-30% sur M2 16GB (marginal car MPS, pas CUDA)

### 3. Async Entity Extraction (`extract_entities.py`)

Extraction parallèle des entités avec semaphore pour contrôle de concurrence :

```python
async def extract_from_chunks_dir_async(
    chunks_dir: str = None,
    config: Optional[GraphConfig] = None,
    include_parent_chunks: bool = False,
    max_concurrent: int = 4,
    show_progress: bool = True,
) -> List[ExtractionResult]:
    """Async version with concurrent extraction using semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_chunk(chunk_file: Path) -> Optional[ExtractionResult]:
        async with semaphore:
            result = await extract_from_chunk_async(...)

    tasks = [process_chunk(cf) for cf in chunk_files]
    results_raw = await async_tqdm.gather(*tasks, desc="Entity extraction")
```

**Nouveaux paramètres CLI:**
```bash
--entity-max-concurrent 4   # Requêtes LLM simultanées (défaut: 4)
--no-async-extraction       # Désactiver l'extraction async
```

**Gain estimé:** 2-4x speedup pour GraphRAG

### 4. json_mode pour llm_chat_async (`utils/llm_client.py`)

Support du mode JSON pour les réponses structurées :

```python
async def llm_chat_async(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
```

### 5. TOP_K_REFS Configuration (`constants.py`)

Nouvelle constante pour afficher toutes les références :

```python
TOP_K_RETRIEVED_CHILDREN = int(os.getenv("TOP_K_RETRIEVED_CHILDREN", "10"))
TOP_K_RERANKED_PARENTS = int(os.getenv("TOP_K_RERANKED_PARENTS", "5"))
# Number of references to display (defaults to TOP_K_RERANKED_PARENTS)
TOP_K_REFS = int(os.getenv("TOP_K_REFS", str(TOP_K_RERANKED_PARENTS)))
```

### 6. README pip install workflow

Réécriture de la section "Getting Started" pour montrer pip install comme méthode principale :

```bash
# Installation via pip (pas besoin de cloner)
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Créer un nouveau projet
mkdir my-doc-assistant && cd my-doc-assistant
mkdir -p data/sources
```

### 7. Commits session 6

| Hash | Description |
|------|-------------|
| `9312a1e` | Add YOLO batching and async entity extraction for improved performance |
| `a22ce32` | Add detailed usage instructions and fix TOP_K_REFS default |
| `bd4d954` | Rewrite README.md for clarity and logical flow |

### 8. Tests vérifiés

```bash
# Pipeline testé avec succès sur 1 page
YOLO batching: 1 image en 1.17s ✅
Entity extraction async: 2 chunks en parallèle ✅
```

## Améliorations implémentées (session 7)

### 1. Suite de tests E2E pytest (`tests/test_00_e2e_pipeline.py`)

Création d'une suite de tests E2E réutilisable pour les futures mises à jour :

```python
# Structure des tests
class TestE2EQueryOnly:           # Tests rapides sur indexes existants (~30s)
class TestE2ETestDocumentContent: # Validation du document fixture
class TestE2EFullPipeline:        # Tests complets avec ingestion (@slow)
class TestE2EEdgeCases:           # Cas limites et gestion d'erreurs
```

**Commandes:**
```bash
# Tests rapides E2E (~30s)
pytest tests/test_00_e2e_pipeline.py -v

# Tests complets avec ingestion (~2-5 min)
pytest tests/test_00_e2e_pipeline.py -v --run-slow
```

### 2. Configuration pytest (`tests/conftest.py`)

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)
```

### 3. Document fixture (`tests/fixtures/test_article.md`)

Article de test sur l'IA en médecine (français) utilisable pour les tests E2E.

### 4. Nettoyage fichiers obsolètes

Suppression des anciens fichiers de test :
- `test_advanced_rag.py`
- `test_e2e 2.py`
- `test_e2e/` directory
- `test_e2e_script.py`

### 5. Fix conflit Qdrant embedded

**Problème:** Qdrant embedded n'autorise qu'un seul client par dossier. Quand tous les tests s'exécutaient ensemble, les tests E2E échouaient car un autre module verrouillait Qdrant.

**Solution:**
1. Renommé `test_e2e_pipeline.py` → `test_00_e2e_pipeline.py` (s'exécute en premier alphabétiquement)
2. Ajouté fixture `cognidoc_session` session-scoped dans `conftest.py`
3. Les tests E2E partagent une seule instance CogniDoc

```python
# conftest.py - Session-scoped fixture
@pytest.fixture(scope="session")
def cognidoc_session():
    """Shared CogniDoc instance across ALL test modules."""
    global _session_cognidoc
    if _session_cognidoc is None:
        from cognidoc import CogniDoc
        _session_cognidoc = CogniDoc(...)
    return _session_cognidoc
```

### 6. Documentation API REST

Ajout de la documentation pour intégrer CogniDoc depuis une autre application :

```bash
# Endpoint principal
POST http://localhost:7860/api/submit_handler

# Exemple curl
curl -X POST http://localhost:7860/api/submit_handler \
  -H "Content-Type: application/json" \
  -d '{"data": ["Question?", [], true, true]}'
```

**Endpoints disponibles:**
- `/api/submit_handler` - Poser une question
- `/api/reset_conversation` - Réinitialiser la conversation
- `/api/refresh_metrics` - Métriques de performance
- `/api/export_csv` / `/api/export_json` - Export métriques

### 7. Commits session 7

| Hash | Description |
|------|-------------|
| `3435f36` | Add proper pytest E2E test suite for future updates |
| `9d6aa5f` | Update documentation with session 7 E2E test suite |
| `311f7c2` | Fix Qdrant lock conflict in tests |
| `5e69771` | Update docs with renamed E2E test file |
| `ba3738e` | Update documentation with Qdrant fix details |
| `11373a7` | Add API Integration documentation to README |

### 8. Tests vérifiés

```bash
# Tous les tests passent maintenant ensemble
pytest tests/ -v
# 134 passed, 2 skipped in ~27s
```

## Améliorations implémentées (session 8)

### 1. Support multilingue ES/DE (`cognidoc_app.py`)

Extension du support linguistique de FR/EN à FR/EN/ES/DE :

```python
def detect_query_language(query: str) -> str:
    """
    Heuristic to detect query language.
    Returns 'es' for Spanish, 'de' for German, 'fr' for French, 'en' for English (default).
    Detection uses indicator counting - highest count wins (threshold: 2+).
    """
    spanish_indicators = [...]  # ~50 indicateurs
    german_indicators = [...]   # ~45 indicateurs
    french_indicators = [...]   # ~30 indicateurs
```

**Messages localisés:**
```python
prefixes = {
    "fr": "**Clarification requise :**",
    "es": "**Se necesita aclaración:**",
    "de": "**Klärung erforderlich:**",
}

messages = {
    "fr": "Je n'ai pas trouvé d'informations pertinentes...",
    "es": "No he encontrado información relevante en la base documental...",
    "de": "Ich habe keine relevanten Informationen in der Dokumentenbasis gefunden...",
}
```

### 2. DATABASE_META_PATTERNS ES/DE (`complexity.py`)

Ajout de 34 nouveaux patterns pour l'espagnol et l'allemand :

**Espagnol (17 patterns):**
- `cuántos documentos`, `número de documentos`, `tamaño de la base`
- `lista los documentos`, `qué documentos`, `cuáles documentos`

**Allemand (17 patterns):**
- `wie viele Dokumente`, `Anzahl der Dokumente`, `Größe der Datenbank`
- `liste die Dokumente`, `welche Dokumente`, `zeige mir die Dokumente`

### 3. Prompts multilingues

Mise à jour de 4 fichiers de prompts avec règles ES/DE :
- `system_prompt_generate_final_answer.md`
- `user_prompt_generate_final_answer.md`
- `system_prompt_rewrite_query.md`
- `agent.py` SYSTEM_PROMPT

### 4. Tests ES/DE (`test_e2e_language_and_count.py`)

5 nouvelles classes de test (14 nouveaux tests) :
- `TestSpanishLanguageDetection`
- `TestGermanLanguageDetection`
- `TestSpanishDatabaseMetaPatterns`
- `TestGermanDatabaseMetaPatterns`
- `TestLanguageAmbiguity`

### 5. Fichiers modifiés

| Fichier | Modifications |
|---------|---------------|
| `src/cognidoc/cognidoc_app.py` | `detect_query_language()`, `get_clarification_prefix()`, `get_no_info_message()` |
| `src/cognidoc/complexity.py` | 34 nouveaux DATABASE_META_PATTERNS ES/DE |
| `src/cognidoc/agent.py` | SYSTEM_PROMPT language rules ES/DE |
| `src/cognidoc/prompts/*.md` | 3 fichiers avec règles ES/DE |
| `tests/test_e2e_language_and_count.py` | 5 nouvelles classes de test |

### 6. Tests vérifiés

```bash
# 150 tests passent (14 nouveaux tests ES/DE)
pytest tests/ -v
# 148 passed, 2 skipped in ~28s
```

### 7. MODEL_SPECS avec paramètres officiels (`constants.py`)

Ajout d'un dictionnaire centralisé avec les paramètres officiels des providers :

```python
MODEL_SPECS = {
    "gemini-2.5-flash": {
        "provider": "gemini",
        "context_window": 1_048_576,      # 1M tokens
        "max_output_tokens": 65_536,
        "default_temperature": 1.0,
        "default_top_p": 0.95,
        "supports_vision": True,
        "supports_json_mode": True,
    },
    "gpt-4o": {
        "context_window": 128_000,
        "max_output_tokens": 16_384,
        ...
    },
    "claude-sonnet-4-20250514": {
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        ...
    },
    # + Anthropic, Ollama models
}
```

**Avantages:**
- Paramètres officiels des providers (pas de valeurs arbitraires)
- `LLMConfig.from_model("gemini-2.5-flash")` charge automatiquement les specs
- Fallback gracieux pour modèles inconnus

### 8. MEMORY_WINDOW dynamique (`helpers.py`)

La mémoire de conversation s'adapte maintenant au modèle LLM :

```python
def get_memory_window() -> int:
    """Returns 50% of the model's context_window."""
    client = get_llm_client()
    if client.config.context_window:
        return int(client.config.context_window * 0.5)
    return MEMORY_WINDOW  # Fallback: 64K

# Exemples:
# Gemini 2.5 Flash (1M context) → 524K memory window
# GPT-4o (128K context) → 64K memory window
# Claude Sonnet 4 (200K context) → 100K memory window
```

### 9. Dark mode (`cognidoc_app.py`)

Toggle dans le header pour basculer entre mode clair et sombre :

- Bouton 🌙/☀️ dans le header
- Détection automatique de la préférence système
- Persistance du choix dans `localStorage`
- ~240 lignes de CSS pour le dark mode

### 10. Commits session 8

| Hash | Description |
|------|-------------|
| `541886d` | Add Spanish and German language support |
| `83951bc` | Add MODEL_SPECS with official provider parameters |
| `6b10105` | Make MEMORY_WINDOW dynamic based on LLM context_window |
| `72e852e` | Update documentation with ES/DE, MODEL_SPECS, MEMORY_WINDOW |
| `e765f00` | Add dark mode toggle to web interface |
| `07a498f` | Update documentation with dark mode feature |
| `4be2600` | Update CLAUDE.md and IMPLEMENTATION_PLAN.md with dark mode |
| `7a55da3` | Fix dark mode toggle for Gradio compatibility |

## Améliorations futures

1. ~~**Support langues additionnelles** - Espagnol, Allemand, etc.~~ ✅ Fait
2. ~~**Cache persistant** - Utiliser Redis ou SQLite pour le cache~~ ✅ Fait
3. ~~**Métriques de performance** - Dashboard temps de réponse, cache hits~~ ✅ Fait
4. **Tests de charge** - Benchmarks avec multiple requêtes simultanées
5. ~~**Export métriques** - CSV/JSON pour analyse externe~~ ✅ Fait
6. **Alerting** - Notifications si latence > seuil
7. ~~**Parallélisation ingestion** - PDF→images, embeddings~~ ✅ Fait
8. ~~**YOLO batching** - Batch inference GPU pour détection~~ ✅ Fait
9. ~~**Entity extraction async** - Queue LLM avec workers~~ ✅ Fait
10. **Pipeline streaming** - Démarrer chunking pendant extraction (risqué sur 16GB)
11. ~~**Fix document count vs chunk count**~~ ✅ Fait
12. ~~**Tests E2E réutilisables**~~ ✅ Fait
13. ~~**Documentation API REST**~~ ✅ Fait
14. ~~**MODEL_SPECS** - Paramètres officiels des providers~~ ✅ Fait
15. ~~**MEMORY_WINDOW dynamique** - Adapté au context_window du LLM~~ ✅ Fait
16. ~~**Dark mode** - Toggle clair/sombre dans l'interface~~ ✅ Fait
17. ~~**Documentation YOLO model** - Clarifier le requirement du modèle~~ ✅ Fait
18. ~~**Cohérence paths documentation/code** - data/sources partout~~ ✅ Fait

## Améliorations implémentées (session 9)

### 1. Documentation CLAUDE.md améliorée

- Ajout d'exemples d'exécution de tests individuels
- Correction des paths de modules (`src.xxx` → `cognidoc.xxx`)
- Ajout section "Source Layout" expliquant la structure du package
- Réorganisation section Commands avec sous-sections claires

### 2. Documentation YOLO Model

Clarification que le modèle YOLO (~109 MB) n'est **pas inclus** dans le repo :

```
models/YOLOv11/yolov11x_best.pt  # Requis pour YOLO, sinon fallback
```

**Comportement:**
- Modèle présent → YOLO detection activée
- Modèle absent → Fallback extraction simple page-par-page
- `--skip-yolo` → Désactiver explicitement

### 3. Structure projet mise à jour

Ajout de `models/` et `config/` dans la documentation :

```
your-project/
├── .env
├── data/sources/           # Documents source
├── models/YOLOv11/         # Modèle YOLO (optionnel)
├── config/graph_schema.yaml
└── data/                   # Généré après ingestion
```

### 4. Clarification Setup Wizards

| Commande | Type | Description |
|----------|------|-------------|
| `python -m cognidoc.setup` | **Interactif** | Wizard complet (providers, API keys, ingestion) |
| `cognidoc init --schema` | Non-interactif | Copie templates seulement |
| Schema Wizard | **Auto** | Se lance pendant `ingest()` si pas de schema |

### 5. Fix cohérence paths `data/sources`

**Documentation (README.md, CLAUDE.md):**
- `./documents` → `./data/sources`
- `./docs` → `./data/sources`

**Code:**
- `cli.py`: Crée `data/sources` au lieu de `data/pdfs`
- `cli.py`: Messages "Next steps" corrigés
- `api.py`: Docstring corrigée

### 6. Table CONTEXT_WINDOW / MEMORY_WINDOW

| Model | CONTEXT_WINDOW | MEMORY_WINDOW (50%) |
|-------|----------------|---------------------|
| Gemini 2.5 Flash | 1,048,576 (1M) | 524,288 |
| GPT-4o | 128,000 | 64,000 |
| Claude Sonnet 4 | 200,000 | 100,000 |
| Granite 3.3:8b | 128,000 | 64,000 |

### 7. Commits session 9

| Hash | Description |
|------|-------------|
| `3560bf8` | Improve CLAUDE.md with correct module paths and test examples |
| `e96aa30` | Document YOLO model requirement in README and CLAUDE.md |
| `306ee1b` | Add models/ directory to project structure documentation |
| `395068a` | Add CONTEXT_WINDOW and MEMORY_WINDOW values table to CLAUDE.md |
| `526feda` | Clarify setup wizards and CLI commands in documentation |
| `aed47e9` | Fix inconsistent document paths in documentation |
| `8ce25b9` | Fix inconsistent source paths in code (data/pdfs -> data/sources) |

### 8. Tests vérifiés

```bash
# 148 passed, 2 skipped in 29.33s
pytest tests/ -v
```

---

## Session 10 - 13 janvier 2026

### Problème corrigé : Paths pointant vers le package d'installation

**Symptôme:** Lorsque CogniDoc est installé comme package (`pip install cognidoc`) et utilisé dans un nouveau projet, toutes les commandes (`cognidoc ingest`, `cognidoc serve`) pointaient vers les répertoires du package d'installation au lieu du répertoire de travail courant.

**Cause racine:** Dans `constants.py`, la variable `BASE_DIR` utilisait `Path(__file__)` qui pointe vers le répertoire d'installation du package, pas vers `Path.cwd()`.

### Modifications

| Fichier | Changement |
|---------|------------|
| `constants.py` | `BASE_DIR` utilise maintenant `Path.cwd()` au lieu de `Path(__file__)` |
| `constants.py` | `load_dotenv()` charge explicitement `.env` depuis `Path.cwd()` |
| `constants.py` | Nouvelle variable `DATA_DIR` (remplace l'ancien `BASE_DIR`) |
| `constants.py` | Support de `COGNIDOC_DATA_DIR` env var pour override |
| `graph_config.py` | Import `DATA_DIR` au lieu de `BASE_DIR` |

### Code avant/après

**Avant (`constants.py`):**
```python
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Package install dir!
```

**Après (`constants.py`):**
```python
load_dotenv(Path.cwd() / ".env")  # Explicit cwd
PACKAGE_DIR = Path(__file__).resolve().parent  # For prompts/templates
DATA_DIR = Path(os.getenv("COGNIDOC_DATA_DIR")) if os.getenv("COGNIDOC_DATA_DIR") else Path.cwd()
BASE_DIR = DATA_DIR  # Backward compatibility alias
```

### Architecture des paths

| Variable | Utilisation | Pointe vers |
|----------|-------------|-------------|
| `PACKAGE_DIR` | Prompts, templates embarqués | Installation du package |
| `DATA_DIR` / `BASE_DIR` | Données utilisateur | Répertoire de travail (cwd) |
| `COGNIDOC_DATA_DIR` | Override env var | Custom directory |

### Corrections documentation supplémentaires

**README.md:**
- Structure projet: consolidé `data/` en un seul répertoire avec `sources/` à l'intérieur
- Remplacé `./docs/` par `./data/sources/` (Schema Wizard section, CLI Options)
- Remplacé `GEMINI_API_KEY` par `GOOGLE_API_KEY` (variable principale dans le code)
- Supprimé `COMPLEXITY_THRESHOLD=0.55` (variable inexistante)

**cli.py:**
- Corrigé `COGNIDOC_LLM_PROVIDER` → `DEFAULT_LLM_PROVIDER` dans le template `.env`

**CLAUDE.md:**
- Corrigé `src/constants.py` → `src/cognidoc/constants.py`
- Corrigé `src/prompts/` → `src/cognidoc/prompts/`
- Ajouté section "Path Resolution" documentant `DATA_DIR`, `PACKAGE_DIR`, `BASE_DIR`

### Commits session 10

| Hash | Description |
|------|-------------|
| `316ae78` | Fix paths to use cwd instead of package installation directory |
| `ffceb60` | Fix path inconsistencies in README.md |
| `547588d` | Fix configuration inconsistencies in documentation and CLI |
| `aaa22a5` | Update CLAUDE.md with correct paths and DATA_DIR documentation |
| `2b31e86` | Fix path resolution: separate PROJECT_DIR and DATA_DIR |

### Fix final des chemins (commit 2b31e86)

Le problème `COGNIDOC_DATA_DIR=./data` causait des chemins doublés (`data/data/sources`).

**Solution:** Séparation des variables:

| Variable | Défaut | Usage |
|----------|--------|-------|
| `PROJECT_DIR` | `cwd()` | Racine projet (models/, config/) |
| `DATA_DIR` | `PROJECT_DIR/data` | Données (sources/, pdfs/, etc.) |

### Variables .env corrigées

| Variable | Statut |
|----------|--------|
| `COGNIDOC_EMBEDDING_PROVIDER` | ✓ Correct |
| `COGNIDOC_EMBEDDING_MODEL` | ✓ Correct |
| `DEFAULT_EMBEDDING_PROVIDER` | ✗ N'existe pas |
| `DEFAULT_EMBEDDING_MODEL` | ✗ N'existe pas |
| `COMPLEXITY_THRESHOLD` | ✗ N'existe pas |
| `COGNIDOC_DATA_DIR` | Optionnel (défaut = ./data) |

### Tests vérifiés

```bash
# 148 passed, 2 skipped in 30.05s
pytest tests/ -v
```

---

## Session 11 - 13 janvier 2026

### 1. Rapport d'ingestion complet (`run_ingestion_pipeline.py`)

Ajout d'un rapport détaillé affiché à la fin de chaque ingestion :

```
======================================================================
                    INGESTION REPORT
======================================================================

┌────────────────────────────────────────────────────────────────────┐
│                             DOCUMENTS                              │
├──────────────────────────────────┬─────────────────────────────────┤
│ Files processed                  │                               1 │
│   Documents converted            │                               1 │
│ PDFs converted to images         │                               1 │
│ Total pages generated            │                               2 │
└──────────────────────────────────┴─────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           YOLO DETECTION                           │
├──────────────────────────────────┬─────────────────────────────────┤
│ Images processed                 │                               2 │
│ Text regions detected            │                               2 │
│ Table regions detected           │                               0 │
│ Picture regions detected         │                               0 │
└──────────────────────────────────┴─────────────────────────────────┘

... (Content Extraction, Chunking & Embeddings, GraphRAG, Timing)
```

**Sections du rapport:**
- **DOCUMENTS**: Fichiers traités, PDFs copiés, documents convertis, pages générées
- **YOLO DETECTION**: Images traitées, régions texte/table/picture détectées
- **CONTENT EXTRACTION**: Régions texte extraites, tables extraites, images décrites
- **CHUNKING & EMBEDDINGS**: Child chunks, from cache, newly embedded, parent chunks
- **KNOWLEDGE GRAPH**: Chunks traités, entités, relations, nœuds, arêtes, communautés, types d'entités
- **TIMING**: Durée de chaque étape avec total formaté

### 2. Fix tiktoken duplicate plugin

**Problème:** Erreur `ValueError: Duplicate encoding name gpt2 in tiktoken plugin tiktoken_ext.openai_public` empêchant le chunking.

**Cause:** Fichier dupliqué `openai_public 2.py` dans le répertoire `tiktoken_ext/` du venv.

**Solution:** Suppression du fichier dupliqué.

```bash
rm ".venv/lib/python3.12/site-packages/tiktoken_ext/openai_public 2.py"
```

### 3. Fix stats embeddings dans le rapport

**Problème:** Le rapport affichait 0 chunks alors que des chunks étaient créés.

**Cause:** Mauvaises clés utilisées pour accéder aux stats de `create_embeddings()`.

**Solution:** Correction des clés dans `format_ingestion_report()`:

```python
# Avant (incorrect)
total_chunks = embed_stats.get("total_chunks", 0)
from_cache = embed_stats.get("from_cache", 0)

# Après (correct)
from_cache = embed_stats.get("cached", 0)
to_embed = embed_stats.get("to_embed", 0)
newly_embedded = embed_stats.get("embedded", 0)
total_chunks = from_cache + to_embed
```

### 4. Fichiers modifiés

| Fichier | Modifications |
|---------|---------------|
| `src/cognidoc/run_ingestion_pipeline.py` | `format_ingestion_report()` function + integration at pipeline end |
| `src/cognidoc/run_ingestion_pipeline.py` | Capture des stats PDF conversion |

### 5. Commits session 11

| Hash | Description |
|------|-------------|
| `fa2b1fd` | Add comprehensive ingestion report at pipeline end |
| `0bca983` | Fix tiktoken duplicate plugin and embedding stats in report |

### 6. Tests vérifiés

```bash
# Tests sur test_article.md et test_document.txt
# Pipeline complet avec GraphRAG: ✅
# Rapport affiché correctement: ✅
# App lancée et requête testée: ✅
```

**Exemple test_document.txt:**
- 1 document → 1 page → 5 child chunks
- GraphRAG: 40 entités → 31 nœuds, 22 arêtes, 14 communautés
- Temps total: 1m 37.7s
