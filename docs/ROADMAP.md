# CogniDoc — Roadmap

Fichier centralisé de suivi des plans d'implémentation.

---

## Fonctionnalités implémentées

### Entity Resolution sémantique
- **Statut:** Fait
- **Module:** `src/cognidoc/entity_resolution.py`
- **Tests:** `tests/test_entity_resolution.py` (34 tests)
- **Description:** Déduplication sémantique d'entités en 4 phases (blocking par embeddings, matching LLM, clustering Union-Find, merging enrichi). Intégré au pipeline d'ingestion (étape 12).

### Metrics Dashboard SaaS-style
- **Statut:** Fait
- **Commits:** `c5c033c`, `400d48e`
- **Description:** Refonte du dashboard Metrics avec cartes à bordures colorées + icones SVG, charts Plotly stylés (bar, spline timeline, donut), conversion ms vers secondes, grille CSS responsive, support dark mode.

### Dark/Light mode toggle + UI responsive
- **Statut:** Fait
- **Commit:** `7b240ad`
- **Description:** Fix du toggle dark/light mode (Gradio 6.x ignore `css=` dans `gr.Blocks()`). CSS+JS injectés via `ThemeInjectionMiddleware`. Layout full-width responsive (`max-width: 100%`, chatbot `calc(100vh - 320px)`). Dark mode complet sur Chat + Metrics (sélecteurs agressifs `!important` pour les composants Svelte de Gradio).

### Formatage structuré des réponses LLM
- **Statut:** Fait
- **Commit:** `6454aa7`
- **Description:** Post-traitement des réponses LLM pour un affichage structuré type ChatGPT/Claude :
  - `_format_markdown()` : insertion de `\n\n` avant les listes/bullets inline (fix Gemini Flash qui compacte tout)
  - `_indent_sublists()` : détection titre/sous-items avec suppression des lignes vides pour le nesting Markdown
  - CSS `ol + ul { padding-left: 2rem }` : indentation visuelle des sous-items (contourne le rendering plat de Gradio)
  - Fix `answer.split(' ')` au lieu de `split()` pour préserver les `\n\n` dans le streaming agent
  - Prompts mis à jour (system, user, agent) avec instructions de listes imbriquées

### Visualisation interactive du Knowledge Graph
- **Statut:** Fait
- **Description:** Onglet "🕸 Graph" dans l'UI Gradio avec visualisation D3.js force-directed du knowledge graph.
  - Page HTML standalone (`src/cognidoc/static/graph-viewer.html`) servie via FastAPI à `/graph-viewer`, intégrée en iframe
  - Endpoint API `GET /api/graph/data` : sérialisation JSON du graphe (nodes, edges, communities, stats)
  - Stats cards (nodes, edges, communities, entity types), filtre par type, recherche, toggle communautés/labels
  - Nodes colorés par type (`d3.schemeTableau10`), taille proportionnelle au degree
  - Panneau détails au clic : name, type, description, attributes, aliases, community, connections
  - Community hulls (enveloppes convexes) avec toggle
  - Dark mode synchronisé via `localStorage('cognidoc-dark-mode')`
  - Drag, zoom, pan, tooltips sur edges

### Bugfixes : sources chat + graph viewer vide
- **Statut:** Fait
- **Commit:** `d85faa0`
- **Description:** Deux bugs corrigés dans `cognidoc_app.py` :
  - **Sources manquantes dans le chat :** `hybrid_result` n'était pas initialisé dans le chemin vector-only fallback, causant un `NameError` lors de l'accès à `hybrid_result.metadata` pour la confidence. Fix : `hybrid_result = None` + guard conditionnel.
  - **Graph viewer vide :** L'endpoint `/api/graph/data` accédait directement à `hybrid_retriever._graph_retriever` qui est `None` en lazy loading. Fix : appel à `_ensure_graph_loaded()` avant l'accès pour déclencher le chargement du graphe.

---

## Phase 1 : Enrichissement extraction + Outil d'agrégation (NetworkX)

**Statut :** Fait

**Objectif :** Permettre les requêtes de comptage et d'agrégation sur le graphe NetworkX existant. Extraire des attributs structurés (dates, lieux, quantités, statuts) en plus du nom et de la description.

**Motivation :** CogniDoc ne peut pas répondre aux requêtes comme "combien de documents traitent de X ?" ou "quelle est la moyenne de Y ?". L'extraction d'entités capture noms et descriptions mais ignore les attributs numériques (dates, quantités, scores), et NetworkX n'a pas de langage de requête.

### 1.1 Enrichir le prompt d'extraction d'entités — **Fait**

**Fichier :** `src/cognidoc/extract_entities.py`

**Changements effectués :**
- `build_entity_extraction_prompt()` : ajout d'instructions pour extraire dates, lieux, quantités, statuts dans le champ `attributes`
- Le prompt inclut les attributs typés définis dans le schéma (si `EntityType.attributes` est non vide)
- Parsing dans `extract_entities_from_text()` et `extract_entities_from_text_async()` : ajout de `attributes=e.get("attributes", {})` avec validation (dict uniquement)
- `extract_json_from_response()` : normalisation `attributs` → `attributes` (FR→EN)

### 1.2 Enrichir le schéma graphe avec des définitions d'attributs typés — **Fait**

**Fichier :** `src/cognidoc/graph_config.py`

**Changements effectués :**
- Nouvelle dataclass `EntityAttribute(name, type, description)` — types: string, number, date, boolean
- `EntityType.attributes` changé de `List[str]` à `List[EntityAttribute]`
- Parser YAML rétrocompatible : gère `["field"]` (ancien format string) et `[{name, type, description}]` (nouveau format typé)
- Les attributs typés sont affichés dans le prompt d'extraction (`Expected attributes: publication_date (date), ...`)

### 1.3 Ajouter l'outil agent `aggregate_graph` — **Fait**

**Fichier :** `src/cognidoc/agent_tools.py`

**Changements effectués :**
- `AGGREGATE_GRAPH = "aggregate_graph"` ajouté à `ToolName`
- `AggregateGraphTool(BaseTool)` : opérations COUNT, COUNT_BY, LIST, GROUP_BY, STATS sur `kg.nodes`
- Filtrage case-insensitive par entity_type
- Observation formatée dans `ToolResult.observation`
- Enregistré dans `create_tool_registry()` (conditionné à `graph_retriever` disponible)

### 1.4 Mettre à jour l'évaluation de complexité — **Fait**

**Fichier :** `src/cognidoc/complexity.py`

**Changements effectués :**
- `AGGREGATION_PATTERNS` : 23 patterns FR/EN/ES/DE (combien, how many, count, nombre, moyenne, average, etc.)
- `is_aggregation_question()` : détecte les requêtes d'agrégation
- Intégré dans `evaluate_complexity()` : force le chemin agent (score >= AGENT_THRESHOLD + 0.1)

### 1.5 Tests — **Fait**

- `test_extract_entities.py` : 7 tests (attributs parsés, dict/non-dict, normalisation FR, prompt schema, round-trip save/load)
- `test_agent_tools.py` : 15 tests AggregateGraphTool (COUNT, COUNT_BY, LIST, GROUP_BY, STATS, erreurs, observation)
- `test_complexity.py` : 22 tests (17 patterns détectés, 4 non-détectés, intégration agent path)
- `test_graph_config.py` : 5 tests EntityAttribute (defaults, backward compat strings, typed dicts, mixed formats)

### 1.6 Ré-ingérer un corpus pour vérifier l'extraction d'attributs — **Fait**

Ré-ingestion complète (`--full-reindex`) sur le corpus de test (3 documents). Résultat : 6/20 entités (30%) ont des attributs structurés extraits (application, category, alias). Les 14 entités sans attributs sont des concepts abstraits (formats, techniques génériques) cohérents avec le texte source.

### Scope Phase 1 — **Fait**

| Fichier | Changements |
|---------|-------------|
| `extract_entities.py` | Prompt enrichi + parsing attributes (sync + async) |
| `graph_config.py` | `EntityAttribute` dataclass + parser YAML rétrocompat |
| `agent_tools.py` | `AggregateGraphTool` (COUNT/COUNT_BY/LIST/GROUP_BY/STATS) |
| `complexity.py` | `AGGREGATION_PATTERNS` + `is_aggregation_question()` |
| Tests | 49 nouveaux tests sur 4 fichiers |

---

## Phase 2 : Migration vers Kùzu (BD graphe embarquée avec Cypher)

**Statut :** Fait

**Objectif :** Introduire une abstraction `GraphBackend` (Strategy pattern) permettant de brancher NetworkX (défaut) ou Kùzu (BD graphe embarquée, Cypher natif). Cela débloque les requêtes graphe Cypher tout en gardant la rétrocompatibilité NetworkX.

### Pourquoi Kùzu (pas Neo4j)

| Critère | Kùzu | Neo4j |
|---------|------|-------|
| Déploiement | Embarqué (comme SQLite) | Serveur séparé |
| Dépendances | `pip install kuzu` | Serveur + driver |
| Coût | Gratuit, MIT | Community/Enterprise |
| Support Cypher | Complet | Complet |
| Performance | Stockage colonne, analytique rapide | Optimisé OLTP |
| Production | Bon pour embarqué | Mieux pour multi-tenant cloud |

**Décision :** Kùzu pour dev/embarqué. Neo4j réservé pour la production cloud (Phase 3).

### 2.1 Ajouter la dépendance Kùzu — **Fait**

**Fichier :** `pyproject.toml`

- Groupe optionnel : `kuzu = ["kuzu>=0.8"]`
- Ajouté au groupe `all`
- Override mypy pour les 3 nouveaux modules

### 2.2 Ajouter la config GRAPH_BACKEND — **Fait**

**Fichier :** `src/cognidoc/constants.py`

- `GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx").lower()`
- `KUZU_DB_DIR` : chemin configurable via `KUZU_DB_DIR` env var

### 2.3 Créer le GraphBackend ABC — **Fait**

**Nouveau fichier :** `src/cognidoc/graph_backend.py` (~90 lignes)

Interface complète : nœuds (add/has/remove/update/get_attrs/count), arêtes (add/has/get_data/update/iter/count), traversée (successors/predecessors/find_all_simple_paths/degree), export (to_undirected_networkx/to_node_link_data/from_node_link_data).

### 2.4 Extraire NetworkXBackend — **Fait**

**Nouveau fichier :** `src/cognidoc/graph_backend_networkx.py` (~95 lignes)

Wraps `nx.DiGraph()` — méthodes 1:1 avec NetworkX, extraites de `knowledge_graph.py`.

### 2.5 Refactorer KnowledgeGraph — **Fait**

**Fichier :** `src/cognidoc/knowledge_graph.py`

- Remplacé `self.graph = nx.DiGraph()` par `self._backend: GraphBackend`
- Factory `_create_backend()` : lit `GRAPH_BACKEND`, retourne `NetworkXBackend()` ou `KuzuBackend()`
- Ajouté wrappers pour `entity_resolution.py` : `has_node()`, `get_successors()`, `get_predecessors()`, `get_edge_data()`, `has_edge()`, `add_edge_raw()`, `remove_graph_node()`, `update_graph_node_attrs()`, `update_edge_attrs()`, `iter_edges()`, `number_of_edges()`
- `detect_communities()` : `self._backend.to_undirected_networkx()` pour Louvain
- `save()`/`load()` : adapté pour les deux backends via node_link_data

### 2.6 Adapter entity_resolution.py — **Fait**

**Fichier :** `src/cognidoc/entity_resolution.py`

~20 accès `graph.graph.*` remplacés par les wrappers KnowledgeGraph. `_redirect_edges()` refactoré : construction de dict explicite + `update_edge_attrs()` au lieu de mutation in-place.

### 2.7 Adapter graph_retrieval.py et pipeline — **Fait**

- `graph_retrieval.py:329` : `kg.graph.edges(data=True)` → `kg.iter_edges(data=True)`
- `run_ingestion_pipeline.py:934` : `kg.graph.number_of_edges()` → `kg.number_of_edges()`

### 2.8 Implémenter KuzuBackend — **Fait**

**Nouveau fichier :** `src/cognidoc/graph_backend_kuzu.py` (~350 lignes)

- Schéma générique : table `Entity` (id, name, type, description, attrs JSON) + table `Relates` (relationship_type, description, weight, source_chunks JSON)
- Attributs stockés en JSON string (pas de migration de schéma)
- `find_all_simple_paths` : Cypher récursif `MATCH path = (a)-[*1..N]->(b)`
- `to_undirected_networkx()` : export pour Louvain
- Import conditionnel : `KUZU_AVAILABLE` flag

### 2.9 Adapter AggregateGraphTool pour Cypher — **Fait**

**Fichier :** `src/cognidoc/agent_tools.py`

- Branche conditionnelle Cypher pour COUNT et LIST quand backend = Kùzu
- Méthode `_execute_cypher()` avec fallback Python si Cypher échoue
- COUNT_BY, GROUP_BY, STATS restent en Python (logique complexe)

### 2.10 Commande CLI migrate-graph — **Fait**

**Fichier :** `src/cognidoc/cli.py`

- Sous-commande `migrate-graph` avec `--graph-path` et `--kuzu-path`
- Charge le graphe NetworkX (JSON), exporte via node_link_data, importe dans KuzuBackend
- Validation de l'installation kuzu

### 2.11 Tests — **Fait**

**Nouveau fichier :** `tests/test_graph_backend.py` (~250 lignes)

- Fixture parametrized `@pytest.fixture(params=["networkx", "kuzu"])` — skips kuzu si non installé
- 47 tests (24 NetworkX, 23 Kùzu skippés sans kuzu) : CRUD nœuds/arêtes, traversée, paths, degree, export/import
- Tests existants adaptés (`test_knowledge_graph.py`, `test_entity_resolution.py`) : `kg.graph.*` → wrappers

### Scope Phase 2 — **Fait**

| Fichier | Changements |
|---------|-------------|
| `pyproject.toml` | Dépendance kuzu optionnelle |
| `constants.py` | `GRAPH_BACKEND`, `KUZU_DB_DIR` (~5 lignes) |
| `graph_backend.py` | **Nouveau** — ABC (~90 lignes) |
| `graph_backend_networkx.py` | **Nouveau** — NetworkX impl (~95 lignes) |
| `graph_backend_kuzu.py` | **Nouveau** — Kùzu impl (~350 lignes) |
| `knowledge_graph.py` | Refactoring backend + wrappers (~80 lignes modifiées, ~30 ajoutées) |
| `entity_resolution.py` | ~20 lignes modifiées (wrappers) |
| `graph_retrieval.py` | 1 ligne modifiée |
| `run_ingestion_pipeline.py` | 1 ligne modifiée |
| `agent_tools.py` | Branche Cypher (~30 lignes ajoutées) |
| `cli.py` | Commande migrate-graph (~60 lignes) |
| `test_graph_backend.py` | **Nouveau** — 47 tests (~250 lignes) |

---

## Phase 3 (futur) : Neo4j pour la production cloud

Hors scope pour l'instant. Quand nécessaire :

- Ajouter une implémentation `Neo4jStorage(GraphStorage)`
- Utiliser le driver Python `neo4j`
- Configuration : `GRAPH_BACKEND=neo4j`, `NEO4J_URI`, `NEO4J_AUTH`
- L'abstraction `GraphStorage` de la Phase 2 en fait un remplacement drop-in

---

## Ordre d'implémentation

```
Phase 1 (enrichissement NetworkX) :          ✅ FAIT
  1.1  Enrichir le prompt d'extraction (attributs + dates/lieux)  ✅
  1.2  Mettre à jour graph_config.py (EntityAttribute typé)       ✅
  1.3  Ajouter AggregateGraphTool                                 ✅
  1.4  Mettre à jour les patterns de complexité                   ✅
  1.5  Tests                                                      ✅
  1.6  Ré-ingérer un corpus de test                               ✅

Phase 2 (migration Kùzu) :                   ✅ FAIT
  2.1   Ajouter la dépendance kuzu                                    ✅
  2.2   Ajouter config GRAPH_BACKEND                                  ✅
  2.3   Créer l'abstraction GraphBackend (ABC)                        ✅
  2.4   Extraire NetworkXBackend                                      ✅
  2.5   Refactorer KnowledgeGraph                                     ✅
  2.6   Adapter entity_resolution.py                                  ✅
  2.7   Adapter graph_retrieval.py + pipeline                         ✅
  2.8   Implémenter KuzuBackend                                       ✅
  2.9   Adapter AggregateGraphTool pour Cypher                        ✅
  2.10  Commande CLI migrate-graph                                    ✅
  2.11  Tests                                                         ✅
  2.12  Documentation                                                 ✅
```

---

## Comment reprendre ce travail

Pour continuer l'implémentation dans une future session Claude Code :

**Phase 1 :**
```
Implémente la Phase 1 du roadmap dans docs/ROADMAP.md.
Commence par l'étape 1.1 (enrichir le prompt d'extraction dans extract_entities.py).
```

**Phase 3 :**
```
Implémente la Phase 3 du roadmap dans docs/ROADMAP.md.
Ajoute une implémentation Neo4jBackend(GraphBackend) dans graph_backend_neo4j.py.
```
