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

### 1.6 Ré-ingérer un corpus pour vérifier l'extraction d'attributs — **À faire**

Après ré-ingestion, vérifier que les attributs sont bien peuplés sur les entités du graphe.

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

**Objectif :** Remplacer NetworkX par Kùzu, une BD graphe embarquée supportant Cypher nativement. Cela débloque les vraies requêtes graphe (pattern matching, agrégation, recherche de chemins) sans code Python de traversée custom.

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

### 2.1 Ajouter la dépendance Kùzu

**Fichier :** `pyproject.toml`

```toml
[project.optional-dependencies]
graph = ["kuzu>=0.7"]
```

Garder `networkx` comme fallback.

### 2.2 Créer l'abstraction GraphStorage

**Nouveau fichier :** `src/cognidoc/graph_storage.py`

```python
class GraphStorage(ABC):
    """Abstract graph storage backend."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]: ...

    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]: ...

    @abstractmethod
    def query_cypher(self, cypher: str, params: dict = None) -> List[dict]: ...

    @abstractmethod
    def aggregate(self, entity_type: str, operation: str, attribute: str = None, filters: dict = None) -> Any: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
```

Deux implémentations :
- `NetworkXStorage` — wraps la logique actuelle de `KnowledgeGraph`
- `KuzuStorage` — Kùzu natif avec Cypher complet

### 2.3 Implémenter `KuzuStorage`

**Nouveau fichier :** `src/cognidoc/graph_kuzu.py`

```python
class KuzuStorage(GraphStorage):
    def __init__(self, db_path: str):
        import kuzu
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        """Create node/edge tables from graph_schema.yaml."""
        config = get_graph_config()
        for et in config.entities:
            cols = "id STRING, name STRING, description STRING, community_id INT64"
            for attr in et.attributes:
                cols += f", {attr.name} {self._kuzu_type(attr.type)}"
            self.conn.execute(f"CREATE NODE TABLE IF NOT EXISTS {et.name}({cols}, PRIMARY KEY(id))")

        for rt in config.relationships:
            self.conn.execute(
                f"CREATE REL TABLE IF NOT EXISTS {rt.name}("
                f"FROM {rt.source_types[0]} TO {rt.target_types[0]}, "
                f"description STRING, weight DOUBLE, confidence DOUBLE)"
            )

    def query_cypher(self, cypher: str, params: dict = None) -> List[dict]:
        result = self.conn.execute(cypher, params or {})
        columns = result.get_column_names()
        return [dict(zip(columns, row)) for row in result.get_as_df().values]
```

### 2.4 Remplacer le backend de `aggregate_graph` par Cypher

**Fichier :** `src/cognidoc/agent_tools.py`

En Phase 2, le LLM génère du Cypher directement :

```python
class AggregateGraphTool(BaseTool):
    def execute(self, ...):
        cypher = self._generate_cypher(query, schema_context)
        results = graph_storage.query_cypher(cypher)
        return self._format_results(results)
```

Le prompt LLM inclut le schéma Kùzu (tables de noeuds, tables de relations, noms/types d'attributs). Une étape de validation vérifie la syntaxe Cypher avant exécution.

### 2.5 Ajouter l'outil agent `cypher_query` (optionnel)

Pour les requêtes avancées, exposer Cypher brut à l'agent :

```python
class CypherQueryTool(BaseTool):
    """Execute a Cypher query on the knowledge graph (Kùzu backend only)."""
    name = ToolName.CYPHER_QUERY
```

### 2.6 Chemin de migration

Pour les utilisateurs existants avec des graphes NetworkX :

1. Commande CLI : `cognidoc migrate-graph --backend kuzu`
2. Lit le `knowledge_graph/` existant (NetworkX JSON)
3. Crée une DB Kùzu et importe tous les noeuds/arêtes
4. Préserve les community assignments

Configuration dans `.env` ou `constants.py` :

```
GRAPH_BACKEND=kuzu  # ou "networkx" (défaut pour rétrocompatibilité)
```

### 2.7 Adapter les opérations graphe existantes

| Fichier | Usage actuel | Migration |
|---------|-------------|-----------|
| `knowledge_graph.py` | `nx.DiGraph()` directement | Wrap en `NetworkXStorage` ou déléguer à `KuzuStorage` |
| `graph_retrieval.py` | `kg.graph.neighbors()`, `kg.get_node()` | Utiliser `GraphStorage.get_neighbors()` |
| `agent_tools.py` | `LookupEntityTool`, `CompareEntitiesTool` | Utiliser les méthodes `GraphStorage` |
| `hybrid_retriever.py` | `self.graph_retriever` | Pas de changement (passe par `graph_retrieval.py`) |

### 2.8 Détection de communautés

Kùzu n'a pas de Louvain intégré. Approche recommandée : export vers NetworkX pour la détection de communautés uniquement (Kùzu -> NetworkX subgraph -> Louvain -> écriture des community_id). La détection de communautés ne s'exécute qu'une fois après l'ingestion.

### 2.9 Tests

- `test_graph_kuzu.py` : Nouveau fichier de tests pour le backend Kùzu
- `test_knowledge_graph.py` : Paramétrer pour tester les deux backends
- `test_agent_tools.py` : Tester `aggregate_graph` avec le backend Cypher

### Scope Phase 2

| Fichier | Changements |
|---------|-------------|
| `graph_storage.py` | Nouvelle base abstraite (~80 lignes) |
| `graph_kuzu.py` | Nouvelle implémentation Kùzu (~250 lignes) |
| `knowledge_graph.py` | Refactoring `GraphStorage` / wrap `NetworkXStorage` (~100 lignes) |
| `graph_retrieval.py` | Adaptations mineures (~20 lignes) |
| `agent_tools.py` | Mise à jour backend `AggregateGraphTool` (~40 lignes) |
| `constants.py` | Ajout config `GRAPH_BACKEND` (~5 lignes) |
| `cli.py` | Ajout commande `migrate-graph` (~40 lignes) |
| Tests | ~150 lignes sur 2-3 fichiers |

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
  1.6  Ré-ingérer un corpus de test                               ⬜ À faire

Phase 2 (migration Kùzu) :
  2.1  Ajouter la dépendance kuzu
  2.2  Créer l'abstraction GraphStorage
  2.3  Implémenter KuzuStorage
  2.4  Remplacer le backend de l'outil aggregate
  2.5  Ajouter l'outil cypher_query (optionnel)
  2.6  Commande CLI de migration
  2.7  Adapter les opérations graphe existantes
  2.8  Adaptateur détection de communautés
  2.9  Tests
```

---

## Comment reprendre ce travail

Pour continuer l'implémentation dans une future session Claude Code :

**Phase 1 :**
```
Implémente la Phase 1 du roadmap dans docs/ROADMAP.md.
Commence par l'étape 1.1 (enrichir le prompt d'extraction dans extract_entities.py).
```

**Phase 2 :**
```
Implémente la Phase 2 du roadmap dans docs/ROADMAP.md.
Commence par l'étape 2.1 (ajouter la dépendance kuzu).
```
