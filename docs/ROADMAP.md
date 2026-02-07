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

**Objectif :** Permettre les requêtes de comptage et d'agrégation sur le graphe NetworkX existant.

**Motivation :** CogniDoc ne peut pas répondre aux requêtes comme "combien de documents traitent de X ?" ou "quelle est la moyenne de Y ?". L'extraction d'entités capture noms et descriptions mais ignore les attributs numériques (dates, quantités, scores), et NetworkX n'a pas de langage de requête.

### 1.1 Enrichir le prompt d'extraction d'entités

**Fichier :** `src/cognidoc/extract_entities.py`

Le prompt actuel demande `name`, `type`, `description`, `confidence`. Il ne demande PAS d'attributs structurés, bien que le champ `Entity.attributes: Dict[str, Any]` existe déjà mais soit inutilisé.

**Changements :**

Ajouter une section `attributes` au format de sortie du prompt :

```json
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "EntityType",
      "description": "Brief description",
      "confidence": 0.95,
      "attributes": {
        "date": "1962",
        "quantity": 42,
        "status": "approved"
      }
    }
  ]
}
```

Mettre à jour les instructions de `build_entity_extraction_prompt()` :

```
5. Extract key attributes as structured key-value pairs in the "attributes" field:
   - Dates (birth_date, publication_date, etc.)
   - Quantities (count, amount, population, etc.)
   - Status or categorical values (status, type, category)
   - Only include attributes explicitly stated in the text
```

**Impact :** Minimal — `Entity.attributes` et `GraphNode.attributes` existent déjà. Le parsing à la ligne 1216 gère déjà `attributes=e.get("attributes", {})`. Aucun changement de schéma nécessaire.

### 1.2 Enrichir le schéma graphe avec des définitions d'attributs typés

**Fichier :** `src/cognidoc/graph_config.py`

`EntityType` a déjà `attributes: List[str]` (inutilisé). L'étendre pour supporter des attributs typés :

```python
@dataclass
class EntityAttribute:
    """Definition of an entity attribute."""
    name: str
    type: str = "string"  # string, number, date, boolean
    description: str = ""

@dataclass
class EntityType:
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    attributes: List[EntityAttribute] = field(default_factory=list)
```

Format `graph_schema.yaml` :

```yaml
entities:
- name: Document
  description: A source document in the corpus
  examples: [...]
  attributes:
    - name: publication_date
      type: date
      description: Date of publication
    - name: page_count
      type: number
      description: Number of pages
```

Passer les définitions d'attributs dans le prompt d'extraction pour que le LLM sache quoi chercher.

### 1.3 Ajouter l'outil agent `aggregate_graph`

**Fichier :** `src/cognidoc/agent_tools.py`

Nouvel outil qui traduit les requêtes d'agrégation en opérations NetworkX :

```python
class AggregateGraphTool(BaseTool):
    """
    Performs counting and aggregation queries on the knowledge graph.

    Supports:
    - COUNT: How many entities of type X?
    - COUNT_BY: How many entities of type X with attribute Y = Z?
    - LIST: List all entities of type X
    - GROUP_BY: Group entities of type X by attribute Y
    - STATS: Min/max/avg of numeric attribute Y on type X
    """
    name = ToolName.AGGREGATE_GRAPH
```

**Approche :**

1. Le LLM parse la requête utilisateur en opération structurée :
   ```json
   {"operation": "COUNT", "entity_type": "Document", "filter": {"topic": "theology"}}
   ```
2. Exécution sur NetworkX :
   ```python
   nodes = [n for n in kg.nodes.values()
            if n.type == entity_type
            and all(n.attributes.get(k) == v for k, v in filters.items())]
   return {"count": len(nodes), "entities": [n.name for n in nodes[:10]]}
   ```
3. Cache via `ToolCache` (TTL: 600s)

**Registration :** Ajouter `AGGREGATE_GRAPH = "aggregate_graph"` à `ToolName`, enregistrer dans `ToolRegistry`.

### 1.4 Mettre à jour l'évaluation de complexité

**Fichier :** `src/cognidoc/complexity.py`

Ajouter des patterns d'agrégation (pour déclencher le mode agent, score >= 0.55) :

```python
AGGREGATION_PATTERNS = [
    r"combien\s+(de|d')",
    r"how\s+many",
    r"count\s+(of|the|all)",
    r"nombre\s+(de|d'|total)",
    r"average|moyenne",
    r"total\s+(de|d'|of)",
    r"list\s+all",
    r"liste[rz]?\s+(tous|toutes|les)",
]
```

### 1.5 Tests

- `test_extract_entities.py` : Vérifier que les attributs sont extraits et parsés
- `test_agent_tools.py` : Tester `AggregateGraphTool` avec des données de graphe mockées
- `test_complexity.py` : Tester les nouveaux patterns d'agrégation

### Scope Phase 1

| Fichier | Changements |
|---------|-------------|
| `extract_entities.py` | Mise à jour de 2 fonctions de prompt (~30 lignes) |
| `graph_config.py` | Ajout `EntityAttribute`, mise à jour parser (~20 lignes) |
| `agent_tools.py` | Nouvelle classe `AggregateGraphTool` (~120 lignes) |
| `complexity.py` | Ajout `AGGREGATION_PATTERNS` (~15 lignes) |
| `agent.py` | Enregistrement du nouvel outil (1 ligne) |
| Tests | ~60 lignes sur 3 fichiers de test |

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
Phase 1 (enrichissement NetworkX) :
  1.1  Enrichir le prompt d'extraction (attributs)      <- commencer ici
  1.2  Mettre à jour graph_config.py (EntityAttribute)
  1.3  Ajouter AggregateGraphTool
  1.4  Mettre à jour les patterns de complexité
  1.5  Tests
  1.6  Ré-ingérer un corpus de test pour vérifier l'extraction d'attributs

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
