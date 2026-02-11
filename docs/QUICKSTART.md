# CogniDoc — Quick Start Guide

Guide pas-à-pas pour déployer CogniDoc sur un nouveau projet avec vos propres documents.

---

## Prérequis

| Prérequis | Pourquoi |
|-----------|----------|
| Python 3.10+ | Runtime |
| [Ollama](https://ollama.ai/) en cours d'exécution (`ollama serve`) | Embeddings + parsing local |
| Clé API Gemini | LLM par défaut (ou OpenAI/Anthropic) |

### Modèles Ollama requis

```bash
ollama pull qwen3-embedding:4b-q8_0          # Embeddings (obligatoire)
ollama pull ibm/granite-docling:258m-bf16     # Parsing documents (obligatoire)
ollama pull dengcao/Qwen3-Reranker-0.6B:F16   # Reranking (optionnel, améliore la pertinence)
ollama pull granite3.3:8b                      # LLM local (si provider=ollama)
```

---

## Étape 1 — Créer le projet et installer CogniDoc

```bash
mkdir mon-projet && cd mon-projet

# Installer CogniDoc depuis GitHub
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Pin Gradio (6.3+ cause des écrans blancs avec Svelte 5)
pip install "gradio==6.2.0"

# Initialiser la structure du projet
cognidoc init --schema
```

Cela crée :

```
mon-projet/
├── data/sources/              ← vos documents ici
├── config/graph_schema.yaml   ← schéma par défaut (générique)
└── .env.example
```

> **Note :** `cognidoc init --schema` crée un schéma générique (Concept, Person, Organization). L'étape 4 le remplacera par un schéma adapté à votre corpus.

---

## Étape 2 — Configurer l'environnement

```bash
cp .env.example .env
```

Éditez `.env` et ajoutez votre clé API et vos modèles :

```bash
# Clé API (obligatoire pour Gemini)
GOOGLE_API_KEY=votre-clé-ici

# Provider + modèle LLM (la CLI --llm choisit le provider, le .env choisit le modèle)
DEFAULT_LLM_PROVIDER=gemini
GEMINI_LLM_MODEL=gemini-3-flash-preview

# Provider + modèle Embedding
COGNIDOC_EMBEDDING_PROVIDER=ollama
OLLAMA_EMBED_MODEL=qwen3-embedding:4b-q8_0
```

> **Autres providers :** les modèles se configurent par provider dans `.env` :
>
> | Provider | Var LLM | Var Embedding | Clé API |
> |----------|---------|---------------|---------|
> | gemini | `GEMINI_LLM_MODEL` | — | `GOOGLE_API_KEY` |
> | ollama | `OLLAMA_LLM_MODEL` | `OLLAMA_EMBED_MODEL` | aucune (local) |
> | openai | `OPENAI_LLM_MODEL` | — | `OPENAI_API_KEY` |
> | anthropic | `ANTHROPIC_LLM_MODEL` | — | `ANTHROPIC_API_KEY` |
>
> La CLI (`--llm gemini --embedding ollama`) choisit le **provider**, le `.env` choisit le **modèle** chez ce provider.

---

## Étape 3 — Ajouter vos documents

```bash
cp /chemin/vers/mes/documents/* data/sources/
```

Formats supportés : **PDF, DOCX, PPTX, XLSX, HTML, Markdown, images** (PNG, JPG).

> **Tip :** Organisez vos documents en sous-dossiers dans `data/sources/` si vous avez plusieurs thématiques. Le générateur de schéma échantillonnera équitablement chaque sous-dossier.

---

## Étape 4 — Générer le schéma du graphe

```bash
cognidoc schema-generate ./data/sources --language fr
```

Le LLM analyse un échantillon de vos documents (jusqu'à 100, 3 premières pages chacun) et génère `config/graph_schema.yaml` avec les types d'entités et de relations adaptés à votre corpus.

> **Corpus multilingue :** `--language` contrôle uniquement la langue de sortie du schéma (noms d'entités, descriptions, exemples), pas la langue d'analyse. Le LLM lit et comprend vos documents quelle que soit leur langue. Choisissez simplement la langue dans laquelle vous voulez que le schéma soit rédigé.

**Vérifier et ajuster le schéma :**

```bash
# Voir le schéma généré
cat config/graph_schema.yaml

# Si le schéma ne convient pas, éditez-le manuellement
nano config/graph_schema.yaml

# Ou régénérez avec d'autres paramètres
cognidoc schema-generate ./data/sources \
    --language fr \
    --max-docs 50 \
    --max-pages 5 \
    --regenerate
```

> **Note :** Si vous sautez cette étape, `cognidoc ingest` générera automatiquement le schéma au premier lancement s'il n'en trouve pas.

---

## Étape 5 — Ingérer les documents

```bash
cognidoc ingest ./data/sources --llm gemini --embedding ollama
```

Pipeline complet : conversion PDF → images 600 DPI → détection YOLO → extraction texte/tableaux → chunking sémantique → embeddings Qdrant + BM25 → knowledge graph + entity resolution.

**Options utiles :**

```bash
# Dry-run : vérifier la config sans rien ingérer
cognidoc ingest ./data/sources --dry-run

# Sans YOLO (plus rapide, extraction page-level)
cognidoc ingest ./data/sources --no-yolo

# Forcer la régénération du schéma pendant l'ingestion
cognidoc ingest ./data/sources --regenerate-schema

# Ré-ingestion complète (ignorer le cache incrémental)
cognidoc ingest ./data/sources --full-reindex
```

> **Ingestion incrémentale :** Les ingestions suivantes ne traitent que les fichiers nouveaux ou modifiés (tracking SHA-256). Ajoutez simplement des fichiers dans `data/sources/` et relancez `cognidoc ingest`.

---

## Étape 6 — Lancer l'interface web

```bash
cognidoc serve --port 7860
```

Ouvrez **http://localhost:7860** dans votre navigateur.

L'interface propose 3 onglets :
- **Chat** — Posez des questions sur vos documents
- **Metrics** — Dashboard de performance (latence, distribution des requêtes)
- **Graph** — Visualisation interactive du knowledge graph (D3.js)

**Options :**

```bash
# Lien public partageable (via Gradio)
cognidoc serve --share

# Désactiver le reranking LLM (plus rapide, moins précis)
cognidoc serve --no-rerank
```

---

## Résumé

```bash
# 6 commandes pour démarrer
mkdir mon-projet && cd mon-projet
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git" "gradio==6.2.0"
cognidoc init --schema
# → Copier vos documents dans data/sources/
# → Créer .env avec GOOGLE_API_KEY
cognidoc schema-generate ./data/sources --language fr
cognidoc ingest ./data/sources --llm gemini --embedding ollama
cognidoc serve
```

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `Storage folder ... already accessed by another instance` | Un processus Qdrant orphelin verrouille le dossier. Fix : `lsof -ti :7860 \| xargs kill -9` |
| Écran blanc dans l'interface | Gradio 6.3+ incompatible. Fix : `pip install "gradio==6.2.0"` |
| `GOOGLE_API_KEY not set` | Ajoutez la clé dans `.env` ou exportez-la : `export GOOGLE_API_KEY=...` |
| `Cannot reach Ollama` | Lancez Ollama : `ollama serve` |
| Schéma trop générique | Régénérez avec `--regenerate` ou éditez `config/graph_schema.yaml` manuellement |
