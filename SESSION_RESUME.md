# Session CogniDoc - 6 janvier 2026 (Package Transformation)

## Résumé

CogniDoc transformé en package Python installable avec providers flexibles.

## Commits de cette session

1. `1b55ef4` - Add implementation plan for Python package transformation
2. `f323a76` - Transform CogniDoc into installable Python package
3. `ae02d45` - Update session resume
4. `4b955af` - Fix paths after package restructuring

## Ce qui fonctionne

```bash
# Installation
pip install git+https://github.com/arielibaba/cognidoc.git
pip install "cognidoc[all] @ git+https://github.com/arielibaba/cognidoc.git"

# Import Python
from cognidoc import CogniDoc, __version__  # OK

# CLI
cognidoc --help      # OK
cognidoc info        # OK
cognidoc init        # OK
cognidoc serve       # OK (testé, app se lance)

# App Gradio
python -m cognidoc.cognidoc_app --no-rerank  # OK, port 7860
```

## Structure du package

```
src/cognidoc/
├── __init__.py          # Exports: CogniDoc, CogniDocConfig
├── __main__.py          # python -m cognidoc
├── api.py               # Classe CogniDoc principale
├── cli.py               # Interface CLI
├── constants.py         # Chemins (FIX: BASE_DIR corrigé)
├── graph_config.py      # Config GraphRAG (FIX: path corrigé)
├── cognidoc_app.py      # Interface Gradio
├── hybrid_retriever.py  # Retriever hybride
├── utils/
│   ├── llm_providers.py       # Providers LLM
│   ├── embedding_providers.py # Providers Embeddings (NOUVEAU)
│   └── ...
└── ...
```

## Providers flexibles

```python
# Gemini LLM + Ollama Embeddings (défaut)
CogniDoc(llm_provider="gemini", embedding_provider="ollama")

# Full cloud
CogniDoc(llm_provider="openai", embedding_provider="openai")

# Full local
CogniDoc(llm_provider="ollama", embedding_provider="ollama")
```

## Dépendances modulaires (pyproject.toml)

```toml
[project.optional-dependencies]
ui = ["gradio>=4.0"]
yolo = ["ultralytics", "opencv-python", "torch"]
ollama = ["ollama>=0.4"]
cloud = ["google-generativeai", "openai", "anthropic"]
all = ["cognidoc[ui,yolo,ollama,cloud,conversion]"]
```

## État des index

- **Vector**: 11,484 documents (Qdrant)
- **BM25**: 11,484 documents
- **Graph**: 15,183 noeuds, 20,568 arêtes, 3,912 communautés
- **PDFs**: 133 fichiers

## Points à améliorer (prochaine session)

1. **`doc.ingest()`** - Connecter à `run_ingestion_pipeline.py`
2. **Warning Gemini** - Migrer `google.generativeai` → `google.genai`
3. **Tests unitaires** - Ajouter tests pour nouveaux providers
4. **Bug retrieve** - `NameError: name 'LLM' is not defined` dans hybrid_retriever.py:325

## Commandes utiles

```bash
# Lancer l'app (avec index existants)
python -m cognidoc.cognidoc_app --no-rerank

# CLI
cognidoc info
cognidoc serve --port 7860

# Développement
pip install -e ".[all,dev]"
```

## Fichiers clés modifiés

| Fichier | Description |
|---------|-------------|
| `pyproject.toml` | Dépendances modulaires, CLI entry point |
| `src/cognidoc/__init__.py` | Exports publics |
| `src/cognidoc/api.py` | Classe CogniDoc (NOUVEAU) |
| `src/cognidoc/cli.py` | Interface CLI (NOUVEAU) |
| `src/cognidoc/utils/embedding_providers.py` | Providers embeddings (NOUVEAU) |
| `src/cognidoc/constants.py` | Fix BASE_DIR pour nouvelle structure |
| `src/cognidoc/graph_config.py` | Fix chemin config |
