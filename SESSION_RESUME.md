# Session CogniDoc - 6 janvier 2026

## Résumé de la session

Cette session a complété l'ingestion pipeline complète pour les documents de bioéthique catholique.

## Travaux accomplis

### 1. Fix du parsing JSON pour GraphRAG
- **Problème**: Gemini ignorait l'instruction "Output ONLY valid JSON" et retournait du texte
- **Solution**: Ajout du paramètre `json_mode` aux providers LLM
  - `src/utils/llm_providers.py`: Ajout de `json_mode` à `LLMConfig` et aux méthodes `chat()`
  - `src/utils/llm_client.py`: Ajout du paramètre `json_mode` à `llm_chat()`
  - `src/extract_entities.py`: Utilisation de `json_mode=True` + normalisation des champs FR→EN
- **Commit**: `ebb98e3` - "Add JSON mode support for reliable entity extraction"

### 2. Pipeline d'ingestion complète
- **Index vectoriel**: 11,484 documents (Qdrant + BM25)
- **Knowledge Graph**:
  - 15,183 noeuds (entités)
  - 20,568 arêtes (relations)
  - 3,912 communautés (Louvain)
- **Temps total**: ~13h 24min

### 3. Types d'entités extraites
| Type | Nombre |
|------|--------|
| ConceptMedical | 3,634 |
| ConceptEthique | 3,597 |
| ConceptTheologique | 1,836 |
| Personne | 1,491 |
| Institution | 1,048 |
| SituationVie | 1,016 |
| ProcedureMedicale | 906 |
| ArgumentEthique | 534 |
| CadreJuridique | 414 |
| DocumentMagisteriel | 386 |
| MethodeContraceptive | 192 |
| ConceptPhilosophique | 103 |

### 4. Tests de l'application
Questions testées avec succès:
- Avortement: Position de l'Église catholique
- Euthanasie: Définition et différence avec sédation palliative
- Contraception: Opposition de l'Église à la contraception artificielle
- PMA: Enjeux éthiques selon l'enseignement catholique
- Embryon: Statut de l'embryon humain

## État actuel du projet

### Fichiers d'index (data/indexes/)
- `child_documents/` - Index vectoriel Qdrant
- `parent_documents/` - Index des documents parents
- `bm25_index.json` - Index BM25 pour recherche keyword
- `knowledge_graph/` - Graphe NetworkX avec communautés

### Commandes utiles
```bash
# Lancer l'application
python -m src.cognidoc_app

# Lancer sans reranking (plus rapide)
python -m src.cognidoc_app --no-rerank

# Relancer seulement le graphe (si besoin)
python -m src.run_ingestion_pipeline --skip-conversion --skip-pdf --skip-yolo \
  --skip-extraction --skip-descriptions --skip-chunking --skip-embeddings --skip-indexing
```

### Configuration
- **LLM par défaut**: Gemini 2.0 Flash
- **Embeddings**: Ollama qwen3-embedding:0.6b (local)
- **Port de l'app**: 7860

## Notes pour la prochaine session

1. L'application est fonctionnelle et prête à l'emploi
2. Le Knowledge Graph améliore significativement les réponses sur les questions relationnelles
3. Les références PDF sont cliquables dans l'interface Gradio
4. Possibilité d'ajouter de nouveaux documents dans `data/pdfs/` et relancer le pipeline

## Statistiques finales

- **Documents source**: ~100+ PDFs de bioéthique catholique
- **Pages traitées**: ~3,448
- **Chunks générés**: 11,742 (parent + child)
- **Entités extraites**: 49,993 (fusionnées en 15,183)
- **Relations extraites**: 30,808
- **Appels LLM totaux**: ~27,400
