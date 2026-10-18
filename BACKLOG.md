# CogniDoc — Roadmap

Fichier centralisé de suivi des plans d'implémentation.
Ce qui est fait est dans `BACKLOG-ARCHIVE.md` (à parcourir avec `grep`, pas à lire en entier).

---

## Fonctionnalités implémentées

Section close, déplacée telle quelle dans `BACKLOG-ARCHIVE.md` le 2026-10-19 (statut : fait).

---

## Phase 1 : Enrichissement extraction + Outil d'agrégation (NetworkX)

Section close, déplacée telle quelle dans `BACKLOG-ARCHIVE.md` le 2026-10-19 (statut : fait).

---

## Phase 2 : Migration vers Kùzu (BD graphe embarquée avec Cypher)

Section close, déplacée telle quelle dans `BACKLOG-ARCHIVE.md` le 2026-10-19 (statut : fait).

---

## Phase 3 (futur) : Neo4j pour la production cloud

Hors scope pour l'instant. Quand nécessaire :

- Ajouter une implémentation `Neo4jStorage(GraphStorage)`
- Utiliser le driver Python `neo4j`
- Configuration : `GRAPH_BACKEND=neo4j`, `NEO4J_URI`, `NEO4J_AUTH`
- L'abstraction `GraphStorage` de la Phase 2 en fait un remplacement drop-in

---

## Ordre d'implémentation

Photo datée de l'état, déplacée dans `BACKLOG-ARCHIVE.md` le 2026-10-19 : l'état de chaque tâche vit dans sa section.

---

## Comment reprendre ce travail

Pour continuer l'implémentation dans une future session Claude Code :

**Phase 1 :**
```
Implémente la Phase 1 du roadmap dans BACKLOG.md.
Commence par l'étape 1.1 (enrichir le prompt d'extraction dans extract_entities.py).
```

**Phase 3 :**
```
Implémente la Phase 3 du roadmap dans BACKLOG.md.
Ajoute une implémentation Neo4jBackend(GraphBackend) dans graph_backend_neo4j.py.
```
