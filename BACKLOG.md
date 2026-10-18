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

- [ ] `TODO#neo4j-backend` — hors scope pour l'instant. Quand nécessaire :

- Ajouter une implémentation `Neo4jBackend(GraphBackend)`
- Utiliser le driver Python `neo4j`
- Configuration : `GRAPH_BACKEND=neo4j`, `NEO4J_URI`, `NEO4J_AUTH`
- L'abstraction `GraphBackend` (`graph_backend.py`) de la Phase 2 en fait un remplacement drop-in

---

## Feedback Loop (futur)

- [ ] `TODO#feedback-loop` — collecte du retour utilisateur pour améliorer la qualité du retrieval
  (reporté du Roadmap de `README.md` le 2026-10-19).

---

## Graph Viewer Search (futur)

- [ ] `TODO#graph-viewer-search` — recherche et filtrage par attribut dans la visualisation du
  graphe (reporté du Roadmap de `README.md` le 2026-10-19).

---

## Défaut : le thème Soft de l'interface n'est jamais appliqué

- [ ] `TODO#theme-soft-perdu` — `cognidoc_app.py` passe `theme=gr.themes.Soft(...)` à
  `gr.Blocks()`, mais l'app est montée par `gr.mount_gradio_app`, qui l'ignore (Gradio 6.2) :
  après montage, `demo.theme` est le thème par défaut. Passer `theme=` à `mount_gradio_app`.
  Relevé par revue le 2026-10-19.

---

## Ordre d'implémentation

Photo datée de l'état, déplacée dans `BACKLOG-ARCHIVE.md` le 2026-10-19 : l'état de chaque tâche vit dans sa section.

---

## Comment reprendre ce travail

Pour continuer l'implémentation dans une future session Claude Code :

**Phase 3 :**
```
Implémente la Phase 3 du roadmap dans BACKLOG.md.
Ajoute une implémentation Neo4jBackend(GraphBackend) dans graph_backend_neo4j.py.
```
