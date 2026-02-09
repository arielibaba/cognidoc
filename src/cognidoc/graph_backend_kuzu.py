"""
Kùzu implementation of GraphBackend.

Uses Kùzu (embedded graph database) with native Cypher queries.
Kùzu persists automatically to disk — like SQLite for graphs.
"""

import json
from typing import Any, Dict, Iterator, List, Optional

import networkx as nx

from .graph_backend import GraphBackend
from .constants import KUZU_DB_DIR
from .utils.logger import logger

try:
    import kuzu

    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False


def _serialize(value: Any) -> str:
    """Serialize a Python value to a JSON string for storage."""
    return json.dumps(value, ensure_ascii=False)


def _deserialize(value: str) -> Any:
    """Deserialize a JSON string back to a Python value."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


class KuzuBackend(GraphBackend):
    """Graph backend backed by Kùzu embedded graph database."""

    _SCHEMA_INITIALIZED_KEY = "__schema_ok__"

    def __init__(self, db_path: str = None):
        if not KUZU_AVAILABLE:
            raise ImportError(
                "kuzu is not installed. Install it with: pip install 'cognidoc[kuzu]'"
            )

        if db_path is None:
            db_path = KUZU_DB_DIR

        self._db_path = str(db_path)
        self._db = kuzu.Database(self._db_path)
        self._conn = kuzu.Connection(self._db)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create node/rel tables if they don't exist."""
        try:
            self._conn.execute("MATCH (n:Entity) RETURN count(n)")
        except Exception:
            # Tables don't exist yet — create them
            self._conn.execute(
                """
                CREATE NODE TABLE Entity(
                    id STRING,
                    name STRING,
                    type STRING,
                    description STRING,
                    attrs STRING,
                    PRIMARY KEY(id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE REL TABLE Relates(
                    FROM Entity TO Entity,
                    relationship_type STRING,
                    description STRING,
                    weight DOUBLE,
                    source_chunks STRING
                )
                """
            )
            logger.debug("Kùzu schema initialized")

    def close(self) -> None:
        """Close the database connection."""
        self._conn = None
        self._db = None

    # ── Nodes ────────────────────────────────────────────────────────────

    def add_node(self, node_id: str, **attrs) -> None:
        self._conn.execute(
            "CREATE (n:Entity {id: $id, name: $name, type: $type, "
            "description: $desc, attrs: $attrs})",
            parameters={
                "id": node_id,
                "name": attrs.get("name", ""),
                "type": attrs.get("type", ""),
                "desc": attrs.get("description", ""),
                "attrs": _serialize(
                    {k: v for k, v in attrs.items() if k not in ("name", "type", "description")}
                ),
            },
        )

    def has_node(self, node_id: str) -> bool:
        result = self._conn.execute(
            "MATCH (n:Entity) WHERE n.id = $id RETURN count(n) AS cnt",
            parameters={"id": node_id},
        )
        while result.has_next():
            row = result.get_next()
            return row[0] > 0
        return False

    def remove_node(self, node_id: str) -> None:
        # Kùzu automatically removes incident edges when a node is deleted
        self._conn.execute(
            "MATCH (n:Entity) WHERE n.id = $id DELETE n",
            parameters={"id": node_id},
        )

    def update_node_attrs(self, node_id: str, **attrs) -> None:
        for key, value in attrs.items():
            if key in ("name", "type", "description"):
                self._conn.execute(
                    f"MATCH (n:Entity) WHERE n.id = $id SET n.{key} = $val",
                    parameters={"id": node_id, "val": value},
                )
            else:
                # Store in the JSON attrs field — merge with existing
                current = self._get_extra_attrs(node_id)
                current[key] = value
                self._conn.execute(
                    "MATCH (n:Entity) WHERE n.id = $id SET n.attrs = $val",
                    parameters={"id": node_id, "val": _serialize(current)},
                )

    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        result = self._conn.execute(
            "MATCH (n:Entity) WHERE n.id = $id " "RETURN n.name, n.type, n.description, n.attrs",
            parameters={"id": node_id},
        )
        while result.has_next():
            row = result.get_next()
            attrs = {"name": row[0], "type": row[1], "description": row[2]}
            extra = _deserialize(row[3])
            if isinstance(extra, dict):
                attrs.update(extra)
            return attrs
        return {}

    def _get_extra_attrs(self, node_id: str) -> Dict[str, Any]:
        result = self._conn.execute(
            "MATCH (n:Entity) WHERE n.id = $id RETURN n.attrs",
            parameters={"id": node_id},
        )
        while result.has_next():
            row = result.get_next()
            val = _deserialize(row[0])
            return val if isinstance(val, dict) else {}
        return {}

    def number_of_nodes(self) -> int:
        result = self._conn.execute("MATCH (n:Entity) RETURN count(n) AS cnt")
        while result.has_next():
            return result.get_next()[0]
        return 0

    # ── Edges ────────────────────────────────────────────────────────────

    def add_edge(self, src: str, tgt: str, **attrs) -> None:
        self._conn.execute(
            "MATCH (a:Entity), (b:Entity) "
            "WHERE a.id = $src AND b.id = $tgt "
            "CREATE (a)-[:Relates {"
            "  relationship_type: $rtype, description: $desc, "
            "  weight: $w, source_chunks: $sc"
            "}]->(b)",
            parameters={
                "src": src,
                "tgt": tgt,
                "rtype": attrs.get("relationship_type", ""),
                "desc": attrs.get("description", ""),
                "w": float(attrs.get("weight", 1.0)),
                "sc": _serialize(attrs.get("source_chunks", [])),
            },
        )

    def has_edge(self, src: str, tgt: str) -> bool:
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:Relates]->(b:Entity) "
            "WHERE a.id = $src AND b.id = $tgt "
            "RETURN count(r) AS cnt",
            parameters={"src": src, "tgt": tgt},
        )
        while result.has_next():
            return result.get_next()[0] > 0
        return False

    def get_edge_data(self, src: str, tgt: str) -> Optional[Dict[str, Any]]:
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:Relates]->(b:Entity) "
            "WHERE a.id = $src AND b.id = $tgt "
            "RETURN r.relationship_type, r.description, r.weight, r.source_chunks",
            parameters={"src": src, "tgt": tgt},
        )
        while result.has_next():
            row = result.get_next()
            return {
                "relationship_type": row[0],
                "description": row[1],
                "weight": row[2],
                "source_chunks": _deserialize(row[3]) or [],
            }
        return None

    def update_edge_attrs(self, src: str, tgt: str, **attrs) -> None:
        for key, value in attrs.items():
            if key == "source_chunks":
                value = _serialize(value)
            if key in ("relationship_type", "description", "weight", "source_chunks"):
                self._conn.execute(
                    f"MATCH (a:Entity)-[r:Relates]->(b:Entity) "
                    f"WHERE a.id = $src AND b.id = $tgt "
                    f"SET r.{key} = $val",
                    parameters={"src": src, "tgt": tgt, "val": value},
                )

    def iter_edges(self, data: bool = False) -> Iterator:
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:Relates]->(b:Entity) "
            "RETURN a.id, b.id, r.relationship_type, r.description, r.weight, r.source_chunks"
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            if data:
                rows.append(
                    (
                        row[0],
                        row[1],
                        {
                            "relationship_type": row[2],
                            "description": row[3],
                            "weight": row[4],
                            "source_chunks": _deserialize(row[5]) or [],
                        },
                    )
                )
            else:
                rows.append((row[0], row[1]))
        return iter(rows)

    def number_of_edges(self) -> int:
        result = self._conn.execute("MATCH ()-[r:Relates]->() RETURN count(r) AS cnt")
        while result.has_next():
            return result.get_next()[0]
        return 0

    # ── Traversal ────────────────────────────────────────────────────────

    def successors(self, node_id: str) -> List[str]:
        result = self._conn.execute(
            "MATCH (a:Entity)-[:Relates]->(b:Entity) " "WHERE a.id = $id RETURN b.id",
            parameters={"id": node_id},
        )
        ids = []
        while result.has_next():
            ids.append(result.get_next()[0])
        return ids

    def predecessors(self, node_id: str) -> List[str]:
        result = self._conn.execute(
            "MATCH (a:Entity)-[:Relates]->(b:Entity) " "WHERE b.id = $id RETURN a.id",
            parameters={"id": node_id},
        )
        ids = []
        while result.has_next():
            ids.append(result.get_next()[0])
        return ids

    def find_all_simple_paths(self, src: str, tgt: str, cutoff: int = 5) -> List[List[str]]:
        """Find all simple paths using Kùzu recursive Cypher."""
        # Kùzu supports variable-length paths
        result = self._conn.execute(
            "MATCH path = (a:Entity)-[:Relates*1.." + str(cutoff) + "]->(b:Entity) "
            "WHERE a.id = $src AND b.id = $tgt "
            "RETURN nodes(path)",
            parameters={"src": src, "tgt": tgt},
        )
        paths = []
        while result.has_next():
            row = result.get_next()
            node_list = row[0]
            # Extract IDs from nodes — Kùzu returns dicts with properties
            path_ids = []
            for node_info in node_list:
                if isinstance(node_info, dict):
                    path_ids.append(node_info.get("id", ""))
                else:
                    # Fallback for different Kùzu return types
                    path_ids.append(str(node_info))
            # Filter out duplicate paths (cycles)
            if len(path_ids) == len(set(path_ids)):
                paths.append(path_ids)
        return paths

    def degree(self) -> Dict[str, int]:
        # Count outgoing + incoming for each node
        result = self._conn.execute(
            "MATCH (n:Entity) "
            "OPTIONAL MATCH (n)-[r1:Relates]->() "
            "OPTIONAL MATCH ()-[r2:Relates]->(n) "
            "RETURN n.id, count(DISTINCT r1) + count(DISTINCT r2) AS deg"
        )
        degrees = {}
        while result.has_next():
            row = result.get_next()
            degrees[row[0]] = row[1]
        return degrees

    # ── Export / import ──────────────────────────────────────────────────

    def to_undirected_networkx(self) -> nx.Graph:
        """Export as undirected NetworkX graph for Louvain community detection."""
        G = nx.Graph()

        # Add nodes
        result = self._conn.execute("MATCH (n:Entity) RETURN n.id, n.name, n.type, n.description")
        while result.has_next():
            row = result.get_next()
            G.add_node(row[0], name=row[1], type=row[2], description=row[3])

        # Add edges (undirected)
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:Relates]->(b:Entity) "
            "RETURN a.id, b.id, r.relationship_type, r.weight"
        )
        while result.has_next():
            row = result.get_next()
            G.add_edge(row[0], row[1], relationship_type=row[2], weight=row[3])

        return G

    def to_node_link_data(self) -> dict:
        """Serialize to NetworkX node-link JSON format (for save compatibility)."""
        G = nx.DiGraph()

        result = self._conn.execute(
            "MATCH (n:Entity) RETURN n.id, n.name, n.type, n.description, n.attrs"
        )
        while result.has_next():
            row = result.get_next()
            attrs = {"name": row[1], "type": row[2], "description": row[3]}
            extra = _deserialize(row[4])
            if isinstance(extra, dict):
                attrs.update(extra)
            G.add_node(row[0], **attrs)

        result = self._conn.execute(
            "MATCH (a:Entity)-[r:Relates]->(b:Entity) "
            "RETURN a.id, b.id, r.relationship_type, r.description, r.weight, r.source_chunks"
        )
        while result.has_next():
            row = result.get_next()
            G.add_edge(
                row[0],
                row[1],
                relationship_type=row[2],
                description=row[3],
                weight=row[4],
                source_chunks=_deserialize(row[5]) or [],
            )

        return nx.node_link_data(G)

    def from_node_link_data(self, data: dict) -> None:
        """Import from NetworkX node-link JSON format."""
        G = nx.node_link_graph(data)

        # Clear existing data
        try:
            self._conn.execute("MATCH (n:Entity) DELETE n")
        except Exception:
            pass

        # Import nodes
        for node_id, attrs in G.nodes(data=True):
            self.add_node(str(node_id), **attrs)

        # Import edges
        for src, tgt, attrs in G.edges(data=True):
            self.add_edge(str(src), str(tgt), **attrs)
