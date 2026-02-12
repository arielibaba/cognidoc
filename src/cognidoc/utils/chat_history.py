"""
Persistent chat history with SQLite backend.

Stores conversations and messages for the CogniDoc chat UI,
enabling conversation management (list, rename, delete, export).
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import logger


class ChatHistory:
    """SQLite-backed persistent chat history."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from ..constants import CHAT_HISTORY_DB

            db_path = str(CHAT_HISTORY_DB)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Chat history initialized at {self.db_path}")

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conv "
                "ON messages(conversation_id, timestamp)"
            )
            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()

    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation. Returns its UUID."""
        conv_id = str(uuid.uuid4())
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title or "New Conversation", now, now),
            )
            conn.commit()
        return conv_id

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations ordered by most recent update."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation, ordered by timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, role, content, sources, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.get("sources"):
                try:
                    msg["sources"] = json.loads(msg["sources"])
                except json.JSONDecodeError:
                    pass
            result.append(msg)
        return result

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
    ) -> int:
        """Add a message to a conversation. Returns message ID."""
        now = time.time()
        sources_json = json.dumps(sources) if sources else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sources, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, sources_json, now),
            )
            # Update conversation timestamp and auto-title
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            # Auto-title from first user message
            if role == "user":
                row = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if row and row[0] == "New Conversation":
                    auto_title = content[:50] + ("..." if len(content) > 50 else "")
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (auto_title, conversation_id),
                    )
            conn.commit()
            return cursor.lastrowid or 0

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        """Rename a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, time.time(), conversation_id),
            )
            conn.commit()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages (CASCADE)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

    def export_conversation(self, conversation_id: str, fmt: str = "json") -> str:
        """Export a conversation as JSON or Markdown string."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conv = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not conv:
                return ""

        messages = self.get_messages(conversation_id)

        if fmt == "markdown":
            lines = [f"# {conv['title']}\n"]
            for msg in messages:
                role = "**User**" if msg["role"] == "user" else "**Assistant**"
                lines.append(f"{role}:\n{msg['content']}\n")
            return "\n".join(lines)

        # Default: JSON
        return json.dumps(
            {
                "id": conv["id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "messages": messages,
            },
            indent=2,
            ensure_ascii=False,
        )


# Global instance (lazy)
_chat_history: Optional[ChatHistory] = None


def get_chat_history(db_path: Optional[str] = None) -> ChatHistory:
    """Get the global ChatHistory instance."""
    global _chat_history
    if _chat_history is None:
        _chat_history = ChatHistory(db_path)
    return _chat_history


__all__ = ["ChatHistory", "get_chat_history"]
