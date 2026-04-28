from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage


class ChatRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('human', 'ai')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations (user_id, updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id
                ON messages (conversation_id, id)
                """
            )
        self._ensure_schema_migrations()

    def _ensure_schema_migrations(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("PRAGMA table_info(conversations)").fetchall()
            has_is_pinned = any(row["name"] == "is_pinned" for row in rows)
            if not has_is_pinned:
                connection.execute(
                    """
                    ALTER TABLE conversations
                    ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_pinned_updated
                ON conversations (user_id, is_pinned DESC, updated_at DESC)
                """
            )

    def create_user(self, username: str, password_hash: str) -> bool:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (?, ?)
                    """,
                    (username, password_hash),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user_by_username(self, username: str):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT id, username, password_hash, created_at
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

    def get_user_by_id(self, user_id: int):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT id, username, password_hash, created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

    def create_conversation(self, user_id: int, title: str = "New chat") -> str:
        conversation_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, user_id, title)
                VALUES (?, ?, ?)
                """,
                (conversation_id, user_id, title),
            )
        return conversation_id

    def _conversation_summary_query(self) -> str:
        return """
            SELECT
                c.id,
                c.user_id,
                c.title,
                c.is_pinned,
                c.created_at,
                c.updated_at,
                COALESCE(
                    (
                        SELECT m.content
                        FROM messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.id DESC
                        LIMIT 1
                    ),
                    ''
                ) AS preview,
                COALESCE(
                    (
                        SELECT strftime('%Y-%m-%d %H:%M', m.created_at)
                        FROM messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.id DESC
                        LIMIT 1
                    ),
                    strftime('%Y-%m-%d %H:%M', c.updated_at)
                ) AS updated_label,
                (
                    SELECT COUNT(*)
                    FROM messages m
                    WHERE m.conversation_id = c.id
                ) AS message_count
            FROM conversations c
        """

    def list_conversations(self, user_id: int):
        with self._connect() as connection:
            return connection.execute(
                self._conversation_summary_query()
                + """
                WHERE c.user_id = ?
                ORDER BY c.is_pinned DESC, c.updated_at DESC, c.created_at DESC
                """,
                (user_id,),
            ).fetchall()

    def get_conversation(self, user_id: int, conversation_id: str):
        with self._connect() as connection:
            return connection.execute(
                self._conversation_summary_query()
                + """
                WHERE c.user_id = ? AND c.id = ?
                LIMIT 1
                """,
                (user_id, conversation_id),
            ).fetchone()

    def get_most_recent_conversation(self, user_id: int):
        with self._connect() as connection:
            return connection.execute(
                self._conversation_summary_query()
                + """
                WHERE c.user_id = ?
                ORDER BY c.is_pinned DESC, c.updated_at DESC, c.created_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

    def get_messages(self, user_id: int, conversation_id: str):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    m.id,
                    m.role,
                    m.content,
                    m.created_at,
                    strftime('%H:%M', m.created_at) AS time_label
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ? AND c.id = ?
                ORDER BY m.id ASC
                """,
                (user_id, conversation_id),
            ).fetchall()

    def get_last_message(self, user_id: int, conversation_id: str):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT m.id, m.role, m.content, m.created_at
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ? AND c.id = ?
                ORDER BY m.id DESC
                LIMIT 1
                """,
                (user_id, conversation_id),
            ).fetchone()

    def load_recent_history(
        self,
        user_id: int,
        conversation_id: str,
        limit: int = 20,
    ) -> List[HumanMessage | AIMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT m.id, m.role, m.content
                    FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE c.user_id = ? AND c.id = ?
                    ORDER BY m.id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (user_id, conversation_id, limit),
            ).fetchall()

        history: List[HumanMessage | AIMessage] = []
        for row in rows:
            if row["role"] == "human":
                history.append(HumanMessage(content=row["content"]))
            else:
                history.append(AIMessage(content=row["content"]))
        return history

    def add_exchange(
        self,
        user_id: int,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            raise ValueError("Conversation not found for this user.")

        should_set_title = conversation["message_count"] == 0 and conversation["title"] == "New chat"
        title = self._build_title(user_message) if should_set_title else None

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
                """,
                [
                    (conversation_id, "human", user_message),
                    (conversation_id, "ai", assistant_message),
                ],
            )
            if title:
                connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (title, conversation_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (conversation_id,),
                )

    def _build_title(self, user_message: str) -> str:
        normalized = " ".join(user_message.strip().split())
        if not normalized:
            return "New chat"
        if len(normalized) <= 60:
            return normalized
        return normalized[:57].rstrip() + "..."

    def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                """
                DELETE FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            )
            return result.rowcount > 0

    def rename_conversation(self, user_id: int, conversation_id: str, title: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (title, conversation_id, user_id),
            )
            return result.rowcount > 0

    def set_conversation_pin(self, user_id: int, conversation_id: str, pinned: bool) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE conversations
                SET is_pinned = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (1 if pinned else 0, conversation_id, user_id),
            )
            return result.rowcount > 0

    def clear_conversation_messages(self, user_id: int, conversation_id: str) -> bool:
        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            return False
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            connection.execute(
                """
                UPDATE conversations
                SET title = 'New chat', updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            )
        return True

    def replace_last_ai_message(self, user_id: int, conversation_id: str, content: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT m.id
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ? AND c.id = ? AND m.role = 'ai'
                ORDER BY m.id DESC
                LIMIT 1
                """,
                (user_id, conversation_id),
            ).fetchone()
            if not row:
                return False
            connection.execute(
                """
                UPDATE messages
                SET content = ?, created_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (content, row["id"]),
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            )
        return True

    def serialize_conversation(self, row) -> Optional[dict]:
        if not row:
            return None
        return {
            "id": row["id"],
            "title": row["title"],
            "preview": row["preview"],
            "updated_label": row["updated_label"],
            "message_count": row["message_count"],
            "is_pinned": bool(row["is_pinned"]),
        }
