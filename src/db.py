"""
Day 3 — Data persistence layer (SQLite).

Stores chat conversations and their messages so users can revisit
previous sessions.

Schema:
  conversations(id, title, created_at, updated_at)
  messages(id, conversation_id, role, content, sources, run_id, created_at)

Run directly to (re)initialize the database file:
  python -m src.db
"""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "dar.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    run_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
"""


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def create_conversation(conversation_id: str, title: str, created_at: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conversation_id, title, created_at, created_at),
    )
    conn.commit()
    conn.close()


def list_conversations() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> dict | None:
    conn = get_db()
    convo = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    if convo is None:
        conn.close()
        return None

    messages = conn.execute(
        "SELECT id, role, content, sources, run_id, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()

    return {
        **dict(convo),
        "messages": [
            {**dict(m), "sources": json.loads(m["sources"]) if m["sources"] else []}
            for m in messages
        ],
    }


def add_message(
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    sources: list,
    run_id: str | None,
    created_at: str,
):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, sources, run_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (message_id, conversation_id, role, content, json.dumps(sources), run_id, created_at),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (created_at, conversation_id),
    )
    conn.commit()
    conn.close()


def rename_conversation(conversation_id: str, title: str, updated_at: str):
    conn = get_db()
    conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title, updated_at, conversation_id),
    )
    conn.commit()
    conn.close()


def delete_conversation(conversation_id: str):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
