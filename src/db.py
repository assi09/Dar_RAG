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

CREATE TABLE IF NOT EXISTS message_versions (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    version_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    run_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS eval_metrics (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    question TEXT NOT NULL,
    metric TEXT NOT NULL,
    score REAL NOT NULL,
    success INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    score INTEGER NOT NULL,
    reason TEXT,
    comment TEXT,
    question TEXT,
    answer TEXT,
    sources TEXT,
    created_at TEXT NOT NULL
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

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    if "active_version" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN active_version INTEGER NOT NULL DEFAULT 1")

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
        "SELECT id, role, content, sources, run_id, created_at, active_version FROM messages "
        "WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()

    out_messages = []
    for m in messages:
        version_count = conn.execute(
            "SELECT COUNT(*) FROM message_versions WHERE message_id = ?", (m["id"],)
        ).fetchone()[0]
        out_messages.append({
            **dict(m),
            "sources": json.loads(m["sources"]) if m["sources"] else [],
            "version_count": version_count or 1,
        })
    conn.close()

    return {
        **dict(convo),
        "messages": out_messages,
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


def get_message(message_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT id, conversation_id, role, content, sources, run_id, created_at, active_version "
        "FROM messages WHERE id = ?",
        (message_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {**dict(row), "sources": json.loads(row["sources"]) if row["sources"] else []}


def get_preceding_user_message(conversation_id: str, message_id: str) -> dict | None:
    """Find the nearest prior role='user' message before the given message."""
    conn = get_db()
    target = conn.execute(
        "SELECT created_at FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    if target is None:
        conn.close()
        return None

    row = conn.execute(
        "SELECT id, content, created_at FROM messages "
        "WHERE conversation_id = ? AND role = 'user' AND created_at < ? "
        "ORDER BY created_at DESC LIMIT 1",
        (conversation_id, target["created_at"]),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_message_versions(message_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT id, message_id, version_index, content, sources, run_id, created_at "
        "FROM message_versions WHERE message_id = ? ORDER BY version_index ASC",
        (message_id,),
    ).fetchall()
    conn.close()
    return [
        {**dict(r), "sources": json.loads(r["sources"]) if r["sources"] else []}
        for r in rows
    ]


def count_message_versions(message_id: str) -> int:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM message_versions WHERE message_id = ?", (message_id,)
    ).fetchone()[0]
    conn.close()
    return count


def add_message_version(
    version_id: str,
    message_id: str,
    version_index: int,
    content: str,
    sources: list,
    run_id: str | None,
    created_at: str,
):
    conn = get_db()
    conn.execute(
        "INSERT INTO message_versions (id, message_id, version_index, content, sources, run_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (version_id, message_id, version_index, content, json.dumps(sources), run_id, created_at),
    )
    conn.commit()
    conn.close()


def update_message(
    message_id: str,
    content: str,
    sources: list,
    run_id: str | None,
    created_at: str,
    active_version: int,
):
    conn = get_db()
    conn.execute(
        "UPDATE messages SET content = ?, sources = ?, run_id = ?, created_at = ?, active_version = ? "
        "WHERE id = ?",
        (content, json.dumps(sources), run_id, created_at, active_version, message_id),
    )
    conn.commit()
    conn.close()


def get_run_context(run_id: str) -> dict | None:
    """Look up question/answer/sources for a run_id from persisted messages.

    Used as a fallback when the in-memory _runs cache has been cleared by a
    backend restart but the run_id still belongs to a message stored earlier.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT id, conversation_id, content, sources FROM messages WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT mv.message_id AS id, m.conversation_id, mv.content, mv.sources "
            "FROM message_versions mv JOIN messages m ON m.id = mv.message_id "
            "WHERE mv.run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        conn.close()
        return None

    prev_user = conn.execute(
        "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' "
        "AND created_at < (SELECT created_at FROM messages WHERE id = ?) "
        "ORDER BY created_at DESC LIMIT 1",
        (row["conversation_id"], row["id"]),
    ).fetchone()
    conn.close()

    return {
        "question": prev_user["content"] if prev_user else "",
        "answer": row["content"],
        "sources": json.loads(row["sources"]) if row["sources"] else [],
    }


def add_feedback(
    feedback_id: str,
    run_id: str,
    score: int,
    reason: str | None,
    comment: str,
    question: str,
    answer: str,
    sources: list,
    created_at: str,
):
    conn = get_db()
    conn.execute(
        "INSERT INTO feedback (id, run_id, score, reason, comment, question, answer, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (feedback_id, run_id, score, reason, comment, question, answer, json.dumps(sources), created_at),
    )
    conn.commit()
    conn.close()


def list_feedback() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT id, run_id, score, reason, comment, question, answer, sources, created_at "
        "FROM feedback ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return [
        {**dict(r), "sources": json.loads(r["sources"]) if r["sources"] else []}
        for r in rows
    ]


def add_eval_metrics(run_id: str, question: str, results: list[dict], created_at: str):
    """Persist live DeepEval metric scores for a 'rag' classified query."""
    import uuid as _uuid

    conn = get_db()
    for r in results:
        conn.execute(
            "INSERT INTO eval_metrics (id, run_id, question, metric, score, success, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(_uuid.uuid4()), run_id, question, r["metric"], r["score"], int(r["success"]), created_at),
        )
    conn.commit()
    conn.close()


def list_eval_metrics() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT id, run_id, question, metric, score, success, created_at "
        "FROM eval_metrics ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
