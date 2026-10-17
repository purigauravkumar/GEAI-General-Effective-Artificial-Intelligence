"""Persistent task queue primitives for GEAI.

This module deliberately does not execute arbitrary shell commands or model-generated
code. It provides a durable, auditable queue that a permission-gated worker can use.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    home = Path(os.getenv("GEAI_HOME", Path.home() / "GEAI")).expanduser().resolve()
    return home / "tasks" / "tasks.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK(status IN
                ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at)"
    )
    return connection


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    item["result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
    return item


def create_task(title: str, description: str = "") -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise ValueError("title must not be empty")
    task_id = str(uuid.uuid4())
    now = _now()
    with _connect() as db:
        db.execute(
            """INSERT INTO tasks (id, title, description, status, created_at, updated_at)
               VALUES (?, ?, ?, 'queued', ?, ?)""",
            (task_id, title, description, now, now),
        )
        row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _decode(row)  # type: ignore[return-value]


def get_task(task_id: str) -> dict[str, Any] | None:
    with _connect() as db:
        row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _decode(row)


def list_tasks(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    if status is not None and status not in STATUSES:
        raise ValueError(f"unsupported status: {status}")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    with _connect() as db:
        if status is None:
            rows = db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
    return [_decode(row) for row in rows]  # type: ignore[misc]


def claim_next_task() -> dict[str, Any] | None:
    """Atomically claim the oldest queued task; safe against competing workers."""
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT id FROM tasks WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
        ).fetchone()
        if row is None:
            db.execute("COMMIT")
            return None
        task_id = row["id"]
        now = _now()
        db.execute(
            """UPDATE tasks SET status = 'running', updated_at = ?, attempts = attempts + 1
               WHERE id = ? AND status = 'queued'""",
            (now, task_id),
        )
        claimed = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        db.execute("COMMIT")
        return _decode(claimed)
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()


def complete_task(task_id: str, result: Any = None) -> dict[str, Any]:
    return _finish(task_id, "succeeded", result=result)


def fail_task(task_id: str, error: str) -> dict[str, Any]:
    if not error.strip():
        raise ValueError("error must not be empty")
    return _finish(task_id, "failed", error=error)


def _finish(
    task_id: str, status: str, result: Any = None, error: str | None = None
) -> dict[str, Any]:
    with _connect() as db:
        row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task not found: {task_id}")
        if row["status"] != "running":
            raise ValueError(f"cannot finish task in status {row['status']!r}")
        db.execute(
            """UPDATE tasks SET status = ?, updated_at = ?, result_json = ?, error = ?
               WHERE id = ?""",
            (
                status,
                _now(),
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                error,
                task_id,
            ),
        )
        updated = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _decode(updated)  # type: ignore[return-value]


def cancel_task(task_id: str) -> dict[str, Any]:
    with _connect() as db:
        row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task not found: {task_id}")
        if row["status"] != "queued":
            raise ValueError(f"only queued tasks can be cancelled (got {row['status']!r})")
        db.execute(
            "UPDATE tasks SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (_now(), task_id),
        )
        updated = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _decode(updated)  # type: ignore[return-value]
