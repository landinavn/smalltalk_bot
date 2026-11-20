"""Создание базы Small Talk. Повторный запуск сохраняет существующие данные."""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DB_PATH = ROOT / os.environ.get("SMALLTALK_DB_PATH", "data/smalltalk.sqlite3")


@contextmanager
def database():
    connection = sqlite3.connect(DB_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with database() as connection:
        connection.executescript("""
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                UNIQUE (source, chat_id, message_id)
            );

            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS history_by_user
                ON history (source, user_id, id DESC);
        """)


if __name__ == "__main__":
    init_db()
    print(f"SQLite: {DB_PATH}")
