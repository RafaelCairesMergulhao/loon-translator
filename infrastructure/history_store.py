from __future__ import annotations

import sqlite3
from pathlib import Path

from core.config import APP_DIR
from domain.entities import TranslationRecord


class HistoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or APP_DIR / "loon_translator.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL
                )
                """
            )

    def add(self, record: TranslationRecord) -> None:
        data = record.to_dict()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO translation_history (
                    timestamp, direction, source_text, translated_text,
                    source_lang, target_lang, provider, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["timestamp"],
                    data["direction"],
                    data["source_text"],
                    data["translated_text"],
                    data["source_lang"],
                    data["target_lang"],
                    data["provider"],
                    data["latency_ms"],
                ),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM translation_history")
