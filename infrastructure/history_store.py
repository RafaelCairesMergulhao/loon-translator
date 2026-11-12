from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from datetime import datetime

from core.config import APP_DIR
from domain.entities import TranslationRecord
from domain.history import LEGACY_CONVERSATION


class HistoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or APP_DIR / "loon_translator.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
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
                        latency_ms INTEGER NOT NULL,
                        conversation TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(translation_history)")
                }
                if "conversation" not in columns:
                    connection.execute(
                        "ALTER TABLE translation_history ADD COLUMN conversation TEXT NOT NULL DEFAULT ''"
                    )
                pending = connection.execute(
                    """
                    SELECT 1 FROM translation_history
                    WHERE conversation IS NULL OR trim(conversation) = ''
                    LIMIT 1
                    """
                ).fetchone()
                if pending is not None:
                    connection.execute(
                        """
                        UPDATE translation_history
                        SET conversation = ?
                        WHERE conversation IS NULL OR trim(conversation) = ''
                        """,
                        (LEGACY_CONVERSATION,),
                    )

    def add(self, record: TranslationRecord) -> None:
        data = record.to_dict()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO translation_history (
                        timestamp, direction, source_text, translated_text,
                        source_lang, target_lang, provider, latency_ms, conversation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        data["conversation"],
                    ),
                )

    def recent(self, limit: int = 200) -> list[TranslationRecord]:
        count = max(0, int(limit))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT timestamp, direction, source_text, translated_text,
                       source_lang, target_lang, provider, latency_ms, conversation
                FROM translation_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (count,),
            ).fetchall()
        records = [TranslationRecord.from_dict(dict(row)) for row in rows]
        records.reverse()
        return records

    def conversations(self) -> list[str]:
        """Nomes já usados, da conversa mais recente para a mais antiga."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT conversation
                FROM translation_history
                WHERE trim(conversation) != ''
                GROUP BY conversation
                ORDER BY MAX(id) DESC
                """
            ).fetchall()
        return [str(row["conversation"]) for row in rows]

    def search(
        self,
        *,
        conversation: str = "",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 5000,
    ) -> list[TranslationRecord]:
        query = """
            SELECT timestamp, direction, source_text, translated_text,
                   source_lang, target_lang, provider, latency_ms, conversation
            FROM translation_history
            WHERE 1 = 1
        """
        params: list[object] = []
        if conversation.strip():
            query += " AND conversation = ?"
            params.append(conversation.strip())
        query += " ORDER BY id ASC LIMIT ?"
        params.append(max(0, int(limit)))
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        records = [TranslationRecord.from_dict(dict(row)) for row in rows]
        if start is None and end is None:
            return records
        return [
            record
            for record in records
            if (start is None or record.timestamp >= start)
            and (end is None or record.timestamp <= end)
        ]

    def clear(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM translation_history")
