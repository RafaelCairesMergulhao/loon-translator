from datetime import UTC, datetime
from pathlib import Path

from domain.entities import AudioDirection, TranslationRecord
from domain.history import (
    LEGACY_CONVERSATION,
    format_history_entry,
    format_history_time,
    parse_history_bound,
    suggest_conversation_name,
)
from infrastructure.history_store import HistoryStore


def _record(source: str, translated: str, when: datetime | None = None) -> TranslationRecord:
    return TranslationRecord(
        source_text=source,
        translated_text=translated,
        source_lang="pt",
        target_lang="en",
        direction=AudioDirection.OUTGOING,
        provider="teste",
        latency_ms=123,
        timestamp=when or datetime(2026, 3, 10, 14, 5, tzinfo=UTC),
    )


def test_record_round_trip_through_dict() -> None:
    original = _record("olá", "hello")
    restored = TranslationRecord.from_dict(original.to_dict())
    assert restored.source_text == "olá"
    assert restored.translated_text == "hello"
    assert restored.direction == AudioDirection.OUTGOING
    assert restored.timestamp == original.timestamp
    assert restored.latency_ms == 123


def test_history_store_round_trip_ordering_limit_and_clear(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    first = _record("um", "one", datetime(2026, 1, 1, tzinfo=UTC))
    second = _record("dois", "two", datetime(2026, 1, 2, tzinfo=UTC))
    third = _record("três", "three", datetime(2026, 1, 3, tzinfo=UTC))
    store.add(first)
    store.add(second)
    store.add(third)

    loaded = store.recent(10)
    assert [item.source_text for item in loaded] == ["um", "dois", "três"]
    assert loaded[0].translated_text == "one"
    assert loaded[-1].direction == AudioDirection.OUTGOING

    newest = store.recent(2)
    assert [item.source_text for item in newest] == ["dois", "três"]

    store.clear()
    assert store.recent(10) == []


def test_old_rows_without_a_name_become_teste_1(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE translation_history (
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
        connection.execute(
            """
            INSERT INTO translation_history (
                timestamp, direction, source_text, translated_text,
                source_lang, target_lang, provider, latency_ms
            ) VALUES ('2026-01-01T00:00:00+00:00', 'outgoing', 'antiga', 'old', 'pt', 'en', 'teste', 1)
            """
        )
    store = HistoryStore(path)
    assert store.recent(10)[0].conversation == LEGACY_CONVERSATION
    named = _record("nova", "new")
    named.conversation = "Cliente"
    store.add(named)
    assert store.conversations() == ["Cliente", LEGACY_CONVERSATION]
    only_named = store.search(conversation="Cliente")
    assert [item.source_text for item in only_named] == ["nova"]


def test_time_range_keeps_the_middle(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    early = _record("cedo", "early", datetime(2026, 3, 1, 10, tzinfo=UTC))
    middle = _record("meio", "middle", datetime(2026, 3, 1, 12, tzinfo=UTC))
    late = _record("tarde", "late", datetime(2026, 3, 1, 18, tzinfo=UTC))
    for record in (early, middle, late):
        record.conversation = "Conversa 1"
        store.add(record)
    found = store.search(
        conversation="Conversa 1",
        start=datetime(2026, 3, 1, 11, tzinfo=UTC),
        end=datetime(2026, 3, 1, 15, tzinfo=UTC),
    )
    assert [item.source_text for item in found] == ["meio"]
    parsed, ok = parse_history_bound("01/03/2026", end=True)
    assert ok and parsed is not None and parsed.astimezone().hour == 23
    assert parse_history_bound("ontem", end=False) == (None, False)
    assert parse_history_bound("", end=True) == (None, True)


def test_next_conversation_name_skips_names_already_used() -> None:
    assert suggest_conversation_name(["Conversa 1", "teste 1"], "Conversa {number}") == "Conversa 2"
    assert suggest_conversation_name([], "Conversa {number}") == "Conversa 1"


def test_format_history_entry_includes_time_and_texts() -> None:
    record = _record("bom dia", "good morning")
    record.portuguese_text = "bom dia"
    record.english_text = "good morning"
    text = format_history_entry(record, direction="EU → CHAMADA", provider="argos")
    assert format_history_time(record.timestamp) in text
    assert "[EU → CHAMADA] bom dia" in text
    assert "→ good morning" in text
    assert "PT: bom dia" in text
    assert "EN: good morning" in text
    assert "argos" in text
