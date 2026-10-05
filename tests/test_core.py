from pathlib import Path

from core.config import AppSettings, SettingsStore
from domain.entities import AudioDirection, TranslationRecord
from domain.structures import CircularBuffer
from infrastructure.history_store import HistoryStore


def test_circular_buffer_keeps_chronological_capacity() -> None:
    buffer = CircularBuffer[int](capacity=3)
    buffer.extend([1, 2, 3, 4])
    assert buffer.get_all() == [2, 3, 4]


def test_settings_round_trip(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    expected = AppSettings(
        outgoing_language="Japonês",
        save_history=True,
        phrase_silence_ms=800,
    )
    store.save(expected)
    assert store.load() == expected


def test_existing_settings_skip_welcome(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"outgoing_language": "Japonês"}', encoding="utf-8")
    loaded = SettingsStore(path).load()
    assert loaded.onboarding_complete is True
    assert loaded.outgoing_language == "Japonês"
    assert loaded.native_language == "Português"
    assert loaded.conversation_context == "casual"
    assert loaded.voice_pitch == 0.0


def test_missing_settings_require_welcome(tmp_path: Path) -> None:
    loaded = SettingsStore(tmp_path / "settings.json").load()
    assert loaded.onboarding_complete is False


def test_invalid_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{inválido", encoding="utf-8")
    assert SettingsStore(path).load() == AppSettings()


def test_history_store_persists_record(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    record = TranslationRecord(
        source_text="olá",
        translated_text="hello",
        source_lang="pt",
        target_lang="en",
        direction=AudioDirection.OUTGOING,
        provider="teste",
        latency_ms=123,
    )
    store.add(record)
    with store._connect() as connection:
        row = connection.execute("SELECT * FROM translation_history").fetchone()
    assert row["source_text"] == "olá"
    assert row["translated_text"] == "hello"
    assert row["direction"] == "outgoing"
