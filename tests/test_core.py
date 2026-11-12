import json
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


def test_old_default_voice_moves_to_piper_once(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"voice_engine": "natural"}', encoding="utf-8")
    store = SettingsStore(path)
    settings = store.load()
    assert settings.voice_engine == "piper"
    settings.voice_engine = "natural"
    store.save(settings)
    assert store.load().voice_engine == "natural", "quem volta ao Edge depois fica com ele"
    path.write_text('{"voice_engine": "windows"}', encoding="utf-8")
    assert store.load().voice_engine == "windows"
    assert AppSettings().voice_engine == "piper"


def test_missing_settings_require_welcome(tmp_path: Path) -> None:
    loaded = SettingsStore(tmp_path / "settings.json").load()
    assert loaded.onboarding_complete is False


def test_a_cpu_with_memory_uses_the_balanced_ear_and_nllb(monkeypatch) -> None:
    from core.config import AppSettings
    from core.machine import apply_pc_ideal, detect_pc, deviation_keys

    monkeypatch.setattr("core.machine.cuda_available", lambda: False)
    monkeypatch.setattr("core.machine.ram_gb", lambda: 32.0)
    monkeypatch.setattr("core.machine.nllb_ready", lambda: True)
    plan = detect_pc()
    assert (plan.whisper_model, plan.offline_asr, plan.offline_mt, plan.voice_engine) == (
        "balanced",
        "alto",
        "nllb",
        "piper",
    )
    settings = AppSettings(whisper_model="accurate", voice_engine="natural", translation_mode="IA")
    apply_pc_ideal(settings)
    assert settings.whisper_model == "balanced"
    assert settings.translation_mode == "Local"
    assert settings.microphone_device == ""
    assert deviation_keys(settings) == []
    settings.whisper_model = "fast"
    settings.offline_asr = ""
    assert deviation_keys(settings) == ["tip_stt_down"]


def test_a_small_pc_stays_on_the_light_ear(monkeypatch) -> None:
    from core.machine import detect_pc

    monkeypatch.setattr("core.machine.cuda_available", lambda: False)
    monkeypatch.setattr("core.machine.ram_gb", lambda: 8.0)
    monkeypatch.setattr("core.machine.nllb_ready", lambda: True)
    plan = detect_pc()
    assert plan.whisper_model == "fast"
    assert plan.offline_mt == ""


def test_a_foreign_settings_file_is_brought_back_to_known_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "whisper_model": "gigante",
                "translation_mode": "mágico",
                "voice_engine": "robot",
                "appearance": "sand",
                "phrase_silence_ms": "50",
                "speech_rate": "rápido",
                "follow_pc_ideal": "sim",
                "offline_mt": "watson",
                "microphone_device": "sd:1",
            }
        ),
        encoding="utf-8",
    )
    loaded = SettingsStore(path).load()
    assert loaded.whisper_model == "auto"
    assert loaded.translation_mode == "Local"
    assert loaded.voice_engine == "piper"
    assert loaded.appearance == "night"
    assert loaded.phrase_silence_ms == 120
    assert loaded.speech_rate == 1.0
    assert loaded.follow_pc_ideal is True
    assert loaded.offline_mt == ""
    assert loaded.microphone_device == "sd:1"


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
    loaded = store.recent(1)
    assert len(loaded) == 1
    assert loaded[0].source_text == "olá"
    assert loaded[0].translated_text == "hello"
    assert loaded[0].direction == AudioDirection.OUTGOING
