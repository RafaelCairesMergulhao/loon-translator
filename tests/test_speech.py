import numpy as np

from core.config import SettingsStore
from core.whisper_engine import resolve_model
from domain.metrics import LatencyTracker, ProviderHealth
from services.audio_manager import PhraseSegmenter
from services.sapi_worker import sapi_rate
from services.speech_prosody import edge_pitch, sapi_ssml, spoken_text
from services.tts_engine import edge_rate, split_for_speech, trim_silence
import core.whisper_engine as whisper_module


def test_short_reply_stays_in_one_piece() -> None:
    assert split_for_speech("Hi. How are you?") == ["Hi. How are you?"]
    assert split_for_speech("") == []


def test_long_reply_is_split_at_sentence_ends() -> None:
    text = "I will be a little late for the meeting today. Please start without me and I will join soon."
    assert split_for_speech(text) == [
        "I will be a little late for the meeting today.",
        "Please start without me and I will join soon.",
    ]


def test_very_long_sentence_is_split_at_a_comma() -> None:
    sentence = ("palavra " * 30).strip() + ", " + ("outra " * 10).strip()
    chunks = split_for_speech(sentence, max_chars=160)
    assert all(len(chunk) <= 161 for chunk in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_trim_silence_keeps_a_small_pad() -> None:
    rate = 16_000
    audio = np.concatenate([np.zeros(8_000), np.full(1_600, 0.3), np.zeros(8_000)]).astype(np.float32)
    trimmed = trim_silence(audio, rate, pad_ms=40)
    assert trimmed.size == 1_600 + 2 * 640
    assert trim_silence(np.zeros(100, dtype=np.float32), rate).size == 100


def test_speech_rate_mappings() -> None:
    assert sapi_rate(1.0) == 0
    assert sapi_rate(1.1) == 1
    assert sapi_rate(3.0) == 6
    assert edge_rate(1.1) == "+10%"
    assert edge_rate(0.9) == "-10%"
    assert edge_pitch(0) == "+0Hz"
    assert edge_pitch(1) == "+12Hz"
    assert edge_pitch(-2) == "-24Hz"


def test_spoken_text_closes_the_sentence() -> None:
    assert spoken_text("  Hey ,  what's up  ") == "Hey, what's up."
    ssml = sapi_ssml("Hey, what's up?", "en-US", 1.1)
    assert "en-US" in ssml
    assert "break time" in ssml
    assert "rate='1.10'" in ssml


def test_pre_roll_keeps_the_first_consonant() -> None:
    segmenter = PhraseSegmenter(sample_rate=16_000, silence_ms=100, min_speech_ms=30, pre_roll_ms=100)
    for _ in range(5):
        assert segmenter.feed(np.zeros(512, dtype=np.float32)) is None
    onset = np.full(512, 0.004, dtype=np.float32)
    assert segmenter.feed(onset) is None
    segmenter.feed(np.full(800, 0.2, dtype=np.float32))
    phrase = segmenter.feed(np.zeros(1_600, dtype=np.float32))
    assert phrase is not None
    assert np.any(np.isclose(phrase, 0.004)), "o início abaixo do limiar precisa entrar na frase"
    assert phrase.size >= 800 + 1_600 + 1_600


def test_legacy_english_only_model_is_migrated(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"whisper_model": "distil-large-v3"}', encoding="utf-8")
    assert SettingsStore(path).load().whisper_model == "auto"


def test_auto_model_follows_the_hardware(monkeypatch) -> None:
    monkeypatch.setattr(whisper_module, "cuda_available", lambda: False)
    monkeypatch.setattr(whisper_module.os, "cpu_count", lambda: 12)
    assert resolve_model("auto").name == "base"
    assert resolve_model("auto").threads == 8
    assert resolve_model("balanced").name == "small"
    assert resolve_model("distil-large-v3").name == "distil-large-v3"
    monkeypatch.setattr(whisper_module, "cuda_available", lambda: True)
    plan = resolve_model("auto")
    assert (plan.name, plan.device, plan.compute_type) == ("large-v3-turbo", "cuda", "float16")


def test_latency_tracker_percentiles() -> None:
    tracker = LatencyTracker(capacity=5)
    for value in (900, 100, 500, 300, 700, 1_100):
        tracker.add(value)
    summary = tracker.summary()
    assert summary["count"] == 5
    assert summary["p50"] == 500
    assert summary["p95"] == 1_100


def test_failed_provider_goes_last_until_cooldown_ends() -> None:
    now = {"t": 0.0}
    health = ProviderHealth(cooldown_s=30, clock=lambda: now["t"])
    order = ["Argos local", "DeepL", "Google"]
    assert health.order(order) == order
    health.failure("Argos local")
    assert health.order(order) == ["DeepL", "Google", "Argos local"]
    now["t"] = 31.0
    assert health.order(order) == order
    health.failure("DeepL")
    health.success("DeepL", 120)
    assert health.order(order) == order
    assert health.latency_ms("DeepL") == 120
