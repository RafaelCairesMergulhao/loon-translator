import numpy as np

from core.config import SettingsStore
from core.whisper_engine import resolve_model
from domain.metrics import LatencyTracker, ProviderHealth
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


def test_whisper_receives_the_rolling_prompt_and_hotwords(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace

    seen: dict = {}

    class FakeModel:
        def transcribe(self, _audio, **kwargs):
            seen.update(kwargs)
            info = SimpleNamespace(language="pt", language_probability=0.9)
            return iter([SimpleNamespace(text=" Abre o deploy.")]), info

    monkeypatch.setattr(whisper_module, "_shared_model", lambda _plan: (FakeModel(), threading.Lock()))
    engine = whisper_module.WhisperEngine("base")
    audio = np.zeros(16_000, dtype=np.float32)
    transcript = engine.transcribe(audio, "pt", prompt="falávamos do servidor", hotwords="Kubernetes")
    assert transcript.text == "Abre o deploy."
    assert seen["initial_prompt"] is None
    assert seen["hotwords"] is None
    assert seen["beam_size"] == 1
    assert seen["temperature"] == 0.0
    assert seen["vad_parameters"]["min_silence_duration_ms"] == 500
    assert seen["vad_parameters"]["speech_pad_ms"] == 200
    engine.transcribe(audio, "pt")
    assert seen["initial_prompt"] is None
    assert seen["hotwords"] is None


def test_rejected_neural_voice_is_skipped_until_the_cooldown(monkeypatch, tmp_path) -> None:
    import services.tts_engine as tts_module
    from services.tts_engine import TTSEngine, TTSError

    attempts: list[str] = []

    class Rejected(Exception):
        status = 403

    class RejectingCommunicate:
        def __init__(self, text, *_args, **_kwargs):
            attempts.append(text)

        async def save(self, _path):
            raise Rejected("Invalid response status")

    def no_piper(self, *_args, **_kwargs):
        raise TTSError("sem Piper")

    monkeypatch.setattr(tts_module.edge_tts, "Communicate", RejectingCommunicate)
    monkeypatch.setattr(TTSEngine, "prune_cache", staticmethod(lambda: None))
    monkeypatch.setattr(TTSEngine, "_piper", no_piper)
    monkeypatch.setattr(
        TTSEngine, "_windows_sapi", classmethod(lambda cls, text, *_args, **_kwargs: tmp_path / "sapi.wav")
    )
    engine = TTSEngine(cache_dir=tmp_path)
    assert engine.synthesize("primeira frase", "Inglês", "Feminina", "Híbrido") == tmp_path / "sapi.wav"
    assert engine.synthesize("segunda frase", "Inglês", "Feminina", "Híbrido") == tmp_path / "sapi.wav"
    assert len(attempts) == 1, "um 403 não se repete, e a frase seguinte vai direto para a voz local"


def test_local_mode_never_sends_the_text_to_the_online_voice(monkeypatch, tmp_path) -> None:
    from services.tts_engine import TTSEngine, TTSError

    online: list[str] = []

    def installing_piper(self, *_args, **_kwargs):
        raise TTSError("A voz Piper de Inglês ainda está sendo instalada")

    def edge(self, text, *_args, **_kwargs):
        online.append(text)
        return tmp_path / "edge.mp3"

    monkeypatch.setattr(TTSEngine, "prune_cache", staticmethod(lambda: None))
    monkeypatch.setattr(TTSEngine, "_piper", installing_piper)
    monkeypatch.setattr(TTSEngine, "_edge", edge)
    monkeypatch.setattr(
        TTSEngine, "_windows_sapi", classmethod(lambda cls, text, *_args, **_kwargs: tmp_path / "sapi.wav")
    )
    engine = TTSEngine(cache_dir=tmp_path)
    assert engine.synthesize("bom dia", "Inglês", "Feminina", "Local") == tmp_path / "sapi.wav"
    assert online == [], "o modo Local com Piper não pediu consentimento de nuvem"
    assert engine.synthesize("boa tarde", "Inglês", "Feminina", "Híbrido") == tmp_path / "edge.mp3"


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
    health.failure("Google")
    health.success("Google", 80, started_at=-1.0)
    assert health.is_open("Google")
    health.success("Google", 80, started_at=now["t"])
    assert health.is_open("Google") is False


def test_grok_voice_writes_the_mp3_and_asks_for_ara(monkeypatch, tmp_path) -> None:
    from services.tts_engine import TTSEngine

    seen: dict[str, object] = {}

    class Response:
        content = b"ID3grok"

        def raise_for_status(self) -> None:
            return None

    def post(url, headers, json, timeout):
        seen["url"] = url
        seen["voice"] = json["voice_id"]
        seen["language"] = json["language"]
        seen["auth"] = headers["Authorization"]
        assert timeout == 8.0
        return Response()

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    monkeypatch.setattr("services.tts_engine.httpx.post", post)
    monkeypatch.setattr(TTSEngine, "prune_cache", staticmethod(lambda: None))
    path = TTSEngine(cache_dir=tmp_path).synthesize(
        "boa noite", "Português", "Feminina", "IA", voice_engine="grok"
    )
    assert path.read_bytes() == b"ID3grok"
    assert path.name.endswith(".grok.mp3")
    assert seen["url"] == "https://api.x.ai/v1/tts"
    assert seen["voice"] == "ara"
    assert seen["language"] == "pt-BR"
    assert seen["auth"] == "Bearer xai-test"
