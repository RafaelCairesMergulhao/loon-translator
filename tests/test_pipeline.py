from dataclasses import replace
from pathlib import Path

import numpy as np

import core.pipeline as pipeline_module
from core.pipeline import DuplexTranslationPipeline, PipelineOptions
from core.whisper_engine import Transcript
from domain.entities import AudioDirection
from services.audio_devices import AudioDevice
from services.translation_engine import TranslationResult
from services.tts_engine import TTSEngine, TTSError


def _device(uid: str, kind: str) -> AudioDevice:
    return AudioDevice(uid=uid, name=uid, kind=kind, channels=2, sample_rate=48_000)


class FakeTTS:
    def synthesize(self, *_args, **_kwargs):
        return Path("fake.wav")


class FakePlayer:
    def __init__(self):
        self.calls = []

    def play(self, path, uid, pitch_semitones: float = 0.0):
        self.calls.append((path, uid, pitch_semitones))


class FakeSTT:
    def transcribe(self, _audio, _language):
        return Transcript("olá", "pt", 0.99)


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return TranslationResult("hello", "teste")


def _pipeline(monkeypatch, records, **option_overrides):
    monkeypatch.setattr(pipeline_module, "TTSEngine", FakeTTS)
    options = PipelineOptions(
        microphone=_device("sd:1", "microfone"),
        virtual_output=_device("sd:2", "virtual"),
        monitor_output=_device("sd:3", "saída"),
        loopback=_device("wasapi:4", "loopback"),
        outgoing_source="Português",
        outgoing_target="Inglês",
        incoming_source="Inglês",
        incoming_target="Português",
        voice_gender="Feminina",
        translation_mode="Híbrido",
    )
    if option_overrides:
        options = replace(options, **option_overrides)
    pipeline = DuplexTranslationPipeline(
        options,
        on_record=records.append,
        on_status=lambda *_args: None,
        on_error=lambda error: (_ for _ in ()).throw(AssertionError(error)),
    )
    pipeline._stt = FakeSTT()
    pipeline._translator = FakeTranslator()
    pipeline._player = FakePlayer()
    return pipeline


def test_missing_voice_still_writes_the_translation(monkeypatch) -> None:
    class SilentTTS:
        def synthesize(self, *_args, **_kwargs):
            raise TTSError("missing_voice:Espanhol")

    errors: list[str] = []
    records: list = []
    pipeline = _pipeline(monkeypatch, records)
    pipeline.on_error = errors.append
    pipeline._tts = SilentTTS()
    pipeline._process(AudioDirection.OUTGOING, np.ones(16_000, dtype=np.float32))
    assert records[0].translated_text == "hello"
    assert records[0].source_text == "olá"
    assert errors == ["missing_voice:Espanhol"]
    assert pipeline._player.calls == []


def test_spoken_language_check_ignores_installed_voices(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.tts_engine.installed_voice_prefixes",
        lambda: {"pt", "en"},
    )
    assert TTSEngine.missing_spoken_languages(["Português", "Espanhol", "Inglês"]) == ["Espanhol"]


def test_outgoing_pipeline_builds_record_and_uses_virtual_output(monkeypatch) -> None:
    records = []
    pipeline = _pipeline(monkeypatch, records)
    pipeline._process(AudioDirection.OUTGOING, np.ones(16_000, dtype=np.float32))
    assert records[0].translated_text == "hello"
    assert records[0].direction == AudioDirection.OUTGOING
    assert pipeline._player.calls[0][1] == "sd:2"


def test_bounded_queue_discards_oldest_without_growing(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch, [])
    for index in range(10):
        pipeline._enqueue(
            AudioDirection.OUTGOING,
            np.array([index], dtype=np.float32),
        )
    assert pipeline._queues[AudioDirection.OUTGOING].qsize() == 4
    assert pipeline._queues[AudioDirection.INCOMING].qsize() == 0


def test_repeated_phrase_is_not_translated_again(monkeypatch) -> None:
    records = []
    pipeline = _pipeline(monkeypatch, records)
    audio = np.ones(16_000, dtype=np.float32)
    pipeline._process(AudioDirection.OUTGOING, audio)
    pipeline._process(AudioDirection.OUTGOING, audio)
    assert len(records) == 1


def test_incoming_failure_keeps_the_outgoing_side(monkeypatch) -> None:
    errors = []
    pipeline = _pipeline(monkeypatch, [])
    pipeline.on_error = errors.append
    pipeline._capture_failed(AudioDirection.INCOMING, "retorno caiu")
    assert errors == ["retorno caiu"]
    assert pipeline._stop.is_set() is False


def test_pipeline_forwards_context_and_pitch(monkeypatch) -> None:
    pipeline = _pipeline(
        monkeypatch,
        [],
        conversation_context="formal",
        voice_pitch=3.5,
    )
    pipeline._process(AudioDirection.OUTGOING, np.ones(16_000, dtype=np.float32))
    _args, kwargs = pipeline._translator.calls[0]
    assert kwargs["context"] == "formal"
    assert pipeline._player.calls[0][2] == 3.5


def test_long_translation_is_spoken_sentence_by_sentence(monkeypatch) -> None:
    class LongTranslator(FakeTranslator):
        def translate(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return TranslationResult(
                "I will be a little late for the meeting today. Please start without me and I will join soon.",
                "teste",
            )

    spoken: list[str] = []

    class RecordingTTS:
        def synthesize(self, text, *_args, **_kwargs):
            spoken.append(text)
            return Path(f"{len(spoken)}.wav")

    records: list = []
    pipeline = _pipeline(monkeypatch, records)
    pipeline._translator = LongTranslator()
    pipeline._tts = RecordingTTS()
    pipeline._process(AudioDirection.OUTGOING, np.ones(16_000, dtype=np.float32))
    assert len(spoken) == 2
    assert [call[0] for call in pipeline._player.calls] == [Path("1.wav"), Path("2.wav")]
    assert len(records) == 1
    assert records[0].stt_ms >= 0 and records[0].translation_ms >= 0


def test_running_speaker_receives_audio_without_blocking(monkeypatch) -> None:
    import threading

    pipeline = _pipeline(monkeypatch, [])
    alive = threading.Event()
    speaker = threading.Thread(target=alive.wait, daemon=True)
    speaker.start()
    pipeline._speakers[AudioDirection.OUTGOING] = speaker
    try:
        pipeline._process(AudioDirection.OUTGOING, np.ones(16_000, dtype=np.float32))
        assert pipeline._player.calls == []
        assert pipeline._speech[AudioDirection.OUTGOING].get_nowait() == Path("fake.wav")
    finally:
        alive.set()


def test_voice_speeds_up_when_phrases_are_waiting(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch, [], speech_rate=1.1)
    assert pipeline._speech_speed(AudioDirection.OUTGOING) == 1.1
    for index in range(3):
        pipeline._speech[AudioDirection.OUTGOING].put(Path(f"{index}.wav"))
    assert pipeline._speech_speed(AudioDirection.OUTGOING) == 1.35
    for index in range(10):
        pipeline._speech[AudioDirection.OUTGOING].put(Path(f"extra{index}.wav"))
    assert pipeline._speech_speed(AudioDirection.OUTGOING) == 1.5


def test_incoming_suppression_survives_overlapping_playback(monkeypatch) -> None:
    class ImmediateTimer:
        def __init__(self, _delay, function):
            self.function = function
            self.daemon = True

        def start(self):
            self.function()

    monkeypatch.setattr(pipeline_module.threading, "Timer", ImmediateTimer)
    pipeline = _pipeline(monkeypatch, [])
    with pipeline._suppressing():
        with pipeline._suppressing():
            assert pipeline._incoming_suppressed.is_set()
        assert pipeline._incoming_suppressed.is_set()
    assert not pipeline._incoming_suppressed.is_set()
