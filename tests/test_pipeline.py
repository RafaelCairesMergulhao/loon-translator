import asyncio
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import core.pipeline as pipeline_module
from core.pipeline import DuplexTranslationPipeline, PipelineOptions, _Stream
from core.whisper_engine import Transcript
from domain.entities import AudioDirection
from services.audio_devices import AudioDevice
from services.audio_manager import SpeechSegment
from services.lexicon import Lexicon
from services.translation_engine import TranslationResult
from services.tts_engine import TTSEngine, TTSError

OUT = AudioDirection.OUTGOING
IN = AudioDirection.INCOMING


def _device(uid: str, kind: str) -> AudioDevice:
    return AudioDevice(uid=uid, name=uid, kind=kind, channels=2, sample_rate=48_000)


def _segment(kind: str, utterance: int = 1, version: int = 10, size: int = 16_000, **fields) -> SpeechSegment:
    return SpeechSegment(kind, utterance, version, np.full(size, 0.1, dtype=np.float32), **fields)


def _final(utterance: int = 1, version: int = 10, size: int = 16_000, **fields) -> SpeechSegment:
    fields.setdefault("reason", "pause")
    return _segment("final", utterance, version, size, **fields)


async def _until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("a condição não aconteceu a tempo")
        await asyncio.sleep(0.005)


class FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text, *_args, **_kwargs):
        self.spoken.append(text)
        return Path(f"{len(self.spoken)}.wav")

    def warm_up(self, *_args) -> None:
        pass


class FakePlayer:
    def __init__(self):
        self.calls = []

    def play(self, path, uid, pitch_semitones: float = 0.0):
        self.calls.append((path, uid, pitch_semitones))

    @staticmethod
    def cancel() -> None:
        pass

    @staticmethod
    def resume() -> None:
        pass


class FakeSTT:
    """Devolve os textos em ordem; um texto pode ser função do áudio recebido."""

    def __init__(self, *texts, language: str = "pt"):
        self.texts = texts or ("olá",)
        self.language = language
        self.calls: list[dict] = []

    def transcribe(self, audio, language, **kwargs):
        self.calls.append({"size": audio.size, "language": language, **kwargs})
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        return Transcript(text(audio) if callable(text) else text, self.language, 0.99)

    def warm_up(self) -> float:
        return 0.0


class FakeTranslator:
    def __init__(self, reply: str = "hello"):
        self.reply = reply
        self.calls: list[tuple] = []

    def translate(self, text, *args, **kwargs):
        self.calls.append((text, args, kwargs))
        return TranslationResult(self.reply, "teste")

    def preview(self, text, *_args):
        return f"[{text}]"

    def warm_up(self, *_args) -> None:
        pass

    def close(self) -> None:
        pass


class Harness:
    def __init__(self, monkeypatch, captions: bool = False, **overrides):
        monkeypatch.setattr(pipeline_module, "TTSEngine", FakeTTS)
        monkeypatch.setattr(pipeline_module, "AudioPlayer", FakePlayer)
        self.records: list = []
        self.errors: list[str] = []
        self.statuses: list[tuple] = []
        self.captions: list = []
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
        self.pipeline = DuplexTranslationPipeline(
            replace(options, **overrides),
            on_record=self.records.append,
            on_status=lambda direction, status: self.statuses.append((direction, status)),
            on_error=self.errors.append,
            on_partial=self.captions.append if captions else None,
        )
        self.pipeline._stt = FakeSTT()
        self.pipeline._translator = FakeTranslator()
        self.pipeline._lexicon = Lexicon()

    @property
    def stt(self) -> FakeSTT:
        return self.pipeline._stt

    @property
    def translator(self) -> FakeTranslator:
        return self.pipeline._translator

    @property
    def tts(self) -> FakeTTS:
        return self.pipeline._tts

    @property
    def player(self) -> FakePlayer:
        return self.pipeline._player

    def replay(self, *segments: SpeechSegment, direction: AudioDirection = OUT) -> None:
        feed = [(direction, segment) for segment in segments]
        asyncio.run(asyncio.wait_for(self.pipeline.replay(feed), 10))

    def offer(self, segment: SpeechSegment, direction: AudioDirection = OUT) -> None:
        self.pipeline._offer(direction, segment)

    def drive(self, script, timeout: float = 10.0) -> None:
        """Roda ``script`` dentro de uma sessão viva, para cenários que dependem do tempo."""

        async def main() -> None:
            session = asyncio.create_task(self.pipeline._session())
            await _until(self.pipeline._loop_ready.is_set)
            try:
                await script()
                await self.pipeline._drain()
            finally:
                self.pipeline._close_session()
                await session

        asyncio.run(asyncio.wait_for(main(), timeout))


def test_missing_voice_still_writes_the_translation(monkeypatch) -> None:
    class SilentTTS:
        def synthesize(self, *_args, **_kwargs):
            raise TTSError("missing_voice:Espanhol")

    h = Harness(monkeypatch)
    h.pipeline._tts = SilentTTS()
    h.replay(_final())
    assert h.records[0].translated_text == "hello"
    assert h.records[0].source_text == "olá"
    assert h.errors == ["missing_voice:Espanhol"]
    assert h.player.calls == []


def test_spoken_language_check_ignores_installed_voices(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.tts_engine.installed_voice_prefixes",
        lambda: {"pt", "en"},
    )
    assert TTSEngine.missing_spoken_languages(["Português", "Espanhol", "Inglês"]) == ["Espanhol"]


def test_outgoing_phrase_is_recorded_and_played_on_the_virtual_cable(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.replay(_final())
    record = h.records[0]
    assert record.translated_text == "hello"
    assert record.direction == OUT
    assert record.latency_ms >= 0
    assert h.player.calls[0][1] == "sd:2"
    assert h.errors == []


def test_outgoing_playback_mutes_the_loopback_on_the_same_cable(monkeypatch) -> None:
    cable = "CABLE Input (VB-Audio Virtual Cable)"
    h = Harness(
        monkeypatch,
        virtual_output=AudioDevice(uid="sd:16", name=cable, kind="virtual", channels=2, sample_rate=48_000),
        loopback=AudioDevice(uid="wasapi:16", name=cable, kind="loopback", channels=2, sample_rate=48_000),
    )
    seen: list[bool] = []

    class WatchingPlayer(FakePlayer):
        def play(self, path, uid, pitch_semitones: float = 0.0):
            seen.append(h.pipeline._incoming_suppressed.is_set())
            super().play(path, uid, pitch_semitones)

    h.pipeline._player = WatchingPlayer()
    h.replay(_final())
    assert seen and all(seen)


def test_played_translation_is_not_translated_again(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline._stt = FakeSTT("olá", "Hello!")

    async def scenario() -> None:
        h.offer(_final(1), OUT)
        await _until(lambda: len(h.records) >= 1)
        h.offer(_final(1), IN)
        await _until(lambda: any(direction == IN and status == "no_speech" for direction, status in h.statuses))

    h.drive(scenario)
    assert [record.direction for record in h.records] == [OUT]


def _harmonic(pitch: float, samples: int) -> np.ndarray:
    time_axis = np.arange(samples) / 16_000
    wave = sum((0.55**harmonic) * np.sin(2 * np.pi * (harmonic + 1) * pitch * time_axis) for harmonic in range(8))
    return (wave / np.max(np.abs(wave)) * 0.4).astype(np.float32)


def _mix(*pieces: np.ndarray) -> SpeechSegment:
    audio = np.sum(pieces, axis=0).astype(np.float32)
    return SpeechSegment("final", 1, 10, audio, reason="pause", created_at=time.monotonic())


def test_incoming_phrase_plays_on_the_headphones(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.replay(_final(), direction=IN)
    assert h.records[0].direction == IN
    assert h.player.calls[0][1] == "sd:3"


def test_bounded_segment_queue_drops_the_oldest(monkeypatch) -> None:
    h = Harness(monkeypatch)
    size = DuplexTranslationPipeline.SEGMENT_QUEUE_SIZE

    async def scenario() -> tuple[int, int]:
        h.pipeline._streams = {
            direction: _Stream(direction, asyncio.Queue(size), asyncio.Queue(), asyncio.Queue())
            for direction in AudioDirection
        }
        for utterance in range(1, size + 5):
            h.offer(_final(utterance))
        queue = h.pipeline._streams[OUT].segments
        return queue.qsize(), queue.get_nowait().utterance

    queued, oldest = asyncio.run(scenario())
    assert queued == size
    assert oldest == 5
    assert h.statuses.count((OUT, "queue_full")) == 4
    assert h.pipeline._streams[IN].segments.qsize() == 0


def test_empty_interpretation_is_not_spoken(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.translator.reply = ""
    h.replay(_final())
    assert h.player.calls == []
    assert h.records == []
    assert (OUT, "no_speech") in h.statuses


def test_a_loop_tail_and_a_very_late_line_are_not_spoken(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.stt.texts = (
        "se o bom se vai jogar nela agora",
        "se o bom se vai jogar",
        "vamos ver o que aconteceu lá",
    )
    h.replay(
        _final(1),
        _final(2),
        _final(3, created_at=time.monotonic() - 30),
    )
    assert [call[0] for call in h.translator.calls] == ["se o bom se vai jogar nela agora"]
    assert len(h.records) == 1


def test_repeated_phrase_is_not_translated_again(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.replay(_final(1), _final(2))
    assert len(h.records) == 1
    assert len(h.translator.calls) == 1
    assert (OUT, "no_speech") in h.statuses


def test_incoming_overlap_is_spoken_one_voice_at_a_time(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.stt.texts = ("voz clara", "segunda pessoa", "terceira pessoa")
    heard: list[np.ndarray] = []
    original = h.stt.transcribe

    def capture(audio, language, **kwargs):
        heard.append(np.asarray(audio).copy())
        return original(audio, language, **kwargs)

    h.stt.transcribe = capture
    single = _mix(_harmonic(150, 9_600))
    h.replay(single, direction=IN)
    assert [record.source_text for record in h.records] == ["voz clara"]

    h.records.clear()
    h.stt.calls.clear()
    heard.clear()
    length = 12_800
    second = np.zeros(length, dtype=np.float32)
    second[4_000:] = _harmonic(230, length - 4_000)
    started = time.perf_counter()
    h.replay(_mix(_harmonic(130, length), second), direction=IN)
    from services.voice_overlap import CROWD_HOLD_S

    assert time.perf_counter() - started < CROWD_HOLD_S
    assert [record.source_text for record in h.records] == ["voz clara", "segunda pessoa"]
    from services.voice_overlap import split_voices

    pitches = [round(split_voices(audio).clips[0].pitch) for audio in heard]
    assert abs(pitches[0] - 130) <= 8
    assert abs(pitches[1] - 230) <= 8


def test_three_incoming_voices_use_the_crowd_hold(monkeypatch) -> None:
    monkeypatch.setattr("services.voice_overlap.CROWD_HOLD_S", 0.2)
    h = Harness(monkeypatch)
    h.stt.texts = ("primeira", "segunda", "terceira")
    length = 11_200
    started = time.perf_counter()
    h.replay(
        _mix(_harmonic(110, length), _harmonic(170, length), _harmonic(250, length)),
        direction=IN,
    )
    assert time.perf_counter() - started >= 0.2
    assert [record.source_text for record in h.records] == ["primeira", "segunda", "terceira"]


def test_incoming_noise_is_not_translated(monkeypatch) -> None:
    h = Harness(monkeypatch)
    noise = (np.random.default_rng(1).standard_normal(8_000) * 0.2).astype(np.float32)
    h.replay(SpeechSegment("final", 1, 10, noise, reason="pause", created_at=time.monotonic()), direction=IN)
    assert h.translator.calls == []
    assert h.records == []
    assert (IN, "no_speech") in h.statuses


def test_outgoing_mix_is_not_split(monkeypatch) -> None:
    h = Harness(monkeypatch)
    length = 11_200
    mix = _harmonic(110, length) + _harmonic(220, length)
    h.replay(SpeechSegment("final", 1, 10, mix, reason="pause", created_at=time.monotonic()))
    assert len(h.translator.calls) == 1
    assert h.stt.calls[0]["size"] == mix.size


def test_incoming_failure_keeps_the_outgoing_side(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline._capture_failed(IN, "retorno caiu")
    assert h.errors == ["retorno caiu"]
    assert h.pipeline._stop.is_set() is False


def test_pipeline_forwards_context_and_pitch(monkeypatch) -> None:
    h = Harness(monkeypatch, conversation_context="formal", voice_pitch=3.5)
    h.replay(_final())
    _text, _args, kwargs = h.translator.calls[0]
    assert kwargs["context"] == "formal"
    assert kwargs["speculative"] is False
    assert h.player.calls[0][2] == 3.5


def test_long_translation_is_spoken_sentence_by_sentence(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.translator.reply = (
        "I will be a little late for the meeting today. Please start without me and I will join soon."
    )
    h.replay(_final())
    assert len(h.tts.spoken) == 2
    assert [call[0] for call in h.player.calls] == [Path("1.wav"), Path("2.wav")]
    assert len(h.records) == 1
    record = h.records[0]
    assert record.stt_ms >= 0 and record.translation_ms >= 0 and record.speech_ms >= 0


def test_playback_runs_beside_the_next_translation(monkeypatch) -> None:
    release = threading.Event()

    class HeldPlayer(FakePlayer):
        def play(self, path, uid, pitch_semitones: float = 0.0):
            assert release.wait(5), "a segunda frase esperou a primeira terminar de tocar"
            super().play(path, uid, pitch_semitones)

    class SignallingTTS(FakeTTS):
        def synthesize(self, text, *args, **kwargs):
            path = super().synthesize(text, *args, **kwargs)
            if len(self.spoken) == 2:
                release.set()
            return path

    h = Harness(monkeypatch)
    h.pipeline._stt = FakeSTT("primeira frase", "segunda frase")
    h.pipeline._player = HeldPlayer()
    h.pipeline._tts = SignallingTTS()
    h.replay(_final(1), _final(2))
    assert [call[0] for call in h.player.calls] == [Path("1.wav"), Path("2.wav")]
    assert h.errors == []


def test_records_keep_the_spoken_order(monkeypatch) -> None:
    class SlowHistory(FakeTranslator):
        def translate(self, text, source, target, *args, **kwargs):
            if text == "uno" and target in {"Português", "Inglês"}:
                time.sleep(0.3)
            return super().translate(text, source, target, *args, **kwargs)

    h = Harness(monkeypatch, outgoing_source="Espanhol", outgoing_target="Francês")
    h.pipeline._stt = FakeSTT("uno", "dos", "tres", language="es")
    h.pipeline._translator = SlowHistory()
    h.replay(_final(1), _final(2), _final(3))
    assert [record.source_text for record in h.records] == ["uno", "dos", "tres"]


def test_voice_speeds_up_when_phrases_are_waiting(monkeypatch) -> None:
    h = Harness(monkeypatch, speech_rate=1.1)
    stream = _Stream(OUT, asyncio.Queue(), asyncio.Queue(), asyncio.Queue())
    assert h.pipeline._speech_speed(stream) == 1.1
    for index in range(3):
        stream.speech.put_nowait(Path(f"{index}.wav"))
    assert h.pipeline._speech_speed(stream) == 1.35
    for index in range(10):
        stream.speech.put_nowait(Path(f"extra{index}.wav"))
    assert h.pipeline._speech_speed(stream) == 1.5


def test_incoming_suppression_survives_overlapping_playback(monkeypatch) -> None:
    class ImmediateTimer:
        def __init__(self, _delay, function):
            self.function = function
            self.daemon = True

        def start(self):
            self.function()

    monkeypatch.setattr(pipeline_module.threading, "Timer", ImmediateTimer)
    h = Harness(monkeypatch)
    with h.pipeline._suppressing():
        with h.pipeline._suppressing():
            assert h.pipeline._incoming_suppressed.is_set()
        assert h.pipeline._incoming_suppressed.is_set()
    assert not h.pipeline._incoming_suppressed.is_set()


@pytest.mark.parametrize(("loopback", "guarded"), [("wasapi:4", False), ("sd:3", True)])
def test_echo_guard_only_mutes_a_loopback_that_hears_the_headphones(monkeypatch, loopback, guarded) -> None:
    h = Harness(monkeypatch, loopback=_device(loopback, "loopback"))
    seen: list[bool] = []

    class WatchingPlayer(FakePlayer):
        def play(self, path, uid, pitch_semitones: float = 0.0):
            seen.append(h.pipeline._incoming_suppressed.is_set())

    h.pipeline._player = WatchingPlayer()
    h.replay(_final(), direction=IN)
    assert seen == [guarded]


def test_pause_transcript_is_reused_when_the_silence_holds(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.replay(_segment("pause", version=10, size=12_000), _final(version=10, size=16_000))
    assert [call["size"] for call in h.stt.calls] == [12_000]
    assert len(h.records) == 1


def test_speech_after_the_pause_needs_a_fresh_transcript(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline._stt = FakeSTT(lambda audio: f"frase de {audio.size} amostras")
    h.replay(
        _segment("pause", version=10, size=12_000),
        _segment("resume", version=11, size=0),
        _final(version=20, size=24_000),
    )
    assert [call["size"] for call in h.stt.calls] == [24_000]
    assert h.records[0].source_text == "frase de 24000 amostras"


def test_voice_is_prepared_during_the_pause_but_spoken_only_after_it(monkeypatch) -> None:
    h = Harness(monkeypatch)

    async def script() -> None:
        h.offer(_segment("pause", version=10))
        await _until(lambda: h.tts.spoken)
        assert h.records == [] and h.player.calls == []
        h.offer(_final(version=10))
        await _until(lambda: h.records)

    h.drive(script)
    assert len(h.stt.calls) == 1
    assert [call[2]["speculative"] for call in h.translator.calls] == [True, False]
    assert len(h.player.calls) == 1


def test_live_caption_keeps_the_agreed_words_and_shows_a_preview(monkeypatch) -> None:
    h = Harness(monkeypatch, captions=True, live_captions="on")
    h.pipeline._stt = FakeSTT("eu vou", "eu vou para casa")

    async def script() -> None:
        h.offer(_segment("partial", version=10, speech_ms=600), direction=IN)
        await _until(lambda: any(caption.translation for caption in h.captions))
        h.offer(_segment("partial", version=30, speech_ms=1_200), direction=IN)
        await _until(lambda: any(caption.stable == "eu vou" for caption in h.captions))

    h.drive(script)
    first = h.captions[0]
    assert (first.stable, first.pending) == ("", "eu vou")
    assert h.captions[1].translation == "[eu vou]"
    agreed = next(caption for caption in h.captions if caption.stable == "eu vou")
    assert agreed.pending == "para casa"
    assert h.records == []


def test_a_slow_whisper_does_not_caption_ahead_of_the_voice(monkeypatch) -> None:
    h = Harness(monkeypatch, captions=True)
    h.pipeline._warm_stt_ms = 600.0
    h.replay(_segment("partial", version=10, speech_ms=600), direction=IN)
    assert h.stt.calls == []
    assert h.captions == []
    assert h.records == []
    assert h.player.calls == []


def test_rolling_context_and_glossary_reach_whisper(monkeypatch, tmp_path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"terms": ["Loon Translator"]}), encoding="utf-8")
    h = Harness(monkeypatch)
    h.pipeline._lexicon = Lexicon(glossary)
    h.pipeline._stt = FakeSTT("vamos falar do deploy", "ele saiu ontem")
    h.replay(_final(1), _final(2))
    first, second = h.stt.calls
    assert first["prompt"] is None
    assert second["prompt"] is None
    assert second["hotwords"] is None
    assert second["language"] == "pt"


def test_glossary_fixes_what_was_heard_and_what_was_translated(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline._stt = FakeSTT("please open a pull request on git hub", language="en")
    h.translator.reply = "por favor, abra uma solicitação de pull no git hub"
    h.replay(_final(), direction=IN)
    record = h.records[0]
    assert h.translator.calls[0][0] == "please open a pull request on GitHub"
    assert record.source_text == "please open a pull request on GitHub"
    assert record.translated_text == "por favor, abra uma pull request no GitHub"


def test_the_microphone_waits_out_a_breath_inside_the_sentence(monkeypatch) -> None:
    h = Harness(monkeypatch, silence_ms=280)
    outgoing = h.pipeline._segmenter(OUT)
    incoming = h.pipeline._segmenter(IN)
    assert outgoing.silence_frames > incoming.silence_frames
    assert outgoing.silence_frames * 32 >= 900
    assert outgoing.partial_frames == incoming.partial_frames
    assert outgoing.partial_frames * 32 < 1_000


def test_the_rest_of_your_own_sentence_is_not_dropped(monkeypatch) -> None:
    h = Harness(monkeypatch)
    late = time.monotonic() - 6
    own = _Stream(OUT, asyncio.Queue(), asyncio.Queue(), asyncio.Queue())
    own.texts.put_nowait(None)
    heard = _Stream(IN, asyncio.Queue(), asyncio.Queue(), asyncio.Queue())
    heard.texts.put_nowait(None)
    item = type("Late", (), {"speech_end": late})()
    assert h.pipeline._too_late(own, item) is False
    assert h.pipeline._too_late(heard, item) is True


def test_the_cable_finishes_before_the_headphones_play(monkeypatch) -> None:
    h = Harness(monkeypatch, monitor_outgoing=True)
    h.replay(_final())
    assert [call[1] for call in h.player.calls] == ["sd:2", "sd:3"]


def test_a_short_breath_is_not_recognized(monkeypatch) -> None:
    h = Harness(monkeypatch)
    breath = SpeechSegment("final", 1, 10, np.full(4_000, 0.001, dtype=np.float32), reason="pause")
    h.replay(breath)
    assert h.stt.calls == []
    assert h.records == []


def test_a_sentence_is_recognized_once_when_silence_closes_it(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline._stt = FakeSTT("a frase inteira")
    h.replay(_segment("partial", version=30, size=28_800), _final(version=45, size=36_000))
    assert [call["size"] for call in h.stt.calls] == [36_000]
    assert [call[0] for call in h.translator.calls] == ["a frase inteira"]
    assert len(h.player.calls) == 1
    assert h.tts.spoken == ["hello"]


def test_a_long_incoming_phrase_waits_until_the_speaker_finishes(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline._stt = FakeSTT("a nota inteira")
    opening = SpeechSegment("partial", 1, 30, _harmonic(130, 40_000), created_at=time.monotonic())
    closing = SpeechSegment("final", 1, 45, _harmonic(130, 56_000), reason="pause", created_at=time.monotonic())
    h.replay(opening, closing, direction=IN)
    assert [call["size"] for call in h.stt.calls] == [56_000]
    assert h.stt.calls[0]["prompt"] is None
    spoken = [call for call in h.translator.calls if call[1][1] == "Português"]
    assert [call[0] for call in spoken] == ["a nota inteira"]
    assert spoken[0][2]["fragment"] is False
    assert spoken[0][2]["context"] == "casual"


def test_start_without_devices_and_stop_cleanly(monkeypatch) -> None:
    h = Harness(monkeypatch)
    h.pipeline.start(listen=False)
    try:
        assert h.pipeline.running
        h.pipeline.submit(OUT, _final())
        deadline = time.monotonic() + 5
        while not h.records and time.monotonic() < deadline:
            time.sleep(0.01)
        assert h.records and h.records[0].translated_text == "hello"
    finally:
        assert h.pipeline.stop(wait_timeout=5)
    assert h.pipeline.terminated
    assert (None, "listening_stopped") in h.statuses
