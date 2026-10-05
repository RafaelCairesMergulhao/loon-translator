from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.whisper_engine import WhisperEngine
from domain.entities import AudioDirection, TranslationRecord
from domain.metrics import LatencyTracker
from services.audio_devices import AudioDevice
from services.audio_manager import LoopbackCapture, MicrophoneCapture, PhraseSegmenter
from services.translation_engine import LANGUAGES_DICT, TranslationEngine, TranslationResult
from services.tts_engine import AudioPlayer, TTSEngine, TTSError, split_for_speech
from services.voice_likeness import is_neural_voice, match_file

logger = logging.getLogger(__name__)

RecordCallback = Callable[[TranslationRecord], None]
StatusCallback = Callable[[AudioDirection | None, str], None]
ErrorCallback = Callable[[str], None]


class PipelineCancelled(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    microphone: AudioDevice
    virtual_output: AudioDevice
    monitor_output: AudioDevice
    loopback: AudioDevice
    outgoing_source: str
    outgoing_target: str
    incoming_source: str
    incoming_target: str
    voice_gender: str
    translation_mode: str
    whisper_model: str = "auto"
    silence_ms: int = 380
    max_phrase_seconds: float = 8.0
    monitor_outgoing: bool = False
    conversation_context: str = "casual"
    voice_pitch: float = 0.0
    speech_rate: float = 1.0
    voice_engine: str = "natural"


class DuplexTranslationPipeline:
    """Reconhecimento, tradução e voz em estágios concorrentes, por sentido da chamada.

    Cada sentido tem uma fila de frases, um processador e um tocador. Enquanto uma
    frase é falada, a próxima já está sendo transcrita; uma resposta do outro lado
    não espera a sua terminar; e uma tradução longa começa a tocar pela primeira
    frase enquanto as seguintes ainda são sintetizadas.
    """

    PHRASE_QUEUE_SIZE = 4
    CATCH_UP_STEP = 0.08
    CATCH_UP_LIMIT = 1.5
    ECHO_TAIL_S = 0.2

    def __init__(
        self,
        options: PipelineOptions,
        on_record: RecordCallback,
        on_status: StatusCallback,
        on_error: ErrorCallback,
    ) -> None:
        self.options = options
        self.on_record = on_record
        self.on_status = on_status
        self.on_error = on_error
        self._stt = WhisperEngine(options.whisper_model)
        self._translator = TranslationEngine()
        self._tts = TTSEngine()
        self._player = AudioPlayer()
        self._queues: dict[AudioDirection, queue.Queue[np.ndarray | None]] = {
            direction: queue.Queue(maxsize=self.PHRASE_QUEUE_SIZE) for direction in AudioDirection
        }
        self._speech: dict[AudioDirection, queue.Queue[Path | None]] = {
            direction: queue.Queue() for direction in AudioDirection
        }
        self._queue_lock = threading.Lock()
        self._stop = threading.Event()
        self._warm = threading.Event()
        self._incoming_suppressed = threading.Event()
        self._suppress_depth = 0
        self._suppress_lock = threading.Lock()
        self._capture: list[MicrophoneCapture | LoopbackCapture] = []
        self._threads: list[threading.Thread] = []
        self._speakers: dict[AudioDirection, threading.Thread] = {}
        self._last_text: dict[AudioDirection, str] = {}
        self.latency = LatencyTracker()

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads) and not self._stop.is_set()

    @property
    def terminated(self) -> bool:
        return not any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._warm.clear()
        AudioPlayer.resume()
        self._threads = []
        self._speakers = {}
        for direction in AudioDirection:
            self._threads.append(self._spawn(self._work, direction, f"loon-{direction.value}"))
            speaker = self._spawn(self._speak_loop, direction, f"loon-voice-{direction.value}")
            self._speakers[direction] = speaker
            self._threads.append(speaker)
        threading.Thread(target=self._warm_up, name="loon-warmup", daemon=True).start()
        self._capture = [
            MicrophoneCapture(
                self.options.microphone,
                lambda audio: self._enqueue(AudioDirection.OUTGOING, audio),
                lambda message: self._capture_failed(AudioDirection.OUTGOING, message),
                PhraseSegmenter(
                    silence_ms=self.options.silence_ms,
                    max_seconds=self.options.max_phrase_seconds,
                ),
            ),
            LoopbackCapture(
                self.options.loopback,
                lambda audio: self._enqueue(AudioDirection.INCOMING, audio),
                lambda message: self._capture_failed(AudioDirection.INCOMING, message),
                PhraseSegmenter(
                    silence_ms=self.options.silence_ms,
                    max_seconds=self.options.max_phrase_seconds,
                ),
                suppressed=self._incoming_suppressed,
            ),
        ]
        for capture in self._capture:
            capture.start()
        threading.Thread(target=self._confirm_captures, daemon=True).start()

    @staticmethod
    def _spawn(target: Callable[[AudioDirection], None], direction: AudioDirection, name: str) -> threading.Thread:
        thread = threading.Thread(target=target, args=(direction,), name=name, daemon=True)
        thread.start()
        return thread

    def stop(self, wait_timeout: float = 15.0) -> bool:
        if self._stop.is_set():
            self._join(wait_timeout)
            return self.terminated
        self._stop.set()
        AudioPlayer.cancel()
        for capture in self._capture:
            capture.stop()
        self._capture.clear()
        for direction in AudioDirection:
            self._put_sentinel(self._queues[direction])
            self._speech[direction].put(None)
        self._join(wait_timeout)
        stopped = self.terminated
        if stopped:
            self._translator.close()
        self.on_status(None, "listening_stopped" if stopped else "stop_pending")
        return stopped

    def _join(self, wait_timeout: float) -> None:
        deadline = time.monotonic() + wait_timeout
        for thread in self._threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _put_sentinel(self, phrases: queue.Queue[np.ndarray | None]) -> None:
        with self._queue_lock:
            try:
                phrases.put_nowait(None)
            except queue.Full:
                self._drop_oldest(phrases)
                phrases.put_nowait(None)

    def _enqueue(self, direction: AudioDirection, audio: np.ndarray) -> None:
        if self._stop.is_set():
            return
        phrases = self._queues[direction]
        dropped = False
        with self._queue_lock:
            try:
                phrases.put_nowait(audio)
            except queue.Full:
                self._drop_oldest(phrases)
                phrases.put_nowait(audio)
                dropped = True
        if dropped:
            self.on_status(direction, "queue_full")

    @staticmethod
    def _drop_oldest(phrases: queue.Queue) -> None:
        try:
            phrases.get_nowait()
            phrases.task_done()
        except queue.Empty:
            pass

    def _warm_up(self) -> None:
        started = time.perf_counter()
        self.on_status(None, "warming")
        try:
            self._stt.warm_up()
            self._tts.warm_up(
                self.options.voice_engine,
                self.options.outgoing_target,
                self.options.voice_gender,
            )
            self._translator.warm_up(
                [
                    (LANGUAGES_DICT.get(self.options.outgoing_source, "auto"), self.options.outgoing_target),
                    (LANGUAGES_DICT.get(self.options.incoming_source, "auto"), self.options.incoming_target),
                ],
                self.options.translation_mode,
            )
        except Exception:
            logger.warning("Aquecimento dos modelos incompleto", exc_info=True)
        finally:
            self._warm.set()
            logger.info("Modelos aquecidos em %.1fs", time.perf_counter() - started)

    def _work(self, direction: AudioDirection) -> None:
        phrases = self._queues[direction]
        while not self._stop.is_set():
            audio = phrases.get()
            try:
                if audio is None:
                    return
                self._process(direction, audio)
            except PipelineCancelled:
                logger.info("Operação cancelada durante a parada")
            except Exception as exc:
                logger.exception("Falha no pipeline")
                self.on_error(str(exc))
            finally:
                phrases.task_done()

    def _speak_loop(self, direction: AudioDirection) -> None:
        pending = self._speech[direction]
        while True:
            audio_file = pending.get()
            if audio_file is None or self._stop.is_set():
                return
            try:
                self._play(direction, audio_file)
            except Exception as exc:
                if not self._stop.is_set():
                    logger.exception("Falha na reprodução")
                    self.on_error(str(exc))

    def _process(self, direction: AudioDirection, audio: np.ndarray) -> None:
        self._ensure_active()
        started = time.perf_counter()
        source_name, target_name = self._language_pair(direction)
        source_code = LANGUAGES_DICT.get(source_name, "auto")
        whisper_language = None if source_code == "auto" else source_code.split("-")[0].lower()

        self.on_status(direction, "transcribing")
        transcript = self._stt.transcribe(audio, whisper_language)
        stt_ms = self._elapsed_ms(started)
        self._ensure_active()
        transcript_text = " ".join(transcript.text.split())
        if len(transcript_text) > 280:
            transcript_text = transcript_text[:280].rsplit(" ", 1)[0]
        if not transcript_text or transcript_text == self._last_text.get(direction):
            self.on_status(direction, "no_speech")
            return
        self._last_text[direction] = transcript_text

        detected_language = transcript.language or source_code
        self.on_status(direction, "translating")
        translation = self._translator.translate(
            transcript_text,
            detected_language,
            target_name,
            self.options.translation_mode,
            context=self.options.conversation_context,
        )
        translation_ms = self._elapsed_ms(started) - stt_ms
        self._ensure_active()

        self.on_status(direction, "synthesizing")
        voice_ms = 0
        recorded = False
        for sentence in split_for_speech(translation.text):
            audio_file = self._synthesize(direction, sentence, target_name, audio)
            if audio_file is None:
                break
            self._ensure_active()
            if not voice_ms:
                voice_ms = self._elapsed_ms(started)
                self.latency.add(voice_ms)
                self.on_status(direction, f"ready:{voice_ms}")
            self._speak(direction, audio_file)
            if not recorded:
                self._record(direction, transcript_text, detected_language, translation, target_name,
                             voice_ms, stt_ms, translation_ms)
                recorded = True
        if not recorded:
            self._record(direction, transcript_text, detected_language, translation, target_name,
                         self._elapsed_ms(started), stt_ms, translation_ms, speech=False)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((time.perf_counter() - started) * 1000)

    def _speech_speed(self, direction: AudioDirection) -> float:
        """Como um intérprete, fala um pouco mais rápido quando há frases esperando."""
        base = float(self.options.speech_rate)
        backlog = self._speech[direction].qsize() + self._queues[direction].qsize()
        if not backlog:
            return base
        faster = min(self.CATCH_UP_LIMIT, base + self.CATCH_UP_STEP * backlog)
        return round(max(base, faster) * 20) / 20

    def _synthesize(
        self,
        direction: AudioDirection,
        sentence: str,
        target_name: str,
        reference: np.ndarray,
    ) -> Path | None:
        try:
            audio_file = self._tts.synthesize(
                sentence,
                target_name,
                self.options.voice_gender,
                self.options.translation_mode,
                self._speech_speed(direction),
                pitch_semitones=self.options.voice_pitch,
                voice_engine=self.options.voice_engine,
            )
        except TTSError as exc:
            self.on_error(str(exc))
            return None
        return match_file(audio_file, reference, 16_000)

    def _speak(self, direction: AudioDirection, audio_file: Path) -> None:
        speaker = self._speakers.get(direction)
        if speaker is not None and speaker.is_alive() and not self._stop.is_set():
            self._speech[direction].put(audio_file)
        else:
            self._play(direction, audio_file)

    def _playback_pitch(self, audio_file: Path) -> float:
        if is_neural_voice(audio_file):
            return 0.0
        return self.options.voice_pitch

    def _play(self, direction: AudioDirection, audio_file: Path) -> None:
        if direction == AudioDirection.INCOMING:
            with self._suppressing():
                self._player.play(
                    audio_file,
                    self.options.monitor_output.uid,
                    pitch_semitones=self._playback_pitch(audio_file),
                )
        else:
            self._play_outgoing(audio_file)

    @contextmanager
    def _suppressing(self) -> Iterator[None]:
        """Pausa o retorno da chamada enquanto o fone toca, para não retraduzir o próprio áudio."""
        with self._suppress_lock:
            self._suppress_depth += 1
            self._incoming_suppressed.set()
        try:
            yield
        finally:
            tail = threading.Timer(self.ECHO_TAIL_S, self._release_suppression)
            tail.daemon = True
            tail.start()

    def _release_suppression(self) -> None:
        with self._suppress_lock:
            self._suppress_depth = max(0, self._suppress_depth - 1)
            if self._suppress_depth == 0:
                self._incoming_suppressed.clear()

    def _record(
        self,
        direction: AudioDirection,
        transcript_text: str,
        detected_language: str,
        translation: TranslationResult,
        target_name: str,
        latency_ms: int,
        stt_ms: int,
        translation_ms: int,
        speech: bool = True,
    ) -> None:
        target_code = LANGUAGES_DICT[target_name]
        self.on_record(
            TranslationRecord(
                source_text=transcript_text,
                translated_text=translation.text,
                source_lang=detected_language,
                target_lang=target_code,
                direction=direction,
                provider=translation.provider,
                latency_ms=latency_ms,
                portuguese_text=self._written_in(
                    "Português", transcript_text, detected_language, translation.text, target_code
                ),
                english_text=self._written_in(
                    "Inglês", transcript_text, detected_language, translation.text, target_code
                ),
                stt_ms=stt_ms,
                translation_ms=translation_ms,
                speech_ms=max(0, latency_ms - stt_ms - translation_ms) if speech else 0,
            )
        )

    def _written_in(
        self,
        language_name: str,
        source_text: str,
        source_code: str,
        translated_text: str,
        target_code: str,
    ) -> str:
        code = LANGUAGES_DICT[language_name]
        if source_code.split("-")[0].lower() == code:
            return source_text
        if target_code.split("-")[0].lower() == code:
            return translated_text
        # A versão do histórico não passa pelo LLM: não gasta tokens nem entra na memória da conversa.
        mode = "Híbrido" if self.options.translation_mode == "IA" else self.options.translation_mode
        try:
            return self._translator.translate(
                source_text,
                source_code,
                language_name,
                mode,
                context=self.options.conversation_context,
            ).text
        except Exception as exc:
            logger.info("Sem versão em %s para o histórico: %s", language_name, exc)
            return ""

    def _play_outgoing(self, audio_file: Path) -> None:
        errors: list[Exception] = []

        def play(device_uid: str) -> None:
            try:
                self._player.play(
                    audio_file,
                    device_uid,
                    pitch_semitones=self._playback_pitch(audio_file),
                )
            except Exception as exc:
                errors.append(exc)

        targets = [self.options.virtual_output.uid]
        monitoring = (
            self.options.monitor_outgoing
            and self.options.monitor_output.uid != self.options.virtual_output.uid
        )
        if monitoring:
            targets.append(self.options.monitor_output.uid)
        threads = [threading.Thread(target=play, args=(uid,), daemon=True) for uid in targets]
        with self._suppressing() if monitoring else nullcontext():
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if errors:
            raise errors[0]

    def _language_pair(self, direction: AudioDirection) -> tuple[str, str]:
        if direction == AudioDirection.OUTGOING:
            return self.options.outgoing_source, self.options.outgoing_target
        return self.options.incoming_source, self.options.incoming_target

    def _ensure_active(self) -> None:
        if self._stop.is_set():
            raise PipelineCancelled

    def _capture_failed(self, direction: AudioDirection, message: str) -> None:
        self.on_error(message)
        if direction == AudioDirection.INCOMING:
            self.on_status(direction, "incoming_down")
            return
        if not self._stop.is_set():
            threading.Thread(target=self.stop, daemon=True).start()

    def _confirm_captures(self) -> None:
        microphone_ok = False
        for capture in self._capture:
            ready = capture.ready.wait(timeout=4) and not capture.failed.is_set()
            direction = (
                AudioDirection.OUTGOING
                if isinstance(capture, MicrophoneCapture)
                else AudioDirection.INCOMING
            )
            if ready and direction == AudioDirection.OUTGOING:
                microphone_ok = True
            elif not ready and not self._stop.is_set():
                self._capture_failed(direction, f"Não foi possível iniciar {capture.device.name}")
        self._warm.wait(timeout=120)
        if microphone_ok and not self._stop.is_set():
            self.on_status(None, "listening")
