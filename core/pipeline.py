from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Coroutine, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from core.whisper_engine import Transcript
from domain.entities import AudioDirection, LiveCaption, TranslationRecord
from domain.metrics import LatencyTracker
from domain.transcripts import RollingTranscript, drop_repeated_prefix, repeats_previous, same_utterance, stable_prefix
from services.audio_devices import AudioDevice, AudioDeviceManager
from services.audio_manager import LoopbackCapture, MicrophoneCapture, SpeechSegment, SpeechSegmenter
from services.lexicon import USER_GLOSSARY, Lexicon
from services.translation_engine import LANGUAGES_DICT, TranslationEngine, TranslationResult, tidy_for_machine
from services.tts_engine import AudioPlayer, TTSEngine, TTSError, split_for_speech
from services.providers import SentenceGate, apply_offline, build_asr, translation_mode_for, voice_engine_for
from services.providers.translate import RoutedTranslator
from services.vad import create_vad
from services.voice_likeness import is_neural_voice, match_file
from services.voice_overlap import merge_clips, split_voices
from services.win_com import com_apartment

logger = logging.getLogger(__name__)

RecordCallback = Callable[[TranslationRecord], None]
StatusCallback = Callable[[AudioDirection | None, str], None]
ErrorCallback = Callable[[str], None]
CaptionCallback = Callable[[LiveCaption], None]


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
    silence_ms: int = 280
    max_phrase_seconds: float = 8.0
    monitor_outgoing: bool = False
    conversation_context: str = "casual"
    voice_pitch: float = 0.0
    speech_rate: float = 1.0
    voice_engine: str = "piper"
    live_captions: str = "auto"
    speculative: bool = True
    pipeline_profile: str = ""
    offline_asr: str = ""
    offline_mt: str = ""
    offline_tts: str = ""


@dataclass(eq=False)
class _SttJob:
    stream: _Stream
    kind: str
    segment: SpeechSegment
    prompt: str
    future: asyncio.Future
    started: bool = False
    claimed: bool = False
    follow_up: asyncio.Task | None = None


@dataclass(frozen=True, slots=True)
class _Recognized:
    segment: SpeechSegment
    transcript: Transcript
    text: str
    stt_ms: int
    speech_end: float
    speculation: _SttJob | None = None


@dataclass(eq=False)
class _Stream:
    """Estado de um sentido da chamada; só o laço de eventos mexe nele, então não há travas."""

    direction: AudioDirection
    segments: asyncio.Queue[SpeechSegment]
    texts: asyncio.Queue[_Recognized]
    speech: asyncio.Queue[Path]
    context: RollingTranscript = field(default_factory=RollingTranscript)
    utterance: int = 0
    version: int = 0
    speculation: _SttJob | None = None
    partial: _SttJob | None = None
    partial_utterance: int = 0
    partial_speech_ms: int = 0
    caption_utterance: int = 0
    caption_words: list[str] = field(default_factory=list)
    preview_request: tuple[int, LiveCaption, str] | None = None
    previewing: bool = False
    delivered: int = 0
    last_text: str = ""
    last_spoken: str = ""
    last_spoken_at: float = 0.0
    last_record: asyncio.Task | None = None
    chunk_samples: int = 0

    def track(self, segment: SpeechSegment) -> None:
        if segment.utterance != self.utterance:
            self.utterance, self.version = segment.utterance, segment.version
            self.chunk_samples = 0
        else:
            self.version = max(self.version, segment.version)

    def current(self, segment: SpeechSegment) -> bool:
        """Nada foi dito depois deste segmento."""
        return segment.utterance == self.utterance and segment.version == self.version


class DuplexTranslationPipeline:
    """Tradução simultânea nos dois sentidos, em estágios produtor–consumidor.

    * Estágio A (uma thread por dispositivo): áudio → 16 kHz → Silero VAD → segmentos.
    * Estágio B (asyncio): segmentos → Whisper, num executor de uma thread → texto.
    * Estágio C (asyncio): texto → tradução contextual → voz, em executores.
    * Estágio D (asyncio): voz → reprodução, em ordem, por sentido.

    As filas entre estágios são ``asyncio.Queue`` e o trabalho pesado vai para
    ``ThreadPoolExecutor`` (CTranslate2, ONNX e HTTP soltam o GIL). O Whisper
    começa quando a fala para por ~100 ms, antes de a pausa se confirmar: se a
    pessoa não voltar a falar, texto, tradução e voz já estão prontos quando a
    janela de silêncio termina. Uma frase só é falada depois de confirmada.
    """

    SEGMENT_QUEUE_SIZE = 8
    TEXT_QUEUE_SIZE = 4
    CATCH_UP_STEP = 0.08
    CATCH_UP_LIMIT = 1.5
    ECHO_TAIL_S = 0.2
    PARTIAL_MS = 480
    FAST_STT_MS = 350.0
    STT_PRIORITY = {"final": 0, "pause": 1, "partial": 2}
    # Uma frase da chamada que já esperou demais, com outra mais nova na fila, só atrapalha.
    # O que a pessoa mesma disse não entra nessa pressa: o resto da frase tem de sair no cabo.
    STALE_IF_WAITING_S = 4.0
    STALE_ALONE_S = 12.0
    OUTGOING_SILENCE_MS = 900
    # Menos de 0,6 s e quase sem voz: respiração, não uma palavra. A frase inteira espera o silêncio.
    GHOST_SAMPLES = 9_600
    BREATH_RMS = 0.02

    def __init__(
        self,
        options: PipelineOptions,
        on_record: RecordCallback,
        on_status: StatusCallback,
        on_error: ErrorCallback,
        on_partial: CaptionCallback | None = None,
    ) -> None:
        self.options = options
        self.on_record = on_record
        self.on_status = on_status
        self.on_error = on_error
        self.on_partial = on_partial
        self._stack, whisper_model = apply_offline(
            options.pipeline_profile,
            options.translation_mode,
            options.whisper_model,
            options.offline_asr,
            options.offline_mt,
            options.offline_tts,
        )
        self._gate = SentenceGate()
        self._stt = build_asr(self._stack.asr, whisper_model)
        self._translator = TranslationEngine()
        if self._stack.mt not in {"", "follow"}:
            self._translator = RoutedTranslator(self._translator, self._stack.mt)
        self._tts = TTSEngine()
        self._player = AudioPlayer()
        self._lexicon = Lexicon(USER_GLOSSARY)
        self._cheap_translation = self._translation_is_cheap()
        self._echo_guard = AudioDeviceManager.shares_endpoint(options.loopback.name, options.monitor_output.name)
        self._cable_guard = AudioDeviceManager.cable_feeds_loopback(
            options.loopback.name, options.virtual_output.name
        )
        self._stop = threading.Event()
        self._warm = threading.Event()
        self._loop_ready = threading.Event()
        self._incoming_suppressed = threading.Event()
        self._suppress_depth = 0
        self._suppress_lock = threading.Lock()
        self._capture: list[MicrophoneCapture | LoopbackCapture] = []
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing: asyncio.Event | None = None
        self._streams: dict[AudioDirection, _Stream] = {}
        self._tasks: set[asyncio.Task] = set()
        self._pools: dict[str, ThreadPoolExecutor] = {}
        self._stt_jobs: asyncio.PriorityQueue[tuple[int, int, _SttJob]] | None = None
        self._stt_sequence = 0
        self._stt_ms: float | None = None
        self._warm_stt_ms: float | None = None
        self.latency = LatencyTracker()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def terminated(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def start(self, listen: bool = True) -> None:
        """``listen=False`` liga os estágios B, C e D sem abrir dispositivos (benchmark)."""
        if self.running:
            return
        self._stop.clear()
        self._warm.clear()
        self._loop_ready.clear()
        AudioPlayer.resume()
        self._thread = threading.Thread(target=self._run_loop, name="loon-pipeline", daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=5)
        threading.Thread(target=self._warm_up, name="loon-warmup", daemon=True).start()
        if not listen:
            return
        self._capture = [
            MicrophoneCapture(
                self.options.microphone,
                lambda segment: self.submit(AudioDirection.OUTGOING, segment),
                lambda message: self._capture_failed(AudioDirection.OUTGOING, message),
                self._segmenter(AudioDirection.OUTGOING),
            ),
            LoopbackCapture(
                self.options.loopback,
                lambda segment: self.submit(AudioDirection.INCOMING, segment),
                lambda message: self._capture_failed(AudioDirection.INCOMING, message),
                self._segmenter(AudioDirection.INCOMING),
                suppressed=self._incoming_suppressed,
            ),
        ]
        for capture in self._capture:
            capture.start()
        threading.Thread(target=self._confirm_captures, daemon=True).start()

    def submit(self, direction: AudioDirection, segment: SpeechSegment) -> None:
        """Entrada do Estágio B; segura para chamar das threads de captura."""
        loop = self._loop
        if loop is None or self._stop.is_set():
            return
        try:
            loop.call_soon_threadsafe(self._offer, direction, segment)
        except RuntimeError:
            pass

    def stop(self, wait_timeout: float = 15.0) -> bool:
        if self._stop.is_set():
            self._join(wait_timeout)
            return self.terminated
        self._stop.set()
        AudioPlayer.cancel()
        for capture in self._capture:
            capture.stop()
        self._capture.clear()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._close_session)
            except RuntimeError:
                pass
        self._join(wait_timeout)
        stopped = self.terminated
        if stopped:
            self._translator.close()
        self.on_status(None, "listening_stopped" if stopped else "stop_pending")
        return stopped

    async def replay(self, segments: Iterable[tuple[AudioDirection, SpeechSegment]]) -> None:
        """Passa segmentos prontos pelos estágios B, C e D e espera tudo terminar."""
        await self._session(list(segments))

    def _segmenter(self, direction: AudioDirection) -> SpeechSegmenter:
        silence = self.options.silence_ms
        partial_ms = 0 if self.options.live_captions == "off" else self.PARTIAL_MS
        if direction == AudioDirection.OUTGOING:
            # Uma respiração no meio da frase não pode fechá-la. O retorno da
            # chamada continua curto; o microfone espera a frase terminar.
            silence = max(silence, self.OUTGOING_SILENCE_MS)
        return SpeechSegmenter(
            silence_ms=silence,
            max_seconds=self.options.max_phrase_seconds,
            partial_ms=partial_ms,
        )

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._session())
        except Exception as exc:
            logger.exception("Falha no pipeline")
            self.on_error(str(exc))
        finally:
            self._loop_ready.set()
            for pool in self._pools.values():
                pool.shutdown(wait=True, cancel_futures=True)

    def _close_session(self) -> None:
        if self._closing is not None:
            self._closing.set()

    def _join(self, wait_timeout: float) -> None:
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, wait_timeout))

    async def _session(self, feed: list[tuple[AudioDirection, SpeechSegment]] | None = None) -> None:
        self._loop = asyncio.get_running_loop()
        self._closing = asyncio.Event()
        self._pools = {
            "stt": ThreadPoolExecutor(1, thread_name_prefix="loon-stt"),
            "work": ThreadPoolExecutor(4, thread_name_prefix="loon-work"),
            "play": ThreadPoolExecutor(len(AudioDirection), thread_name_prefix="loon-play"),
        }
        self._stt_jobs = asyncio.PriorityQueue()
        self._streams = {
            direction: _Stream(
                direction,
                asyncio.Queue(self.SEGMENT_QUEUE_SIZE),
                asyncio.Queue(self.TEXT_QUEUE_SIZE),
                asyncio.Queue(),
            )
            for direction in AudioDirection
        }
        stages = [asyncio.create_task(self._stt_worker(), name="loon-stt")]
        for stream in self._streams.values():
            stages += [
                asyncio.create_task(self._recognize_stage(stream), name=f"loon-stt-{stream.direction}"),
                asyncio.create_task(self._translate_stage(stream), name=f"loon-mt-{stream.direction}"),
                asyncio.create_task(self._playback_stage(stream), name=f"loon-voice-{stream.direction}"),
            ]
        self._loop_ready.set()
        try:
            if feed is None:
                await self._closing.wait()
            else:
                for direction, segment in feed:
                    await self._streams[direction].segments.put(segment)
                await self._drain()
        finally:
            pending = [*stages, *self._tasks]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for pool in self._pools.values():
                pool.shutdown(wait=False, cancel_futures=True)
            self._loop = None

    async def _drain(self) -> None:
        for stream in self._streams.values():
            await stream.segments.join()
        for stream in self._streams.values():
            await stream.texts.join()
        # Tarefa pronta só sai de _tasks no próximo giro do laço, e um gather de
        # tarefas prontas termina sem ceder a vez: esperar só as pendentes.
        while pending := [task for task in self._tasks if not task.done()]:
            await asyncio.wait(pending)
        for stream in self._streams.values():
            await stream.speech.join()

    def _spawn(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled() or task.exception() is None:
            return
        logger.error("Falha numa tarefa do pipeline", exc_info=task.exception())
        if not self._stop.is_set():
            self.on_error(str(task.exception()))

    def _run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> asyncio.Future:
        return asyncio.get_running_loop().run_in_executor(
            self._pools["work"], functools.partial(function, *args, **kwargs)
        )

    def _offer(self, direction: AudioDirection, segment: SpeechSegment) -> None:
        stream = self._streams.get(direction)
        if stream is None or self._stop.is_set():
            return
        if stream.segments.full():
            dropped = stream.segments.get_nowait()
            stream.segments.task_done()
            if dropped.kind == "final":
                self.on_status(direction, "queue_full")
        stream.segments.put_nowait(segment)

    # Estágio B: segmentos → texto

    async def _recognize_stage(self, stream: _Stream) -> None:
        while True:
            segment = await stream.segments.get()
            try:
                await self._consume_segment(stream, segment)
            except Exception as exc:
                logger.exception("Falha no reconhecimento")
                if not self._stop.is_set():
                    self.on_error(str(exc))
            finally:
                stream.segments.task_done()

    async def _consume_segment(self, stream: _Stream, segment: SpeechSegment) -> None:
        stream.track(segment)
        if stream.direction == AudioDirection.OUTGOING and segment.kind in {"partial", "final"}:
            await self._enqueue_outgoing_chunk(stream, segment, final=segment.kind == "final")
            return
        if stream.direction == AudioDirection.INCOMING and segment.kind in {"partial", "final"}:
            if await self._enqueue_incoming_chunk(stream, segment, final=segment.kind == "final"):
                return
        if segment.kind == "final" and stream.direction == AudioDirection.INCOMING and segment.audio.size:
            await self._consume_incoming(stream, segment)
            return
        if segment.kind == "final":
            await self._recognize_final(stream, segment)
        elif segment.kind == "pause":
            self._speculate(stream, segment)
        elif segment.kind == "resume":
            self._abandon_speculation(stream)
        elif segment.kind == "partial":
            self._request_partial(stream, segment)

    async def _consume_incoming(self, stream: _Stream, segment: SpeechSegment, hold: bool = True) -> None:
        """A chamada chega numa faixa só. Duas vozes viram duas falas, em ordem.

        Com três ou mais ao mesmo tempo, espera a espera curta desta rajada para
        juntar o que for da mesma pessoa. O microfone de quem usa o app não passa aqui.
        """
        split = await self._run(split_voices, segment.audio)
        if split.voice_count <= 0:
            self._release_speculation(stream)
            self.on_status(stream.direction, "no_speech")
            return
        if split.voice_count == 1:
            await self._recognize_final(stream, segment)
            return
        groups = [split]
        deferred: list[SpeechSegment] = []
        if split.hold_s > 0 and hold:
            extra, deferred = await self._collect_crowd(stream, split.hold_s)
            groups.extend(extra)
        self._release_speculation(stream)
        for index, clip in enumerate(merge_clips(groups)):
            child = replace(
                segment,
                utterance=segment.utterance * 10 + index + 1,
                version=segment.version + index + 1,
                audio=clip.audio,
                reason="overlap",
                continues=index > 0,
                created_at=time.monotonic(),
                speech_ms=max(1, int(clip.audio.size / 16)),
            )
            stream.track(child)
            await self._recognize_final(stream, child)
        for item in deferred:
            await self._consume_segment(stream, item)

    async def _collect_crowd(
        self, stream: _Stream, hold_s: float
    ) -> tuple[list, list[SpeechSegment]]:
        groups = []
        deferred: list[SpeechSegment] = []
        deadline = time.monotonic() + hold_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(stream.segments.get(), remaining)
            except TimeoutError:
                break
            try:
                if nxt.kind != "final" or nxt.audio.size == 0:
                    deferred.append(nxt)
                    break
                extra = await self._run(split_voices, nxt.audio)
                if extra.voice_count >= 2:
                    groups.append(extra)
                    if extra.hold_s > 0:
                        deadline = time.monotonic() + extra.hold_s
                else:
                    deferred.append(nxt)
                    break
            finally:
                stream.segments.task_done()
        return groups, deferred

    async def _enqueue_outgoing_chunk(self, stream: _Stream, segment: SpeechSegment, final: bool) -> None:
        """O microfone só é reconhecido quando o silêncio fecha a frase.

        Cortar a cada 1,8 s parte a palavra e o pedaço seguinte se perde
        ("Hoje o dia está bonito", o resto de "acompanhamento do curso").
        """
        if not self._gate.speak("final" if final else "partial"):
            if self._gate.caption("partial"):
                self._request_partial(stream, segment)
            return
        stream.chunk_samples = 0
        await self._recognize_final(stream, segment)

    async def _enqueue_incoming_chunk(self, stream: _Stream, segment: SpeechSegment, final: bool) -> bool:
        """A chamada e a nota do WhatsApp esperam o silêncio, como o microfone.

        Um corte a cada 2,5 s entrega um pedaço sem sentido e o Whisper completa
        com outra frase. Devolve False para o caminho da frase inteira.
        """
        del stream, segment, final
        return False

    def _release_speculation(self, stream: _Stream) -> None:
        job = stream.speculation
        if job is None:
            return
        job.claimed = True
        if not job.started:
            job.future.cancel()
        stream.speculation = None

    def _submit_stt(self, stream: _Stream, kind: str, segment: SpeechSegment) -> _SttJob:
        job = _SttJob(
            stream,
            kind,
            segment,
            stream.context.last_phrases(time.monotonic()),
            asyncio.get_running_loop().create_future(),
        )
        self._stt_sequence += 1
        self._stt_jobs.put_nowait((self.STT_PRIORITY[kind], self._stt_sequence, job))
        return job

    async def _stt_worker(self) -> None:
        """Um Whisper por vez: quem fecha frase passa na frente de quem só adianta ou legenda."""
        while True:
            _priority, _sequence, job = await self._stt_jobs.get()
            if job.future.done():
                continue
            job.started = True
            started = time.perf_counter()
            try:
                transcript = await asyncio.get_running_loop().run_in_executor(
                    self._pools["stt"], self._transcribe, job.stream.direction, job.segment.audio, job.prompt
                )
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
                continue
            elapsed = (time.perf_counter() - started) * 1000
            self._stt_ms = elapsed if self._stt_ms is None else self._stt_ms + 0.3 * (elapsed - self._stt_ms)
            if not job.future.done():
                job.future.set_result((transcript, round(elapsed)))

    def _transcribe(self, direction: AudioDirection, audio: np.ndarray, prompt: str) -> Transcript:
        del prompt  # a fala anterior e a lista de gírias faziam o modelo escrever palavras que ninguém disse
        source_code = self._source_code(direction)
        language = None if source_code == "auto" else source_code.split("-")[0].lower()
        return self._stt.transcribe(audio, language, prompt=None, hotwords=None)

    def _speculate(self, stream: _Stream, segment: SpeechSegment) -> None:
        if not self.options.speculative:
            return
        previous = stream.speculation
        if previous is not None and not previous.started:
            previous.future.cancel()
        job = self._submit_stt(stream, "pause", segment)
        stream.speculation = job
        job.follow_up = self._spawn(self._after_speculation(stream, job))

    def _abandon_speculation(self, stream: _Stream) -> None:
        """A fala voltou: o Whisper que nem começou fica livre para quem precisa."""
        job = stream.speculation
        if job is not None and not job.started:
            job.future.cancel()
            stream.speculation = None

    async def _after_speculation(self, stream: _Stream, job: _SttJob) -> None:
        try:
            transcript, _stt_ms = await job.future
        except (asyncio.CancelledError, Exception):
            return
        if job.claimed:
            return
        text = self._clean(stream, transcript, job.segment)
        if not text:
            return
        self._caption(stream, job.segment, text, transcript.language)
        if stream.current(job.segment) and self._cheap_translation:
            await self._prepare_voice(stream, job.segment, text, transcript)

    async def _prepare_voice(
        self, stream: _Stream, segment: SpeechSegment, text: str, transcript: Transcript
    ) -> None:
        """Adianta tradução e voz da primeira sentença; os caches devolvem tudo na confirmação."""
        source_code = self._source_code(stream.direction)
        target_name = self._language_pair(stream.direction)[1]
        try:
            translation = await self._run(
                self._translator.translate,
                text,
                transcript.language or source_code,
                target_name,
                self._translation_mode(),
                context=self.options.conversation_context,
                speculative=True,
            )
            if not stream.current(segment):
                return
            sentences = split_for_speech(self._lexicon.fix_translation(translation.text, LANGUAGES_DICT[target_name]))
            if sentences:
                await self._run(
                    self._synthesize,
                    stream.direction,
                    sentences[0],
                    target_name,
                    segment.audio,
                    self._speech_speed(stream),
                    quiet=True,
                )
        except Exception as exc:
            logger.debug("Preparação antecipada descartada: %s", exc)

    def _request_partial(self, stream: _Stream, segment: SpeechSegment) -> None:
        if not self._partials_allowed():
            return
        fresh = stream.partial_utterance != segment.utterance
        if not fresh and segment.speech_ms - stream.partial_speech_ms < self._partial_interval_ms():
            return
        pending = stream.partial
        if pending is not None and not pending.future.done():
            if pending.started:
                return
            pending.future.cancel()
        stream.partial_utterance = segment.utterance
        stream.partial_speech_ms = segment.speech_ms
        job = self._submit_stt(stream, "partial", segment)
        stream.partial = job
        job.follow_up = self._spawn(self._after_partial(stream, job))

    def _partials_allowed(self) -> bool:
        """A legenda ao vivo só roda quando o reconhecimento é rápido.

        No processador uma passada extra segura o modelo. A frase final
        esperaria essa legenda acabar para só então começar a voz.
        """
        if self.on_partial is None or self.options.live_captions == "off":
            return False
        stt = self._stt_ms or self._warm_stt_ms
        return stt is None or stt <= self.FAST_STT_MS

    def _partial_interval_ms(self) -> float:
        stt = self._stt_ms or self._warm_stt_ms or self.FAST_STT_MS
        floor = float(self.PARTIAL_MS) if stt <= self.FAST_STT_MS else 1_200.0
        return max(floor, 2.5 * stt)

    async def _after_partial(self, stream: _Stream, job: _SttJob) -> None:
        try:
            transcript, _stt_ms = await job.future
        except (asyncio.CancelledError, Exception):
            return
        text = self._clean(stream, transcript, job.segment)
        if text:
            self._caption(stream, job.segment, text, transcript.language)

    async def _recognize_final(self, stream: _Stream, segment: SpeechSegment) -> None:
        if self._is_ghost(segment.audio):
            self._release_speculation(stream)
            self.on_status(stream.direction, "no_speech")
            return
        job, stream.speculation = stream.speculation, None
        partial = stream.partial
        if partial is not None and not partial.started:
            partial.future.cancel()
        speech_end = (segment.created_at or time.monotonic()) - segment.trailing_ms / 1000
        result = None
        reusable = (
            job is not None
            and segment.reason == "pause"
            and job.segment.utterance == segment.utterance
            and job.segment.version == segment.version
            and not job.future.cancelled()
        )
        if reusable:
            job.claimed = True
            if not job.future.done():
                self.on_status(stream.direction, "transcribing")
            try:
                result = await job.future
            except Exception:
                result = None
        elif job is not None and not job.started:
            job.future.cancel()
        if result is None:
            job = None
            self.on_status(stream.direction, "transcribing")
            fresh = self._submit_stt(stream, "final", segment)
            fresh.claimed = True
            result = await fresh.future
        transcript, stt_ms = result
        # O contexto avança aqui, e não na tradução, para a frase seguinte já ir ao Whisper com ele.
        text = self._clean(stream, transcript, segment)
        if text == stream.last_text:
            text = ""
        elif text:
            stream.last_text = text
            stream.context.add(text, time.monotonic())
        await stream.texts.put(_Recognized(segment, transcript, text, stt_ms, speech_end, job))

    # Estágio C: texto → tradução → voz

    async def _translate_stage(self, stream: _Stream) -> None:
        while True:
            item = await stream.texts.get()
            try:
                await self._deliver(stream, item)
            except Exception as exc:
                logger.exception("Falha no pipeline")
                if not self._stop.is_set():
                    self.on_error(str(exc))
            finally:
                stream.texts.task_done()

    async def _deliver(self, stream: _Stream, item: _Recognized) -> None:
        segment, transcript, text = item.segment, item.transcript, item.text
        stream.delivered = max(stream.delivered, segment.utterance)
        follow_up = item.speculation.follow_up if item.speculation is not None else None
        if follow_up is not None and not follow_up.done():
            await asyncio.wait([follow_up])
        if not text or self._too_late(stream, item):
            self._clear_caption(stream)
            self.on_status(stream.direction, "no_speech")
            return

        direction = stream.direction
        target_name = self._language_pair(direction)[1]
        detected_language = transcript.language or self._source_code(direction)
        self.on_status(direction, "translating")
        started = time.perf_counter()
        translation = await self._run(
            self._translator.translate,
            text,
            detected_language,
            target_name,
            self._translation_mode(),
            context=self.options.conversation_context,
            speculative=False,
            fragment=segment.reason == "chunk" or segment.continues,
        )
        translation = TranslationResult(
            self._lexicon.fix_translation(translation.text, LANGUAGES_DICT[target_name]),
            translation.provider,
        )
        if not translation.text.strip():
            self._clear_caption(stream)
            self.on_status(stream.direction, "no_speech")
            return
        details = {
            "transcript_text": text,
            "detected_language": detected_language,
            "translation": translation,
            "target_name": target_name,
            "stt_ms": item.stt_ms,
            "translation_ms": _elapsed_ms(started),
        }

        stream.last_spoken = translation.text
        stream.last_spoken_at = time.monotonic()
        self.on_status(direction, "synthesizing")
        voice_started = time.perf_counter()
        spoken = False
        for sentence in split_for_speech(translation.text):
            audio_file = await self._run(
                self._synthesize, direction, sentence, target_name, segment.audio, self._speech_speed(stream)
            )
            if audio_file is None:
                break
            if not spoken:
                spoken = True
                latency_ms = round((time.monotonic() - item.speech_end) * 1000)
                self.latency.add(latency_ms)
                self.on_status(direction, f"ready:{latency_ms}")
                self._emit_record(stream, latency_ms=latency_ms, speech_ms=_elapsed_ms(voice_started), **details)
            await stream.speech.put(audio_file)
        if not spoken:
            latency_ms = round((time.monotonic() - item.speech_end) * 1000)
            self._emit_record(stream, latency_ms=latency_ms, speech_ms=0, **details)

    @staticmethod
    def _is_ghost(audio: np.ndarray) -> bool:
        """Menos de 0,6 s e quase sem voz: respiração ou ruído, não uma palavra."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0 or samples.size >= DuplexTranslationPipeline.GHOST_SAMPLES:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms < DuplexTranslationPipeline.BREATH_RMS

    def _clean(self, stream: _Stream, transcript: Transcript, segment: SpeechSegment) -> str:
        text = tidy_for_machine(" ".join(transcript.text.split()))
        if len(text) > 600:
            text = text[:600].rsplit(" ", 1)[0]
        text = self._lexicon.fix_transcript(text, transcript.language or self._source_code(stream.direction))
        if segment.continues and stream.last_text:
            text = drop_repeated_prefix(stream.last_text, text)
        if text and (
            self._hears_itself(stream, text)
            or repeats_previous(stream.last_text, text)
            or stream.context.already_said(text, time.monotonic())
        ):
            return ""
        return text

    @classmethod
    def _too_late(cls, stream: _Stream, item: _Recognized) -> bool:
        age = time.monotonic() - item.speech_end
        if age > cls.STALE_ALONE_S:
            return True
        segment = getattr(item, "segment", None)
        if stream.direction == AudioDirection.OUTGOING or getattr(segment, "reason", "") == "chunk":
            return False
        return age > cls.STALE_IF_WAITING_S and stream.texts.qsize() > 0

    def _hears_itself(self, stream: _Stream, text: str) -> bool:
        """A frase que acabamos de falar, ouvida de volta pelo cabo ou pelo microfone."""
        other_direction = (
            AudioDirection.INCOMING
            if stream.direction == AudioDirection.OUTGOING
            else AudioDirection.OUTGOING
        )
        other = self._streams.get(other_direction)
        if other is None or not other.last_spoken:
            return False
        if time.monotonic() - other.last_spoken_at > 6:
            return False
        return same_utterance(text, other.last_spoken)

    def _synthesize(
        self,
        direction: AudioDirection,
        sentence: str,
        target_name: str,
        reference: np.ndarray,
        speed: float,
        quiet: bool = False,
    ) -> Path | None:
        try:
            audio_file = self._tts.synthesize(
                sentence,
                target_name,
                self.options.voice_gender,
                self._translation_mode(),
                speed,
                pitch_semitones=self.options.voice_pitch,
                voice_engine=self._voice_engine(),
            )
        except TTSError as exc:
            if not quiet:
                self.on_error(str(exc))
            return None
        return match_file(audio_file, reference, 16_000)

    def _speech_speed(self, stream: _Stream) -> float:
        """Como um intérprete, fala um pouco mais rápido quando há frases esperando."""
        base = float(self.options.speech_rate)
        backlog = stream.speech.qsize() + stream.texts.qsize()
        if not backlog:
            return base
        faster = min(self.CATCH_UP_LIMIT, base + self.CATCH_UP_STEP * backlog)
        return round(max(base, faster) * 20) / 20

    def _translation_mode(self) -> str:
        return translation_mode_for(self._stack, self.options.translation_mode)

    def _voice_engine(self) -> str:
        return voice_engine_for(self._stack, self.options.voice_engine)

    def _translation_is_cheap(self) -> bool:
        """Adiantar a tradução às vezes joga uma chamada fora; só vale onde ela não custa cota."""
        mode = self._translation_mode()
        if mode == "Nuvem":
            return not os.getenv("DEEPL_API_KEY", "").strip()
        # No modo IA a pausa curta já manda a frase ao modelo. Quando o silêncio
        # fecha, a tradução confirmada sai do cache.
        return True

    # Legendas ao vivo

    def _caption(self, stream: _Stream, segment: SpeechSegment, text: str, language: str) -> None:
        if self.on_partial is None or segment.utterance <= stream.delivered:
            return
        words = text.split()
        if stream.caption_utterance != segment.utterance:
            stream.caption_utterance, stream.caption_words = segment.utterance, []
        stable = stable_prefix(stream.caption_words, words)
        stream.caption_words = words
        caption = LiveCaption(stream.direction, " ".join(words[:stable]), " ".join(words[stable:]))
        self.on_partial(caption)
        stream.preview_request = (segment.utterance, caption, language)
        if not stream.previewing:
            stream.previewing = True
            self._spawn(self._preview_loop(stream))

    async def _preview_loop(self, stream: _Stream) -> None:
        target_name = self._language_pair(stream.direction)[1]
        try:
            while stream.preview_request is not None:
                utterance, caption, language = stream.preview_request
                stream.preview_request = None
                translated = await self._run(
                    self._translator.preview,
                    caption.heard,
                    language or self._source_code(stream.direction),
                    target_name,
                )
                fresh = (
                    stream.preview_request is None
                    and utterance > stream.delivered
                    and " ".join(stream.caption_words) == caption.heard
                )
                if translated and fresh:
                    translated = self._lexicon.fix_translation(translated, LANGUAGES_DICT[target_name])
                    self.on_partial(replace(caption, translation=translated))
        finally:
            stream.previewing = False

    def _clear_caption(self, stream: _Stream) -> None:
        if self.on_partial is not None:
            self.on_partial(LiveCaption(stream.direction, "", ""))

    # Histórico

    def _emit_record(self, stream: _Stream, **details: Any) -> None:
        """As versões em português e inglês do histórico saem fora do caminho da voz."""
        previous = stream.last_record

        async def publish() -> None:
            record = await self._run(self._build_record, stream.direction, **details)
            if previous is not None:
                await asyncio.wait([previous])
            self.on_record(record)

        stream.last_record = self._spawn(publish())

    def _build_record(
        self,
        direction: AudioDirection,
        transcript_text: str,
        detected_language: str,
        translation: TranslationResult,
        target_name: str,
        latency_ms: int,
        stt_ms: int,
        translation_ms: int,
        speech_ms: int,
    ) -> TranslationRecord:
        target_code = LANGUAGES_DICT[target_name]
        return TranslationRecord(
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
            speech_ms=speech_ms,
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
        mode = "Híbrido" if self._translation_mode() == "IA" else self._translation_mode()
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

    # Estágio D: voz → fone ou microfone virtual

    async def _playback_stage(self, stream: _Stream) -> None:
        while True:
            audio_file = await stream.speech.get()
            try:
                if not self._stop.is_set():
                    await asyncio.get_running_loop().run_in_executor(
                        self._pools["play"], self._play, stream.direction, audio_file
                    )
            except Exception as exc:
                if not self._stop.is_set():
                    logger.exception("Falha na reprodução")
                    self.on_error(str(exc))
            finally:
                stream.speech.task_done()

    def _playback_pitch(self, audio_file: Path) -> float:
        if is_neural_voice(audio_file):
            return 0.0
        return self.options.voice_pitch

    def _play(self, direction: AudioDirection, audio_file: Path) -> None:
        with com_apartment():
            if direction == AudioDirection.INCOMING:
                with self._suppressing() if self._echo_guard else nullcontext():
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

        monitoring = (
            self.options.monitor_outgoing
            and self.options.monitor_output.uid != self.options.virtual_output.uid
        )
        # Os dois ao mesmo tempo fazem o cabo virtual ficar só com o começo.
        # O cabo termina primeiro; o fone ouve a mesma frase em seguida.
        guard_loopback = self._cable_guard or (monitoring and self._echo_guard)
        with self._suppressing() if guard_loopback else nullcontext():
            play(self.options.virtual_output.uid)
            if monitoring:
                play(self.options.monitor_output.uid)
        if errors:
            raise errors[0]

    # Ciclo de vida

    def _warm_up(self) -> None:
        started = time.perf_counter()
        self.on_status(None, "warming")
        try:
            create_vad()
            self._warm_stt_ms = self._stt.warm_up() * 1000
            for target in dict.fromkeys((self.options.outgoing_target, self.options.incoming_target)):
                self._tts.warm_up(
                    self._voice_engine(),
                    target,
                    self.options.voice_gender,
                    self.options.speech_rate,
                )
            self._translator.warm_up(
                [
                    (LANGUAGES_DICT.get(self.options.outgoing_source, "auto"), self.options.outgoing_target),
                    (LANGUAGES_DICT.get(self.options.incoming_source, "auto"), self.options.incoming_target),
                ],
                self._translation_mode(),
            )
        except Exception:
            logger.warning("Aquecimento dos modelos incompleto", exc_info=True)
        finally:
            self._warm.set()
            logger.info("Modelos aquecidos em %.1fs", time.perf_counter() - started)

    def _language_pair(self, direction: AudioDirection) -> tuple[str, str]:
        if direction == AudioDirection.OUTGOING:
            return self.options.outgoing_source, self.options.outgoing_target
        return self.options.incoming_source, self.options.incoming_target

    def _source_code(self, direction: AudioDirection) -> str:
        return LANGUAGES_DICT.get(self._language_pair(direction)[0], "auto")

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


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
