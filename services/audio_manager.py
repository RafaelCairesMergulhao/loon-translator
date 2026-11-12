from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from services.audio_devices import AudioDevice, AudioDeviceManager, pyaudio
from services.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, VoiceDetector, create_vad
from services.win_com import com_apartment

logger = logging.getLogger(__name__)

_EMPTY = np.zeros(0, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Evento da captura para o pipeline.

    * ``partial``: a frase continua; serve para a legenda ao vivo.
    * ``pause``: a fala parou há ~100 ms; dá para transcrever antes de a pausa se confirmar.
    * ``resume``: a fala voltou; o que foi adiantado vale só como legenda.
    * ``final``: a pausa se confirmou, a frase bateu no limite ou a captura foi interrompida.

    ``version`` conta os quadros de fala da frase: se ``final`` chega com a mesma
    versão de ``pause``, nada foi dito entre os dois e o texto adiantado vale.
    """

    kind: str
    utterance: int
    version: int
    audio: np.ndarray = field(default_factory=lambda: _EMPTY)
    reason: str = ""
    trailing_ms: int = 0
    speech_ms: int = 0
    continues: bool = False
    created_at: float = 0.0


def _frames(milliseconds: float) -> int:
    return max(1, math.ceil(milliseconds / FRAME_MS))


class FrameRing:
    """Buffer circular de quadros com memória fixa: o pré-roll antes de o VAD disparar."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._audio = np.zeros((self.capacity, FRAME_SAMPLES), dtype=np.float32)
        self._probs = np.zeros(self.capacity, dtype=np.float32)
        self._next = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def push(self, frame: np.ndarray, probability: float) -> None:
        self._audio[self._next] = frame
        self._probs[self._next] = probability
        self._next = (self._next + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def drain(self) -> tuple[np.ndarray, np.ndarray]:
        order = (np.arange(self._count) + self._next - self._count) % self.capacity
        audio = self._audio[order].reshape(-1)
        probs = self._probs[order]
        self._count = 0
        return audio, probs

    def clear(self) -> None:
        self._count = 0


class SpeechSegmenter:
    """Corta a fala em frases com VAD, pré-roll e "flush por pausa".

    Trabalha em quadros de 32 ms, o tamanho que o Silero aceita a 16 kHz.
    O pré-roll devolve à frase os ~320 ms anteriores ao disparo, onde ficam
    as consoantes surdas (p, t, k, s, f). Uma frase longa demais aceita uma
    pausa mais curta depois de 60% do limite e, no limite, é cortada no
    quadro de menor probabilidade de voz; a frase seguinte recomeça ~100 ms
    antes do corte para não engolir a sílaba da emenda.
    """

    def __init__(
        self,
        vad: VoiceDetector | None = None,
        *,
        silence_ms: int = 380,
        max_seconds: float = 12.0,
        min_speech_ms: int = 180,
        pre_roll_ms: int = 320,
        pause_ms: int = 96,
        partial_ms: int = 0,
        overlap_ms: int = 96,
        tail_ms: int = 192,
        short_silence_ms: int = 200,
        valley_ms: int = 1_500,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.vad = vad
        self.silence_frames = _frames(silence_ms)
        self.short_silence_frames = min(self.silence_frames, _frames(short_silence_ms))
        self.pause_frames = min(_frames(pause_ms), self.silence_frames)
        self.max_frames = max(_frames(1_000), round(max_seconds * 1000 / FRAME_MS))
        self.soft_frames = self.max_frames * 3 // 5
        self.min_speech_frames = _frames(min_speech_ms)
        self.partial_frames = _frames(partial_ms) if partial_ms > 0 else 0
        self.overlap_frames = _frames(overlap_ms)
        self.tail_frames = _frames(tail_ms)
        self.valley_frames = _frames(valley_ms)
        self.threshold = threshold
        self.neg_threshold = min(neg_threshold, threshold)
        self._clock = clock
        self._pre_roll = FrameRing(_frames(pre_roll_ms))
        capacity = self.max_frames + self._pre_roll.capacity + 1
        self._audio = np.zeros(capacity * FRAME_SAMPLES, dtype=np.float32)
        self._probs = np.zeros(capacity, dtype=np.float32)
        self._pending = _EMPTY
        self._utterance = 0
        self._continues = False
        self._reset()

    @property
    def active(self) -> bool:
        return self._active

    def feed(self, samples: np.ndarray) -> list[SpeechSegment]:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._pending.size:
            data = np.concatenate((self._pending, data))
        usable = data.size - data.size % FRAME_SAMPLES
        self._pending = data[usable:].copy()
        if self.vad is None:
            self.vad = create_vad()
        events: list[SpeechSegment] = []
        for start in range(0, usable, FRAME_SAMPLES):
            frame = data[start : start + FRAME_SAMPLES]
            self._step(frame, self.vad.probability(frame), events)
        return events

    def flush(self) -> list[SpeechSegment]:
        events: list[SpeechSegment] = []
        if self._active:
            self._finish("flush", events)
        return events

    def discard(self) -> None:
        """Esquece a frase, o pré-roll e o estado do VAD: o áudio seguinte não emenda no anterior."""
        self._reset()
        self._pre_roll.clear()
        self._pending = _EMPTY
        if self.vad is not None:
            self.vad.reset()

    def _reset(self) -> None:
        self._frames = 0
        self._active = False
        self._version = 0
        self._silence_run = 0
        self._last_speech = -1
        self._pause_sent = False
        self._since_partial = 0

    def _step(self, frame: np.ndarray, probability: float, events: list[SpeechSegment]) -> None:
        if not self._active:
            self._pre_roll.push(frame, probability)
            if probability >= self.threshold:
                self._begin()
            return
        self._append(frame, probability)
        if probability >= self.threshold or (probability >= self.neg_threshold and self._silence_run == 0):
            self._speech_frame(events, probability)
        else:
            self._silence_frame(events)
            if not self._active:
                return
        if self._frames >= self.max_frames:
            self._split(events)

    def _begin(self) -> None:
        audio, probs = self._pre_roll.drain()
        self._audio[: audio.size] = audio
        self._probs[: probs.size] = probs
        self._frames = int(probs.size)
        self._active = True
        self._utterance += 1
        self._continues = False
        self._version = 1
        self._last_speech = self._frames - 1
        self._since_partial = 1

    def _append(self, frame: np.ndarray, probability: float) -> None:
        start = self._frames * FRAME_SAMPLES
        self._audio[start : start + FRAME_SAMPLES] = frame
        self._probs[self._frames] = probability
        self._frames += 1

    def _speech_frame(self, events: list[SpeechSegment], probability: float) -> None:
        self._version += 1
        self._last_speech = self._frames - 1
        self._silence_run = 0
        if self._pause_sent:
            self._pause_sent = False
            events.append(self._event("resume"))
        self._since_partial += 1
        if (
            self.partial_frames
            and self._since_partial >= self.partial_frames
            and self._version >= self.min_speech_frames
        ):
            # No meio da palavra, espera um vale curto. A frase não é cortada em "ultrapass" / "sando".
            grace = self.partial_frames + _frames(600)
            if self.partial_frames >= _frames(1000) and probability >= self.threshold and self._since_partial < grace:
                return
            self._since_partial = 0
            events.append(self._event("partial", self._snapshot()))

    def _silence_frame(self, events: list[SpeechSegment]) -> None:
        self._silence_run += 1
        if (
            self._silence_run == self.pause_frames
            and not self._pause_sent
            and self._version >= self.min_speech_frames
        ):
            self._pause_sent = True
            events.append(self._event("pause", self._snapshot()))
        limit = self.short_silence_frames if self._frames >= self.soft_frames else self.silence_frames
        if self._silence_run >= limit:
            self._finish("pause", events)

    def _finish(self, reason: str, events: list[SpeechSegment]) -> None:
        end = min(self._frames, self._last_speech + 1 + self.tail_frames)
        if self._version >= self.min_speech_frames:
            events.append(self._event("final", self._audio[: end * FRAME_SAMPLES].copy(), reason))
        # O silêncio depois da frase vira pré-roll da próxima; a fala já enviada, não.
        for index in range(max(end, self._frames - self._pre_roll.capacity), self._frames):
            start = index * FRAME_SAMPLES
            self._pre_roll.push(self._audio[start : start + FRAME_SAMPLES], float(self._probs[index]))
        self._reset()

    def _split(self, events: list[SpeechSegment]) -> None:
        # Procurar só na metade final garante que o corte avance mesmo com a sobreposição.
        window = min(self._frames // 2, self.valley_frames)
        first = self._frames - window
        cut = first + int(np.argmin(self._probs[first : self._frames])) + 1
        if self._version >= self.min_speech_frames:
            events.append(self._event("final", self._audio[: cut * FRAME_SAMPLES].copy(), "max_length"))
        keep_from = max(0, cut - self.overlap_frames)
        kept = self._frames - keep_from
        self._audio[: kept * FRAME_SAMPLES] = self._audio[keep_from * FRAME_SAMPLES : self._frames * FRAME_SAMPLES]
        self._probs[:kept] = self._probs[keep_from : self._frames]
        speech = np.flatnonzero(self._probs[:kept] >= self.threshold)
        self._frames = kept
        self._utterance += 1
        self._continues = True
        self._version = int(speech.size)
        self._last_speech = int(speech[-1]) if speech.size else -1
        self._silence_run = kept - 1 - self._last_speech
        self._pause_sent = False
        self._since_partial = 0

    def _snapshot(self) -> np.ndarray:
        return self._audio[: self._frames * FRAME_SAMPLES].copy()

    def _event(self, kind: str, audio: np.ndarray = _EMPTY, reason: str = "") -> SpeechSegment:
        return SpeechSegment(
            kind=kind,
            utterance=self._utterance,
            version=self._version,
            audio=audio,
            reason=reason,
            trailing_ms=max(0, self._frames - 1 - self._last_speech) * FRAME_MS,
            speech_ms=self._version * FRAME_MS,
            continues=self._continues,
            created_at=self._clock(),
        )


def resample_mono(audio: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.reshape(-1)
    if source_rate == target_rate or samples.size < 2:
        return samples
    target_size = max(1, round(samples.size * target_rate / source_rate))
    old_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


class StreamResampler:
    """Reamostra blocos contínuos para 16 kHz mono sem aliasing e sem estalo na emenda.

    Sem o passa-baixa, 48 kHz → 16 kHz vira "uma amostra a cada três": pratos,
    chiado e sibilantes de 8 a 24 kHz dobram para dentro da banda da voz e
    atrapalham o VAD e o Whisper. O estado do filtro e a fase da interpolação
    passam de um bloco para o seguinte.
    """

    TAPS = 127

    def __init__(self, source_rate: int, target_rate: int = SAMPLE_RATE) -> None:
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._step = self.source_rate / self.target_rate
        self._kernel: np.ndarray | None = None
        self._history = _EMPTY
        if self.source_rate > self.target_rate:
            cutoff = 0.46 * self.target_rate / self.source_rate
            taps = np.arange(self.TAPS) - (self.TAPS - 1) / 2
            kernel = np.sinc(2 * cutoff * taps) * np.blackman(self.TAPS)
            self._kernel = (kernel / kernel.sum()).astype(np.float32)
            self._history = np.zeros(self.TAPS - 1, dtype=np.float32)
        self._tail = _EMPTY
        self._position = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1, dtype=np.float32)
        samples = samples.reshape(-1)
        if self.source_rate == self.target_rate or samples.size == 0:
            return samples
        if self._kernel is not None:
            padded = np.concatenate((self._history, samples))
            self._history = padded[-(self.TAPS - 1) :]
            samples = np.convolve(padded, self._kernel, mode="valid").astype(np.float32)
        data = np.concatenate((self._tail, samples)) if self._tail.size else samples
        last = data.size - 1
        if last < self._position:
            self._tail = data
            return _EMPTY
        count = int((last - self._position) // self._step) + 1
        positions = self._position + self._step * np.arange(count)
        output = np.interp(positions, np.arange(data.size), data).astype(np.float32)
        following = self._position + self._step * count
        keep = min(int(following), data.size)
        self._tail = data[keep:].copy()
        self._position = following - keep
        return output


SegmentCallback = Callable[[SpeechSegment], None]
ErrorCallback = Callable[[str], None]


class AudioCapture:
    """Estágio A: lê o dispositivo numa thread própria e entrega segmentos já cortados pelo VAD."""

    def __init__(
        self,
        device: AudioDevice,
        on_segment: SegmentCallback,
        on_error: ErrorCallback,
        segmenter: SpeechSegmenter,
        suppressed: threading.Event | None = None,
    ) -> None:
        self.device = device
        self.on_segment = on_segment
        self.on_error = on_error
        self.segmenter = segmenter
        self.suppressed = suppressed
        self._stop = threading.Event()
        self.ready = threading.Event()
        self.failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._resampler: StreamResampler | None = None
        self._muted = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.ready.clear()
        self.failed.clear()
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.failed.set()
            logger.exception("Falha na captura de %s", self.device.name)
            self.on_error(f"{self.device.name}: {exc}")

    def _accept(self, chunk: np.ndarray, source_rate: int) -> None:
        if self._resampler is None or self._resampler.source_rate != source_rate:
            self._resampler = StreamResampler(source_rate)
        mono = self._resampler.process(chunk)
        if self.suppressed is not None and self.suppressed.is_set():
            if not self._muted:
                self._muted = True
                # O que chegou antes de a tradução começar a tocar ainda é fala limpa.
                self._emit(self.segmenter.flush())
            return
        if self._muted:
            self._muted = False
            self.segmenter.discard()
        self._emit(self.segmenter.feed(mono))

    def _emit(self, segments: Iterable[SpeechSegment]) -> None:
        for segment in segments:
            if self._stop.is_set():
                return
            self.on_segment(segment)

    def _run(self) -> None:
        raise NotImplementedError


class MicrophoneCapture(AudioCapture):
    def _run(self) -> None:
        index = AudioDeviceManager.index_from_uid(self.device.uid, "sd")
        block_size = int(self.device.sample_rate * 0.032)
        with com_apartment():
            with sd.InputStream(
                samplerate=self.device.sample_rate,
                channels=1,
                dtype="float32",
                device=index,
                blocksize=block_size,
            ) as stream:
                self.ready.set()
                while not self._stop.is_set():
                    chunk, overflowed = stream.read(block_size)
                    if overflowed:
                        logger.debug("Overflow no microfone %s", self.device.name)
                    self._accept(chunk, self.device.sample_rate)


class LoopbackCapture(AudioCapture):
    def _run(self) -> None:
        if pyaudio is None:
            raise RuntimeError("PyAudioWPatch não está instalado")
        index = AudioDeviceManager.index_from_uid(self.device.uid, "wasapi")
        block_size = max(256, int(self.device.sample_rate * 0.032))
        with com_apartment():
            self._run_loopback(index, block_size)

    def _run_loopback(self, index: int, block_size: int) -> None:
        with pyaudio.PyAudio() as audio:
            stream = audio.open(
                format=pyaudio.paFloat32,
                channels=self.device.channels,
                rate=self.device.sample_rate,
                input=True,
                input_device_index=index,
                frames_per_buffer=block_size,
            )
            host_failed = False
            try:
                self.ready.set()
                while not self._stop.is_set():
                    try:
                        raw = stream.read(block_size, exception_on_overflow=False)
                    except OSError as exc:
                        host_failed = True
                        raise RuntimeError(
                            "O retorno da chamada caiu. Atualize os dispositivos e deixe o CABLE Output selecionado."
                        ) from exc
                    chunk = np.frombuffer(raw, dtype=np.float32)
                    if self.device.channels > 1:
                        chunk = chunk.reshape(-1, self.device.channels)
                    self._accept(chunk, self.device.sample_rate)
            finally:
                LoopbackCapture._release_stream(audio, stream, host_failed)

    @staticmethod
    def _release_stream(audio, stream, host_failed: bool) -> None:
        """O WASAPI já morto quebra se o PortAudio tentar pará-lo de novo."""
        stream._is_running = False
        if host_failed:
            audio._streams.discard(stream)
            return
        try:
            stream.close()
        except Exception:
            logger.warning("Não foi possível fechar o retorno da chamada", exc_info=True)
