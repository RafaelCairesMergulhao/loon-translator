from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from services.audio_devices import AudioDevice, AudioDeviceManager, pyaudio

logger = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], None]
ErrorCallback = Callable[[str], None]


class PhraseSegmenter:
    """Segmenta frases usando energia adaptativa; o STT aplica Silero VAD depois.

    O pré-roll guarda os últimos milissegundos antes do limiar de energia ser
    cruzado: consoantes surdas (p, t, k, s, f) começam abaixo dele e seriam cortadas.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        silence_ms: int = 450,
        max_seconds: float = 8.0,
        min_speech_ms: int = 200,
        pre_roll_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.max_samples = int(sample_rate * max_seconds)
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.pre_roll_samples = int(sample_rate * pre_roll_ms / 1000)
        self.noise_floor = 0.006
        self._chunks: list[np.ndarray] = []
        self._total = 0
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_total = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._active = False

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return None
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
        threshold = max(0.012, self.noise_floor * 3.2)
        is_speech = rms >= threshold

        if not self._active and not is_speech:
            self.noise_floor = (self.noise_floor * 0.97) + (rms * 0.03)
            self._remember_pre_roll(mono)
            return None

        if is_speech:
            if not self._active:
                self._chunks = list(self._pre_roll)
                self._total = self._pre_roll_total
                self._pre_roll.clear()
                self._pre_roll_total = 0
            self._active = True
            self._speech_samples += mono.size
            self._silence_samples = 0
        elif self._active:
            self._silence_samples += mono.size

        if self._active:
            self._chunks.append(mono)
            self._total += mono.size

        phrase_done = self._silence_samples >= self.silence_samples or self._total >= self.max_samples
        if phrase_done:
            return self.flush()
        return None

    def _remember_pre_roll(self, mono: np.ndarray) -> None:
        if self.pre_roll_samples <= 0:
            return
        self._pre_roll.append(mono)
        self._pre_roll_total += mono.size
        while self._pre_roll and self._pre_roll_total - self._pre_roll[0].size >= self.pre_roll_samples:
            self._pre_roll_total -= self._pre_roll.popleft().size

    def flush(self) -> np.ndarray | None:
        self._pre_roll.clear()
        self._pre_roll_total = 0
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks)
        valid = self._speech_samples >= self.min_speech_samples
        self._chunks = []
        self._total = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._active = False
        return audio if valid else None


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


class AudioCapture:
    def __init__(
        self,
        device: AudioDevice,
        on_phrase: AudioCallback,
        on_error: ErrorCallback,
        segmenter: PhraseSegmenter,
        suppressed: threading.Event | None = None,
    ) -> None:
        self.device = device
        self.on_phrase = on_phrase
        self.on_error = on_error
        self.segmenter = segmenter
        self.suppressed = suppressed
        self._stop = threading.Event()
        self.ready = threading.Event()
        self.failed = threading.Event()
        self._thread: threading.Thread | None = None

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
        if not self._thread or not self._thread.is_alive():
            trailing = self.segmenter.flush()
            if trailing is not None:
                self.on_phrase(trailing)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.failed.set()
            logger.exception("Falha na captura de %s", self.device.name)
            self.on_error(f"{self.device.name}: {exc}")

    def _accept(self, chunk: np.ndarray, source_rate: int) -> None:
        if self.suppressed and self.suppressed.is_set():
            self.segmenter.flush()
            return
        phrase = self.segmenter.feed(resample_mono(chunk, source_rate))
        if phrase is not None and not self._stop.is_set():
            self.on_phrase(phrase)

    def _run(self) -> None:
        raise NotImplementedError


class MicrophoneCapture(AudioCapture):
    def _run(self) -> None:
        index = AudioDeviceManager.index_from_uid(self.device.uid, "sd")
        block_size = int(self.device.sample_rate * 0.032)
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