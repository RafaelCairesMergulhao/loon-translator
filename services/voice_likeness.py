"""Aproxima o tom da voz traduzida da pessoa que acabou de falar.

Isso não clona a identidade. Em vozes neurais o arquivo original é mantido:
o vocoder de fase deixaria o timbre metálico. Na voz do Windows, só o tom
sobe ou desce no máximo 3 semitons.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from services.pitch_shift import shift_pitch
from services.tts_engine import trim_silence

logger = logging.getLogger(__name__)


def match_file(
    path: Path,
    reference: np.ndarray,
    reference_rate: int,
    extra_semitones: float = 0.0,
) -> Path:
    if is_neural_voice(path):
        return path
    try:
        spoken, rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        return path
    spoken = trim_silence(spoken, rate)
    reference_mono = _mono(reference)
    if reference_mono.size < reference_rate // 5 or spoken.size < rate // 5:
        return path
    reference_mono = _resample(reference_mono, reference_rate, rate)
    spoken_mono = _mono(spoken)
    semitones = _pitch_delta(spoken_mono, reference_mono, rate) + float(extra_semitones)
    if abs(semitones) < 0.8:
        return path
    shifted = shift_pitch(spoken, semitones)
    target = path.with_name(f"{path.stem}.like.wav")
    sf.write(target, shifted, rate)
    return target


def is_neural_voice(path: Path) -> bool:
    """Voz neural já tem timbre humano; o vocoder de fase deixaria ela metálica."""
    name = path.name.lower()
    return ".neural." in name or ".piper." in name or path.suffix.lower() == ".mp3"


def _pitch_delta(spoken: np.ndarray, reference: np.ndarray, rate: int) -> float:
    spoken_hz = _fundamental_hz(spoken, rate)
    reference_hz = _fundamental_hz(reference, rate)
    if not spoken_hz or not reference_hz:
        return 0.0
    return float(np.clip(12.0 * np.log2(reference_hz / spoken_hz), -3.0, 3.0))


def _fundamental_hz(samples: np.ndarray, rate: int) -> float | None:
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    mono = mono - float(np.mean(mono))
    window = min(mono.size, int(rate * 0.45))
    if window < int(rate * 0.15):
        return None
    if mono.size > window:
        # Energia em janela deslizante por soma de prefixos: O(n) em vez da convolução O(n·janela).
        prefix = np.concatenate(([0.0], np.cumsum(mono * mono)))
        energy = prefix[window:] - prefix[:-window]
        start = int(np.argmax(energy))
        mono = mono[start : start + window]
    else:
        mono = mono[:window]
    if float(np.sqrt(np.mean(mono * mono))) < 0.01:
        return None
    mono = mono * np.hanning(mono.size)
    correlation = np.fft.irfft(np.abs(np.fft.rfft(mono)) ** 2)
    minimum = max(1, int(rate / 400))
    maximum = min(correlation.size - 1, int(rate / 70))
    if maximum <= minimum:
        return None
    lag = minimum + int(np.argmax(correlation[minimum:maximum]))
    if correlation[lag] < 0.25 * correlation[0]:
        return None
    return rate / lag


def _mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1).astype(np.float32)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size < 2:
        return samples.astype(np.float32)
    target_size = max(2, int(round(samples.size * target_rate / source_rate)))
    source = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    target = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target, source, samples).astype(np.float32)
