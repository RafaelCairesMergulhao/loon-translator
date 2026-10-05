"""Desloca o tom da voz sem mudar a duração da frase.

O vocoder de fase alonga ou encurta o áudio preservando o tom; em seguida
a leitura volta ao tamanho original, o que sobe ou desce a frequência.
"""

from __future__ import annotations

import numpy as np


def shift_pitch(samples: np.ndarray, semitones: float) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0 or abs(float(semitones)) < 0.05:
        return np.array(data, dtype=np.float32, copy=True)
    semitones = float(np.clip(semitones, -12.0, 12.0))
    if data.ndim == 1:
        return _shift_channel(data, semitones)
    channels = [_shift_channel(data[:, index], semitones) for index in range(data.shape[1])]
    length = max(channel.shape[0] for channel in channels)
    aligned = [
        channel if channel.shape[0] == length else np.pad(channel, (0, length - channel.shape[0]))
        for channel in channels
    ]
    return np.column_stack(aligned).astype(np.float32)


def _shift_channel(samples: np.ndarray, semitones: float) -> np.ndarray:
    factor = 2.0 ** (semitones / 12.0)
    if samples.shape[0] < 1024:
        return _resample_pitch(samples, factor)
    stretched = _time_stretch(samples, factor)
    if stretched.shape[0] < 2:
        return samples.copy()
    positions = np.linspace(0.0, stretched.shape[0] - 1, num=samples.shape[0])
    shifted = np.interp(positions, np.arange(stretched.shape[0]), stretched).astype(np.float32)
    fade = min(64, max(1, shifted.shape[0] // 20))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    shifted[:fade] *= ramp
    shifted[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
    if peak > 1.0:
        shifted /= peak
    return shifted


def _resample_pitch(samples: np.ndarray, factor: float) -> np.ndarray:
    """Atalho para trechos curtos demais para o vocoder. A duração muda junto com o tom."""
    new_length = max(1, int(round(samples.shape[0] / factor)))
    if samples.shape[0] < 2 or new_length < 2:
        return samples.copy()
    source = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False)
    target = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(target, source, samples).astype(np.float32)


def _time_stretch(samples: np.ndarray, stretch: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """stretch > 1 deixa o áudio mais longo e mantém o tom."""
    window = np.hanning(n_fft).astype(np.float64)
    frame_count = 1 + (samples.shape[0] - n_fft) // hop
    if frame_count < 2:
        return samples.astype(np.float64)
    synthesis_hop = max(1, int(round(hop * stretch)))
    bins = n_fft // 2 + 1
    expected = 2.0 * np.pi * hop * np.arange(bins) / n_fft
    output = np.zeros(n_fft + synthesis_hop * (frame_count - 1), dtype=np.float64)
    window_sum = np.zeros_like(output)
    previous = np.zeros(bins, dtype=np.float64)
    phase = np.zeros(bins, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop
        frame = samples[start : start + n_fft].astype(np.float64) * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        angle = np.angle(spectrum)
        if index == 0:
            phase = angle
        else:
            delta = angle - previous - expected
            delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
            phase = phase + (expected + delta) * stretch
        previous = angle
        grain = np.fft.irfft(magnitude * np.exp(1j * phase)).real * window
        out_start = index * synthesis_hop
        output[out_start : out_start + n_fft] += grain
        window_sum[out_start : out_start + n_fft] += window**2
    window_sum = np.maximum(window_sum, 1e-8)
    return output / window_sum
