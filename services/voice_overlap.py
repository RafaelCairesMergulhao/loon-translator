"""Separa vozes misturadas numa única faixa da chamada.

Não há um modelo de diarização. Cada quadro curto procura um ou mais tons
(F0). Tons que não são harmônicos um do outro viram pessoas diferentes.
Uma pessoa só devolve o áudio original, sem atraso. Duas saem em ordem.
Três ou mais ao mesmo tempo pedem uma espera curta para juntar os pedaços
da mesma voz antes de traduzir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
CROWD_HOLD_S = 1.5
_MIN_F0 = 85.0
_MAX_F0 = 360.0
_N_FFT = 2048
_HOP = 512
_PITCH_FFT = 4096
_MIN_FRAMES = 4


@dataclass
class VoiceClip:
    pitch: float
    onset: float
    audio: np.ndarray


@dataclass
class OverlapSplit:
    voice_count: int
    simultaneous: int
    hold_s: float
    clips: list[VoiceClip] = field(default_factory=list)


def split_voices(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> OverlapSplit:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < sample_rate // 5:
        return OverlapSplit(1 if samples.size else 0, 1 if samples.size else 0, 0.0, [VoiceClip(0.0, 0.0, samples)] if samples.size else [])
    padded, frame_count = _padded(samples)
    pitches = [_frame_pitches(padded[index * _HOP : index * _HOP + _N_FFT], sample_rate) for index in range(frame_count)]
    tracks = _tracks(pitches)
    if not tracks:
        if _is_noise(samples):
            return OverlapSplit(0, 0, 0.0, [])
        return OverlapSplit(1, 1, 0.0, [VoiceClip(0.0, 0.0, samples)])
    if len(tracks) == 1:
        return OverlapSplit(1, 1, 0.0, [VoiceClip(_pitch_of(tracks[0]), tracks[0][0][0] * _HOP / sample_rate, samples)])
    simultaneous = _simultaneous(tracks, frame_count)
    clips = _isolate(samples, padded, frame_count, tracks, sample_rate)
    clips.sort(key=lambda clip: (int(clip.onset / 0.15), clip.pitch))
    hold = CROWD_HOLD_S if simultaneous > 2 else 0.0
    return OverlapSplit(len(clips), simultaneous, hold, clips)


def merge_clips(groups: list[OverlapSplit]) -> list[VoiceClip]:
    """Junta pedaços do mesmo tom, na ordem em que cada pessoa começou."""
    merged: list[VoiceClip] = []
    for group in groups:
        for clip in group.clips:
            host = next((item for item in merged if _same_voice(item.pitch, clip.pitch)), None)
            if host is None:
                merged.append(VoiceClip(clip.pitch, clip.onset, clip.audio.copy()))
                continue
            host.audio = np.concatenate((host.audio, clip.audio))
            host.onset = min(host.onset, clip.onset)
            host.pitch = float(np.median((host.pitch, clip.pitch)))
    merged.sort(key=lambda clip: (clip.onset, clip.pitch))
    return merged


def _padded(audio: np.ndarray) -> tuple[np.ndarray, int]:
    pad = _N_FFT // 2
    padded = np.pad(audio, (pad, pad))
    count = 1 + max(0, (padded.size - _N_FFT) // _HOP)
    return padded, count


def _frame_pitches(frame: np.ndarray, sample_rate: int) -> list[float]:
    """Cada voz é um pente de harmônicas. Dá para achar mais de um tom no mesmo quadro."""
    if frame.size != _N_FFT or float(np.dot(frame, frame)) < 1e-5:
        return []
    windowed = np.pad(frame * np.hanning(frame.size), (0, _PITCH_FFT - frame.size))
    magnitude = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(_PITCH_FFT, d=1.0 / sample_rate)
    speech = magnitude[(freqs >= _MIN_F0) & (freqs <= 1200)]
    peak = float(speech.max()) if speech.size else 0.0
    if peak < 1e-3:
        return []
    grid = np.arange(_MIN_F0, _MAX_F0 + 1.0, 1.0)
    work = magnitude.copy()
    chosen: list[float] = []
    first_score = 0.0
    for _attempt in range(5):
        scores = np.array([_score_f0(freqs, work, pitch, peak) for pitch in grid])
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score <= 0:
            break
        if chosen and score < 0.62 * first_score:
            break
        pitch = float(grid[best])
        _notch(work, freqs, pitch)
        if any(_same_voice(pitch, other) for other in chosen):
            continue
        if not chosen:
            first_score = score
        chosen.append(pitch)
        if len(chosen) == 3:
            break
    return chosen


def _score_f0(freqs: np.ndarray, magnitude: np.ndarray, pitch: float, peak: float) -> float:
    index = int(np.argmin(np.abs(freqs - pitch)))
    neighborhood = np.abs(freqs - pitch) < 40
    neighborhood[index] = False
    local = float(np.median(magnitude[neighborhood])) if np.any(neighborhood) else 0.0
    if magnitude[index] < 4.0 * (local + 1e-9) or magnitude[index] < 0.12 * peak:
        return 0.0
    score = 0.0
    hits = 0
    for harmonic in range(1, 8):
        center = harmonic * pitch
        if center >= freqs[-1]:
            break
        bin_index = int(np.argmin(np.abs(freqs - center)))
        score += float(np.log1p(magnitude[bin_index]))
        hits += 1
    return score if hits >= 3 else 0.0


def _notch(magnitude: np.ndarray, freqs: np.ndarray, pitch: float) -> None:
    for harmonic in range(1, 14):
        center = harmonic * pitch
        if center >= freqs[-1]:
            break
        band = np.abs(freqs - center) < max(16.0, 0.045 * center)
        magnitude[band] *= 0.05


def _tracks(frame_pitches: list[list[float]]) -> list[list[tuple[int, float]]]:
    tracks: list[list[tuple[int, float]]] = []
    for index, pitches in enumerate(frame_pitches):
        used: set[int] = set()
        for pitch in pitches:
            match = None
            for track_index, track in enumerate(tracks):
                if track_index in used or index - track[-1][0] > 10:
                    continue
                recent = [item[1] for item in track[-5:]]
                if _same_voice(float(np.median(recent)), pitch):
                    match = track_index
                    break
            if match is None:
                tracks.append([(index, pitch)])
                used.add(len(tracks) - 1)
            else:
                tracks[match].append((index, pitch))
                used.add(match)
    return [track for track in tracks if len(track) >= _MIN_FRAMES]


def _simultaneous(tracks: list[list[tuple[int, float]]], frame_count: int) -> int:
    counts = np.zeros(frame_count, dtype=np.int16)
    for track in tracks:
        start = track[0][0]
        end = track[-1][0]
        counts[start : end + 1] += 1
    return int(counts.max()) if counts.size else 0


def _isolate(
    audio: np.ndarray,
    padded: np.ndarray,
    frame_count: int,
    tracks: list[list[tuple[int, float]]],
    sample_rate: int,
) -> list[VoiceClip]:
    window = np.hanning(_N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sample_rate)
    masks = np.zeros((len(tracks), frame_count, freqs.size), dtype=np.complex64)
    active = [{index for index, _pitch in track} for track in tracks]
    pitches = [_pitch_of(track) for track in tracks]
    owned = np.zeros(frame_count, dtype=np.int16)
    for index in range(frame_count):
        present = [voice for voice, frames in enumerate(active) if index in frames]
        # Consoante grudada no quadro de voz mais próximo, até ~80 ms.
        if not present:
            nearest = _nearest_voice(index, active, distance=5)
            if nearest is None:
                continue
            present = [nearest]
        owned[index] = len(present)
        frame = padded[index * _HOP : index * _HOP + _N_FFT] * window
        spectrum = np.fft.rfft(frame)
        if len(present) == 1:
            masks[present[0], index] = spectrum
            continue
        weights = np.zeros((len(present), freqs.size), dtype=np.float32)
        for slot, voice in enumerate(present):
            weights[slot] = _harmonic_weight(freqs, pitches[voice])
        total = weights.sum(axis=0)
        total[total < 1e-6] = 1.0
        for slot, voice in enumerate(present):
            masks[voice, index] = spectrum * (weights[slot] / total)
    clips: list[VoiceClip] = []
    for voice, track in enumerate(tracks):
        restored = _istft(masks[voice], window, audio.size)
        if float(np.sqrt(np.mean(restored**2))) < 1e-3:
            continue
        onset = track[0][0] * _HOP / sample_rate
        clips.append(VoiceClip(pitches[voice], onset, restored))
    return clips


def _nearest_voice(index: int, active: list[set[int]], distance: int) -> int | None:
    best: tuple[int, int] | None = None
    for voice, frames in enumerate(active):
        if not frames:
            continue
        gap = min(abs(index - frame) for frame in frames)
        if gap <= distance and (best is None or gap < best[0]):
            best = (gap, voice)
    return None if best is None else best[1]


def _harmonic_weight(freqs: np.ndarray, pitch: float) -> np.ndarray:
    weight = np.zeros(freqs.shape, dtype=np.float32)
    if pitch <= 0:
        return weight
    width = max(28.0, pitch * 0.18)
    for harmonic in range(1, 18):
        center = harmonic * pitch
        if center >= freqs[-1]:
            break
        weight += np.exp(-0.5 * ((freqs - center) / width) ** 2).astype(np.float32)
    return weight


def _istft(spectra: np.ndarray, window: np.ndarray, length: int) -> np.ndarray:
    pad = _N_FFT // 2
    output = np.zeros(length + _N_FFT + pad, dtype=np.float32)
    normal = np.zeros_like(output)
    for index in range(spectra.shape[0]):
        start = index * _HOP
        frame = np.fft.irfft(spectra[index], n=_N_FFT).astype(np.float32) * window
        output[start : start + _N_FFT] += frame
        normal[start : start + _N_FFT] += window**2
    normal[normal < 1e-6] = 1.0
    restored = output[pad : pad + length] / normal[pad : pad + length]
    return restored.astype(np.float32)


def _pitch_of(track: list[tuple[int, float]]) -> float:
    return float(np.median([pitch for _index, pitch in track]))


def _same_voice(left: float, right: float) -> bool:
    if left <= 0 or right <= 0:
        return False
    ratio = abs(np.log2(left / right))
    nearest = round(ratio)
    if nearest == 0:
        return ratio < 0.12
    if nearest > 3:
        return False
    return abs(ratio - nearest) < 0.07


def _is_noise(audio: np.ndarray) -> bool:
    windowed = audio * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum = spectrum[1:]
    spectrum = spectrum[spectrum > 0]
    if spectrum.size < 8:
        return True
    geometric = float(np.exp(np.mean(np.log(spectrum + 1e-12))))
    arithmetic = float(np.mean(spectrum))
    return geometric / (arithmetic + 1e-12) > 0.28
