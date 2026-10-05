import numpy as np
import soundfile as sf

from services.pitch_shift import shift_pitch
from services.voice_likeness import match_file


def _peak_hz(samples: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(samples.shape[0]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples * window))
    return float(np.argmax(spectrum) * sample_rate / samples.shape[0])


def test_zero_semitones_keep_the_signal() -> None:
    tone = np.sin(2 * np.pi * 220 * np.arange(8_000) / 8_000).astype(np.float32)
    assert np.allclose(shift_pitch(tone, 0), tone)


def test_pitch_up_raises_frequency_without_changing_length() -> None:
    sample_rate = 16_000
    tone = np.sin(2 * np.pi * 200 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    shifted = shift_pitch(tone, 12)
    assert shifted.shape == tone.shape
    assert np.isfinite(shifted).all()
    base = _peak_hz(tone, sample_rate)
    higher = _peak_hz(shifted, sample_rate)
    assert higher > base * 1.7
    assert higher < base * 2.4


def test_translated_voice_moves_toward_the_speaker(tmp_path) -> None:
    rate = 16_000
    seconds = np.arange(rate) / rate
    spoken = np.sin(2 * np.pi * 120 * seconds).astype(np.float32)
    reference = np.sin(2 * np.pi * 200 * seconds).astype(np.float32)
    source = tmp_path / "voz.wav"
    sf.write(source, spoken, rate)
    matched = match_file(source, reference, rate)
    heard, heard_rate = sf.read(matched, dtype="float32")
    assert heard_rate == rate
    assert _peak_hz(heard, rate) > 130


def test_neural_voice_is_left_untouched(tmp_path) -> None:
    rate = 16_000
    spoken = np.sin(2 * np.pi * 120 * np.arange(rate) / rate).astype(np.float32)
    reference = np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    source = tmp_path / "abc.neural.mp3"
    source.write_bytes(b"not-audio")
    assert match_file(source, reference, rate) == source


def test_pitch_down_lowers_frequency() -> None:
    sample_rate = 16_000
    tone = np.sin(2 * np.pi * 400 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    shifted = shift_pitch(tone, -12)
    base = _peak_hz(tone, sample_rate)
    lower = _peak_hz(shifted, sample_rate)
    assert lower < base * 0.65
    assert lower > base * 0.35
