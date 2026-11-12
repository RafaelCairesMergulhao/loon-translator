import numpy as np

from services.voice_overlap import CROWD_HOLD_S, split_voices


def _voice(pitch: float, samples: int, sample_rate: int = 16_000) -> np.ndarray:
    time_axis = np.arange(samples) / sample_rate
    wave = sum((0.55**harmonic) * np.sin(2 * np.pi * (harmonic + 1) * pitch * time_axis) for harmonic in range(8))
    return (wave / np.max(np.abs(wave)) * 0.4).astype(np.float32)


def test_one_voice_stays_whole_and_fast() -> None:
    audio = _voice(150, 9_600)
    split = split_voices(audio)
    assert split.voice_count == 1
    assert split.simultaneous == 1
    assert split.hold_s == 0
    assert np.array_equal(split.clips[0].audio, audio)
    assert abs(split.clips[0].pitch - 150) <= 4


def test_two_voices_come_out_in_the_order_they_started() -> None:
    length = 12_800
    first = _voice(130, length)
    second = np.zeros(length, dtype=np.float32)
    second[4_000:] = _voice(230, length - 4_000)
    split = split_voices(first + second)
    assert split.voice_count == 2
    assert split.simultaneous == 2
    assert split.hold_s == 0
    assert abs(split.clips[0].pitch - 130) <= 6
    assert abs(split.clips[1].pitch - 230) <= 6
    assert split.clips[0].onset < split.clips[1].onset


def test_three_voices_at_once_ask_for_the_longer_hold() -> None:
    length = 11_200
    mix = _voice(110, length) + _voice(170, length) + _voice(250, length)
    split = split_voices(mix)
    assert split.voice_count == 3
    assert split.simultaneous >= 3
    assert split.hold_s == CROWD_HOLD_S == 1.5
    assert [round(clip.pitch) for clip in split.clips] == [110, 170, 250]


def test_noise_that_is_not_a_voice_is_dropped() -> None:
    noise = (np.random.default_rng(1).standard_normal(8_000) * 0.2).astype(np.float32)
    split = split_voices(noise)
    assert split.voice_count == 0
    assert split.clips == []
