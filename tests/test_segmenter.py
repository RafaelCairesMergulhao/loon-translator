import numpy as np
import pytest

from services.audio_manager import FrameRing, SpeechSegmenter, StreamResampler
from services.vad import FRAME_MS, FRAME_SAMPLES, EnergyVAD, SileroVAD


class EchoVAD:
    """A probabilidade de voz de cada quadro é o próprio valor das amostras."""

    def __init__(self) -> None:
        self.resets = 0

    def probability(self, frame: np.ndarray) -> float:
        return float(frame[0])

    def reset(self) -> None:
        self.resets += 1


def frames(*values: float) -> np.ndarray:
    return np.concatenate([np.full(FRAME_SAMPLES, value, dtype=np.float32) for value in values])


def segmenter(**overrides) -> SpeechSegmenter:
    """Quadros de 32 ms: silêncio de 4 quadros, pausa de 2, pré-roll de 4, cauda de 2."""
    options = {
        "silence_ms": 128,
        "min_speech_ms": 64,
        "pre_roll_ms": 128,
        "pause_ms": 64,
        "tail_ms": 64,
        "clock": lambda: 10.0,
    }
    options.update(overrides)
    return SpeechSegmenter(EchoVAD(), **options)


def test_phrase_ends_after_the_silence_and_keeps_a_short_tail() -> None:
    seg = segmenter()
    events = seg.feed(frames(*[0.9] * 5, *[0.0] * 6))
    assert [event.kind for event in events] == ["pause", "final"]
    pause, final = events
    assert pause.version == final.version == 5
    assert final.reason == "pause"
    assert final.audio.size == (5 + 2) * FRAME_SAMPLES
    assert final.trailing_ms == 4 * FRAME_MS
    assert final.speech_ms == 5 * FRAME_MS
    assert final.created_at == 10.0
    assert not seg.active


def test_short_noise_is_not_a_phrase() -> None:
    seg = segmenter(min_speech_ms=160)
    assert seg.feed(frames(0.9, 0.9, *[0.0] * 6)) == []
    assert not seg.active


def test_pre_roll_keeps_the_first_consonant() -> None:
    seg = segmenter()
    events = seg.feed(frames(0.0, 0.0, 0.0, 0.3, *[0.9] * 3, *[0.0] * 6))
    final = events[-1]
    assert final.kind == "final"
    assert np.any(np.isclose(final.audio, 0.3)), "o ataque abaixo do limiar precisa entrar na frase"
    assert final.audio.size == (2 + 1 + 3 + 2) * FRAME_SAMPLES


def test_samples_wait_for_a_whole_frame() -> None:
    seg = segmenter()
    half = np.full(FRAME_SAMPLES // 2 + 44, 0.9, dtype=np.float32)
    assert seg.feed(half) == []
    assert not seg.active
    seg.feed(half)
    assert seg.active


def test_speech_after_a_short_pause_invalidates_it() -> None:
    seg = segmenter(silence_ms=320)
    events = seg.feed(frames(*[0.9] * 4, 0.0, 0.0, 0.0, *[0.9] * 3, *[0.0] * 10))
    assert [event.kind for event in events] == ["pause", "resume", "pause", "final"]
    first_pause, resume, second_pause, final = events
    assert first_pause.version == 4
    assert resume.version == 5
    assert second_pause.version == final.version == 7
    assert len({event.utterance for event in events}) == 1


def test_hysteresis_keeps_a_soft_syllable_but_not_a_late_one() -> None:
    seg = segmenter()
    events = seg.feed(frames(0.9, 0.9, 0.4, 0.9, 0.0, 0.4, *[0.0] * 4))
    assert [event.kind for event in events] == ["pause", "final"]
    assert events[-1].version == 4


def test_partials_carry_the_phrase_so_far() -> None:
    seg = segmenter(partial_ms=96)
    partials = [event for event in seg.feed(frames(*[0.9] * 7)) if event.kind == "partial"]
    assert [event.audio.size // FRAME_SAMPLES for event in partials] == [3, 6]


def test_long_speech_is_cut_at_the_quietest_frame_with_overlap() -> None:
    seg = segmenter(max_seconds=1.0)
    probabilities = [0.9] * 32
    probabilities[24] = 0.6
    [cut] = seg.feed(frames(*probabilities))
    assert cut.kind == "final" and cut.reason == "max_length"
    assert cut.audio.size == 25 * FRAME_SAMPLES
    assert not cut.continues

    events = seg.feed(frames(*[0.9] * 3, *[0.0] * 6))
    final = events[-1]
    assert final.kind == "final" and final.continues
    assert final.utterance == cut.utterance + 1
    overlap = 3 * FRAME_SAMPLES
    np.testing.assert_array_equal(final.audio[:overlap], cut.audio[-overlap:])


def test_endless_speech_never_overflows_the_buffer() -> None:
    seg = segmenter(max_seconds=1.0)
    cuts = [event for event in seg.feed(frames(*[0.9] * 400)) if event.kind == "final"]
    assert len(cuts) >= 10
    assert all(event.audio.size <= 32 * FRAME_SAMPLES for event in cuts)


def test_flush_closes_the_phrase_and_discard_forgets_it() -> None:
    seg = segmenter()
    seg.feed(frames(*[0.9] * 4))
    [final] = seg.flush()
    assert final.kind == "final" and final.reason == "flush"
    assert final.audio.size == 4 * FRAME_SAMPLES
    assert seg.flush() == []

    seg.feed(frames(*[0.9] * 4))
    seg.discard()
    assert not seg.active
    assert seg.flush() == []
    assert seg.vad.resets == 1


def test_frame_ring_returns_frames_in_order() -> None:
    ring = FrameRing(3)
    for value in range(5):
        ring.push(np.full(FRAME_SAMPLES, value, dtype=np.float32), value / 10)
    audio, probabilities = ring.drain()
    assert audio.reshape(-1, FRAME_SAMPLES)[:, 0].tolist() == [2, 3, 4]
    assert probabilities.tolist() == pytest.approx([0.2, 0.3, 0.4])
    assert len(ring) == 0


@pytest.mark.parametrize("rate", [48_000, 44_100, 16_000])
def test_resampler_does_not_drift(rate: int) -> None:
    resampler = StreamResampler(rate)
    sizes = np.random.default_rng(1).integers(100, 2_000, 300)
    produced = sum(resampler.process(np.zeros(size, dtype=np.float32)).size for size in sizes)
    assert abs(produced - sizes.sum() * 16_000 / rate) <= 2


def test_resampler_blocks_aliasing_but_keeps_the_voice_band() -> None:
    rate = 48_000
    time_axis = np.arange(rate) / rate

    def level(frequency: float) -> float:
        resampler = StreamResampler(rate)
        tone = np.sin(2 * np.pi * frequency * time_axis).astype(np.float32)
        output = np.concatenate([resampler.process(block) for block in np.array_split(tone, 47)])
        return float(np.sqrt(np.mean(output[1_000:] ** 2)))

    assert level(1_000) > 0.65
    assert level(12_000) < 0.01, "12 kHz voltaria como 4 kHz, no meio da voz"


def test_resampler_is_seamless_across_blocks() -> None:
    rate = 44_100
    signal = np.random.default_rng(3).standard_normal(rate).astype(np.float32)
    whole = StreamResampler(rate).process(signal)
    streamed = StreamResampler(rate)
    parts = np.concatenate([streamed.process(block) for block in np.array_split(signal, 97)])
    assert abs(parts.size - whole.size) <= 1
    size = min(parts.size, whole.size)
    np.testing.assert_allclose(parts[:size], whole[:size], atol=1e-4)


def test_resampler_mixes_stereo_to_mono() -> None:
    stereo = np.column_stack([np.full(320, 0.2), np.full(320, 0.4)]).astype(np.float32)
    output = StreamResampler(16_000).process(stereo)
    assert output.shape == (320,)
    assert np.allclose(output, 0.3)


def test_silero_vad_is_stateful_and_resettable() -> None:
    try:
        vad = SileroVAD()
    except Exception as exc:
        pytest.skip(f"Silero indisponível: {exc}")
    noise = [
        np.random.default_rng(seed).standard_normal(FRAME_SAMPLES).astype(np.float32) * 0.005
        for seed in range(12)
    ]
    first = [vad.probability(frame) for frame in noise]
    vad.reset()
    again = [vad.probability(frame) for frame in noise]
    assert first == pytest.approx(again, abs=1e-6)
    assert all(0.0 <= probability < 0.5 for probability in first), "chiado baixo não é voz"


def test_energy_vad_follows_the_noise_floor() -> None:
    vad = EnergyVAD()
    hiss = np.full(FRAME_SAMPLES, 0.01, dtype=np.float32)
    assert all(vad.probability(hiss) == 0.0 for _ in range(50))
    assert vad.probability(np.full(FRAME_SAMPLES, 0.2, dtype=np.float32)) == 1.0
