import numpy as np
import pytest

import services.tts_engine as tts_module
from services.audio_devices import AudioDevice, AudioDeviceManager
from services.tts_engine import AudioPlayer
from services.audio_manager import PhraseSegmenter, resample_mono


def test_whisper_drops_a_repeated_hallucination() -> None:
    from core.whisper_engine import WhisperEngine

    assert WhisperEngine.usable("Alô, tudo bem?")
    assert not WhisperEngine.usable("obrigado obrigado obrigado obrigado")
    assert WhisperEngine.normalize_speech("Não pode falar. " * 50) == ""
    assert WhisperEngine.normalize_speech("Oi. Oi. Tudo bem.") == "Oi. Tudo bem."
    loop = "Eu não sei se " + "o que é " * 20
    assert WhisperEngine.normalize_speech(loop) == ""


def test_phrase_segmenter_emits_after_silence() -> None:
    segmenter = PhraseSegmenter(
        sample_rate=16_000,
        silence_ms=100,
        min_speech_ms=30,
        max_seconds=2,
    )
    speech = np.full(800, 0.2, dtype=np.float32)
    silence = np.zeros(1_600, dtype=np.float32)
    assert segmenter.feed(speech) is None
    phrase = segmenter.feed(silence)
    assert phrase is not None
    assert phrase.size == 2_400


def test_phrase_segmenter_ignores_short_noise() -> None:
    segmenter = PhraseSegmenter(sample_rate=16_000, silence_ms=50, min_speech_ms=100)
    segmenter.feed(np.full(400, 0.2, dtype=np.float32))
    assert segmenter.feed(np.zeros(800, dtype=np.float32)) is None


def test_resample_stereo_to_mono_16khz() -> None:
    stereo = np.ones((4_800, 2), dtype=np.float32)
    result = resample_mono(stereo, source_rate=48_000, target_rate=16_000)
    assert result.shape == (1_600,)
    assert np.allclose(result, 1.0)


def test_kernel_streaming_host_is_hidden_from_the_device_list() -> None:
    assert AudioDeviceManager.include_hostapi("Windows WASAPI")
    assert AudioDeviceManager.include_hostapi("MME")
    assert not AudioDeviceManager.include_hostapi("Windows WDM-KS")
    assert not AudioDeviceManager.include_hostapi("Windows DirectSound")


def test_saved_wdm_uid_resolves_to_the_readable_headset(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = AudioDeviceManager()

    def query_devices(index: int) -> dict[str, str]:
        assert index == 41
        return {
            "name": r"Headset (@System32\drivers\bthhfenum.sys Hands-Free ;(PFO05BTSG))"
        }

    monkeypatch.setattr("services.audio_devices.sd.query_devices", query_devices)
    speakers = AudioDevice(
        uid="sd:3",
        name="Alto-falantes (Realtek)",
        kind="saída",
        channels=2,
        sample_rate=48_000,
    )
    headset = AudioDevice(
        uid="sd:18",
        name="Fones de ouvido (PFO05BTSG)",
        kind="saída",
        channels=2,
        sample_rate=48_000,
    )
    assert manager.find_equivalent("sd:41", [speakers, headset]) is headset
    assert manager.find_equivalent("sd:missing", [headset]) is None


def test_playback_opens_a_callback_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: dict[str, object] = {}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            opened.update(kwargs)

        def __enter__(self):
            callback = opened["callback"]
            block = np.zeros((128, int(opened["channels"])), dtype=np.float32)
            while True:
                try:
                    callback(block, block.shape[0], None, None)
                except tts_module.sd.CallbackStop:
                    break
            opened["finished_callback"]()
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr(
        tts_module.sd,
        "query_devices",
        lambda _index: {"default_samplerate": 16_000, "max_output_channels": 2},
    )
    monkeypatch.setattr(tts_module.sd, "check_output_settings", lambda **_kwargs: None)
    monkeypatch.setattr(tts_module.sd, "OutputStream", FakeStream)
    AudioPlayer.play_samples(np.zeros(320, dtype=np.float32), 16_000, "sd:18")
    assert opened["device"] == 18
    assert opened["channels"] == 2
    assert opened["latency"] == "high"
    assert callable(opened["callback"])
    assert callable(opened["finished_callback"])


def test_playback_retries_when_the_headset_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("Error starting stream: WdmSyncIoctl [Windows WDM-KS error -9999]")
            callback = self.kwargs["callback"]
            block = np.zeros((64, int(self.kwargs["channels"])), dtype=np.float32)
            try:
                callback(block, block.shape[0], None, None)
            except tts_module.sd.CallbackStop:
                pass
            self.kwargs["finished_callback"]()
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr(
        tts_module.sd,
        "query_devices",
        lambda _index: {"default_samplerate": 16_000, "max_output_channels": 2},
    )
    monkeypatch.setattr(tts_module.sd, "check_output_settings", lambda **_kwargs: None)
    monkeypatch.setattr(tts_module.sd, "OutputStream", FakeStream)
    monkeypatch.setattr(tts_module.time, "sleep", lambda _seconds: None)
    AudioPlayer.play_samples(np.zeros(160, dtype=np.float32), 16_000, "sd:18")
    assert attempts["count"] == 2


def test_cable_input_is_not_the_headphone_or_the_speaker_end() -> None:
    assert AudioDeviceManager.is_cable_input("CABLE In 16 Ch (2- VB-Audio Virtual Cable)")
    assert not AudioDeviceManager.is_cable_input("Fones de ouvido (PFO05BTSG)")
    assert not AudioDeviceManager.is_cable_input("Alto-falantes (2- VB-Audio Virtual Cable)")
    assert not AudioDeviceManager.is_cable_input("CABLE Output (2- VB-Audio Virtual Cable)")
    assert AudioDeviceManager.is_cable_output("CABLE Output (2- VB-Audio Virtual Cable)")


def test_a_saved_headphone_is_replaced_by_the_wasapi_cable() -> None:
    manager = AudioDeviceManager()
    phones = AudioDevice(
        uid="sd:5",
        name="Fones de ouvido (PFO05BTSG)",
        kind="saída",
        channels=2,
        sample_rate=44_100,
        hostapi="MME",
    )
    mme_cable = AudioDevice(
        uid="sd:6",
        name="CABLE In 16 Ch (2- VB-Audio Virtual Cable)",
        kind="virtual",
        channels=16,
        sample_rate=44_100,
        hostapi="MME",
    )
    wasapi_cable = AudioDevice(
        uid="sd:19",
        name="CABLE In 16 Ch (2- VB-Audio Virtual Cable)",
        kind="virtual",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    cables = manager.prefer_shared_mode([mme_cable, wasapi_cable])
    assert cables == [wasapi_cable]
    assert manager.select_device("cable", [phones, *cables], "sd:5") is wasapi_cable


def test_shared_mode_keeps_one_headset_and_skips_the_mapper() -> None:
    manager = AudioDeviceManager()
    mapper = AudioDevice(
        uid="sd:0",
        name="Mapeador de som da Microsoft - Input",
        kind="microfone",
        channels=2,
        sample_rate=44_100,
        hostapi="MME",
    )
    mme_mic = AudioDevice(
        uid="sd:1",
        name="Grupo de microfones",
        kind="microfone",
        channels=2,
        sample_rate=44_100,
        hostapi="MME",
    )
    wasapi_mic = AudioDevice(
        uid="sd:24",
        name="Grupo de microfones",
        kind="microfone",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    microphones = manager.prefer_shared_mode([mapper, mme_mic, wasapi_mic])
    assert [item.uid for item in microphones] == ["sd:0", "sd:24"]
    assert manager.select_device("microphone", microphones, "") is wasapi_mic


def test_playback_falls_back_to_speakers_when_the_headset_is_listed() -> None:
    manager = AudioDeviceManager()
    phones = AudioDevice(
        uid="sd:18",
        name="Fones de ouvido (PFO05BTSG)",
        kind="saída",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    speakers = AudioDevice(
        uid="sd:21",
        name="Alto-falantes (Realtek)",
        kind="saída",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    cable = AudioDevice(
        uid="sd:19",
        name="CABLE In 16 Ch (VB-Audio Virtual Cable)",
        kind="virtual",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    assert manager.playback_fallbacks("sd:18", [phones, cable, speakers]) == ["sd:21"]


def test_saved_headphone_loopback_yields_to_the_cable() -> None:
    manager = AudioDeviceManager()
    phones = AudioDevice(
        uid="wasapi:4",
        name="Fones de ouvido (PFO05BTSG)",
        kind="loopback",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    cable = AudioDevice(
        uid="wasapi:7",
        name="CABLE Output (VB-Audio Virtual Cable)",
        kind="loopback",
        channels=2,
        sample_rate=48_000,
        hostapi="Windows WASAPI",
    )
    assert manager.select_device("loopback", [phones, cable], "wasapi:4") is cable


def test_device_uid_validation() -> None:
    assert AudioDeviceManager.index_from_uid("sd:12", "sd") == 12
    with pytest.raises(ValueError):
        AudioDeviceManager.index_from_uid("wasapi:2", "sd")
