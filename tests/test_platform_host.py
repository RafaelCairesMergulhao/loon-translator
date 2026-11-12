import services.platform_host as host_module
from services.audio_devices import AudioDevice, AudioDeviceManager
from services.platform_host import HostProfile, current_host, host_rank, machine_identity


def test_this_machine_has_a_known_family() -> None:
    host = current_host()
    assert host.family in {"windows", "macos", "linux", "other"}
    assert host.floor
    assert host.loopback in {"wasapi", "monitor"}


def test_native_audio_host_outranks_the_legacy_copy() -> None:
    assert host_rank("Windows WASAPI") == 0
    assert host_rank("Core Audio") == 0
    assert host_rank("PulseAudio") == 0
    assert host_rank("PipeWire") == 0
    assert host_rank("MME") == 1
    assert host_rank("ALSA") == 1


def test_windows_10_is_supported_and_windows_11_has_the_glass_fix(monkeypatch) -> None:
    monkeypatch.setattr(host_module.sys, "platform", "win32")
    monkeypatch.setattr(host_module.platform, "version", lambda: "10.0.19045")
    windows_10 = current_host()
    assert windows_10.family == "windows"
    assert windows_10.supported
    assert windows_10.loopback == "wasapi"
    assert windows_10.dwm_backdrop is False
    monkeypatch.setattr(host_module.platform, "version", lambda: "10.0.22631")
    assert current_host().dwm_backdrop is True


def test_old_windows_is_recognized_but_not_promised(monkeypatch) -> None:
    monkeypatch.setattr(host_module.sys, "platform", "win32")
    monkeypatch.setattr(host_module.platform, "version", lambda: "6.1.7601")
    host = current_host()
    assert host.family == "windows"
    assert host.supported is False
    assert host.floor == "Windows 10"


def test_macos_and_linux_listen_on_a_monitor(monkeypatch) -> None:
    monkeypatch.setattr(host_module.sys, "platform", "darwin")
    monkeypatch.setattr(host_module.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
    mac = current_host()
    assert mac.family == "macos"
    assert mac.supported
    assert mac.loopback == "monitor"
    assert mac.dwm_backdrop is False
    monkeypatch.setattr(host_module.sys, "platform", "linux")
    monkeypatch.setattr(host_module.platform, "release", lambda: "6.8.0")
    linux = current_host()
    assert linux.family == "linux"
    assert linux.loopback == "monitor"


def test_call_audio_on_mac_and_linux_is_the_monitor(monkeypatch) -> None:
    monkeypatch.setattr(
        host_module,
        "current_host",
        lambda: HostProfile("linux", "Linux", "6", True, "Linux", "monitor", False),
    )
    monkeypatch.setattr("services.audio_devices.current_host", host_module.current_host)
    monkeypatch.setattr(
        "services.audio_devices.sd.query_devices",
        lambda: [
            {"name": "Microfone da webcam", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000, "hostapi": 0},
            {"name": "Monitor of Speakers", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000, "hostapi": 0},
            {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 48000, "hostapi": 0},
        ],
    )
    monkeypatch.setattr("services.audio_devices.sd.query_hostapis", lambda: [{"name": "PulseAudio"}])
    manager = AudioDeviceManager()
    loopbacks = manager.list_loopbacks()
    assert [item.name for item in loopbacks] == ["Monitor of Speakers", "BlackHole 2ch"]
    microphones = manager.list_microphones()
    assert [item.name for item in microphones] == ["Microfone da webcam"]
    assert AudioDeviceManager.is_cable_input("BlackHole 2ch")
    assert manager.select_device("loopback", loopbacks, "") == loopbacks[0]


def test_machine_identity_prefers_the_override(monkeypatch) -> None:
    monkeypatch.setenv("LOON_MACHINE_ID", "bancada")
    assert machine_identity() == "bancada"


def test_core_audio_wins_when_the_same_device_is_listed_twice() -> None:
    legacy = AudioDevice("sd:1", "Fones", "saída", 2, 48_000, "ALSA")
    native = AudioDevice("sd:2", "Fones", "saída", 2, 48_000, "PulseAudio")
    assert AudioDeviceManager.prefer_shared_mode([legacy, native]) == [native]
