from services.audio_devices import AudioDevice, AudioDeviceManager
from services.platform_host import HostProfile
from services.virtual_cable import SOURCE_NAME, VirtualCable


class _Result:
    def __init__(self, code: int, out: str = "", err: str = "") -> None:
        self.returncode = code
        self.stdout = out
        self.stderr = err


def _host(family: str) -> HostProfile:
    return HostProfile(family, family, "1", True, family, "monitor", False)


def test_linux_creates_a_speaker_and_a_microphone_and_removes_them(monkeypatch) -> None:
    monkeypatch.setattr("services.virtual_cable.current_host", lambda: _host("linux"))
    calls: list[list[str]] = []

    def runner(args: list[str]) -> _Result:
        calls.append(args)
        if args[1:3] == ["list", "short"]:
            return _Result(0, "")
        if args[1] == "load-module":
            return _Result(0, "42\n" if "module-null-sink" in args else "43\n")
        return _Result(0)

    cable = VirtualCable(runner)
    session = cable.open()
    assert session.created
    assert session.detail == "pulse"
    assert any("module-null-sink" in part for call in calls for part in call)
    assert any(SOURCE_NAME in part for call in calls for part in call)
    cable.close()
    assert ["pactl", "unload-module", "43"] in calls
    assert ["pactl", "unload-module", "42"] in calls


def test_an_existing_linux_cable_is_left_in_place(monkeypatch) -> None:
    monkeypatch.setattr("services.virtual_cable.current_host", lambda: _host("linux"))

    def runner(args: list[str]) -> _Result:
        if "sinks" in args:
            return _Result(0, "loon_translator\n")
        if "sources" in args:
            return _Result(0, "loon_translator_mic\n")
        raise AssertionError(args)

    cable = VirtualCable(runner)
    session = cable.open()
    assert session.created
    assert cable._loaded == []


def test_windows_and_mac_do_not_pretend_the_cable_exists(monkeypatch) -> None:
    monkeypatch.setattr("services.virtual_cable.current_host", lambda: _host("windows"))
    assert VirtualCable(lambda _args: _Result(0)).open().detail == "windows_signed_driver"
    monkeypatch.setattr("services.virtual_cable.current_host", lambda: _host("macos"))
    assert VirtualCable(lambda _args: _Result(0)).open().detail == "macos_signed_driver"


def test_the_loon_output_is_preferred_over_another_virtual_cable() -> None:
    manager = AudioDeviceManager()
    other = AudioDevice("sd:1", "CABLE Input", "virtual", 2, 48_000, "Windows WASAPI")
    ours = AudioDevice("sd:2", "Loon Translator", "virtual", 2, 48_000, "Windows WASAPI")
    assert manager.select_device("cable", [other, ours], "") is ours
    assert AudioDeviceManager.is_cable_input("Loon Translator")
    assert not AudioDeviceManager.is_cable_input("Loon Translator Microphone")
    assert AudioDeviceManager.is_monitor_source("Loon Translator Microphone")
    assert AudioDeviceManager.is_virtual_output("Loon Translator")
