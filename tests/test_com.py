from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pytest

import services.tts_engine as tts_module
from services.tts_engine import AudioPlayer
from services.win_com import RPC_E_CHANGED_MODE, com_apartment


def test_com_apartment_is_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.win_com.os.name", "posix")
    with com_apartment() as hr:
        assert hr == 0


@pytest.mark.skipif(os.name != "nt", reason="COM só existe no Windows")
def test_com_apartment_uninitializes_on_s_ok_and_s_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    class FakeOle:
        def CoInitializeEx(self, _pointer, mode: int) -> int:
            calls.append(("init", mode))
            return 0 if len([item for item in calls if item[0] == "init"]) == 1 else 1

        def CoUninitialize(self) -> None:
            calls.append(("uninit",))

    monkeypatch.setattr("services.win_com._ole32", lambda: FakeOle())
    with com_apartment() as first:
        assert first == 0
        with com_apartment() as second:
            assert second == 1
    assert calls == [("init", 0), ("init", 0), ("uninit",), ("uninit",)]


@pytest.mark.skipif(os.name != "nt", reason="COM só existe no Windows")
def test_com_apartment_skips_uninit_when_thread_is_already_sta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeOle:
        def CoInitializeEx(self, _pointer, _mode: int) -> int:
            calls.append("init")
            return RPC_E_CHANGED_MODE

        def CoUninitialize(self) -> None:
            calls.append("uninit")

    monkeypatch.setattr("services.win_com._ole32", lambda: FakeOle())
    with com_apartment() as hr:
        assert hr < 0
        assert (hr & 0xFFFFFFFF) == RPC_E_CHANGED_MODE
    assert calls == ["init"]


@pytest.mark.skipif(os.name != "nt", reason="COM só existe no Windows")
def test_com_apartment_nested_mta_on_this_thread() -> None:
    with com_apartment() as first:
        assert first >= 0 or (first & 0xFFFFFFFF) == RPC_E_CHANGED_MODE
        with com_apartment() as second:
            if first >= 0:
                assert second == 1


def test_playback_enters_com_apartment(monkeypatch: pytest.MonkeyPatch) -> None:
    entered: list[bool] = []

    @contextmanager
    def fake_com(_mode: int = 0):
        entered.append(True)
        yield 0

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self):
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

    monkeypatch.setattr("services.tts_engine.com_apartment", fake_com)
    monkeypatch.setattr(
        tts_module.sd,
        "query_devices",
        lambda _index: {"default_samplerate": 16_000, "max_output_channels": 2},
    )
    monkeypatch.setattr(tts_module.sd, "check_output_settings", lambda **_kwargs: None)
    monkeypatch.setattr(tts_module.sd, "OutputStream", FakeStream)
    AudioPlayer.play_samples(np.zeros(160, dtype=np.float32), 16_000, "sd:18")
    assert entered
