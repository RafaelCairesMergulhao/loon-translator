"""Inicialização COM para streams WASAPI do PortAudio em threads do Windows."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

COINIT_MULTITHREADED = 0x0
COINIT_APARTMENTTHREADED = 0x2
RPC_E_CHANGED_MODE = 0x80010106


def _ole32() -> Any:
    import ctypes

    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    return ole32


def _hresult(value: int) -> int:
    import ctypes

    return int(ctypes.c_long(value & 0xFFFFFFFF).value)


@contextmanager
def com_apartment(mode: int = COINIT_MULTITHREADED) -> Iterator[int]:
    """MTA no thread atual. Fora do Windows não faz nada.

    S_OK (0) e S_FALSE (1) exigem CoUninitialize. RPC_E_CHANGED_MODE (thread já
    STA) deixa o apartment como está: CoUninitialize quebraria o chamador.
    """
    if os.name != "nt":
        yield 0
        return
    ole32 = _ole32()
    hr = _hresult(int(ole32.CoInitializeEx(None, mode)))
    try:
        yield hr
    finally:
        if hr >= 0:
            ole32.CoUninitialize()


def com_guard(mode: int = COINIT_MULTITHREADED):
    """Atalho para ``with com_apartment():`` ou no-op fora do Windows."""
    return com_apartment(mode) if os.name == "nt" else nullcontext(0)
