"""Mantém a janela opaca no Windows 11.

O Windows 11 coloca um vidro (Mica/Acrylic) atrás de uma janela com a barra
escura. Ao minimizar e voltar, o DWM recoloca esse vidro: a janela fica
layered, com alpha baixo, e o fundo do CustomTkinter some. Os cliques
continuam na janela. Desligar o vidro, devolver o alpha e redesenhar o
frame traz o fundo sólido de volta.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes

logger = logging.getLogger(__name__)

# DWMWA_SYSTEMBACKDROP_TYPE / DWMSBT_NONE: sem Mica e sem Acrylic.
_DWMWA_SYSTEMBACKDROP_TYPE = 38
_DWMSBT_NONE = 1
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_LWA_ALPHA = 0x00000002
# Sem mover, sem redimensionar, sem ativar; aplica a mudança de estilo.
_SWP_FRAME = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020
# Invalida, apaga, redesenha agora, inclui os filhos e o frame.
_REDRAW = 0x0001 | 0x0004 | 0x0080 | 0x0100 | 0x0200 | 0x0400
_RESTORE_DELAYS_MS = (80, 250, 900)


class _MARGINS(ctypes.Structure):
    _fields_ = (
        ("cxLeftWidth", ctypes.c_int),
        ("cxRightWidth", ctypes.c_int),
        ("cyTopHeight", ctypes.c_int),
        ("cyBottomHeight", ctypes.c_int),
    )


_user32 = None
_dwmapi = None


def _win32():
    global _user32, _dwmapi
    if _user32 is not None:
        return _user32, _dwmapi

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    # No Windows de 32 bits o nome Ptr não existe; o Long comum é o ponteiro.
    getter = user32.GetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user32.GetWindowLongW
    setter = user32.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user32.SetWindowLongW
    getter.argtypes = [wintypes.HWND, ctypes.c_int]
    getter.restype = ctypes.c_ssize_t
    setter.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    setter.restype = ctypes.c_ssize_t
    user32.get_exstyle = getter
    user32.set_exstyle = setter
    user32.SetLayeredWindowAttributes.argtypes = [
        wintypes.HWND,
        wintypes.COLORREF,
        wintypes.BYTE,
        wintypes.DWORD,
    ]
    user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.RedrawWindow.argtypes = [
        wintypes.HWND,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    user32.RedrawWindow.restype = wintypes.BOOL
    dwmapi.DwmSetWindowAttribute.argtypes = [
        wintypes.HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
    dwmapi.DwmExtendFrameIntoClientArea.argtypes = [wintypes.HWND, ctypes.POINTER(_MARGINS)]
    dwmapi.DwmExtendFrameIntoClientArea.restype = ctypes.c_long
    _user32, _dwmapi = user32, dwmapi
    return user32, dwmapi


def _top_level_hwnd(window) -> int:
    user32, _dwmapi = _win32()
    hwnd = wintypes.HWND(int(window.winfo_id()))
    parent = user32.GetParent(hwnd)
    return int(parent or hwnd)


def pin_window_opaque(window) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        user32, dwmapi = _win32()
        hwnd = wintypes.HWND(_top_level_hwnd(window))
        backdrop = ctypes.c_int(_DWMSBT_NONE)
        result = dwmapi.DwmSetWindowAttribute(
            hwnd,
            _DWMWA_SYSTEMBACKDROP_TYPE,
            ctypes.byref(backdrop),
            ctypes.sizeof(backdrop),
        )
        # Margem zero: o frame do DWM não entra na área do cliente.
        dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(_MARGINS(0, 0, 0, 0)))
        style = int(user32.get_exstyle(hwnd, _GWL_EXSTYLE))
        if style & _WS_EX_LAYERED:
            # Alpha cheio antes de tirar o bit, senão o vidro continua visível.
            user32.SetLayeredWindowAttributes(hwnd, 0, 255, _LWA_ALPHA)
            user32.set_exstyle(hwnd, _GWL_EXSTYLE, style & ~_WS_EX_LAYERED)
            user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, _SWP_FRAME)
        user32.RedrawWindow(hwnd, None, None, _REDRAW)
        # O atributo do vidro só existe no Windows 11. No 10 a janela já é opaca.
        from services.platform_host import current_host

        return result == 0 or not current_host().dwm_backdrop
    except Exception:
        logger.debug("janela opaca não aplicada", exc_info=True)
        return False


def bind_opaque(window) -> None:
    """Reaplica a opacidade ao ativar, minimizar e restaurar a janela."""

    last_pin = {"at": 0.0}

    def _pin(_event=None) -> None:
        pin_window_opaque(window)

    def _pin_if_due(_event=None) -> None:
        # Ativar a janela várias vezes seguidas não pode redesenhar o vidro em rajada.
        now = time.monotonic()
        if now - last_pin["at"] < 0.45:
            return
        last_pin["at"] = now
        _pin()

    def _pin_after_restore(_event=None) -> None:
        last_pin["at"] = 0.0
        _pin()
        try:
            for delay in _RESTORE_DELAYS_MS:
                window.after(delay, _pin)
        except Exception:
            pass

    def _wrap(name: str, callback) -> None:
        original = getattr(window, name, None)
        if not callable(original) or getattr(original, "_loon_opaque", False):
            return

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            callback()
            return result

        wrapped._loon_opaque = True
        setattr(window, name, wrapped)

    try:
        window.bind("<Activate>", _pin_if_due, add="+")
        window.bind("<Map>", _pin_after_restore, add="+")
        window.bind("<Unmap>", _pin_if_due, add="+")
        _wrap("deiconify", _pin_after_restore)
        _wrap("wm_deiconify", _pin_after_restore)
        _wrap("iconify", _pin)
        _wrap("wm_iconify", _pin)
        # A barra escura do CustomTkinter é aplicada depois do primeiro frame
        # e o Windows recoloca o vidro em seguida. O mesmo atraso vale depois
        # de restaurar (Map / deiconify), porque o DWM chega atrasado.
        window.after(250, _pin)
        window.after(900, _pin)
    except Exception:
        logger.debug("não foi possível vigiar a opacidade", exc_info=True)
