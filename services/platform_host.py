"""O que muda de um sistema para outro fica aqui.

O restante do programa pede um microfone, uma saída virtual e o áudio da
chamada. Este módulo diz como isso se chama em cada sistema e qual versão
ainda consegue rodar o Loon. Versões anteriores ao piso sobem a janela, mas
não prometem o áudio da chamada: o Python 3.11 e o PortAudio não cabem nelas.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostProfile:
    family: str
    label: str
    version: str
    supported: bool
    """Piso em que microfone, saída e áudio da chamada foram pensados."""
    floor: str
    loopback: str
    """``wasapi`` no Windows, ``monitor`` no Mac e no Linux."""
    dwm_backdrop: bool


def current_host() -> HostProfile:
    system = sys.platform
    version = platform.version() or platform.release()
    if system.startswith("win"):
        build = _windows_build(version)
        return HostProfile(
            family="windows",
            label="Windows",
            version=version,
            supported=build >= 10240,
            floor="Windows 10",
            loopback="wasapi",
            dwm_backdrop=build >= 22000,
        )
    if system == "darwin":
        release = platform.mac_ver()[0] or platform.release()
        major = _leading_int(release)
        return HostProfile(
            family="macos",
            label="macOS",
            version=release,
            supported=major >= 12,
            floor="macOS 12",
            loopback="monitor",
            dwm_backdrop=False,
        )
    if system.startswith("linux"):
        return HostProfile(
            family="linux",
            label="Linux",
            version=platform.release(),
            supported=True,
            floor="Linux com PulseAudio ou PipeWire",
            loopback="monitor",
            dwm_backdrop=False,
        )
    return HostProfile(
        family="other",
        label=platform.system() or "sistema",
        version=version,
        supported=False,
        floor="Windows 10, macOS 12 ou Linux",
        loopback="monitor",
        dwm_backdrop=False,
    )


def host_rank(hostapi: str) -> int:
    """O driver nativo fica na frente da cópia antiga do mesmo aparelho."""
    name = hostapi.casefold()
    if any(token in name for token in ("wasapi", "core audio", "pulse", "pipewire")):
        return 0
    return 1


def machine_identity() -> str:
    """Identificador estável desta máquina, sem o nome que a pessoa escolheu."""
    override = os.getenv("LOON_MACHINE_ID", "").strip()
    if override:
        return override
    host = current_host()
    if host.family == "windows":
        guid = _windows_machine_guid()
        if guid:
            return guid
    elif host.family == "linux":
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                value = open(path, encoding="utf-8").read().strip()
            except OSError:
                continue
            if value:
                return value
    elif host.family == "macos":
        uuid = _mac_platform_uuid()
        if uuid:
            return uuid
    return platform.node()


def _windows_build(version: str) -> int:
    parts = version.split(".")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return 0


def _leading_int(text: str) -> int:
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else 0


def _windows_machine_guid() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _kind = winreg.QueryValueEx(key, "MachineGuid")
    except OSError:
        return ""
    return str(value).strip()


def _mac_platform_uuid() -> str:
    try:
        completed = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in completed.stdout.splitlines():
        if "IOPlatformUUID" not in line:
            continue
        _key, _separator, value = line.partition("=")
        return value.strip().strip('"')
    return ""
