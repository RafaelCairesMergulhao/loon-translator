"""Cabo virtual do Loon: uma saída e um microfone com o mesmo nome.

No Linux o cabo é um módulo do PulseAudio ou do PipeWire, criado quando o
programa abre e removido quando fecha. Só existe enquanto a cópia licenciada
está aberta. No Windows e no macOS um cabo que outros programas enxergam é um
driver assinado do sistema; sem essa assinatura o sistema recusa o aparelho,
então aqui não se finge que ele foi instalado.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

from services.platform_host import current_host

logger = logging.getLogger(__name__)

SINK_NAME = "loon_translator"
SOURCE_NAME = "loon_translator_mic"
DEVICE_NAME = "Loon Translator"


@dataclass(frozen=True, slots=True)
class CableSession:
    created: bool
    name: str
    detail: str


class VirtualCable:
    def __init__(self, runner=None) -> None:
        self._runner = runner or _run
        self._loaded: list[str] = []

    def open(self) -> CableSession:
        family = current_host().family
        if family == "linux":
            return self._open_linux()
        if family == "windows":
            return CableSession(False, DEVICE_NAME, "windows_signed_driver")
        if family == "macos":
            return CableSession(False, DEVICE_NAME, "macos_signed_driver")
        return CableSession(False, DEVICE_NAME, "unsupported")

    def close(self) -> None:
        for module_id in reversed(self._loaded):
            result = self._runner(["pactl", "unload-module", module_id])
            if result.returncode != 0:
                logger.info("Não foi possível soltar o cabo %s: %s", module_id, result.stderr.strip())
        self._loaded.clear()

    def _open_linux(self) -> CableSession:
        if shutil.which("pactl") is None and self._runner is _run:
            return CableSession(False, DEVICE_NAME, "no_pactl")
        sinks = self._runner(["pactl", "list", "short", "sinks"])
        if sinks.returncode != 0:
            return CableSession(False, DEVICE_NAME, sinks.stderr.strip() or "pactl_failed")
        if SINK_NAME not in sinks.stdout:
            loaded = self._load(
                "module-null-sink",
                f"sink_name={SINK_NAME}",
                'sink_properties=device.description="Loon Translator"',
            )
            if loaded is None:
                return CableSession(False, DEVICE_NAME, "pactl_failed")
        sources = self._runner(["pactl", "list", "short", "sources"])
        if sources.returncode != 0:
            return CableSession(False, DEVICE_NAME, sources.stderr.strip() or "pactl_failed")
        if SOURCE_NAME not in sources.stdout:
            loaded = self._load(
                "module-remap-source",
                f"master={SINK_NAME}.monitor",
                f"source_name={SOURCE_NAME}",
                'source_properties=device.description="Loon Translator Microphone"',
            )
            if loaded is None:
                return CableSession(False, DEVICE_NAME, "pactl_failed")
        return CableSession(True, DEVICE_NAME, "pulse")

    def _load(self, module: str, *arguments: str) -> str | None:
        result = self._runner(["pactl", "load-module", module, *arguments])
        module_id = result.stdout.strip()
        if result.returncode != 0 or not module_id.isdigit():
            logger.info("Cabo virtual não criado: %s", result.stderr.strip() or result.stdout.strip())
            return None
        self._loaded.append(module_id)
        return module_id


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, timeout=5)
