from __future__ import annotations

import platform
import sys

from services.audio_devices import AudioDeviceManager


def _section(title: str, devices) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not devices:
        print("Nenhum dispositivo encontrado.")
        return
    for device in devices:
        print(
            f"{device.uid:>12} | {device.sample_rate:>6} Hz | "
            f"{device.channels} canais | {device.name}"
        )


def main() -> int:
    print(f"Loon Translator — diagnóstico de áudio")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.platform()}")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print("AVISO: use Python 3.11 ou 3.12 para compatibilidade.")

    manager = AudioDeviceManager()
    _section("Microfones", manager.list_microphones())
    outputs = manager.list_outputs()
    _section("Saídas", outputs)
    loopbacks = manager.list_loopbacks()
    _section("WASAPI loopback", loopbacks)

    cable = manager.find_virtual_cable()
    print("\nResultado")
    print(f"VB-CABLE: {'OK — ' + cable.name if cable else 'NÃO ENCONTRADO'}")
    print(f"WASAPI loopback: {'OK' if loopbacks else 'NÃO ENCONTRADO'}")
    return 0 if cable and loopbacks else 1


if __name__ == "__main__":
    raise SystemExit(main())
