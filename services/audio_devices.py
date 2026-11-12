from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import sounddevice as sd

from domain.sorting import shell_sort
from services.platform_host import current_host, host_rank

logger = logging.getLogger(__name__)

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # Permite abrir a interface antes de instalar as dependências Windows.
    pyaudio = None


@dataclass(frozen=True, slots=True)
class AudioDevice:
    uid: str
    name: str
    kind: str
    channels: int
    sample_rate: int
    hostapi: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} [{self.kind}]"


class AudioDeviceManager:
    @staticmethod
    def include_hostapi(name: str) -> bool:
        """WDM-KS não aceita reprodução bloqueante e o DirectSound só duplica o WASAPI."""
        normalized = name.lower()
        return "wdm-ks" not in normalized and "directsound" not in normalized

    def _allowed_hostapis(self) -> set[int] | None:
        try:
            apis = sd.query_hostapis()
        except Exception:
            return None
        allowed = {
            index
            for index, api in enumerate(apis)
            if self.include_hostapi(str(api["name"]))
        }
        return allowed or None

    def list_microphones(self) -> list[AudioDevice]:
        allowed = self._allowed_hostapis()
        names = self._hostapi_names()
        result: list[AudioDevice] = []
        for index, device in enumerate(sd.query_devices()):
            host_index = int(device["hostapi"])
            if allowed is not None and host_index not in allowed:
                continue
            channels = int(device["max_input_channels"])
            name = str(device["name"])
            if channels > 0 and not self.is_monitor_source(name):
                result.append(
                    AudioDevice(
                        uid=f"sd:{index}",
                        name=str(device["name"]),
                        kind="microfone",
                        channels=channels,
                        sample_rate=int(device["default_samplerate"]),
                        hostapi=names.get(host_index, ""),
                    )
                )
        return self.prefer_shared_mode(result)

    def list_outputs(self) -> list[AudioDevice]:
        allowed = self._allowed_hostapis()
        names = self._hostapi_names()
        result: list[AudioDevice] = []
        for index, device in enumerate(sd.query_devices()):
            host_index = int(device["hostapi"])
            if allowed is not None and host_index not in allowed:
                continue
            channels = int(device["max_output_channels"])
            if channels > 0:
                name = str(device["name"])
                kind = "virtual" if self.is_virtual_output(name) else "saída"
                result.append(
                    AudioDevice(
                        uid=f"sd:{index}",
                        name=name,
                        kind=kind,
                        channels=channels,
                        sample_rate=int(device["default_samplerate"]),
                        hostapi=names.get(host_index, ""),
                    )
                )
        return self.prefer_shared_mode(result)

    def list_loopbacks(self) -> list[AudioDevice]:
        """No Windows o retorno é o WASAPI. No Mac e no Linux é um monitor."""
        if current_host().loopback == "wasapi":
            return self._wasapi_loopbacks()
        return self._monitor_inputs()

    def _wasapi_loopbacks(self) -> list[AudioDevice]:
        if pyaudio is None:
            return []
        result: list[AudioDevice] = []
        try:
            with pyaudio.PyAudio() as audio:
                for device in audio.get_loopback_device_info_generator():
                    result.append(
                        AudioDevice(
                            uid=f"wasapi:{int(device['index'])}",
                            name=str(device["name"]),
                            kind="loopback",
                            channels=max(1, int(device["maxInputChannels"])),
                            sample_rate=int(device["defaultSampleRate"]),
                            hostapi="Windows WASAPI",
                        )
                    )
        except OSError as exc:
            logger.warning("WASAPI loopback indisponível: %s", exc)
        return result

    def _monitor_inputs(self) -> list[AudioDevice]:
        names = self._hostapi_names()
        result: list[AudioDevice] = []
        try:
            devices = sd.query_devices()
        except Exception as exc:
            logger.warning("Monitores de áudio indisponíveis: %s", exc)
            return []
        for index, device in enumerate(devices):
            name = str(device["name"])
            if int(device["max_input_channels"]) <= 0 or not self.is_monitor_source(name):
                continue
            result.append(
                AudioDevice(
                    uid=f"sd:{index}",
                    name=name,
                    kind="loopback",
                    channels=int(device["max_input_channels"]),
                    sample_rate=int(device["default_samplerate"]),
                    hostapi=names.get(int(device["hostapi"]), ""),
                )
            )
        return result

    def playback_fallbacks(self, preferred_uid: str, devices: list[AudioDevice]) -> list[str]:
        """Outras saídas reais para quando o fone recusa a gravação."""
        physical = [
            item
            for item in self._wasapi_first(devices)
            if not self.is_virtual_output(item.name) and not self.is_mapper(item.name)
        ]
        ordered: list[str] = []
        for device in physical:
            if device.uid != preferred_uid and device.uid not in ordered:
                ordered.append(device.uid)
        return ordered

    def find_virtual_cable(self) -> AudioDevice | None:
        return self.select_device("cable", self.list_outputs(), "")

    def _hostapi_names(self) -> dict[int, str]:
        try:
            return {index: str(api["name"]) for index, api in enumerate(sd.query_hostapis())}
        except Exception:
            return {}

    @staticmethod
    def prefer_shared_mode(devices: list[AudioDevice]) -> list[AudioDevice]:
        """Quando o mesmo aparelho aparece no MME e no WASAPI, fica o WASAPI."""
        grouped: dict[str, list[AudioDevice]] = {}
        order: list[str] = []
        for device in devices:
            key = " ".join(device.name.split()).casefold()
            if key not in grouped:
                order.append(key)
                grouped[key] = []
            grouped[key].append(device)
        chosen: list[AudioDevice] = []
        for key in order:
            group = grouped[key]
            native = [item for item in group if host_rank(item.hostapi) == 0]
            chosen.append(native[0] if native else group[0])
        return chosen

    def select_device(
        self,
        role: str,
        devices: list[AudioDevice],
        preferred_uid: str,
    ) -> AudioDevice | None:
        """Mantém a escolha da pessoa e, se ela apontar para o aparelho errado, corrige."""
        if preferred_uid:
            found = self.find_equivalent(preferred_uid, devices)
            if found is not None and self._saved_choice_ok(role, found):
                return found
        ranked = self._wasapi_first(devices)
        if role == "cable":
            candidates = [item for item in ranked if self.is_cable_input(item.name)]
            ours = [item for item in candidates if "loon translator" in item.name.casefold()]
            return (ours or candidates or [None])[0]
        if role == "phones":
            physical = [
                item
                for item in ranked
                if not self.is_virtual_output(item.name) and not self.is_mapper(item.name)
            ]
            if physical:
                return physical[0]
            return next((item for item in ranked if not self.is_virtual_output(item.name)), None)
        if role == "microphone":
            real = [item for item in ranked if not self.is_mapper(item.name)]
            if real:
                return real[0]
            return ranked[0] if ranked else None
        if role == "loopback":
            cable = next((item for item in devices if self.is_cable_output(item.name)), None)
            return cable or (devices[0] if devices else None)
        return ranked[0] if ranked else None

    @staticmethod
    def _wasapi_first(devices: list[AudioDevice]) -> list[AudioDevice]:
        return shell_sort(
            devices,
            key=lambda item: (host_rank(item.hostapi), item.name.casefold(), item.uid),
        )

    def _saved_choice_ok(self, role: str, device: AudioDevice) -> bool:
        if role == "cable":
            return self.is_cable_input(device.name)
        if role == "phones":
            return not self.is_virtual_output(device.name)
        if role == "loopback":
            return self.is_cable_output(device.name)
        return True

    @staticmethod
    def is_cable_input(name: str) -> bool:
        normalized = name.casefold()
        if "cable output" in normalized or "cable out" in normalized:
            return False
        # "CABLE In 16ch" é outro ponto. O microfone da chamada é o CABLE Output,
        # e ele só recebe o que entra no CABLE Input estéreo.
        if "16" in normalized and "cable" in normalized:
            return False
        if "cable input" in normalized or "cable in" in normalized:
            return True
        if "blackhole" in normalized or "soundflower" in normalized:
            return True
        if "loon translator" in normalized and "microphone" not in normalized and "monitor" not in normalized:
            return True
        return "voicemeeter" in normalized and "input" in normalized and "output" not in normalized

    @staticmethod
    def is_cable_output(name: str) -> bool:
        return AudioDeviceManager.is_monitor_source(name)

    @staticmethod
    def is_monitor_source(name: str) -> bool:
        """Onde o áudio da chamada entra: cabo, BlackHole ou monitor do sistema."""
        normalized = name.casefold()
        if "cable input" in normalized or "cable in" in normalized:
            return False
        return any(
            token in normalized
            for token in (
                "cable output",
                "cable out",
                "blackhole",
                "soundflower",
                "monitor of",
                "loopback",
                ".monitor",
                "loon translator microphone",
            )
        )

    @staticmethod
    def shares_endpoint(loopback_name: str, output_name: str) -> bool:
        """O loopback ouve o que o Loon toca no mesmo aparelho; em outro aparelho não há eco.

        O MME corta nomes em 31 caracteres, por isso basta um ser prefixo do outro.
        Na dúvida (nome vazio ou mapeador do Windows), considera o mesmo aparelho.
        """

        def clean(name: str) -> str:
            name = re.sub(r"\[\s*loopback\s*\]", "", name, flags=re.IGNORECASE)
            return " ".join(name.casefold().split())

        loopback, output = clean(loopback_name), clean(output_name)
        if not loopback or not output or AudioDeviceManager.is_mapper(output_name):
            return True
        return loopback.startswith(output) or output.startswith(loopback)

    @staticmethod
    def cable_feeds_loopback(loopback_name: str, output_name: str) -> bool:
        """O CABLE Output ouve o CABLE Input mesmo com nomes diferentes."""
        if AudioDeviceManager.shares_endpoint(loopback_name, output_name):
            return True
        return AudioDeviceManager.is_cable_output(loopback_name) and AudioDeviceManager.is_cable_input(
            output_name
        )

    @staticmethod
    def is_mapper(name: str) -> bool:
        normalized = name.casefold()
        return any(
            token in normalized
            for token in ("mapeador", "mapper", "primary sound", "som primário", "som primario")
        )

    @staticmethod
    def is_virtual_output(name: str) -> bool:
        return AudioDeviceManager._is_virtual(name)

    def find_equivalent(self, preferred_uid: str, devices: list[AudioDevice]) -> AudioDevice | None:
        """Reencontra um aparelho salvo mesmo quando o índice antigo era WDM-KS."""
        direct = next((item for item in devices if item.uid == preferred_uid), None)
        if direct is not None or not preferred_uid.startswith("sd:"):
            return direct
        try:
            info = sd.query_devices(self.index_from_uid(preferred_uid, "sd"))
        except (OSError, ValueError):
            return None
        wanted = self._identity_tokens(str(info["name"]))
        best: AudioDevice | None = None
        best_score = 0
        for device in devices:
            score = len(wanted & self._identity_tokens(device.name))
            if score > best_score:
                best = device
                best_score = score
        return best

    @staticmethod
    def _identity_tokens(name: str) -> set[str]:
        ignored = {
            "system32",
            "drivers",
            "headset",
            "headphones",
            "headphone",
            "microphone",
            "microfone",
            "speakers",
            "speaker",
            "fones",
            "ouvido",
            "audio",
            "input",
            "output",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]{5,}", name.lower())
            if token not in ignored
        }

    @staticmethod
    def index_from_uid(uid: str, expected_backend: str) -> int:
        backend, separator, raw_index = uid.partition(":")
        if not separator or backend != expected_backend:
            raise ValueError(f"Dispositivo inválido para {expected_backend}: {uid}")
        return int(raw_index)

    @staticmethod
    def _is_virtual(name: str) -> bool:
        normalized = name.lower()
        return any(
            token in normalized
            for token in ("cable", "virtual", "voicemeeter", "sonar", "blackhole", "soundflower", "loon translator")
        )
