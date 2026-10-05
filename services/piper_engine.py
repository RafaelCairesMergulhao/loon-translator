"""Piper: voz neural local (ONNX) para quando o Edge não estiver disponível.

O binário oficial do Windows (~21 MB) e as vozes (~60 MB cada) são baixados
uma vez para ``%APPDATA%\\LoonTranslator\\piper``. No app instalado o binário
já vem na pasta do programa; só as vozes faltantes são baixadas.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import httpx

from core.config import APP_DIR
from services.pitch_shift import shift_pitch

logger = logging.getLogger(__name__)

BINARY_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
HF_VOICE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Caminho relativo no repositório Hugging Face → nome do arquivo .onnx
_VOICES: dict[str, dict[str, str]] = {
    "Português": {
        "Feminina": "pt/pt_BR/faber/medium/pt_BR-faber-medium",
        "Masculina": "pt/pt_BR/faber/medium/pt_BR-faber-medium",
    },
    "Inglês": {
        "Feminina": "en/en_US/amy/medium/en_US-amy-medium",
        "Masculina": "en/en_US/lessac/medium/en_US-lessac-medium",
    },
    "Espanhol": {
        "Feminina": "es/es_ES/sharvard/medium/es_ES-sharvard-medium",
        "Masculina": "es/es_ES/davefx/medium/es_ES-davefx-medium",
    },
    "Espanhol (Latino)": {
        "Feminina": "es/es_MX/ald/medium/es_MX-ald-medium",
        "Masculina": "es/es_MX/ald/medium/es_MX-ald-medium",
    },
    "Francês": {
        "Feminina": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
        "Masculina": "fr/fr_FR/upmc/medium/fr_FR-upmc-medium",
    },
    "Alemão": {
        "Feminina": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
        "Masculina": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    },
    "Italiano": {
        "Feminina": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
        "Masculina": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
    },
    "Árabe": {
        "Feminina": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
        "Masculina": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
    },
    "Chinês (Mandarim)": {
        "Feminina": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
        "Masculina": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
    },
    "Russo": {
        "Feminina": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
        "Masculina": "ru/ru_RU/denis/medium/ru_RU-denis-medium",
    },
}


class PiperError(RuntimeError):
    pass


def bundled_root() -> Path | None:
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "piper")
    candidates.append(Path(__file__).resolve().parent.parent / "vendor" / "piper")
    for folder in candidates:
        if find_executable(folder) is not None:
            return folder
    return None


def find_executable(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    direct = folder / "piper.exe" if os.name == "nt" else folder / "piper"
    if direct.exists():
        return direct
    matches = list(folder.rglob("piper.exe" if os.name == "nt" else "piper"))
    return matches[0] if matches else None


class PiperEngine:
    """Garante o binário e a voz, depois sintetiza WAV."""

    def __init__(self, root: Path | None = None) -> None:
        self.user_root = root or APP_DIR / "piper"
        self._lock = threading.Lock()

    def available_for(self, language_name: str) -> bool:
        return language_name in _VOICES or bool(os.getenv("PIPER_MODEL", "").strip())

    def synthesize(
        self,
        text: str,
        language_name: str,
        gender: str,
        output: Path,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> Path:
        if output.exists() and output.stat().st_size > 0:
            return output
        executable = self.ensure_binary()
        model = self.ensure_model(language_name, gender)
        speed = max(0.5, min(2.0, float(speed)))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    "--model",
                    str(model),
                    "--output_file",
                    str(output.resolve()),
                    "--length_scale",
                    f"{1.0 / speed:.3f}",
                    "--sentence_silence",
                    "0.18",
                    "--noise_scale",
                    "0.667",
                    "--noise_w",
                    "0.8",
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=45,
                check=False,
                cwd=str(executable.parent),
                creationflags=flags,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            output.unlink(missing_ok=True)
            raise PiperError(f"Piper não respondeu: {exc}") from exc
        if completed.returncode != 0 or not output.exists() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or b"").decode("utf-8", errors="replace").strip()
            raise PiperError(f"Piper falhou: {detail[:240] or completed.returncode}")
        if abs(pitch_semitones) >= 0.05:
            self._apply_pitch(output, pitch_semitones)
        return output

    def ensure_binary(self) -> Path:
        override = shutil.which("piper")
        if override:
            return Path(override)
        runtime = self.user_root / "runtime"
        with self._lock:
            found = find_executable(runtime)
            if found:
                return found
            bundled = bundled_root()
            source = find_executable(bundled).parent if bundled else None
            runtime.mkdir(parents=True, exist_ok=True)
            if source is not None:
                # O Piper 2023 quebra em caminhos com acento (OneDrive «Área de Trabalho»).
                shutil.copytree(source, runtime, dirs_exist_ok=True)
                found = find_executable(runtime)
                if found:
                    return found
            archive = self.user_root / "piper_windows_amd64.zip"
            logger.info("Baixando o Piper para voz local...")
            _download(BINARY_URL, archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(runtime)
            archive.unlink(missing_ok=True)
            found = find_executable(runtime)
            if found is None:
                raise PiperError("O pacote do Piper não trouxe piper.exe")
            return found

    def ensure_model(self, language_name: str, gender: str) -> Path:
        custom = os.getenv("PIPER_MODEL", "").strip()
        if custom:
            path = Path(custom)
            if not path.exists():
                raise PiperError(f"PIPER_MODEL não encontrado: {path}")
            return path
        voices = _VOICES.get(language_name)
        if not voices:
            raise PiperError(f"Piper ainda não tem voz para {language_name}")
        relative = voices.get(gender) or next(iter(voices.values()))
        name = relative.rsplit("/", 1)[-1]
        voices_dir = self.user_root / "voices"
        name_onnx = f"{name}.onnx"
        search = [voices_dir]
        bundled = bundled_root()
        if bundled:
            search.append(bundled / "voices")
        for folder in search:
            candidate = folder / name_onnx
            config = candidate.with_suffix(".onnx.json")
            if candidate.exists() and config.exists():
                if folder != voices_dir:
                    voices_dir.mkdir(parents=True, exist_ok=True)
                    copied = voices_dir / name_onnx
                    if not copied.exists():
                        shutil.copy2(candidate, copied)
                        shutil.copy2(config, copied.with_suffix(".onnx.json"))
                    return copied
                return candidate
        with self._lock:
            voices_dir.mkdir(parents=True, exist_ok=True)
            onnx = voices_dir / f"{name}.onnx"
            config = voices_dir / f"{name}.onnx.json"
            if onnx.exists() and config.exists():
                return onnx
            logger.info("Baixando voz Piper %s...", name)
            _download(f"{HF_VOICE}/{relative}.onnx", onnx)
            _download(f"{HF_VOICE}/{relative}.onnx.json", config)
            return onnx

    @staticmethod
    def _apply_pitch(path: Path, semitones: float) -> None:
        import numpy as np
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        shifted = shift_pitch(data, semitones)
        sf.write(path, np.asarray(shifted, dtype="float32"), rate)


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 256):
                        handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
